import asyncio
import threading
from collections import deque
from pathlib import Path
from types import SimpleNamespace

from opentelemetry.proto.collector.metrics.v1.metrics_service_pb2 import (
    ExportMetricsServiceRequest,
)
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest

from pllm._cli.app import build_parser
from pllm.runtime.dashboard import DashboardConfig, DashboardRuntime, OTelStore, _http_origin


def _resource_attribute(resource, key: str, value: str) -> None:
    attribute = resource.attributes.add()
    attribute.key = key
    attribute.value.string_value = value


def _integer_attribute(resource, key: str, value: int) -> None:
    attribute = resource.attributes.add()
    attribute.key = key
    attribute.value.int_value = value


def test_otel_store_aggregates_process_and_protocol_metrics() -> None:
    request = ExportMetricsServiceRequest()
    resource = request.resource_metrics.add()
    _resource_attribute(resource.resource, "service.name", "pllm-client")
    scope = resource.scope_metrics.add()

    cpu = scope.metrics.add()
    cpu.name = "process.cpu.utilization"
    cpu.gauge.data_points.add().as_double = 0.42
    memory = scope.metrics.add()
    memory.name = "process.memory.usage"
    memory.gauge.data_points.add().as_int = 64 * 1024 * 1024
    cpu_time = scope.metrics.add()
    cpu_time.name = "process.cpu.time"
    for kind, value in (("user", 8.5), ("system", 4.0)):
        point = cpu_time.sum.data_points.add()
        point.as_double = value
        _resource_attribute(point, "type", kind)
    traffic = scope.metrics.add()
    traffic.name = "pllm.protocol.bytes"
    point = traffic.sum.data_points.add()
    point.as_int = 4096
    _resource_attribute(point, "source", "client")
    _resource_attribute(point, "destination", "inference")

    store = OTelStore()
    store.ingest_metrics(request.SerializeToString())
    snapshot = store.snapshot()
    assert snapshot["services"]["pllm-client"]["cpu_percent"] == 42
    assert snapshot["services"]["pllm-client"]["cpu_seconds"] == 12.5
    assert snapshot["services"]["pllm-client"]["memory_bytes"] == 64 * 1024 * 1024
    assert snapshot["traffic"]["client->inference"] == 4096


def test_otel_store_keeps_bounded_span_details() -> None:
    request = ExportTraceServiceRequest()
    resource = request.resource_spans.add()
    _resource_attribute(resource.resource, "service.name", "pllm-client")
    span = resource.scope_spans.add().spans.add()
    span.name = "pllm.prepared_linear"
    span.start_time_unix_nano = 1_000_000
    span.end_time_unix_nano = 3_500_000
    _resource_attribute(span, "pllm.stage", "layers.0.self_attn.qkv_proj")
    _integer_attribute(span, "pllm.client_inference.bytes", 4096)
    store = OTelStore()
    store.ingest_traces(request.SerializeToString())
    expected = {
        "service": "pllm-client",
        "name": "pllm.prepared_linear",
        "duration_ms": 2.5,
        "status": 0,
        "method": None,
        "route": None,
        "stage": "layers.0.self_attn.qkv_proj",
        "phase": "online",
        "flows": {"client_inference": 4096},
        "start": 0.001,
        "time": 0.0035,
    }
    snapshot = store.snapshot()
    assert snapshot["spans"][-1] == expected
    assert snapshot["protocol_spans"][-1] == {**expected, "sequence": 1}
    assert snapshot["protocol_cursor"] == 1
    assert store.snapshot(protocol_after=1)["protocol_spans"] == []


def test_otel_store_reports_protocol_cursor_gaps() -> None:
    request = ExportTraceServiceRequest()
    resource = request.resource_spans.add()
    _resource_attribute(resource.resource, "service.name", "pllm-client")
    span = resource.scope_spans.add().spans.add()
    span.name = "pllm.prepared_linear"
    span.start_time_unix_nano = 1_000_000
    span.end_time_unix_nano = 2_000_000
    _integer_attribute(span, "pllm.client_inference.bytes", 1)
    store = OTelStore()
    store._protocol_spans = deque(maxlen=1)
    store.ingest_traces(request.SerializeToString())
    store.ingest_traces(request.SerializeToString())

    snapshot = store.snapshot(protocol_after=0)

    assert snapshot["protocol_truncated"] is True
    assert snapshot["protocol_oldest_sequence"] == 2


def test_dashboard_origin_formats_ipv6() -> None:
    assert _http_origin("::1", 8791) == "http://[::1]:8791"


def test_otel_run_window_falls_back_to_local_client_process_samples() -> None:
    store = OTelStore()
    store.begin_run_window("run_missing")

    metrics = store.finish_run_window("run_missing")

    assert metrics["client"]["cpu_seconds"] is not None
    assert metrics["preparation"]["cpu_seconds"] is None
    assert metrics["inference"]["cpu_seconds"] is None


def test_dashboard_assets_are_packaged_beside_python_package() -> None:
    assets = Path(__file__).parents[1] / "python" / "pllm" / "dashboard"
    assert {path.name for path in assets.iterdir()} == {"app.js", "index.html", "style.css"}
    assert "LIVE NETWORK" in (assets / "index.html").read_text()
    script = (assets / "app.js").read_text()
    assert "renderSequence" in script
    assert "if (cpuMetric)" in script
    assert "slice(-6)" not in script
    assert 'flowRow("preparation", "inference"' in script
    assert 'max="512"' in (assets / "index.html").read_text()


def test_dashboard_defaults_to_real_qwen_and_keeps_tiny_explicit() -> None:
    parser = build_parser()
    default = parser.parse_args(["dev", "dashboard"])
    tiny = parser.parse_args(["dev", "dashboard", "--tiny"])
    assert default.model == "Qwen/Qwen2.5-0.5B-Instruct"
    assert default.tiny is False
    assert default.model_id is None
    assert tiny.tiny is True


def test_dashboard_launches_internal_runtime_services(monkeypatch) -> None:
    runtime = DashboardRuntime(
        DashboardConfig(
            host="127.0.0.1",
            port=7777,
            model_path="/tmp/model",
            model_id="model",
            default_max_output_tokens=8,
        ),
        OTelStore(),
    )
    captured = {}

    class Topology:
        inference_url = "http://127.0.0.1:9101"
        preparation_url = "http://127.0.0.1:9102"
        requires_preparation = True

        def start(self):
            captured["started"] = True
            return self

        @staticmethod
        def client(**kwargs):
            captured["client"] = kwargs
            return SimpleNamespace()

    def roles(model, **kwargs):
        captured["model"] = model
        captured["roles"] = kwargs
        return Topology()

    async def stop_before_preparation(*_args, **_kwargs) -> None:
        raise RuntimeError("stop test startup")

    monkeypatch.setattr("pllm.runtime.dashboard.build_roles", roles)
    monkeypatch.setattr(runtime, "_background_call", stop_before_preparation)

    asyncio.run(runtime.start())

    assert captured["started"] is True
    assert captured["model"].source == "/tmp/model"
    assert captured["model"].kind == "huggingface"
    assert captured["roles"]["model_id"] == "model"
    assert captured["roles"]["credential_prefix"] == "dash"
    assert captured["client"]["prepared_inventory_rows"] == 64
    runtime._temporary.cleanup()


def test_dashboard_snapshot_uses_cached_inventory_without_client_io() -> None:
    class Client:
        privacy_audit = None

        @staticmethod
        def prepared_inventory_status(_model: str) -> dict[str, object]:
            raise AssertionError("snapshot must not perform runtime I/O")

    runtime = object.__new__(DashboardRuntime)
    runtime._lock = threading.Lock()
    runtime._state = {
        "started_at": None,
        "first_token_at": None,
        "finished_at": None,
        "tokens": 0,
        "inventory": {"status": "unprepared", "capacity": 0, "available": 0},
    }
    runtime._topology = None
    runtime._client = Client()
    runtime.config = SimpleNamespace(model_id="model")
    runtime.store = OTelStore()
    runtime._inventory_burned_total = 3
    runtime._inventory_consumed_total = 5
    runtime._online_traffic_final = None
    runtime._online_traffic_baseline = {}
    runtime._preparation_online_cpu_final = None
    runtime._preparation_cpu_baseline = None
    runtime._preparation_online_operations = 0

    inventory = runtime.snapshot()["run"]["inventory"]
    assert inventory["burned"] == 3
    assert inventory["consumed"] == 5


def test_completed_dashboard_run_does_not_eagerly_refill(monkeypatch) -> None:
    class Client:
        privacy_audit = SimpleNamespace(preparation_attempts=0)

        def __init__(self) -> None:
            self.preprocess_calls = 0
            self.responses = SimpleNamespace(
                create=lambda **_kwargs: [
                    SimpleNamespace(type="response.output_text.delta", delta="token"),
                    SimpleNamespace(
                        type="response.completed",
                        response=SimpleNamespace(
                            usage=SimpleNamespace(input_tokens=1, output_tokens=1)
                        ),
                    ),
                ]
            )

        def preprocess(self, *_args, **_kwargs) -> None:
            self.preprocess_calls += 1

        @staticmethod
        def prepared_inventory_status(_model: str) -> dict[str, object]:
            return {
                "id": "inventory",
                "status": "ready",
                "capacity": 64,
                "available": 63,
                "reserved": 0,
                "burned": 0,
                "consumed": 1,
            }

    client = Client()
    runtime = object.__new__(DashboardRuntime)
    runtime._lock = threading.Lock()
    runtime._state = {
        "phase": "online",
        "text": "",
        "tokens": 0,
        "first_token_at": None,
        "finished_at": None,
    }
    runtime._client = client
    runtime._topology = SimpleNamespace(
        started=True,
        closed=False,
        requires_preparation=True,
        statuses=(SimpleNamespace(running=True),),
    )
    runtime.config = SimpleNamespace(model_id="model")
    runtime.store = SimpleNamespace(
        snapshot=lambda: {
            "traffic": {},
            "services": {"pllm-preparation": {"cpu_seconds": 0.0}},
        }
    )
    runtime._online_traffic_baseline = {}
    runtime._preparation_cpu_baseline = 0.0
    runtime._preparation_attempts_baseline = 0
    runtime._online_traffic_final = None
    runtime._preparation_online_cpu_final = None
    runtime._preparation_online_operations = 0
    monkeypatch.setattr("pllm.runtime.dashboard.time.sleep", lambda _seconds: None)

    runtime._run_chat("private prompt", 1)

    assert client.preprocess_calls == 0
    assert runtime._state["phase"] == "ready", runtime._state.get("error")
    assert runtime._state["inventory"]["available"] == 63


def test_incomplete_dashboard_stream_is_not_reported_as_success() -> None:
    runtime = DashboardRuntime(
        SimpleNamespace(model_id="model", model_path=None, default_max_output_tokens=8),
        OTelStore(),
    )
    runtime._client = SimpleNamespace(
        privacy_audit=None,
        prepared_inventory_status=lambda _model: {},
        responses=SimpleNamespace(
            create=lambda **_kwargs: [
                SimpleNamespace(type="response.output_text.delta", delta="partial")
            ]
        ),
    )

    runtime._run_chat("prompt", 1)

    assert runtime._state["phase"] == "error"
    assert "terminal response" in runtime._state["error"]


def test_dashboard_stop_does_not_wait_forever_for_client_close() -> None:
    release = threading.Event()
    runtime = DashboardRuntime(
        SimpleNamespace(model_id="model", model_path=None, default_max_output_tokens=8),
        OTelStore(),
    )
    runtime._CLIENT_CLOSE_TIMEOUT_SECONDS = 0.01
    runtime._BACKGROUND_JOIN_TIMEOUT_SECONDS = 0.01
    runtime._client = SimpleNamespace(close=lambda: release.wait())

    asyncio.run(asyncio.wait_for(runtime.stop(), timeout=0.5))

    threads = list(runtime._background_threads)
    assert threads and all(thread.daemon for thread in threads)
    release.set()
    for thread in threads:
        thread.join(timeout=0.5)
