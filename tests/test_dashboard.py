import threading
from pathlib import Path
from types import SimpleNamespace

from opentelemetry.proto.collector.metrics.v1.metrics_service_pb2 import (
    ExportMetricsServiceRequest,
)
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest

from pllm.cli import _build_parser
from pllm.runtime.dashboard import DashboardRuntime, OTelStore


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
    parser = _build_parser()
    default = parser.parse_args(["benchmark", "dashboard"])
    tiny = parser.parse_args(["benchmark", "dashboard", "--tiny"])
    assert default.model == "Qwen/Qwen2.5-0.5B-Instruct"
    assert default.tiny is False
    assert default.model_id is None
    assert tiny.tiny is True


def test_dashboard_snapshot_normalizes_unprepared_inventory() -> None:
    class Client:
        privacy_audit = None

        @staticmethod
        def prepared_inventory_status(_model: str) -> dict[str, object]:
            return {"status": "unprepared", "capacity": 0, "available": 0}

    runtime = object.__new__(DashboardRuntime)
    runtime._lock = threading.Lock()
    runtime._state = {
        "started_at": None,
        "first_token_at": None,
        "finished_at": None,
        "tokens": 0,
    }
    runtime._processes = {}
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
                    SimpleNamespace(type="response.output_text.delta", delta="token")
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
