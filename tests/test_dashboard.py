from pathlib import Path

from opentelemetry.proto.collector.metrics.v1.metrics_service_pb2 import (
    ExportMetricsServiceRequest,
)
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest

from pllm.cli import _build_parser
from pllm.runtime.dashboard import OTelStore


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
        "flows": {"client_inference": 4096},
        "start": 0.001,
        "time": 0.0035,
    }
    snapshot = store.snapshot()
    assert snapshot["spans"][-1] == expected
    assert snapshot["protocol_spans"][-1] == expected


def test_dashboard_assets_are_packaged_beside_python_package() -> None:
    assets = Path(__file__).parents[1] / "python" / "pllm" / "dashboard"
    assert {path.name for path in assets.iterdir()} == {"app.js", "index.html", "style.css"}
    assert "LIVE NETWORK" in (assets / "index.html").read_text()
    assert "renderSequence" in (assets / "app.js").read_text()


def test_dashboard_defaults_to_real_qwen_and_keeps_tiny_explicit() -> None:
    parser = _build_parser()
    default = parser.parse_args(["benchmark", "dashboard"])
    tiny = parser.parse_args(["benchmark", "dashboard", "--tiny"])
    assert default.model == "Qwen/Qwen2.5-0.5B-Instruct"
    assert default.tiny is False
    assert default.model_id is None
    assert tiny.tiny is True
