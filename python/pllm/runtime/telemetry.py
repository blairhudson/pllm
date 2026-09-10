from __future__ import annotations

import os
import threading
from typing import Any

_lock = threading.Lock()
_configured = False
_protocol_bytes: Any = None
_protocol_operations: Any = None
_meter_provider: Any = None
_trace_provider: Any = None


def configure_telemetry(service_name: str | None = None, *, shutdown_on_exit: bool = True) -> bool:
    """Enable OTLP telemetry only when an exporter endpoint is configured."""
    global _configured, _meter_provider, _protocol_bytes, _protocol_operations, _trace_provider
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    if not endpoint:
        return False
    with _lock:
        if _configured:
            return True
        from opentelemetry import metrics, trace
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
        from opentelemetry.instrumentation.system_metrics import SystemMetricsInstrumentor
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.semconv.resource import ResourceAttributes

        name = service_name or os.getenv("OTEL_SERVICE_NAME", "pllm")
        resource = Resource.create({ResourceAttributes.SERVICE_NAME: name})
        base = endpoint.rstrip("/")
        interval = max(250, int(os.getenv("OTEL_METRIC_EXPORT_INTERVAL", "500")))
        meter_provider = MeterProvider(
            resource=resource,
            shutdown_on_exit=shutdown_on_exit,
            metric_readers=[
                PeriodicExportingMetricReader(
                    OTLPMetricExporter(endpoint=f"{base}/v1/metrics", timeout=2),
                    export_interval_millis=interval,
                )
            ],
        )
        metrics.set_meter_provider(meter_provider)
        trace_provider = TracerProvider(resource=resource, shutdown_on_exit=shutdown_on_exit)
        trace_provider.add_span_processor(
            BatchSpanProcessor(
                OTLPSpanExporter(endpoint=f"{base}/v1/traces", timeout=2),
                schedule_delay_millis=200,
            )
        )
        trace.set_tracer_provider(trace_provider)
        meter = metrics.get_meter("pllm.protocol")
        _protocol_bytes = meter.create_counter(
            "pllm.protocol.bytes", unit="By", description="Bytes crossing a PLLM trust boundary"
        )
        _protocol_operations = meter.create_counter(
            "pllm.protocol.operations", description="Prepared linear protocol operations"
        )
        HTTPXClientInstrumentor().instrument()
        SystemMetricsInstrumentor(
            config={
                "process.cpu.time": ["user", "system"],
                "process.cpu.utilization": None,
                "process.memory.usage": None,
                "process.memory.virtual": None,
                "process.thread.count": None,
            }
        ).instrument()
        _meter_provider = meter_provider
        _trace_provider = trace_provider
        _configured = True
        return True


def shutdown_telemetry() -> None:
    """Flush configured providers while their collector is still available."""
    if _meter_provider is not None:
        _meter_provider.shutdown()
    if _trace_provider is not None:
        _trace_provider.shutdown()


def instrument_fastapi(app: Any) -> None:
    if not _configured:
        return
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

    FastAPIInstrumentor.instrument_app(app)


def record_protocol_bytes(source: str, destination: str, amount: int, stage: str) -> None:
    if _protocol_bytes is None or amount <= 0:
        return
    attributes = {"source": source, "destination": destination, "stage": stage}
    _protocol_bytes.add(amount, attributes)


def record_protocol_operation(stage: str) -> None:
    if _protocol_operations is not None:
        _protocol_operations.add(1, {"stage": stage})


def start_protocol_span(
    stage: str,
    preparation_bytes: int,
    inference_bytes: int,
    *,
    phase: str = "online",
) -> Any:
    from opentelemetry import trace

    return trace.get_tracer("pllm.protocol").start_span(
        "pllm.prepared_linear",
        attributes={
            "pllm.stage": stage,
            "pllm.phase": phase,
            "pllm.client_preparation.bytes": preparation_bytes,
            "pllm.client_inference.bytes": inference_bytes,
        },
    )
