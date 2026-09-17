from __future__ import annotations

import asyncio
import json
import os
import re
import secrets
import socket
import subprocess
import sys
import tempfile
import threading
import time
import webbrowser
from collections import defaultdict, deque
from contextlib import asynccontextmanager, suppress
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, cast
from urllib.parse import urlsplit

import httpx
import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, Response
from google.protobuf.message import DecodeError
from opentelemetry.proto.collector.metrics.v1.metrics_service_pb2 import (
    ExportMetricsServiceRequest,
    ExportMetricsServiceResponse,
)
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
    ExportTraceServiceRequest,
    ExportTraceServiceResponse,
)
from safetensors.numpy import save_file

from pllm.runtime.client import OpenAI
from pllm.runtime.benchmark_history import (
    MAX_LIMIT,
    SCHEMA_VERSION,
    BenchmarkHistory,
    BenchmarkRun,
)


_RUN_ID = re.compile(r"[A-Za-z0-9_.:-]{1,128}")


def _http_origin(host: str, port: int) -> str:
    rendered_host = f"[{host}]" if ":" in host else host
    return f"http://{rendered_host}:{port}"


def _create_demo_checkpoint(path: Path) -> Path:
    path.mkdir(parents=True)
    hidden, intermediate, heads, kv_heads, head_dim, vocab = 32, 64, 4, 2, 8, 258
    config = {
        "architectures": ["Gemma4ForCausalLM"],
        "model_type": "gemma4_text",
        "name_or_path": "pllm-otel-demo",
        "vocab_size": vocab,
        "hidden_size": hidden,
        "intermediate_size": intermediate,
        "num_hidden_layers": 1,
        "num_attention_heads": heads,
        "num_key_value_heads": kv_heads,
        "head_dim": head_dim,
        "max_position_embeddings": 1024,
        "sliding_window": 64,
        "layer_types": ["full_attention"],
        "hidden_activation": "silu",
        "rms_norm_eps": 1e-6,
        "tie_word_embeddings": True,
        "bos_token_id": 0,
        "eos_token_id": 1,
        "pad_token_id": 1,
        "rope_parameters": {"full_attention": {"rope_type": "default", "rope_theta": 10000.0}},
        "pllm_test_tokenizer": "byte",
    }
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (path / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "bos_token": "<bos>",
                "eos_token": "<eos>",
                "model_max_length": 1024,
                "chat_template": (
                    "{% for message in messages %}{{ message['role'] }}: "
                    "{{ message['content'] }}\\n{% endfor %}assistant: "
                ),
            }
        ),
        encoding="utf-8",
    )
    rng = np.random.default_rng(17)

    def matrix(rows: int, columns: int, scale: float = 0.08) -> np.ndarray:
        return (rng.standard_normal((rows, columns), dtype=np.float32) * scale).astype(np.float32)

    prefix = "model.layers.0"
    tensors = {
        "model.embed_tokens.weight": matrix(vocab, hidden, 0.12),
        "model.norm.weight": np.ones(hidden, dtype=np.float32),
        f"{prefix}.self_attn.q_proj.weight": matrix(heads * head_dim, hidden),
        f"{prefix}.self_attn.k_proj.weight": matrix(kv_heads * head_dim, hidden),
        f"{prefix}.self_attn.v_proj.weight": matrix(kv_heads * head_dim, hidden),
        f"{prefix}.self_attn.o_proj.weight": matrix(hidden, heads * head_dim),
        f"{prefix}.self_attn.q_norm.weight": np.ones(head_dim, dtype=np.float32),
        f"{prefix}.self_attn.k_norm.weight": np.ones(head_dim, dtype=np.float32),
        f"{prefix}.mlp.gate_proj.weight": matrix(intermediate, hidden),
        f"{prefix}.mlp.up_proj.weight": matrix(intermediate, hidden),
        f"{prefix}.mlp.down_proj.weight": matrix(hidden, intermediate),
        f"{prefix}.input_layernorm.weight": np.ones(hidden, dtype=np.float32),
        f"{prefix}.post_attention_layernorm.weight": np.ones(hidden, dtype=np.float32),
        f"{prefix}.pre_feedforward_layernorm.weight": np.ones(hidden, dtype=np.float32),
        f"{prefix}.post_feedforward_layernorm.weight": np.ones(hidden, dtype=np.float32),
    }
    save_file(tensors, path / "model.safetensors", metadata={"format": "pt"})
    return path


def _attribute_value(value: Any) -> Any:
    kind = value.WhichOneof("value")
    return getattr(value, kind) if kind else None


def _attributes(items: Any) -> dict[str, Any]:
    return {item.key: _attribute_value(item.value) for item in items}


def _point_value(point: Any) -> float:
    kind = point.WhichOneof("value")
    return float(getattr(point, kind)) if kind else 0.0


class OTelStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._metrics: dict[str, dict[tuple[str, tuple[tuple[str, str], ...]], float]] = (
            defaultdict(dict)
        )
        self._history: dict[str, deque[dict[str, float]]] = defaultdict(lambda: deque(maxlen=180))
        self._spans: deque[dict[str, Any]] = deque(maxlen=300)
        self._protocol_spans: deque[dict[str, Any]] = deque(maxlen=65_536)
        self._protocol_sequence = 0
        self._run_windows: dict[str, dict[str, dict[str, float | None]]] = {}

    def _has_metric(self, service: str, name: str) -> bool:
        return any(metric_name == name for metric_name, _attrs in self._metrics.get(service, {}))

    def ingest_metrics(self, payload: bytes) -> None:
        message = ExportMetricsServiceRequest.FromString(payload)
        now = time.time()
        with self._lock:
            touched: set[str] = set()
            for resource_metrics in message.resource_metrics:
                resource = _attributes(resource_metrics.resource.attributes)
                service = str(resource.get("service.name", "unknown"))
                touched.add(service)
                for scope in resource_metrics.scope_metrics:
                    for metric in scope.metrics:
                        data_kind = metric.WhichOneof("data")
                        if not data_kind:
                            continue
                        data = getattr(metric, data_kind)
                        for point in data.data_points:
                            attrs = tuple(
                                sorted(
                                    (key, str(value))
                                    for key, value in _attributes(point.attributes).items()
                                )
                            )
                            if data_kind == "histogram":
                                value = float(point.sum)
                            else:
                                value = _point_value(point)
                            self._metrics[service][(metric.name, attrs)] = value
            for service in touched:
                values = self._service_values(service)
                if values["cpu_percent"] is not None and values["memory_bytes"] is not None:
                    self._history[service].append(
                        {
                            "time": now,
                            "cpu": values["cpu_percent"],
                            "memory": values["memory_bytes"],
                        }
                    )
                for window in self._run_windows.values():
                    memory = values["memory_bytes"]
                    if memory is not None:
                        peaks = window["rss_peaks"]
                        previous = peaks.get(service)
                        peaks[service] = memory if previous is None else max(previous, memory)

    def ingest_traces(self, payload: bytes) -> None:
        message = ExportTraceServiceRequest.FromString(payload)
        with self._lock:
            for resource_spans in message.resource_spans:
                resource = _attributes(resource_spans.resource.attributes)
                service = str(resource.get("service.name", "unknown"))
                for scope in resource_spans.scope_spans:
                    for span in scope.spans:
                        attrs = _attributes(span.attributes)
                        flow_bytes = {
                            key.removeprefix("pllm.").removesuffix(".bytes"): int(value)
                            for key, value in attrs.items()
                            if key.startswith("pllm.")
                            and key.endswith(".bytes")
                            and isinstance(value, int)
                        }
                        item = {
                            "service": service,
                            "name": span.name,
                            "duration_ms": max(
                                0.0, (span.end_time_unix_nano - span.start_time_unix_nano) / 1e6
                            ),
                            "status": int(span.status.code),
                            "method": attrs.get("http.request.method", attrs.get("http.method")),
                            "route": attrs.get("http.route", attrs.get("url.path")),
                            "stage": attrs.get("pllm.stage"),
                            "phase": attrs.get("pllm.phase", "online"),
                            "flows": flow_bytes,
                            "start": span.start_time_unix_nano / 1e9,
                            "time": span.end_time_unix_nano / 1e9,
                        }
                        self._spans.append(item)
                        if span.name == "pllm.prepared_linear":
                            self._protocol_sequence += 1
                            self._protocol_spans.append(
                                {**item, "sequence": self._protocol_sequence}
                            )

    def _sum_metric(self, service: str, name: str, **attributes: str) -> float:
        total = 0.0
        for (metric_name, attrs), value in self._metrics.get(service, {}).items():
            if metric_name != name:
                continue
            current = dict(attrs)
            if all(current.get(key) == expected for key, expected in attributes.items()):
                total += value
        return total

    def _service_values(self, service: str) -> dict[str, float | None]:
        return {
            "cpu_percent": self._sum_metric(service, "process.cpu.utilization") * 100.0
            if self._has_metric(service, "process.cpu.utilization")
            else None,
            "cpu_seconds": self._sum_metric(service, "process.cpu.time")
            if self._has_metric(service, "process.cpu.time")
            else None,
            "memory_bytes": self._sum_metric(service, "process.memory.usage")
            if self._has_metric(service, "process.memory.usage")
            else None,
            "virtual_memory_bytes": self._sum_metric(service, "process.memory.virtual")
            if self._has_metric(service, "process.memory.virtual")
            else None,
            "threads": self._sum_metric(service, "process.thread.count")
            if self._has_metric(service, "process.thread.count")
            else None,
        }

    def snapshot(self, protocol_after: int = 0) -> dict[str, Any]:
        with self._lock:
            services = {
                name: {**self._service_values(name), "history": list(self._history.get(name, ()))}
                for name in ("pllm-client", "pllm-preparation", "pllm-inference")
            }
            traffic: dict[str, float] = defaultdict(float)
            operations = 0.0
            for metrics in self._metrics.values():
                for (name, attrs), value in metrics.items():
                    labels = dict(attrs)
                    if name == "pllm.protocol.bytes":
                        traffic[
                            f"{labels.get('source', '?')}->{labels.get('destination', '?')}"
                        ] += value
                    elif name == "pllm.protocol.operations":
                        operations += value
            oldest_protocol_sequence = (
                int(self._protocol_spans[0]["sequence"])
                if self._protocol_spans
                else self._protocol_sequence + 1
            )
            return {
                "services": services,
                "traffic": dict(traffic),
                "operations": int(operations),
                "spans": list(self._spans)[-30:],
                "protocol_spans": [
                    span for span in self._protocol_spans if int(span["sequence"]) > protocol_after
                ],
                "protocol_cursor": self._protocol_sequence,
                "protocol_oldest_sequence": oldest_protocol_sequence,
                "protocol_truncated": protocol_after < oldest_protocol_sequence - 1,
            }

    def protocol_cursor(self) -> int:
        with self._lock:
            return self._protocol_sequence

    def begin_run_window(self, run_id: str) -> None:
        with self._lock:
            if run_id in self._run_windows:
                raise RuntimeError("OTel run window already exists")
            services = ("pllm-client", "pllm-preparation", "pllm-inference")
            self._run_windows[run_id] = {
                "cpu_baselines": {
                    service: self._service_values(service)["cpu_seconds"] for service in services
                },
                "rss_peaks": {
                    service: self._service_values(service)["memory_bytes"] for service in services
                },
            }

    def finish_run_window(self, run_id: str) -> dict[str, dict[str, float | int | None]]:
        with self._lock:
            window = self._run_windows.pop(run_id, None)
            if window is None:
                return {}
            result: dict[str, dict[str, float | int | None]] = {}
            for role, service in (
                ("client", "pllm-client"),
                ("preparation", "pllm-preparation"),
                ("inference", "pllm-inference"),
            ):
                values = self._service_values(service)
                baseline_cpu = window["cpu_baselines"].get(service)
                current_cpu = values["cpu_seconds"]
                cpu_delta = (
                    max(0.0, current_cpu - baseline_cpu)
                    if current_cpu is not None and baseline_cpu is not None
                    else None
                )
                rss_values = [
                    value
                    for value in (window["rss_peaks"].get(service), values["memory_bytes"])
                    if value is not None
                ]
                result[role] = {
                    "cpu_seconds": cpu_delta,
                    "rss_peak_bytes": int(max(rss_values)) if rss_values else None,
                }
            return result


@dataclass(frozen=True)
class DashboardConfig:
    host: str
    port: int
    model_path: Path | str | None
    model_id: str
    default_max_output_tokens: int
    history_path: Path | str | None = None
    startup_inventory_rows: int | None = None
    experiment: Any | None = None
    otel_token: str = field(default_factory=lambda: secrets.token_urlsafe(32))


@dataclass(slots=True)
class _RunCapture:
    run_id: str
    max_output_tokens: int
    cold: bool
    started_at_ns: int
    audit_before: dict[str, int]
    started_monotonic_ns: int
    protocol_start_cursor: int
    preparation_started_at_ns: int | None = None
    preparation_finished_at_ns: int | None = None
    online_started_at_ns: int | None = None
    first_token_at_ns: int | None = None
    last_token_at_ns: int | None = None
    finished_at_ns: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    token_usage_authoritative: bool = False
    inventory_required: int = 0
    inventory_generated: int = 0
    inventory_reused: int = 0
    inventory_consumed_before: int = 0
    inventory_burned_before: int = 0
    process_metrics: dict[str, dict[str, float | int | None]] = field(default_factory=dict)
    preparation_started_monotonic_ns: int | None = None
    preparation_finished_monotonic_ns: int | None = None
    online_started_monotonic_ns: int | None = None
    first_token_monotonic_ns: int | None = None
    last_token_monotonic_ns: int | None = None
    finished_monotonic_ns: int | None = None


class _IncompleteResponseError(RuntimeError):
    pass


def _free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


class DashboardRuntime:
    _CLIENT_CLOSE_TIMEOUT_SECONDS = 2.0
    _PROCESS_EXIT_TIMEOUT_SECONDS = 2.0
    _WORKER_JOIN_TIMEOUT_SECONDS = 5.0
    _BACKGROUND_JOIN_TIMEOUT_SECONDS = 1.0

    def __init__(
        self,
        config: DashboardConfig,
        store: OTelStore,
        history: BenchmarkHistory | None = None,
    ) -> None:
        self.config = config
        self.store = store
        self.history = history
        self._lock = threading.Lock()
        self._temporary: tempfile.TemporaryDirectory[str] | None = None
        self._processes: dict[str, subprocess.Popen[bytes]] = {}
        self._logs: dict[str, Any] = {}
        self._log_paths: dict[str, Path] = {}
        self._client: OpenAI | None = None
        self._state: dict[str, Any] = {
            "phase": "starting",
            "startup_step": "dashboard",
            "model_id": config.model_id,
            "tiny": config.model_path is None,
            "prompt": "Explain why neither server can see the prompt.",
            "text": "",
            "tokens": 0,
            "run_id": None,
            "started_at": None,
            "first_token_at": None,
            "last_token_at": None,
            "finished_at": None,
            "error": None,
            "inventory": {},
            "online_traffic": {},
            "preparation_online_cpu_seconds": 0.0,
            "configured_max_output_tokens": config.default_max_output_tokens,
            "protocol_start_cursor": 0,
        }
        self._inventory_rows = getattr(config, "startup_inventory_rows", None) or (
            256 if config.model_path is None else 64
        )
        self._online_traffic_baseline: dict[str, float] = {}
        self._preparation_cpu_baseline: float | None = None
        self._online_traffic_final: dict[str, float] | None = None
        self._preparation_online_cpu_final: float | None = None
        self._preparation_attempts_baseline = 0
        self._preparation_online_operations = 0
        self._inventory_burned_total = 0
        self._inventory_consumed_total = 0
        self._worker: threading.Thread | None = None
        self._active_run: _RunCapture | None = None
        self._has_started_run = False
        self._model_fingerprint: str | None = None
        self._stopping = threading.Event()
        self._stop_started = False
        self._background_threads: set[threading.Thread] = set()
        self._last_privacy_delta: dict[str, int] = {}

    def _services_healthy(self) -> bool:
        processes = getattr(self, "_processes", {})
        return self._client is not None and (
            not processes
            or all(
                (process := processes.get(role)) is not None and process.poll() is None
                for role in ("preparation", "inference")
            )
        )

    async def _background_call(
        self, callback: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Any:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[Any] = loop.create_future()

        def deliver(result: Any, error: BaseException | None) -> None:
            if future.done():
                return
            if error is None:
                future.set_result(result)
            else:
                future.set_exception(error)

        def invoke() -> None:
            result: Any = None
            error: BaseException | None = None
            try:
                result = callback(*args, **kwargs)
            except BaseException as exc:
                error = exc
            finally:
                with self._lock:
                    self._background_threads.discard(thread)
            with suppress(RuntimeError):
                loop.call_soon_threadsafe(deliver, result, error)

        thread = threading.Thread(target=invoke, name="pllm-dashboard-call", daemon=True)
        with self._lock:
            self._background_threads.add(thread)
        thread.start()
        return await future

    def _set(self, **values: Any) -> None:
        with self._lock:
            self._state.update(values)

    def _audit_snapshot(self) -> dict[str, int]:
        client = self._client
        audit = client.privacy_audit if client is not None else None
        if audit is None:
            return {}
        if hasattr(audit, "to_dict"):
            values = audit.to_dict()
        else:
            try:
                values = asdict(audit)
            except TypeError:
                values = vars(audit)
        return {
            str(key): int(value) for key, value in values.items() if isinstance(value, (bool, int))
        }

    def _inventory_totals(self, inventory: dict[str, Any]) -> tuple[int, int]:
        return (
            int(inventory.get("consumed", 0)) + self._inventory_consumed_total,
            int(inventory.get("burned", 0)) + self._inventory_burned_total,
        )

    def _discover_model_fingerprint(self) -> str | None:
        if self._client is None:
            return None
        try:
            models = self._client.models.list().get("data", [])
        except Exception:
            return None
        for model in models:
            if not isinstance(model, dict) or model.get("id") != self.config.model_id:
                continue
            raw_runtime = model.get("runtime")
            runtime: dict[str, Any] = raw_runtime if isinstance(raw_runtime, dict) else {}
            value = (
                model.get("fingerprint")
                or runtime.get("fingerprint")
                or runtime.get("body_fingerprint")
            )
            if value is not None:
                fingerprint = str(value)
                if (
                    fingerprint
                    and len(fingerprint) <= 256
                    and not any(ord(char) < 32 for char in fingerprint)
                ):
                    return fingerprint
        return None

    async def start(self) -> None:
        try:
            self._temporary = tempfile.TemporaryDirectory(prefix="pllm-dashboard-")
            root = Path(self._temporary.name)
            model = self.config.model_path
            if model is None:
                model = _create_demo_checkpoint(root / "model")
            inference_port, preparation_port = _free_port(), _free_port()
            inference_url = f"http://127.0.0.1:{inference_port}"
            preparation_url = f"http://127.0.0.1:{preparation_port}"
            inference_key, preparation_key, push_key = (
                "dash_" + secrets.token_urlsafe(24),
                "dash_" + secrets.token_urlsafe(24),
                "dash_" + secrets.token_urlsafe(24),
            )
            common = [sys.executable, "-m", "pllm.runtime.cli"]
            inference = [
                *common,
                "inference",
                "--model",
                str(model),
                "--model-id",
                self.config.model_id,
                "--privacy-mode",
                "public",
                "--host",
                "127.0.0.1",
                "--port",
                str(inference_port),
                "--api-key",
                inference_key,
                "--provider-push-api-key",
                push_key,
                "--rendezvous-capacity",
                "131072",
                "--rendezvous-max-bytes",
                "1073741824",
            ]
            preparation = [
                *common,
                "preparation",
                "--model",
                str(model),
                "--model-id",
                self.config.model_id,
                "--host",
                "127.0.0.1",
                "--port",
                str(preparation_port),
                "--api-key",
                preparation_key,
                "--inference-url",
                inference_url,
                "--push-api-key",
                push_key,
            ]
            if self.config.experiment is not None:
                kernels = self.config.experiment.pipeline.components.get("kernels")
                if kernels is not None and kernels.component == "pllm/cpu":
                    threads = kernels.params.get("threads")
                    if threads is not None:
                        thread_args = ["--engine-threads", str(threads)]
                        inference.extend(thread_args)
                        preparation.extend(thread_args)
            if self._stopping.is_set():
                return
            self._set(startup_step="inference")
            self._spawn("inference", inference, root)
            await self._wait_for_health(inference_url, "inference")
            self._set(startup_step="preparation")
            self._spawn("preparation", preparation, root)
            await self._wait_for_health(preparation_url, "preparation")
            self._client = OpenAI(
                base_url=inference_url,
                api_key=inference_key,
                default_model=self.config.model_id,
                preparation_base_url=preparation_url,
                preparation_api_key=preparation_key,
                bundle_cache_dir=root / "bundle-cache",
                timeout=300,
                prepared_inventory_rows=self._inventory_rows,
                background_inventory_refill=False,
                experiment=self.config.experiment,
            )
            if self._stopping.is_set():
                raise RuntimeError("dashboard stopped during startup")
            self._set(
                phase="preparing",
                startup_step="inventory",
                endpoints={"preparation": preparation_url, "inference": inference_url},
            )
            await self._background_call(
                self._client.preprocess,
                self.config.model_id,
                count=self._inventory_rows,
            )
            self._model_fingerprint = await self._background_call(self._discover_model_fingerprint)
            await asyncio.sleep(0.6)
            if self._stopping.is_set():
                raise RuntimeError("dashboard stopped during startup")
            self._set(
                phase="ready",
                startup_step="ready",
                inventory=self._client.prepared_inventory_status(self.config.model_id),
            )
        except Exception as exc:
            self._set(phase="error", error=f"startup failed: {type(exc).__name__}: {exc}")

    def _spawn(self, role: str, command: list[str], root: Path) -> None:
        if self._stopping.is_set():
            raise RuntimeError("dashboard is stopping")
        log = (root / f"{role}.log").open("wb")
        env = os.environ.copy()
        env.update(
            {
                "OTEL_EXPORTER_OTLP_ENDPOINT": _http_origin(self.config.host, self.config.port),
                "OTEL_INSTRUMENTATION_HTTP_CAPTURE_HEADERS_CLIENT_REQUEST": "",
                "OTEL_INSTRUMENTATION_HTTP_CAPTURE_HEADERS_SERVER_REQUEST": "",
                "OTEL_METRIC_EXPORT_INTERVAL": "500",
                "OTEL_EXPORTER_OTLP_HEADERS": f"x-pllm-otel-token={self.config.otel_token}",
                "OTEL_SERVICE_NAME": f"pllm-{role}",
                "PYTHONUNBUFFERED": "1",
            }
        )
        self._logs[role] = log
        self._log_paths[role] = root / f"{role}.log"
        self._processes[role] = subprocess.Popen(
            command,
            cwd=Path.cwd(),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
        )

    async def _wait_for_health(self, base_url: str, role: str) -> None:
        deadline = time.monotonic() + 300
        async with httpx.AsyncClient(timeout=1) as client:
            while time.monotonic() < deadline:
                if self._stopping.is_set():
                    raise RuntimeError("dashboard stopped during startup")
                process = self._processes[role]
                if process.poll() is not None:
                    log = self._log_paths[role].read_text(errors="replace")[-2000:]
                    raise RuntimeError(
                        f"{role} exited with status {process.returncode}: {log.strip()}"
                    )
                try:
                    response = await client.get(f"{base_url}/health")
                    if response.is_success:
                        return
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(0.2)
        raise TimeoutError(f"{role} did not become healthy")

    def begin(self, prompt: str, max_output_tokens: int, request_id: str | None = None) -> str:
        started_at_ns = time.time_ns()
        started_monotonic_ns = time.monotonic_ns()
        run_id = request_id or "run_" + secrets.token_hex(16)
        if not _RUN_ID.fullmatch(run_id):
            raise ValueError("request_id is invalid")
        with self._lock:
            if self.history is not None and self.history.contains(run_id):
                raise RuntimeError("request_id already belongs to a completed run")
            if self._state["phase"] != "ready":
                raise RuntimeError("prepared inventory is not ready")
            if self._stopping.is_set() or not self._services_healthy():
                raise RuntimeError("benchmark services are not healthy")
            capture = _RunCapture(
                run_id=run_id,
                max_output_tokens=max_output_tokens,
                cold=not self._has_started_run,
                started_at_ns=started_at_ns,
                audit_before=self._audit_snapshot(),
                started_monotonic_ns=started_monotonic_ns,
                protocol_start_cursor=self.store.protocol_cursor(),
            )
            self.store.begin_run_window(run_id)
            self._has_started_run = True
            self._active_run = capture
            self._state.update(
                phase="preparing",
                run_id=run_id,
                prompt=prompt,
                text="",
                tokens=0,
                started_at=None,
                first_token_at=None,
                last_token_at=None,
                finished_at=None,
                error=None,
                max_output_tokens=max_output_tokens,
                full_started_at=started_at_ns / 1_000_000_000,
                protocol_start_cursor=capture.protocol_start_cursor,
                preparation_started_at=None,
                preparation_finished_at=None,
            )
            self._online_traffic_baseline = {}
            self._online_traffic_final = None
            self._preparation_cpu_baseline = None
            self._preparation_online_cpu_final = None
            self._preparation_online_operations = 0
            self._last_privacy_delta = {}
            worker = threading.Thread(
                target=self._prepare_then_run,
                args=(run_id, prompt, max_output_tokens),
                daemon=True,
                name="pllm-dashboard-chat",
            )
            self._worker = worker
            worker.start()
        return run_id

    def _prepare_then_run(self, run_id: str, prompt: str, max_output_tokens: int) -> None:
        assert self._client is not None
        capture = self._active_run
        if capture is None or capture.run_id != run_id:
            return
        capture.preparation_started_at_ns = time.time_ns()
        capture.preparation_started_monotonic_ns = time.monotonic_ns()
        self._set(preparation_started_at=capture.preparation_started_at_ns / 1_000_000_000)
        try:
            previous_inventory = self._client.prepared_inventory_status(self.config.model_id)
            (
                capture.inventory_consumed_before,
                capture.inventory_burned_before,
            ) = self._inventory_totals(previous_inventory)
            required = self._client.prepared_rows_for_response(
                prompt,
                max_output_tokens,
                model=self.config.model_id,
            )
            capture.inventory_required = int(required)
            capture.input_tokens = max(0, required - max(0, max_output_tokens - 1))
            preparation = self._client.preprocess(self.config.model_id, count=required) or {}
            capture.inventory_generated = max(0, int(preparation.get("generated", 0)))
            current_inventory = self._client.prepared_inventory_status(self.config.model_id)
            same_inventory = current_inventory.get("id") == previous_inventory.get("id")
            if not same_inventory:
                self._inventory_burned_total += int(previous_inventory.get("burned", 0))
                self._inventory_consumed_total += int(previous_inventory.get("consumed", 0))
            if same_inventory:
                capture.inventory_reused = min(
                    required,
                    max(0, int(previous_inventory.get("available", 0))),
                )
            (
                capture.inventory_consumed_before,
                capture.inventory_burned_before,
            ) = self._inventory_totals(current_inventory)
            capture.preparation_finished_at_ns = time.time_ns()
            capture.preparation_finished_monotonic_ns = time.monotonic_ns()
            self._set(preparation_finished_at=(capture.preparation_finished_at_ns / 1_000_000_000))

            time.sleep(0.6)
            telemetry = self.store.snapshot()
            capture.online_started_at_ns = time.time_ns()
            capture.online_started_monotonic_ns = time.monotonic_ns()
            audit = self._audit_snapshot()
            preparation_cpu = telemetry["services"]["pllm-preparation"].get("cpu_seconds")
            with self._lock:
                self._online_traffic_baseline = dict(telemetry["traffic"])
                self._preparation_cpu_baseline = (
                    None if preparation_cpu is None else float(preparation_cpu)
                )
                self._preparation_attempts_baseline = audit.get("preparation_attempts", 0)
                self._online_traffic_final = None
                self._preparation_online_cpu_final = None
                self._preparation_online_operations = 0
                self._state.update(
                    phase="online",
                    started_at=capture.online_started_at_ns / 1_000_000_000,
                    inventory=current_inventory,
                )
        except Exception as exc:
            if capture.preparation_finished_at_ns is None:
                capture.preparation_finished_at_ns = time.time_ns()
            capture.finished_at_ns = time.time_ns()
            capture.finished_monotonic_ns = time.monotonic_ns()
            self._finish_run(capture, "failed", exc)
            return
        self._run_chat(prompt, max_output_tokens)

    @staticmethod
    def _completed_usage(event: Any) -> tuple[int, int] | None:
        response = (
            event.get("response") if isinstance(event, dict) else getattr(event, "response", None)
        )
        if response is None:
            return None
        usage = (
            response.get("usage")
            if isinstance(response, dict)
            else getattr(response, "usage", None)
        )
        if usage is None:
            return None
        try:
            if isinstance(usage, dict):
                input_tokens = int(usage["input_tokens"])
                output_tokens = int(usage["output_tokens"])
            else:
                input_tokens = int(usage.input_tokens)
                output_tokens = int(usage.output_tokens)
        except (AttributeError, KeyError, TypeError, ValueError):
            return None
        if input_tokens < 0 or output_tokens < 0:
            return None
        return input_tokens, output_tokens

    def _run_chat(self, prompt: str, max_output_tokens: int) -> None:
        assert self._client is not None
        capture = getattr(self, "_active_run", None)
        delta_count = 0
        authoritative_usage: tuple[int, int] | None = None
        completed_at_ns: int | None = None
        completed_monotonic_ns: int | None = None
        saw_completed = False
        try:
            from opentelemetry import trace

            with trace.get_tracer("pllm.dashboard").start_as_current_span("pllm.chat") as span:
                span.set_attribute("gen_ai.request.model", self.config.model_id)
                span.set_attribute("gen_ai.request.max_tokens", max_output_tokens)
                stream = self._client.responses.create(
                    model=self.config.model_id,
                    input=prompt,
                    max_output_tokens=max_output_tokens,
                    stream=True,
                )
                for event in cast(Iterable[Any], stream):
                    event_type = event.get("type") if isinstance(event, dict) else event.type
                    now_ns = time.time_ns()
                    if event_type in {"response.completed", "response.incomplete"}:
                        saw_completed = True
                        authoritative_usage = self._completed_usage(event)
                        completed_at_ns = now_ns
                        completed_monotonic_ns = time.monotonic_ns()
                        continue
                    if event_type != "response.output_text.delta":
                        continue
                    delta_count += 1
                    if capture is not None:
                        capture.first_token_at_ns = capture.first_token_at_ns or now_ns
                        capture.last_token_at_ns = now_ns
                        token_monotonic_ns = time.monotonic_ns()
                        capture.first_token_monotonic_ns = (
                            capture.first_token_monotonic_ns or token_monotonic_ns
                        )
                        capture.last_token_monotonic_ns = token_monotonic_ns
                    delta = event.get("delta") if isinstance(event, dict) else event.delta
                    with self._lock:
                        self._state["text"] += delta or ""
                        self._state["tokens"] = delta_count
                        self._state["first_token_at"] = (
                            self._state["first_token_at"] or now_ns / 1_000_000_000
                        )
                        self._state["last_token_at"] = now_ns / 1_000_000_000
            if not saw_completed:
                raise _IncompleteResponseError("response stream ended before a terminal response")
        except Exception as exc:
            finished_at_ns = time.time_ns()
            if capture is None:
                self._finish_legacy_run(finished_at_ns, exc)
                return
            capture.finished_at_ns = finished_at_ns
            capture.finished_monotonic_ns = time.monotonic_ns()
            capture.output_tokens = delta_count
            self._finish_run(capture, "failed", exc)
            return

        finished_at_ns = completed_at_ns or time.time_ns()
        if capture is None:
            self._finish_legacy_run(finished_at_ns, None)
            return
        capture.finished_at_ns = finished_at_ns
        capture.finished_monotonic_ns = completed_monotonic_ns or time.monotonic_ns()
        if authoritative_usage is not None:
            capture.input_tokens, capture.output_tokens = authoritative_usage
            capture.token_usage_authoritative = True
        else:
            capture.output_tokens = delta_count
        self._set(
            tokens=capture.output_tokens,
            finished_at=finished_at_ns / 1_000_000_000,
        )
        self._finish_run(capture, "completed", None)

    def _capture_online_telemetry(self) -> None:
        from pllm.runtime.telemetry import force_flush_telemetry

        with suppress(Exception):
            force_flush_telemetry()
        telemetry = self.store.snapshot()
        audit = self._audit_snapshot()
        with self._lock:
            active_run = getattr(self, "_active_run", None)
            audit_before = active_run.audit_before if active_run is not None else {}
            self._online_traffic_final = {
                "client->inference": float(
                    max(
                        0,
                        audit.get("masked_online_upload_bytes", 0)
                        - audit_before.get("masked_online_upload_bytes", 0),
                    )
                ),
                "inference->client": float(
                    max(
                        0,
                        audit.get("masked_online_download_bytes", 0)
                        - audit_before.get("masked_online_download_bytes", 0),
                    )
                ),
            }
            preparation_cpu_value = telemetry["services"]["pllm-preparation"].get("cpu_seconds")
            preparation_cpu = (
                None if preparation_cpu_value is None else float(preparation_cpu_value)
            )
            baseline_cpu = self._preparation_cpu_baseline
            self._preparation_online_cpu_final = (
                None
                if baseline_cpu is None or preparation_cpu is None
                else max(0.0, preparation_cpu - baseline_cpu)
            )
            self._preparation_online_operations = max(
                0,
                audit.get("preparation_attempts", 0) - self._preparation_attempts_baseline,
            )

    def _finish_legacy_run(self, finished_at_ns: int, error: Exception | None) -> None:
        self._set(
            finished_at=finished_at_ns / 1_000_000_000,
            error=f"{type(error).__name__}: {error}" if error is not None else None,
        )
        time.sleep(0.6)
        self._capture_online_telemetry()
        client = self._client
        assert client is not None
        inventory = client.prepared_inventory_status(self.config.model_id)
        self._set(
            phase="ready" if error is None and self._services_healthy() else "error",
            inventory=inventory,
        )

    def _finish_run(
        self,
        capture: _RunCapture,
        status: str,
        error: Exception | None,
    ) -> None:
        if capture.finished_at_ns is None:
            capture.finished_at_ns = time.time_ns()
        if capture.finished_monotonic_ns is None:
            capture.finished_monotonic_ns = time.monotonic_ns()
        display_error = f"{type(error).__name__}: {error}" if error is not None else None
        self._set(
            finished_at=capture.finished_at_ns / 1_000_000_000,
            error=display_error,
        )
        try:
            time.sleep(0.6)
            if capture.online_started_at_ns is not None:
                self._capture_online_telemetry()
        except Exception as telemetry_error:
            if display_error is None:
                display_error = f"telemetry failed: {type(telemetry_error).__name__}"

        try:
            capture.process_metrics = self.store.finish_run_window(capture.run_id)
        except Exception as telemetry_error:
            capture.process_metrics = {}
            if display_error is None:
                display_error = f"telemetry failed: {type(telemetry_error).__name__}"
        inventory: dict[str, Any] = {}
        client = self._client
        if client is not None:
            try:
                inventory = client.prepared_inventory_status(self.config.model_id)
            except Exception:
                pass
        try:
            consumed, burned = self._inventory_totals(inventory)
        except (TypeError, ValueError):
            consumed = capture.inventory_consumed_before
            burned = capture.inventory_burned_before
            if display_error is None:
                display_error = "inventory metrics were invalid"
        try:
            audit_after = self._audit_snapshot()
            privacy_delta = {
                key: max(0, value - capture.audit_before.get(key, 0))
                for key, value in audit_after.items()
            }
            self._last_privacy_delta = privacy_delta
            run = BenchmarkRun(
                run_id=capture.run_id,
                status=status,
                model_id=self.config.model_id,
                model_fingerprint=self._model_fingerprint,
                max_output_tokens=capture.max_output_tokens,
                cold=capture.cold,
                warm=not capture.cold,
                started_at_ns=capture.started_at_ns,
                preparation_started_at_ns=capture.preparation_started_at_ns,
                preparation_finished_at_ns=capture.preparation_finished_at_ns,
                online_started_at_ns=capture.online_started_at_ns,
                first_token_at_ns=capture.first_token_at_ns,
                last_token_at_ns=capture.last_token_at_ns,
                finished_at_ns=capture.finished_at_ns,
                full_seconds=(capture.finished_monotonic_ns - capture.started_monotonic_ns)
                / 1_000_000_000,
                preparation_seconds=(
                    (
                        capture.preparation_finished_monotonic_ns
                        - capture.preparation_started_monotonic_ns
                    )
                    / 1_000_000_000
                    if capture.preparation_started_monotonic_ns is not None
                    and capture.preparation_finished_monotonic_ns is not None
                    else None
                ),
                transition_seconds=(
                    (
                        capture.online_started_monotonic_ns
                        - capture.preparation_finished_monotonic_ns
                    )
                    / 1_000_000_000
                    if capture.preparation_finished_monotonic_ns is not None
                    and capture.online_started_monotonic_ns is not None
                    else None
                ),
                online_seconds=(
                    (capture.finished_monotonic_ns - capture.online_started_monotonic_ns)
                    / 1_000_000_000
                    if capture.online_started_monotonic_ns is not None
                    else None
                ),
                ttft_seconds=(
                    (capture.first_token_monotonic_ns - capture.online_started_monotonic_ns)
                    / 1_000_000_000
                    if capture.online_started_monotonic_ns is not None
                    and capture.first_token_monotonic_ns is not None
                    else None
                ),
                generation_seconds=(
                    (capture.last_token_monotonic_ns - capture.first_token_monotonic_ns)
                    / 1_000_000_000
                    if capture.first_token_monotonic_ns is not None
                    and capture.last_token_monotonic_ns is not None
                    else None
                ),
                input_tokens=capture.input_tokens,
                output_tokens=capture.output_tokens,
                token_usage_authoritative=capture.token_usage_authoritative,
                inventory_required=capture.inventory_required,
                inventory_generated=capture.inventory_generated,
                inventory_reused=capture.inventory_reused,
                inventory_consumed=max(0, consumed - capture.inventory_consumed_before),
                inventory_burned=max(0, burned - capture.inventory_burned_before),
                privacy_delta=privacy_delta,
                process_metrics=capture.process_metrics,
                failure_type=type(error).__name__ if error is not None else None,
            )
            if self.history is not None:
                self.history.add(run)
        except Exception as history_error:
            if display_error is None:
                display_error = f"history write failed: {type(history_error).__name__}"
        with self._lock:
            if self._active_run is capture:
                self._active_run = None
            self._worker = None
            self._state.update(
                phase="ready" if self._services_healthy() else "error",
                inventory=inventory,
                error=display_error,
            )

    def snapshot(self, protocol_after: int = 0) -> dict[str, Any]:
        with self._lock:
            state = dict(self._state)
        now = state.get("last_token_at") or state.get("finished_at") or time.time()
        started = state.get("started_at")
        first = state.get("first_token_at")
        state["ttft_seconds"] = first - started if first and started else None
        generated_after_first = max(0, int(state.get("tokens", 0)) - 1)
        state["tps"] = (
            generated_after_first / max(0.001, now - first)
            if first and generated_after_first
            else None
        )
        state["processes"] = {
            role: {"pid": process.pid, "running": process.poll() is None}
            for role, process in self._processes.items()
        }
        audit = self._audit_snapshot()
        with self._lock:
            active_capture = getattr(self, "_active_run", None)
        state["privacy"] = (
            {
                key: max(0, value - active_capture.audit_before.get(key, 0))
                for key, value in audit.items()
            }
            if active_capture is not None
            else getattr(self, "_last_privacy_delta", {})
            if state.get("run_id")
            else audit
        )
        state["privacy_cumulative"] = audit
        protocol_floor = int(state.get("protocol_start_cursor", 0))
        telemetry = self.store.snapshot(max(protocol_after, protocol_floor))
        if self._client is not None:
            inventory = dict(state.get("inventory", {}))
            inventory["burned"] = int(inventory.get("burned", 0)) + self._inventory_burned_total
            inventory["consumed"] = (
                int(inventory.get("consumed", 0)) + self._inventory_consumed_total
            )
            state["inventory"] = inventory
        state["online_traffic"] = self._online_traffic_final or {
            "client->inference": float(state["privacy"].get("masked_online_upload_bytes", 0)),
            "inference->client": float(state["privacy"].get("masked_online_download_bytes", 0)),
        }
        preparation_cpu_value = telemetry["services"]["pllm-preparation"].get("cpu_seconds")
        preparation_cpu = None if preparation_cpu_value is None else float(preparation_cpu_value)
        state["preparation_online_cpu_seconds"] = (
            self._preparation_online_cpu_final
            if self._preparation_online_cpu_final is not None
            else None
            if self._preparation_cpu_baseline is None or preparation_cpu is None
            else max(0.0, preparation_cpu - self._preparation_cpu_baseline)
        )
        state["preparation_online_operations"] = self._preparation_online_operations
        telemetry["services"].setdefault("pllm-client", {})["status"] = "live"
        for service, role in (
            ("pllm-preparation", "preparation"),
            ("pllm-inference", "inference"),
        ):
            process = state["processes"].get(role)
            status = "live" if process and process["running"] else "offline"
            telemetry["services"].setdefault(service, {})["status"] = status
        return {"run": state, "otel": telemetry}

    async def stop(self) -> None:
        with self._lock:
            if self._stop_started:
                return
            self._stop_started = True
            self._stopping.set()
            worker = self._worker
        for process in reversed(list(self._processes.values())):
            if process.poll() is None:
                process.terminate()
        if self._client is not None:
            try:
                await asyncio.wait_for(
                    self._background_call(self._client.close),
                    timeout=self._CLIENT_CLOSE_TIMEOUT_SECONDS,
                )
            except Exception:
                pass

        async def reap(process: subprocess.Popen[bytes]) -> None:
            try:
                await asyncio.to_thread(process.wait, self._PROCESS_EXIT_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                process.kill()
                with suppress(subprocess.TimeoutExpired):
                    await asyncio.to_thread(process.wait, self._PROCESS_EXIT_TIMEOUT_SECONDS)

        await asyncio.gather(*(reap(process) for process in self._processes.values()))
        if worker is not None and worker.is_alive():
            await asyncio.to_thread(worker.join, self._WORKER_JOIN_TIMEOUT_SECONDS)
        worker_alive = worker is not None and worker.is_alive()
        with self._lock:
            background_threads = list(self._background_threads)
        for thread in background_threads:
            await asyncio.to_thread(thread.join, self._BACKGROUND_JOIN_TIMEOUT_SECONDS)
        background_alive = any(thread.is_alive() for thread in background_threads)
        for log in self._logs.values():
            log.close()
        if self._temporary is not None and not worker_alive and not background_alive:
            self._temporary.cleanup()

    def has_live_worker(self) -> bool:
        with self._lock:
            return self._worker is not None and self._worker.is_alive()


async def _bounded_body(request: Request, maximum: int) -> bytes:
    declared = request.headers.get("content-length")
    if declared is not None:
        try:
            if int(declared) > maximum:
                raise HTTPException(status_code=413, detail="request body is too large")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="invalid content-length") from exc
    chunks: list[bytes] = []
    size = 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > maximum:
            raise HTTPException(status_code=413, detail="request body is too large")
        chunks.append(chunk)
    return b"".join(chunks)


def _loopback_request(request: Request, port: int) -> bool:
    try:
        host = urlsplit(f"http://{request.headers['host']}")
        if host.hostname not in {"127.0.0.1", "localhost", "::1"}:
            return False
        if host.port not in {None, port}:
            return False
        origin_value = request.headers.get("origin")
        if origin_value is None:
            return True
        origin = urlsplit(origin_value)
        return (
            origin.scheme == "http"
            and origin.hostname in {"127.0.0.1", "localhost", "::1"}
            and (origin.port or 80) == port
        )
    except (KeyError, ValueError):
        return False


def create_dashboard_app(config: DashboardConfig) -> FastAPI:
    store = OTelStore()
    history = BenchmarkHistory(config.history_path)
    runtime = DashboardRuntime(config, store, history)
    assets = Path(__file__).resolve().parent.parent / "dashboard"

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        startup = asyncio.create_task(runtime.start())
        try:
            yield
        finally:
            try:
                await runtime.stop()
                await asyncio.wait_for(asyncio.shield(startup), timeout=10)
            except TimeoutError:
                startup.cancel()
            except Exception:
                pass
            if not runtime.has_live_worker():
                history.close()

    app = FastAPI(title="PLLM Live Protocol", docs_url=None, redoc_url=None, lifespan=lifespan)
    app.state.dashboard_runtime = runtime
    app.state.benchmark_history = history

    @app.middleware("http")
    async def enforce_loopback_boundary(request: Request, call_next: Any) -> Response:
        if not _loopback_request(request, config.port):
            return Response("loopback origin required", status_code=403)
        if request.url.path in {"/v1/metrics", "/v1/traces"} and not secrets.compare_digest(
            request.headers.get("x-pllm-otel-token", ""), config.otel_token
        ):
            return Response("invalid telemetry credential", status_code=401)
        return await call_next(request)

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(assets / "index.html")

    @app.get("/assets/{name}")
    async def asset(name: str) -> FileResponse:
        if name not in {"app.js", "style.css"}:
            raise HTTPException(status_code=404)
        return FileResponse(assets / name)

    @app.get("/api/config")
    async def dashboard_config() -> dict[str, Any]:
        return {
            "model_id": config.model_id,
            "default_max_output_tokens": config.default_max_output_tokens,
            "history_limit": MAX_LIMIT,
        }

    @app.get("/api/snapshot")
    async def snapshot(protocol_after: int = 0) -> dict[str, Any]:
        return runtime.snapshot(protocol_after)

    @app.get("/api/runs")
    async def runs(
        limit: int = Query(default=50, ge=1, le=MAX_LIMIT),
        model_id: str | None = None,
        status: str | None = None,
        context_tokens: int | None = Query(default=None, ge=0),
        min_context_tokens: int | None = Query(default=None, ge=0),
        max_context_tokens: int | None = Query(default=None, ge=0),
        cold: bool | None = None,
        before_started_at_ns: int | None = Query(default=None, gt=0),
        before_run_id: str | None = Query(default=None, min_length=1, max_length=128),
    ) -> dict[str, Any]:
        if (
            min_context_tokens is not None
            and max_context_tokens is not None
            and min_context_tokens > max_context_tokens
        ):
            raise HTTPException(
                status_code=400,
                detail="min_context_tokens cannot exceed max_context_tokens",
            )
        if (before_started_at_ns is None) != (before_run_id is None):
            raise HTTPException(
                status_code=400,
                detail="both pagination cursor fields are required",
            )
        try:
            records = history.list(
                limit=limit,
                model_id=model_id,
                status=status,
                context_tokens=context_tokens,
                min_context_tokens=min_context_tokens,
                max_context_tokens=max_context_tokens,
                cold=cold,
                before_started_at_ns=before_started_at_ns,
                before_run_id=before_run_id,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        next_cursor = None
        if len(records) == limit:
            last = records[-1]
            next_cursor = {
                "before_started_at_ns": int(last["timestamps"]["started_at_ns"]),
                "before_run_id": str(last["run_id"]),
            }
        return {
            "schema_version": SCHEMA_VERSION,
            "runs": records,
            "next_cursor": next_cursor,
        }

    @app.get("/api/runs/{run_id}")
    async def get_run(run_id: str) -> dict[str, Any]:
        record = history.get(run_id)
        if record is None:
            raise HTTPException(status_code=404, detail="benchmark run not found")
        return record

    @app.post("/api/run", status_code=202)
    async def run(request: Request) -> dict[str, Any]:
        if request.headers.get("content-type", "").split(";", 1)[0].strip().lower() != (
            "application/json"
        ):
            raise HTTPException(status_code=415, detail="application/json is required")
        try:
            body = json.loads(await _bounded_body(request, 20_000))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise HTTPException(status_code=400, detail="invalid JSON body") from exc
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="JSON body must be an object")
        if set(body).difference({"prompt", "max_output_tokens", "request_id"}):
            raise HTTPException(status_code=400, detail="JSON body contains unsupported fields")
        prompt_value = body.get("prompt", "")
        if not isinstance(prompt_value, str):
            raise HTTPException(status_code=400, detail="prompt must be a string")
        prompt = prompt_value.strip()
        if not prompt or len(prompt.encode()) > 16_384:
            raise HTTPException(status_code=400, detail="prompt must contain 1 to 16384 bytes")
        maximum_value = body.get("max_output_tokens", config.default_max_output_tokens)
        if type(maximum_value) is not int:
            raise HTTPException(status_code=400, detail="max_output_tokens must be an integer")
        maximum = maximum_value
        if not 1 <= maximum <= 512:
            raise HTTPException(
                status_code=400, detail="max_output_tokens must be between 1 and 512"
            )
        request_id = body.get("request_id")
        if request_id is not None and (
            not isinstance(request_id, str) or not _RUN_ID.fullmatch(request_id)
        ):
            raise HTTPException(status_code=400, detail="request_id is invalid")
        try:
            run_id = runtime.begin(prompt, maximum, request_id)
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {
            "status": "started",
            "run_id": run_id,
            "protocol_cursor": store.protocol_cursor(),
        }

    @app.post("/v1/metrics")
    async def metrics(request: Request) -> Response:
        if request.headers.get("content-type", "").split(";", 1)[0].strip().lower() != (
            "application/x-protobuf"
        ):
            raise HTTPException(status_code=415, detail="protobuf body is required")
        try:
            store.ingest_metrics(await _bounded_body(request, 16 * 1024 * 1024))
        except (ValueError, DecodeError) as exc:
            raise HTTPException(status_code=400, detail="invalid metrics payload") from exc
        return Response(
            ExportMetricsServiceResponse().SerializeToString(),
            media_type="application/x-protobuf",
        )

    @app.post("/v1/traces")
    async def traces(request: Request) -> Response:
        if request.headers.get("content-type", "").split(";", 1)[0].strip().lower() != (
            "application/x-protobuf"
        ):
            raise HTTPException(status_code=415, detail="protobuf body is required")
        try:
            store.ingest_traces(await _bounded_body(request, 16 * 1024 * 1024))
        except (ValueError, DecodeError) as exc:
            raise HTTPException(status_code=400, detail="invalid traces payload") from exc
        return Response(
            ExportTraceServiceResponse().SerializeToString(),
            media_type="application/x-protobuf",
        )

    return app


def run_dashboard(args: Any) -> None:
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        raise SystemExit("benchmark dashboard must bind to a loopback host")
    if not 1 <= args.port <= 65_535:
        raise SystemExit("--port must be between 1 and 65535")
    if not 1 <= args.max_output_tokens <= 512:
        raise SystemExit("--max-output-tokens must be between 1 and 512")
    experiment = None
    experiment_config = getattr(args, "experiment_config", None)
    if experiment_config is not None:
        from pllm.configuration import load_configuration

        experiment = load_configuration(experiment_config)
        experiment.resolve()
        if args.tiny:
            raise SystemExit("--experiment-config cannot be combined with --tiny")
        args.model = experiment.pipeline.model.source
        args.model_id = args.model
    model: Path | str | None = None
    if not args.tiny:
        candidate = Path(args.model).expanduser()
        model = candidate.resolve() if candidate.exists() else args.model
    model_id = args.model_id or ("pllm-benchmark-tiny" if args.tiny else str(args.model))
    config = DashboardConfig(
        host=args.host,
        port=args.port,
        model_path=model,
        model_id=model_id,
        default_max_output_tokens=args.max_output_tokens,
        history_path=getattr(args, "history_db", None),
        startup_inventory_rows=getattr(args, "startup_inventory_rows", None),
        experiment=experiment,
    )
    dashboard_origin = _http_origin(args.host, args.port)
    os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = dashboard_origin
    os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"x-pllm-otel-token={config.otel_token}"
    # Child roles override this to 500 ms after the collector is listening.
    os.environ["OTEL_METRIC_EXPORT_INTERVAL"] = "2000"
    os.environ["OTEL_SERVICE_NAME"] = "pllm-client"
    from pllm.runtime.telemetry import configure_telemetry, shutdown_telemetry

    configure_telemetry("pllm-client", shutdown_on_exit=False)
    if not args.no_open:
        threading.Timer(1.0, webbrowser.open, args=(dashboard_origin,)).start()
    import uvicorn

    dashboard_app = create_dashboard_app(config)

    class DashboardServer(uvicorn.Server):
        _flushing = False

        def handle_exit(self, sig: int, frame: Any) -> None:
            if self._flushing:
                super().handle_exit(sig, frame)
                return
            self._flushing = True
            asyncio.get_running_loop().create_task(self._flush_then_exit(sig, frame))

        async def _flush_then_exit(self, sig: int, frame: Any) -> None:
            runtime = dashboard_app.state.dashboard_runtime
            try:
                with suppress(Exception):
                    await runtime.stop()
                with suppress(Exception, asyncio.CancelledError):
                    await asyncio.wait_for(runtime._background_call(shutdown_telemetry), timeout=5)
            finally:
                super().handle_exit(sig, frame)

    server = DashboardServer(
        uvicorn.Config(
            dashboard_app,
            host=args.host,
            port=args.port,
            log_level="warning",
        )
    )
    try:
        server.run()
    except KeyboardInterrupt:
        pass
