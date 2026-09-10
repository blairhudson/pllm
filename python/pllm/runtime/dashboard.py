from __future__ import annotations

import asyncio
import json
import os
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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, cast

import httpx
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response
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
        "he_test_tokenizer": "byte",
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
                self._history[service].append(
                    {
                        "time": now,
                        "cpu": values["cpu_percent"],
                        "memory": values["memory_bytes"],
                    }
                )

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
                                "method": attrs.get(
                                    "http.request.method", attrs.get("http.method")
                                ),
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

    def _service_values(self, service: str) -> dict[str, float]:
        return {
            "cpu_percent": self._sum_metric(service, "process.cpu.utilization") * 100.0,
            "cpu_seconds": self._sum_metric(service, "process.cpu.time"),
            "memory_bytes": self._sum_metric(service, "process.memory.usage"),
            "virtual_memory_bytes": self._sum_metric(service, "process.memory.virtual"),
            "threads": self._sum_metric(service, "process.thread.count"),
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
            return {
                "services": services,
                "traffic": dict(traffic),
                "operations": int(operations),
                "spans": list(self._spans)[-30:],
                "protocol_spans": [
                    span
                    for span in self._protocol_spans
                    if int(span["sequence"]) > protocol_after
                ],
                "protocol_cursor": self._protocol_sequence,
            }


@dataclass(frozen=True)
class DashboardConfig:
    host: str
    port: int
    model_path: Path | str | None
    model_id: str
    default_max_output_tokens: int


def _free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


class DashboardRuntime:
    def __init__(self, config: DashboardConfig, store: OTelStore) -> None:
        self.config = config
        self.store = store
        self._lock = threading.Lock()
        self._temporary: tempfile.TemporaryDirectory[str] | None = None
        self._processes: dict[str, subprocess.Popen[bytes]] = {}
        self._logs: dict[str, Any] = {}
        self._log_paths: dict[str, Path] = {}
        self._client: OpenAI | None = None
        self._state: dict[str, Any] = {
            "phase": "starting",
            "model_id": config.model_id,
            "tiny": config.model_path is None,
            "prompt": "Explain why neither server can see the prompt.",
            "text": "",
            "tokens": 0,
            "started_at": None,
            "first_token_at": None,
            "finished_at": None,
            "error": None,
            "inventory": {},
            "online_traffic": {},
            "preparation_online_cpu_seconds": 0.0,
        }
        self._inventory_rows = 256 if config.model_path is None else 64
        self._online_traffic_baseline: dict[str, float] = {}
        self._preparation_cpu_baseline: float | None = None
        self._online_traffic_final: dict[str, float] | None = None
        self._preparation_online_cpu_final: float | None = None
        self._preparation_attempts_baseline = 0
        self._preparation_online_operations = 0
        self._inventory_burned_total = 0
        self._inventory_consumed_total = 0

    def _set(self, **values: Any) -> None:
        with self._lock:
            self._state.update(values)

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
            common = [sys.executable, "-m", "pllm"]
            inference = [
                *common,
                "serve",
                str(model),
                "--model-id",
                self.config.model_id,
                "--weights",
                "public",
                "--activation-protection",
                "seeded-preparation",
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
                "serve",
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
            self._spawn("inference", inference, root)
            await self._wait_for_health(inference_url, "inference")
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
            )
            self._set(
                phase="preparing",
                endpoints={"preparation": preparation_url, "inference": inference_url},
            )
            await asyncio.to_thread(
                self._client.preprocess,
                self.config.model_id,
                count=self._inventory_rows,
            )
            await asyncio.sleep(0.6)
            self._set(
                phase="ready",
                inventory=self._client.prepared_inventory_status(self.config.model_id),
            )
        except Exception as exc:
            self._set(phase="error", error=f"startup failed: {type(exc).__name__}: {exc}")

    def _spawn(self, role: str, command: list[str], root: Path) -> None:
        log = (root / f"{role}.log").open("wb")
        env = os.environ.copy()
        env.update(
            {
                "OTEL_EXPORTER_OTLP_ENDPOINT": f"http://127.0.0.1:{self.config.port}",
                "OTEL_INSTRUMENTATION_HTTP_CAPTURE_HEADERS_CLIENT_REQUEST": "",
                "OTEL_INSTRUMENTATION_HTTP_CAPTURE_HEADERS_SERVER_REQUEST": "",
                "OTEL_METRIC_EXPORT_INTERVAL": "500",
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

    def begin(self, prompt: str, max_output_tokens: int) -> None:
        with self._lock:
            if self._state["phase"] != "ready":
                raise RuntimeError("prepared inventory is not ready")
            if self._client is None:
                raise RuntimeError("benchmark services are not ready")
            self._state.update(
                phase="preparing",
                prompt=prompt,
                text="",
                tokens=0,
                started_at=None,
                first_token_at=None,
                finished_at=None,
                error=None,
                max_output_tokens=max_output_tokens,
            )
        threading.Thread(
            target=self._prepare_then_run,
            args=(prompt, max_output_tokens),
            daemon=True,
            name="pllm-dashboard-chat",
        ).start()

    def _prepare_then_run(self, prompt: str, max_output_tokens: int) -> None:
        assert self._client is not None
        try:
            previous_inventory = self._client.prepared_inventory_status(self.config.model_id)
            required = self._client.prepared_rows_for_response(
                prompt,
                max_output_tokens,
                model=self.config.model_id,
            )
            self._client.preprocess(self.config.model_id, count=required)
            current_inventory = self._client.prepared_inventory_status(self.config.model_id)
            if current_inventory.get("id") != previous_inventory.get("id"):
                self._inventory_burned_total += int(previous_inventory.get("burned", 0))
                self._inventory_consumed_total += int(previous_inventory.get("consumed", 0))
            time.sleep(0.6)
            telemetry = self.store.snapshot()
            with self._lock:
                self._online_traffic_baseline = dict(telemetry["traffic"])
                self._preparation_cpu_baseline = float(
                    telemetry["services"]["pllm-preparation"].get("cpu_seconds", 0.0)
                )
                self._preparation_attempts_baseline = int(
                    self._client.privacy_audit.preparation_attempts
                )
                self._online_traffic_final = None
                self._preparation_online_cpu_final = None
                self._preparation_online_operations = 0
                self._state.update(
                    phase="online",
                    started_at=time.time(),
                    inventory=self._client.prepared_inventory_status(self.config.model_id),
                )
            self._run_chat(prompt, max_output_tokens)
        except Exception as exc:
            self._set(phase="error", finished_at=time.time(), error=f"{type(exc).__name__}: {exc}")

    def _run_chat(self, prompt: str, max_output_tokens: int) -> None:
        assert self._client is not None
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
                    if event.type != "response.output_text.delta":
                        continue
                    now = time.time()
                    with self._lock:
                        self._state["text"] += event.delta or ""
                        self._state["tokens"] += 1
                        self._state["first_token_at"] = self._state["first_token_at"] or now
            time.sleep(0.6)
            telemetry = self.store.snapshot()
            traffic = telemetry["traffic"]
            with self._lock:
                self._online_traffic_final = {
                    edge: max(
                        0.0,
                        float(value) - self._online_traffic_baseline.get(edge, 0.0),
                    )
                    for edge, value in traffic.items()
                }
                preparation_cpu = float(
                    telemetry["services"]["pllm-preparation"].get("cpu_seconds", 0.0)
                )
                baseline_cpu = self._preparation_cpu_baseline
                self._preparation_online_cpu_final = (
                    0.0
                    if baseline_cpu is None
                    else max(0.0, preparation_cpu - baseline_cpu)
                )
                self._preparation_online_operations = max(
                    0,
                    self._client.privacy_audit.preparation_attempts
                    - self._preparation_attempts_baseline,
                )
            self._set(
                phase="ready",
                finished_at=time.time(),
                inventory=self._client.prepared_inventory_status(self.config.model_id),
            )
        except Exception as exc:
            self._set(phase="error", finished_at=time.time(), error=f"{type(exc).__name__}: {exc}")

    def snapshot(self, protocol_after: int = 0) -> dict[str, Any]:
        with self._lock:
            state = dict(self._state)
        now = state["finished_at"] or time.time()
        started = state["started_at"]
        first = state["first_token_at"]
        state["ttft_seconds"] = first - started if first and started else None
        state["tps"] = max(0, state["tokens"] - 1) / max(0.001, now - first) if first else 0.0
        state["processes"] = {
            role: {"pid": process.pid, "running": process.poll() is None}
            for role, process in self._processes.items()
        }
        audit = self._client.privacy_audit if self._client is not None else None
        state["privacy"] = asdict(audit) if audit is not None else {}
        telemetry = self.store.snapshot(protocol_after)
        if self._client is not None:
            inventory = dict(self._client.prepared_inventory_status(self.config.model_id))
            inventory["burned"] = int(inventory.get("burned", 0)) + self._inventory_burned_total
            inventory["consumed"] = (
                int(inventory.get("consumed", 0)) + self._inventory_consumed_total
            )
            state["inventory"] = inventory
        state["online_traffic"] = self._online_traffic_final or {
            edge: max(0.0, amount - self._online_traffic_baseline.get(edge, 0.0))
            for edge, amount in telemetry["traffic"].items()
        }
        preparation_cpu = float(
            telemetry["services"]["pllm-preparation"].get("cpu_seconds", 0.0)
        )
        state["preparation_online_cpu_seconds"] = (
            self._preparation_online_cpu_final
            if self._preparation_online_cpu_final is not None
            else 0.0
            if self._preparation_cpu_baseline is None
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
        if self._client is not None:
            self._client.close()
        for process in reversed(list(self._processes.values())):
            if process.poll() is None:
                process.terminate()
        for process in self._processes.values():
            try:
                await asyncio.to_thread(process.wait, 5)
            except subprocess.TimeoutExpired:
                process.kill()
        for log in self._logs.values():
            log.close()
        if self._temporary is not None:
            self._temporary.cleanup()


def create_dashboard_app(config: DashboardConfig) -> FastAPI:
    store = OTelStore()
    runtime = DashboardRuntime(config, store)
    assets = Path(__file__).resolve().parent.parent / "dashboard"

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        startup = asyncio.create_task(runtime.start())
        try:
            yield
        finally:
            if not startup.done():
                startup.cancel()
                with suppress(asyncio.CancelledError):
                    await startup
            else:
                await startup
            await runtime.stop()

    app = FastAPI(title="PLLM Live Protocol", docs_url=None, redoc_url=None, lifespan=lifespan)

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(assets / "index.html")

    @app.get("/assets/{name}")
    async def asset(name: str) -> FileResponse:
        if name not in {"app.js", "style.css"}:
            raise HTTPException(status_code=404)
        return FileResponse(assets / name)

    @app.get("/api/snapshot")
    async def snapshot(protocol_after: int = 0) -> dict[str, Any]:
        return runtime.snapshot(protocol_after)

    @app.post("/api/run", status_code=202)
    async def run(request: Request) -> dict[str, str]:
        body = await request.json()
        prompt = str(body.get("prompt", "")).strip()
        if not prompt or len(prompt.encode()) > 16_384:
            raise HTTPException(status_code=400, detail="prompt must contain 1 to 16384 bytes")
        maximum = int(body.get("max_output_tokens", config.default_max_output_tokens))
        if not 1 <= maximum <= 512:
            raise HTTPException(
                status_code=400, detail="max_output_tokens must be between 1 and 512"
            )
        try:
            runtime.begin(prompt, maximum)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"status": "started"}

    @app.post("/v1/metrics")
    async def metrics(request: Request) -> Response:
        store.ingest_metrics(await request.body())
        return Response(
            ExportMetricsServiceResponse().SerializeToString(),
            media_type="application/x-protobuf",
        )

    @app.post("/v1/traces")
    async def traces(request: Request) -> Response:
        store.ingest_traces(await request.body())
        return Response(
            ExportTraceServiceResponse().SerializeToString(),
            media_type="application/x-protobuf",
        )

    return app


def run_dashboard(args: Any) -> None:
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        raise SystemExit("benchmark dashboard must bind to a loopback host")
    os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = f"http://127.0.0.1:{args.port}"
    os.environ["OTEL_METRIC_EXPORT_INTERVAL"] = "500"
    os.environ["OTEL_SERVICE_NAME"] = "pllm-client"
    from pllm.runtime.telemetry import configure_telemetry

    configure_telemetry("pllm-client", shutdown_on_exit=False)
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
        default_max_output_tokens=max(1, args.max_output_tokens),
    )
    if not args.no_open:
        threading.Timer(1.0, webbrowser.open, args=(f"http://{args.host}:{args.port}",)).start()
    import uvicorn

    class DashboardServer(uvicorn.Server):
        _flushing = False

        def handle_exit(self, sig: int, frame: Any) -> None:
            if self._flushing:
                super().handle_exit(sig, frame)
                return
            self._flushing = True
            asyncio.get_running_loop().create_task(self._flush_then_exit(sig, frame))

        async def _flush_then_exit(self, sig: int, frame: Any) -> None:
            from pllm.runtime.telemetry import shutdown_telemetry

            await asyncio.to_thread(shutdown_telemetry)
            super().handle_exit(sig, frame)

    server = DashboardServer(
        uvicorn.Config(
            create_dashboard_app(config),
            host=args.host,
            port=args.port,
            log_level="warning",
        )
    )
    server.run()
