from __future__ import annotations

import asyncio
import math
import os
import re
import secrets
import signal
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx

from pllm.configuration import Experiment, Model, Pipeline


class TopologyError(RuntimeError):
    pass


async def load_role_adapter(
    experiment: Experiment, role_id: str, api_key: str, *,
    preloaded_engine=None, push_key: str | None = None, inference_url: str | None = None,
):
    """Shared installed role factory for local children and authenticated parties.

    Admission never installs a method described by an offer. Exact composition
    determines the existing engine, numeric settings and session protocol.
    """
    from pllm.model_loader import resolve_model
    from pllm.profiles import resolve_runtime_composition
    from pllm.runtime.offset_worker import create_offset_worker_app
    from pllm.runtime.transformer_engine import MaskedTransformerEngine

    profile = experiment.resolve()
    options = resolve_runtime_composition(experiment.pipeline)
    roles = {role.id for role in profile.role_graph.roles} - {"client"}
    if options is None or role_id not in roles or roles not in (
        {"worker_a", "worker_b"}, {"inference", "preparation"},
    ):
        raise TopologyError("UNSUPPORTED_NETWORK_GRAPH: installed network adapter is unavailable")
    kernels = experiment.pipeline.components["kernels"]
    engine = preloaded_engine or MaskedTransformerEngine(
        threads=kernels.params.get("threads", 1),
        weight_bits=options.weight_bits, activation_bits=options.activation_bits,
        metal_min_rows=kernels.params.get("min_rows")
        if kernels.component == "pllm/apple-metal-int8/v1" else None,
        verification_component=options.verification_component or "none",
        verification_target_failure_bits=options.verification_target_failure_bits,
        public_equalization_digest=options.public_equalization_digest,
        remote_output_head=options.remote_output_head,
        client_prefix_layers=options.client_prefix_layers,
        client_linear_roles=options.client_linear_roles,
        prepared_output_encoding=options.prepared_output_encoding,
        weight_residency="provider" if role_id == "preparation" else "provider_and_bundle",
        weight_storage=options.preparation_storage if role_id == "preparation" else "resident",
    )
    if preloaded_engine is None:
        source = resolve_model(experiment.pipeline.model)
        await engine.load(source.manifest)
    if role_id in {"worker_a", "worker_b"}:
        return create_offset_worker_app(engine, model_id=profile.model, role_id=role_id,
                                       api_key=api_key, composition=experiment.pipeline), engine
    from pllm.runtime.config import GatewayConfig
    from pllm.runtime.preparation_server import create_preparation_app
    from pllm.runtime.server import create_app

    push_key = push_key or secrets.token_urlsafe(32)
    if role_id == "inference":
        manifest = engine._model(profile.model).manifest
        return create_app(GatewayConfig(api_keys=(api_key,), provider_push_api_key=push_key),
                          private_models={}, engines={"masked-transformer": engine},
                          preloaded_models=(("masked-transformer", manifest),),
                          owns_engines=False), engine
    return create_preparation_app(GatewayConfig(
        api_keys=(api_key,), preparation_push_api_key=push_key,
        preparation_inference_url=inference_url or "http://127.0.0.1",
    ), engine, owns_engine=False), engine


def role_client(experiment, connections):
    """Common topology-to-SDK routing; uses the existing inference executor."""
    from pllm.runtime.client import OpenAI

    roles = set(connections)
    if roles == {"worker_a", "worker_b"}:
        return OpenAI(default_model=experiment.resolve().model, experiment=experiment,
                       role_connections=connections)
    if roles == {"inference", "preparation"}:
        return OpenAI(default_model=experiment.resolve().model, experiment=experiment,
                      base_url=connections["inference"][0], api_key=connections["inference"][1],
                      preparation_base_url=connections["preparation"][0],
                      preparation_api_key=connections["preparation"][1],
                      background_inventory_refill=False)
    raise TopologyError("UNSUPPORTED_NETWORK_GRAPH: no admitted role client")


def _spawn_role_process(
    command: list[str],
    *,
    environment: dict[str, str],
    stdout: Any = None,
    stderr: Any = None,
    discard_output: bool = False,
) -> subprocess.Popen[Any]:
    """All local role children share one process-group and credential boundary."""
    if discard_output:
        stdout = subprocess.DEVNULL
        stderr = subprocess.DEVNULL
    return subprocess.Popen(
        command,
        cwd=Path.cwd(),
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=stdout,
        stderr=stderr,
        start_new_session=True,
    )


def _stop_role_process(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
        process.wait(timeout=10)
        return
    except (ProcessLookupError, subprocess.TimeoutExpired):
        pass
    if process.poll() is None:
        if os.name == "posix":
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        else:
            process.kill()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired as exc:
            raise TopologyError("local role process did not stop") from exc


@dataclass(frozen=True, slots=True)
class RoleStatus:
    role: str
    url: str
    pid: int | None
    running: bool


class LocalTopology:
    __slots__ = (
        "_activation_bits",
        "_closed",
        "_correlation_mode",
        "_credential_prefix",
        "_engine_threads",
        "_experiment",
        "_guard_max_requests_per_minute",
        "_guard_max_rows_per_owner_stage",
        "_guard_max_rows_per_request",
        "_guard_output_dither",
        "_hf_cache_dir",
        "_inference_key",
        "_inference_url",
        "_log_dir",
        "_logs",
        "_local_engine",
        "_model",
        "_model_id",
        "_preparation_key",
        "_preparation_url",
        "_process_lock",
        "_privacy_mode",
        "_processes",
        "_progress",
        "_proprietary_protocol",
        "_requires_preparation",
        "_push_key",
        "_rendezvous_capacity",
        "_rendezvous_max_bytes",
        "_reserved_ports",
        "_role_credentials",
        "_role_ids",
        "_role_urls",
        "_started",
        "_starting",
        "_startup_timeout",
        "_stopping",
        "_telemetry_endpoint",
        "_telemetry_token",
        "_tenseal_path",
        "_verification_component",
        "_verification_target_failure_bits",
        "_public_equalization_digest",
        "_remote_output_head",
        "_client_prefix_layers",
        "_client_linear_roles",
        "_weight_bits",
    )

    def __init__(
        self,
        model: Model,
        *,
        model_id: str,
        experiment: Experiment | None,
        engine_threads: int | None,
        weight_bits: int,
        activation_bits: int,
        correlation_mode: str,
        privacy_mode: str,
        proprietary_protocol: str,
        role_ids: tuple[str, ...],
        requires_preparation: bool,
        guard_max_rows_per_request: int,
        guard_max_rows_per_owner_stage: int,
        guard_max_requests_per_minute: int,
        guard_output_dither: int,
        verification_component: str | None,
        verification_target_failure_bits: int,
        public_equalization_digest: str | None,
        remote_output_head: bool,
        client_prefix_layers: int,
        client_linear_roles: tuple[str, ...],
        tenseal_path: str | None,
        hf_cache_dir: str | None,
        reserved_ports: tuple[int, ...],
        rendezvous_capacity: int,
        rendezvous_max_bytes: int,
        log_dir: Path | None,
        telemetry_endpoint: str | None,
        telemetry_token: str | None,
        credential_prefix: str,
        startup_timeout: float,
        progress: Callable[[str], None] | None,
    ) -> None:
        self._model = model
        self._model_id = model_id
        self._experiment = experiment
        self._engine_threads = engine_threads
        self._weight_bits = weight_bits
        self._activation_bits = activation_bits
        self._correlation_mode = correlation_mode
        self._privacy_mode = privacy_mode
        self._proprietary_protocol = proprietary_protocol
        self._role_ids = role_ids
        self._role_credentials: dict[str, str] = {}
        self._role_urls: dict[str, str] = {}
        self._local_engine: Any | None = None
        self._requires_preparation = requires_preparation
        self._guard_max_rows_per_request = guard_max_rows_per_request
        self._guard_max_rows_per_owner_stage = guard_max_rows_per_owner_stage
        self._guard_max_requests_per_minute = guard_max_requests_per_minute
        self._guard_output_dither = guard_output_dither
        self._verification_component = verification_component or "none"
        self._verification_target_failure_bits = verification_target_failure_bits
        self._public_equalization_digest = public_equalization_digest
        self._remote_output_head = remote_output_head
        self._client_prefix_layers = client_prefix_layers
        self._client_linear_roles = client_linear_roles
        self._tenseal_path = tenseal_path
        self._hf_cache_dir = hf_cache_dir
        self._reserved_ports = reserved_ports
        self._rendezvous_capacity = rendezvous_capacity
        self._rendezvous_max_bytes = rendezvous_max_bytes
        self._log_dir = log_dir
        self._telemetry_endpoint = telemetry_endpoint
        self._telemetry_token = telemetry_token
        self._credential_prefix = credential_prefix
        self._startup_timeout = startup_timeout
        self._progress = progress
        self._processes: dict[str, subprocess.Popen[Any]] = {}
        self._process_lock = threading.Lock()
        self._logs: dict[str, Any] = {}
        self._inference_url = ""
        self._preparation_url = ""
        self._inference_key = ""
        self._preparation_key = ""
        self._push_key = ""
        self._started = False
        self._starting = False
        self._closed = False
        self._stopping = threading.Event()

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def inference_url(self) -> str:
        if "inference" not in self._role_ids:
            raise TopologyError("local topology has no inference provider")
        if not self._started:
            raise TopologyError("local topology has not started")
        return self._inference_url

    @property
    def preparation_url(self) -> str:
        if not self._requires_preparation:
            raise TopologyError("local topology has no preparation role")
        if not self._started:
            raise TopologyError("local topology has not started")
        return self._preparation_url

    def worker_connections(self) -> dict[str, tuple[str, str]]:
        if set(self._role_ids) != {"worker_a", "worker_b"} or not self.is_healthy():
            raise TopologyError("two-worker topology is not running")
        return {
            role: (self._role_urls[role], self._role_credentials[role]) for role in self._role_ids
        }

    def worker_process_metrics(self) -> dict[str, dict[str, int | None]]:
        """Read authenticated in-process clocks; missing samples are never zero-filled."""
        connections = self.worker_connections()
        result: dict[str, dict[str, int | None]] = {}
        for role, (url, key) in connections.items():
            response = httpx.get(
                f"{url}/v1/offset-reference/metrics",
                headers={"authorization": f"Bearer {key}"},
                timeout=2.0,
            )
            response.raise_for_status()
            value = response.json()
            if (
                type(value) is not dict
                or value.get("schema") != "pllm.offset_worker_process_metrics.v1"
                or type(value.get("cpu_ns")) is not int
                or value["cpu_ns"] < 0
                or (
                    value.get("peak_rss_bytes") is not None
                    and (type(value["peak_rss_bytes"]) is not int or value["peak_rss_bytes"] < 0)
                )
            ):
                raise TopologyError("offset worker process metrics are malformed")
            result[role] = {
                "cpu_ns": value["cpu_ns"],
                "peak_rss_bytes": value["peak_rss_bytes"],
            }
        return result

    def client_model_ownership(self) -> dict[str, int | None]:
        """Snapshot checkpoint storage and distinct loaded weight/local arrays."""
        if self._role_ids or not self.is_healthy() or self._local_engine is None:
            raise TopologyError("client-owned model is not loaded")
        loaded = self._local_engine._model(self._model_id)
        try:
            artifact_bytes = sum(
                item.stat().st_size for item in loaded.store.root.iterdir() if item.is_file()
            )
        except OSError:
            artifact_bytes = None
        seen: set[tuple[int, int]] = set()
        loaded_bytes = 0
        arrays = [
            array
            for stage in loaded.stages.values()
            for array in (stage.weight.values, stage.weight.scales, stage.bias)
            if array is not None
        ]
        arrays.extend(loaded.local_tensors.values())
        for array in arrays:
            key = (int(array.__array_interface__["data"][0]), int(array.nbytes))
            if key not in seen:
                seen.add(key)
                loaded_bytes += key[1]
        return {
            "checkpoint_artifact_bytes": artifact_bytes,
            "loaded_weight_and_local_tensor_bytes": loaded_bytes,
            "cold_checkpoint_transfer_bytes": None,
            "client_peak_memory_bytes": None,
        }

    @property
    def started(self) -> bool:
        return self._started

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def requires_preparation(self) -> bool:
        return self._requires_preparation

    @property
    def statuses(self) -> tuple[RoleStatus, ...]:
        with self._process_lock:
            processes = dict(self._processes)
        roles = tuple(
            role for role in ("inference", "preparation", *self._role_ids) if role in self._role_ids
        )
        roles = tuple(dict.fromkeys(roles))
        return tuple(
            RoleStatus(
                role=role,
                url=self._role_urls.get(role, ""),
                pid=None if process is None else process.pid,
                running=process is not None and process.poll() is None,
            )
            for role in roles
            for process in (processes.get(role),)
        )

    @staticmethod
    def _free_port(excluded: set[int]) -> int:
        for _ in range(64):
            with socket.socket() as listener:
                listener.bind(("127.0.0.1", 0))
                port = int(listener.getsockname()[1])
            if port not in excluded:
                return port
        raise TopologyError("could not allocate a distinct loopback port")

    def _credential(self, used: set[str]) -> str:
        while True:
            value = f"{self._credential_prefix}_{secrets.token_urlsafe(24)}"
            if value not in used:
                used.add(value)
                return value

    def _model_options(self) -> list[str]:
        options = [
            "--model",
            self._model.source,
            "--model-id",
            self._model_id,
            "--model-kind",
            self._model.kind,
            "--weight-bits",
            str(self._weight_bits),
            "--activation-bits",
            str(self._activation_bits),
        ]
        if self._model.revision is not None:
            options.extend(("--revision", self._model.revision))
        if self._model.local_files_only:
            options.append("--local-files-only")
        if self._engine_threads is not None:
            options.extend(("--engine-threads", str(self._engine_threads)))
        if self._experiment is not None:
            kernels = self._experiment.pipeline.components.get("kernels")
            if kernels is not None and kernels.component == "pllm/apple-metal-int8/v1":
                options.extend(("--metal-min-rows", str(kernels.params["min_rows"])))
        if self._tenseal_path is not None:
            options.extend(("--tenseal-path", self._tenseal_path))
        if self._hf_cache_dir is not None:
            options.extend(("--hf-cache-dir", self._hf_cache_dir))
        if self._public_equalization_digest is not None:
            options.extend(("--public-equalization-digest", self._public_equalization_digest))
        if self._remote_output_head:
            options.append("--remote-output-head")
        if self._client_prefix_layers:
            options.extend(("--client-prefix-layers", str(self._client_prefix_layers)))
        if self._client_linear_roles:
            options.extend(("--client-linear-roles", ",".join(self._client_linear_roles)))
        if self._experiment is not None:
            from pllm.profiles import resolve_runtime_composition
            selected = resolve_runtime_composition(self._experiment.pipeline)
            if selected is not None and selected.prepared_output_encoding != "raw":
                options.extend(("--prepared-output-encoding", selected.prepared_output_encoding))
        return options

    def _commands(self, ports: dict[str, int]) -> dict[str, list[str]]:
        if set(self._role_ids) == {"worker_a", "worker_b"}:
            if self._model.kind != "huggingface" or not Path(self._model.source).is_dir():
                raise TopologyError("offset workers require a locally resolved checkpoint")
            return {
                role: [
                    sys.executable,
                    "-m",
                    "pllm.runtime.party",
                    self._model.source,
                    "--model-id",
                    self._model_id,
                    "--role",
                    role,
                    "--port",
                    str(ports[role]),
                    "--weight-bits",
                    str(self._weight_bits),
                    "--activation-bits",
                    str(self._activation_bits),
                ]
                for role in self._role_ids
            }
        inference_port = ports["inference"]
        preparation_port = ports.get("preparation", 0)
        common = [sys.executable, "-m", "pllm", "serve"]
        model = self._model_options()
        verification = [
            "--verification-component",
            self._verification_component,
            "--verification-target-failure-bits",
            str(self._verification_target_failure_bits),
        ]
        inference = [
            *common,
            "inference",
            "--privacy-mode",
            self._privacy_mode,
            "--protocol",
            self._proprietary_protocol,
            "--host",
            "127.0.0.1",
            "--port",
            str(inference_port),
            "--rendezvous-capacity",
            str(self._rendezvous_capacity),
            "--rendezvous-max-bytes",
            str(self._rendezvous_max_bytes),
            *verification,
            *model,
        ]
        if self._privacy_mode == "proprietary" and self._proprietary_protocol == "guarded":
            inference.extend(
                (
                    "--guard-max-rows-per-request",
                    str(self._guard_max_rows_per_request),
                    "--guard-max-rows-per-stage",
                    str(self._guard_max_rows_per_owner_stage),
                    "--guard-max-requests-per-minute",
                    str(self._guard_max_requests_per_minute),
                    "--guard-output-dither",
                    str(self._guard_output_dither),
                )
            )
        if self._privacy_mode == "public" and self._correlation_mode == "local-test":
            inference.append("--allow-insecure-local-correlations")
        if not self._requires_preparation:
            return {"inference": inference}
        preparation = [
            *common,
            "preparation",
            "--privacy-mode",
            "public",
            "--protocol",
            "guarded",
            "--host",
            "127.0.0.1",
            "--port",
            str(preparation_port),
            *verification,
            *model,
        ]
        return {"inference": inference, "preparation": preparation}

    def _environment(self, role: str) -> dict[str, str]:
        if role in {"worker_a", "worker_b"}:
            environment = {
                name: value
                for name, value in os.environ.items()
                if name
                in {
                    "HOME",
                    "PATH",
                    "VIRTUAL_ENV",
                    "PYTHONPATH",
                    "TMPDIR",
                    "LANG",
                    "SSL_CERT_FILE",
                    "SSL_CERT_DIR",
                    "DYLD_LIBRARY_PATH",
                }
            }
            environment["PLLM_OFFSET_WORKER_API_KEY"] = self._role_credentials[role]
            assert self._experiment is not None
            environment["PLLM_OFFSET_EXPERIMENT_JSON"] = self._experiment.canonical_bytes().decode()
            return environment
        environment = os.environ.copy()
        if self._experiment is not None:
            environment["PLLM_ROLE_EXPERIMENT_JSON"] = self._experiment.canonical_bytes().decode()
        if role == "inference":
            environment["PLLM_API_KEY"] = self._inference_key
            if self._push_key:
                environment["PLLM_PROVIDER_PUSH_API_KEY"] = self._push_key
            else:
                environment.pop("PLLM_PROVIDER_PUSH_API_KEY", None)
        else:
            environment.update(
                {
                    "PLLM_API_KEY": self._preparation_key,
                    "PLLM_INFERENCE_URL": self._inference_url,
                    "PLLM_PUSH_API_KEY": self._push_key,
                }
            )
        if self._telemetry_endpoint is not None:
            environment.update(
                {
                    "OTEL_EXPORTER_OTLP_ENDPOINT": self._telemetry_endpoint,
                    "OTEL_INSTRUMENTATION_HTTP_CAPTURE_HEADERS_CLIENT_REQUEST": "",
                    "OTEL_INSTRUMENTATION_HTTP_CAPTURE_HEADERS_SERVER_REQUEST": "",
                    "OTEL_METRIC_EXPORT_INTERVAL": "500",
                    "OTEL_EXPORTER_OTLP_HEADERS": f"x-pllm-otel-token={self._telemetry_token}",
                    "OTEL_SERVICE_NAME": f"pllm-{role}",
                    "PYTHONUNBUFFERED": "1",
                }
            )
        return environment

    def _spawn(self, role: str, command: list[str]) -> subprocess.Popen[Any]:
        with self._process_lock:
            if self._stopping.is_set():
                raise TopologyError("local topology stopped during startup")
            stdout: Any = None
            stderr: Any = None
            if self._log_dir is not None:
                self._log_dir.mkdir(parents=True, exist_ok=True)
                log = (self._log_dir / f"{role}.log").open("wb")
                self._logs[role] = log
                stdout = log
                stderr = subprocess.STDOUT
            process = _spawn_role_process(
                command,
                environment=self._environment(role),
                stdout=stdout,
                stderr=stderr,
            )
            self._processes[role] = process
            return process

    def _redacted_tail(self, role: str) -> str:
        if self._log_dir is None:
            return ""
        path = self._log_dir / f"{role}.log"
        try:
            text = path.read_text(errors="replace")[-4000:]
        except OSError:
            return ""
        for secret in (
            self._inference_key,
            self._preparation_key,
            self._push_key,
            *self._role_credentials.values(),
        ):
            if secret:
                text = text.replace(secret, "<redacted>")
        return re.sub(
            rf"{re.escape(self._credential_prefix)}_[A-Za-z0-9_-]+",
            "<redacted>",
            text,
        ).strip()

    def _wait(self, role: str, expected_role: str, url: str) -> None:
        process = self._processes[role]
        deadline = time.monotonic() + self._startup_timeout
        while time.monotonic() < deadline:
            if self._stopping.is_set():
                raise TopologyError("local topology stopped during startup")
            returncode = process.poll()
            if returncode is not None:
                detail = self._redacted_tail(role)
                suffix = f": {detail}" if detail else ""
                raise TopologyError(
                    f"local {role} service exited during startup ({returncode}){suffix}"
                )
            try:
                response = httpx.get(f"{url}/healthz", timeout=0.5)
                if response.status_code == 200 and response.json().get("role") == expected_role:
                    return
            except (httpx.HTTPError, ValueError):
                pass
            time.sleep(0.1)
        raise TopologyError(f"local {role} service did not become ready before timeout")

    def start(self) -> LocalTopology:
        with self._process_lock:
            if self._started or self._starting or self._closed:
                raise TopologyError("local topology cannot be started again")
            self._starting = True
        try:
            if self._model.kind == "tiny":
                from pllm.model_loader import resolve_model

                resolved = resolve_model(self._model, cache_dir=self._hf_cache_dir)
                if resolved.path is None:
                    raise TopologyError("tiny model did not produce a local source")
                self._model = Model.path(
                    str(resolved.path),
                    model_id=self._model_id,
                )
            if not self._role_ids:
                if self._experiment is None:
                    raise TopologyError("client-only execution requires a bound Experiment")
                from pllm.model_loader import resolve_model
                from .transformer_engine import MaskedTransformerEngine

                engine = MaskedTransformerEngine(
                    threads=self._engine_threads,
                    weight_bits=self._weight_bits,
                    activation_bits=self._activation_bits,
                    public_equalization_digest=self._public_equalization_digest,
                    remote_output_head=self._remote_output_head,
                    client_prefix_layers=self._client_prefix_layers,
                )

                def load() -> None:
                    source = resolve_model(self._model, cache_dir=self._hf_cache_dir)
                    asyncio.run(engine.load(source.manifest))

                # Model import includes async native stage loading. Keep the
                # supervisor usable from both sync CLI and async app factories.
                with ThreadPoolExecutor(max_workers=1) as loader:
                    loader.submit(load).result()
                if self._model_id not in engine.models:
                    raise TopologyError("client-owned model identity differs from Experiment")
                self._local_engine = engine
                with self._process_lock:
                    if self._closed or self._stopping.is_set():
                        raise TopologyError("local topology stopped during startup")
                    self._started = True
                    self._starting = False
                return self
            if set(self._role_ids) == {"worker_a", "worker_b"} and (
                self._model.kind != "huggingface" or not Path(self._model.source).is_dir()
            ):
                from pllm.model_loader import resolve_model

                source = resolve_model(self._model, cache_dir=self._hf_cache_dir)
                if source.path is None:
                    raise TopologyError(
                        "offset worker source did not resolve to a local checkpoint"
                    )
                self._model = Model.path(str(source.path), model_id=self._model_id)
            excluded = set(self._reserved_ports)
            ports: dict[str, int] = {}
            for role in self._role_ids:
                port = self._free_port(excluded)
                excluded.add(port)
                ports[role] = port
            self._role_urls = {role: f"http://127.0.0.1:{port}" for role, port in ports.items()}
            self._inference_url = self._role_urls.get("inference", "")
            self._preparation_url = self._role_urls.get("preparation", "")
            used: set[str] = set()
            self._role_credentials = {role: self._credential(used) for role in self._role_ids}
            self._inference_key = self._role_credentials.get("inference", "")
            self._preparation_key = self._role_credentials.get("preparation", "")
            self._push_key = self._credential(used) if self._requires_preparation else ""
            commands = self._commands(ports)
            for role in self._role_ids:
                if self._progress is not None:
                    self._progress(role)
                self._spawn(role, commands[role])
                expected = (
                    "trusted-preparation"
                    if role == "preparation"
                    else ("offset-worker" if role in {"worker_a", "worker_b"} else "inference")
                )
                self._wait(role, expected, self._role_urls[role])
            with self._process_lock:
                if self._closed or self._stopping.is_set():
                    raise TopologyError("local topology stopped during startup")
                self._started = True
                self._starting = False
            return self
        except BaseException as exc:
            try:
                self.close()
            except Exception as close_exc:
                with self._process_lock:
                    self._starting = False
                raise TopologyError(
                    "local topology startup failed and role cleanup did not complete"
                ) from close_exc
            with self._process_lock:
                self._starting = False
            if not isinstance(exc, Exception) or isinstance(exc, TopologyError):
                raise
            raise TopologyError(f"local topology startup failed: {type(exc).__name__}") from exc

    def is_healthy(self) -> bool:
        if not self._started or self._closed:
            return False
        if not self._role_ids:
            return self._local_engine is not None and self._model_id in self._local_engine.models
        expected_roles = {
            "inference": "inference",
            "preparation": "trusted-preparation",
            "worker_a": "offset-worker",
            "worker_b": "offset-worker",
        }
        for status in self.statuses:
            if not status.running:
                return False
            try:
                response = httpx.get(f"{status.url}/healthz", timeout=0.5)
                if (
                    response.status_code != 200
                    or response.json().get("role") != expected_roles[status.role]
                ):
                    return False
            except (httpx.HTTPError, ValueError):
                return False
        return True

    @staticmethod
    def _reject_overrides(overrides: dict[str, Any], forbidden: set[str]) -> None:
        conflict = sorted(set(overrides) & forbidden)
        if conflict:
            raise TopologyError(f"local topology routing cannot be overridden: {conflict}")

    def client(self, **overrides: Any) -> Any:
        if not self.is_healthy():
            raise TopologyError("local topology is not healthy")
        self._reject_overrides(
            overrides,
            {
                "base_url",
                "api_key",
                "model",
                "default_model",
                "preparation_base_url",
                "preparation_api_key",
                "correlation_mode",
                "experiment",
                "http_client",
                "preparation_http_client",
                "role_connections",
            },
        )
        from pllm.runtime.client import OpenAI

        if not self._role_ids:
            return OpenAI(
                default_model=self._model_id,
                experiment=self._experiment,
                local_engine=self._local_engine,
                **overrides,
            )
        if set(self._role_ids) == {"worker_a", "worker_b"}:
            if not overrides:
                return role_client(self._experiment, self.worker_connections())
            return OpenAI(
                default_model=self._model_id,
                experiment=self._experiment,
                role_connections=self.worker_connections(),
                **overrides,
            )
        options: dict[str, Any] = {
            "base_url": self._inference_url,
            "api_key": self._inference_key,
            "default_model": self._model_id,
            "correlation_mode": self._correlation_mode,
            "experiment": self._experiment,
        }
        if self._requires_preparation:
            options.update(
                {
                    "preparation_base_url": self._preparation_url,
                    "preparation_api_key": self._preparation_key,
                }
            )
        return OpenAI(**options, **overrides)

    def gateway_app(self, *, local_api_key: str, **overrides: Any) -> Any:
        if not self.is_healthy():
            raise TopologyError("local topology is not healthy")
        if type(local_api_key) is not str or not local_api_key:
            raise TopologyError("local gateway API key must be a nonempty string")
        self._reject_overrides(
            overrides,
            {
                "remote_base_url",
                "remote_api_key",
                "preparation_base_url",
                "preparation_api_key",
                "default_model",
                "correlation_mode",
                "experiment",
                "client",
                "http_client",
                "preparation_http_client",
                "role_connections",
                "local_api_key",
            },
        )
        from pllm.runtime.sidecar import create_sidecar_app

        if not self._role_ids:
            return create_sidecar_app(
                local_api_key=local_api_key,
                default_model=self._model_id,
                experiment=self._experiment,
                local_engine=self._local_engine,
                **overrides,
            )
        if set(self._role_ids) == {"worker_a", "worker_b"}:
            return create_sidecar_app(
                local_api_key=local_api_key,
                default_model=self._model_id,
                experiment=self._experiment,
                role_connections=self.worker_connections(),
                **overrides,
            )
        options: dict[str, Any] = {
            "remote_base_url": self._inference_url,
            "remote_api_key": self._inference_key,
            "default_model": self._model_id,
            "correlation_mode": self._correlation_mode,
            "experiment": self._experiment,
            "local_api_key": local_api_key,
        }
        if self._requires_preparation:
            options.update(
                {
                    "preparation_base_url": self._preparation_url,
                    "preparation_api_key": self._preparation_key,
                }
            )
        return create_sidecar_app(**options, **overrides)

    @staticmethod
    def _stop(process: subprocess.Popen[Any]) -> None:
        _stop_role_process(process)

    def close(self) -> None:
        self._stopping.set()
        with self._process_lock:
            processes = dict(self._processes)
        if self._closed and all(process.poll() is not None for process in processes.values()):
            return
        self._closed = True
        error: Exception | None = None
        for role in reversed(self._role_ids):
            process = processes.get(role)
            if process is None:
                continue
            try:
                self._stop(process)
            except Exception as exc:
                error = error or exc
        for log in self._logs.values():
            if not log.closed:
                log.close()
        if error is not None:
            raise error
        if self._local_engine is not None:
            self._local_engine.models.clear()
            self._local_engine = None
        self._inference_key = ""
        self._preparation_key = ""
        self._push_key = ""

    def __enter__(self) -> LocalTopology:
        if self._started and not self._closed:
            return self
        return self.start()

    def __exit__(self, _type: Any, _value: Any, _traceback: Any) -> None:
        self.close()


def build_roles(
    model: Model | Pipeline | Experiment | str,
    *,
    model_id: str | None = None,
    engine_threads: int | None = None,
    weight_bits: int | None = None,
    activation_bits: int | None = None,
    correlation_mode: str = "bfv",
    tenseal_path: str | None = None,
    hf_cache_dir: str | None = None,
    reserved_ports: Iterable[int] = (),
    rendezvous_capacity: int = 262_144,
    rendezvous_max_bytes: int = 2_147_483_648,
    log_dir: str | Path | None = None,
    telemetry_endpoint: str | None = None,
    telemetry_token: str | None = None,
    credential_prefix: str = "local",
    startup_timeout: float = 300.0,
    progress: Callable[[str], None] | None = None,
    docker: bool = False,
    docker_image: str | None = None,
    docker_network=None,
    wan=None,
    memory_limits: dict[str, int] | None = None,
) -> LocalTopology:
    if type(docker) is not bool:
        raise TypeError("docker must be a boolean")
    if wan is not None:
        from pllm.deployment import WanConditions
        if type(wan) is not WanConditions:
            raise TypeError("wan must be WanConditions")
        docker = True
    if docker_image is not None and not docker:
        raise ValueError("docker_image requires docker=True")
    if docker_network is not None and not docker:
        raise ValueError("docker_network requires docker=True")
    experiment = model if isinstance(model, Experiment) else None
    pipeline: Pipeline | None = None
    if experiment is not None:
        if experiment.deployment.kind == "network":
            raise TopologyError("network deployment requires open_execution live admission")
        experiment.resolve()
        pipeline = experiment.pipeline
    elif isinstance(model, Pipeline):
        pipeline = model
    from pllm.profiles import RuntimeComposition, resolve_runtime_composition

    runtime_options = RuntimeComposition(
        "public", "guarded", True, "bfv", "masked_transformer_v1", None
    )
    if pipeline is not None:
        resolved_options = resolve_runtime_composition(pipeline)
        if resolved_options is None:
            raise ValueError("local topology does not support this component composition")
        runtime_options = resolved_options
        for name, explicit, selected in (
            ("weight_bits", weight_bits, runtime_options.weight_bits),
            ("activation_bits", activation_bits, runtime_options.activation_bits),
        ):
            if explicit is not None and explicit != selected:
                raise ValueError(f"{name} conflicts with the pipeline quantization")
        kernels = pipeline.components.get("kernels")
        if kernels is not None and kernels.component == "pllm/cpu":
            configured_threads = kernels.params.get("threads")
            if engine_threads is None:
                engine_threads = configured_threads
            elif configured_threads is not None and engine_threads != configured_threads:
                raise ValueError("engine_threads conflicts with the pipeline kernel component")
        model = pipeline.model
    from pllm.roles.topology import graph_for_runtime

    role_ids = tuple(
        role.id for role in graph_for_runtime(runtime_options).roles if role.id != "client"
    )
    if memory_limits is not None and (
        type(memory_limits) is not dict or set(memory_limits) != set(role_ids)
        or any(type(value) is not int or value < 64 << 20 for value in memory_limits.values())
    ):
        raise ValueError("memory limits must cover each provider role with at least 64 MiB")
    if set(role_ids) not in (
        set(),
        {"inference"},
        {"inference", "preparation"},
        {"worker_a", "worker_b"},
    ):
        raise ValueError("role graph requires an unavailable local role implementation")
    if type(model) is str:
        model = Model(model)
    if not isinstance(model, Model):
        raise TypeError("model must be a Model, Pipeline, Experiment, or source string")
    if model.kind not in {"huggingface", "safetensors", "vllm", "mlx", "mlx-lm", "tiny"}:
        raise ValueError(f"local topology does not support model kind {model.kind!r}")
    resolved_id = model_id if model_id is not None else model.model_id or model.source
    if (
        type(resolved_id) is not str
        or not resolved_id
        or len(resolved_id) > 512
        or any(ord(character) < 32 for character in resolved_id)
    ):
        raise ValueError("model_id must be a nonempty printable string of at most 512 characters")
    if engine_threads is not None and (type(engine_threads) is not int or engine_threads <= 0):
        raise ValueError("engine_threads must be a positive integer")
    if weight_bits is None:
        weight_bits = runtime_options.weight_bits
    if activation_bits is None:
        activation_bits = runtime_options.activation_bits
    if weight_bits not in {4, 8} or activation_bits not in {4, 8}:
        raise ValueError("weight_bits and activation_bits must be 4 or 8")
    if correlation_mode not in {"bfv", "local-test"}:
        raise ValueError("correlation_mode must be bfv or local-test")
    reserved = tuple(reserved_ports)
    if any(type(port) is not int or not 1 <= port <= 65535 for port in reserved):
        raise ValueError("reserved_ports must contain valid TCP ports")
    if len(set(reserved)) != len(reserved):
        raise ValueError("reserved_ports must be unique")
    if type(rendezvous_capacity) is not int or rendezvous_capacity <= 0:
        raise ValueError("rendezvous_capacity must be a positive integer")
    if type(rendezvous_max_bytes) is not int or rendezvous_max_bytes <= 0:
        raise ValueError("rendezvous_max_bytes must be a positive integer")
    if (
        not isinstance(startup_timeout, (int, float))
        or isinstance(startup_timeout, bool)
        or not math.isfinite(startup_timeout)
        or startup_timeout <= 0
    ):
        raise ValueError("startup_timeout must be finite and positive")
    if re.fullmatch(r"[A-Za-z0-9_-]{1,24}", credential_prefix) is None:
        raise ValueError("credential_prefix is invalid")
    if (telemetry_endpoint is None) != (telemetry_token is None):
        raise ValueError("telemetry endpoint and token must be supplied together")
    for name, value in (("tenseal_path", tenseal_path), ("hf_cache_dir", hf_cache_dir)):
        if value is not None and (type(value) is not str or not value):
            raise ValueError(f"{name} must be a nonempty string")
    if telemetry_endpoint is not None:
        if (
            type(telemetry_endpoint) is not str
            or type(telemetry_token) is not str
            or not telemetry_token
        ):
            raise ValueError("telemetry endpoint and token must be nonempty strings")
        endpoint = urlsplit(telemetry_endpoint)
        try:
            endpoint_port = endpoint.port
        except ValueError as exc:
            raise ValueError("telemetry endpoint has an invalid port") from exc
        if (
            endpoint.scheme not in {"http", "https"}
            or endpoint.hostname is None
            or endpoint.username is not None
            or endpoint.password is not None
            or endpoint.path not in {"", "/"}
            or endpoint.query
            or endpoint.fragment
            or endpoint_port is not None
            and not 1 <= endpoint_port <= 65535
        ):
            raise ValueError("telemetry endpoint must be an HTTP(S) URL without credentials")
    if progress is not None and not callable(progress):
        raise TypeError("progress must be callable")
    topology_type = LocalTopology
    extra = {}
    if docker:
        from .docker_roles import DockerTopology
        topology_type = DockerTopology
        extra["docker_image"] = docker_image
        extra["docker_network"] = docker_network
        extra["wan"] = wan
        extra["memory_limits"] = memory_limits
    return topology_type(
        model,
        model_id=resolved_id,
        experiment=experiment,
        engine_threads=engine_threads,
        weight_bits=weight_bits,
        activation_bits=activation_bits,
        correlation_mode=correlation_mode,
        privacy_mode=runtime_options.privacy_mode,
        proprietary_protocol=runtime_options.proprietary_protocol,
        role_ids=role_ids,
        requires_preparation=runtime_options.requires_preparation,
        guard_max_rows_per_request=runtime_options.guard_max_rows_per_request,
        guard_max_rows_per_owner_stage=runtime_options.guard_max_rows_per_owner_stage,
        guard_max_requests_per_minute=runtime_options.guard_max_requests_per_minute,
        guard_output_dither=runtime_options.output_dither_bound,
        verification_component=runtime_options.verification_component,
        verification_target_failure_bits=runtime_options.verification_target_failure_bits,
        public_equalization_digest=runtime_options.public_equalization_digest,
        remote_output_head=runtime_options.remote_output_head,
        client_prefix_layers=runtime_options.client_prefix_layers,
        client_linear_roles=runtime_options.client_linear_roles,
        tenseal_path=tenseal_path,
        hf_cache_dir=hf_cache_dir,
        reserved_ports=reserved,
        rendezvous_capacity=rendezvous_capacity,
        rendezvous_max_bytes=rendezvous_max_bytes,
        log_dir=None if log_dir is None else Path(log_dir),
        telemetry_endpoint=telemetry_endpoint,
        telemetry_token=telemetry_token,
        credential_prefix=credential_prefix,
        startup_timeout=float(startup_timeout),
        progress=progress,
        **extra,
    )


def serve_local(
    model: Model | Pipeline | Experiment | str,
    *,
    model_id: str | None = None,
    engine_threads: int | None = None,
    weight_bits: int | None = None,
    activation_bits: int | None = None,
    correlation_mode: str = "bfv",
    tenseal_path: str | None = None,
    hf_cache_dir: str | None = None,
    reserved_ports: Iterable[int] = (),
    rendezvous_capacity: int = 262_144,
    rendezvous_max_bytes: int = 2_147_483_648,
    log_dir: str | Path | None = None,
    telemetry_endpoint: str | None = None,
    telemetry_token: str | None = None,
    credential_prefix: str = "local",
    startup_timeout: float = 300.0,
    progress: Callable[[str], None] | None = None,
) -> LocalTopology:
    return build_roles(
        model,
        model_id=model_id,
        engine_threads=engine_threads,
        weight_bits=weight_bits,
        activation_bits=activation_bits,
        correlation_mode=correlation_mode,
        tenseal_path=tenseal_path,
        hf_cache_dir=hf_cache_dir,
        reserved_ports=reserved_ports,
        rendezvous_capacity=rendezvous_capacity,
        rendezvous_max_bytes=rendezvous_max_bytes,
        log_dir=log_dir,
        telemetry_endpoint=telemetry_endpoint,
        telemetry_token=telemetry_token,
        credential_prefix=credential_prefix,
        startup_timeout=startup_timeout,
        progress=progress,
    ).start()


__all__ = [
    "LocalTopology",
    "RoleStatus",
    "TopologyError",
    "build_roles",
    "serve_local",
]
