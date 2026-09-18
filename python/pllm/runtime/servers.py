from __future__ import annotations

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
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx

from pllm.configuration import Experiment, Model, Pipeline


class TopologyError(RuntimeError):
    pass


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
        "_hf_cache_dir",
        "_inference_key",
        "_inference_url",
        "_log_dir",
        "_logs",
        "_model",
        "_model_id",
        "_preparation_key",
        "_preparation_url",
        "_process_lock",
        "_processes",
        "_progress",
        "_push_key",
        "_rendezvous_capacity",
        "_rendezvous_max_bytes",
        "_reserved_ports",
        "_started",
        "_starting",
        "_startup_timeout",
        "_stopping",
        "_telemetry_endpoint",
        "_telemetry_token",
        "_tenseal_path",
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
        if not self._started:
            raise TopologyError("local topology has not started")
        return self._inference_url

    @property
    def preparation_url(self) -> str:
        if not self._started:
            raise TopologyError("local topology has not started")
        return self._preparation_url

    @property
    def started(self) -> bool:
        return self._started

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def statuses(self) -> tuple[RoleStatus, ...]:
        with self._process_lock:
            processes = dict(self._processes)
        return tuple(
            RoleStatus(
                role=role,
                url=self._inference_url if role == "inference" else self._preparation_url,
                pid=None if process is None else process.pid,
                running=process is not None and process.poll() is None,
            )
            for role in ("inference", "preparation")
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
        if self._tenseal_path is not None:
            options.extend(("--tenseal-path", self._tenseal_path))
        if self._hf_cache_dir is not None:
            options.extend(("--hf-cache-dir", self._hf_cache_dir))
        return options

    def _commands(self, inference_port: int, preparation_port: int) -> dict[str, list[str]]:
        common = [sys.executable, "-m", "pllm.runtime.cli"]
        model = self._model_options()
        inference = [
            *common,
            "inference",
            "--privacy-mode",
            "public",
            "--protocol",
            "guarded",
            "--host",
            "127.0.0.1",
            "--port",
            str(inference_port),
            "--rendezvous-capacity",
            str(self._rendezvous_capacity),
            "--rendezvous-max-bytes",
            str(self._rendezvous_max_bytes),
            *model,
        ]
        if self._correlation_mode == "local-test":
            inference.append("--allow-insecure-local-correlations")
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
            *model,
        ]
        return {"inference": inference, "preparation": preparation}

    def _environment(self, role: str) -> dict[str, str]:
        environment = os.environ.copy()
        if role == "inference":
            environment.update({
                "PLLM_API_KEY": self._inference_key,
                "PLLM_PROVIDER_PUSH_API_KEY": self._push_key,
            })
        else:
            environment.update({
                "PLLM_API_KEY": self._preparation_key,
                "PLLM_INFERENCE_URL": self._inference_url,
                "PLLM_PUSH_API_KEY": self._push_key,
            })
        if self._telemetry_endpoint is not None:
            environment.update({
                "OTEL_EXPORTER_OTLP_ENDPOINT": self._telemetry_endpoint,
                "OTEL_INSTRUMENTATION_HTTP_CAPTURE_HEADERS_CLIENT_REQUEST": "",
                "OTEL_INSTRUMENTATION_HTTP_CAPTURE_HEADERS_SERVER_REQUEST": "",
                "OTEL_METRIC_EXPORT_INTERVAL": "500",
                "OTEL_EXPORTER_OTLP_HEADERS": f"x-pllm-otel-token={self._telemetry_token}",
                "OTEL_SERVICE_NAME": f"pllm-{role}",
                "PYTHONUNBUFFERED": "1",
            })
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
            process = subprocess.Popen(
                command,
                cwd=Path.cwd(),
                env=self._environment(role),
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
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
        for secret in (self._inference_key, self._preparation_key, self._push_key):
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
            excluded = set(self._reserved_ports)
            inference_port = self._free_port(excluded)
            excluded.add(inference_port)
            preparation_port = self._free_port(excluded)
            self._inference_url = f"http://127.0.0.1:{inference_port}"
            self._preparation_url = f"http://127.0.0.1:{preparation_port}"
            used: set[str] = set()
            self._inference_key = self._credential(used)
            self._preparation_key = self._credential(used)
            self._push_key = self._credential(used)
            commands = self._commands(inference_port, preparation_port)
            if self._progress is not None:
                self._progress("inference")
            self._spawn("inference", commands["inference"])
            self._wait("inference", "inference", self._inference_url)
            if self._progress is not None:
                self._progress("preparation")
            self._spawn("preparation", commands["preparation"])
            self._wait("preparation", "trusted-preparation", self._preparation_url)
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
        for status, expected in zip(
            self.statuses,
            ("inference", "trusted-preparation"),
            strict=True,
        ):
            if not status.running:
                return False
            try:
                response = httpx.get(f"{status.url}/healthz", timeout=0.5)
                if response.status_code != 200 or response.json().get("role") != expected:
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
            },
        )
        from pllm.runtime.client import OpenAI

        return OpenAI(
            base_url=self._inference_url,
            api_key=self._inference_key,
            default_model=self._model_id,
            preparation_base_url=self._preparation_url,
            preparation_api_key=self._preparation_key,
            correlation_mode=self._correlation_mode,
            experiment=self._experiment,
            **overrides,
        )

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
                "local_api_key",
            },
        )
        from pllm.runtime.sidecar import create_sidecar_app

        return create_sidecar_app(
            remote_base_url=self._inference_url,
            remote_api_key=self._inference_key,
            preparation_base_url=self._preparation_url,
            preparation_api_key=self._preparation_key,
            default_model=self._model_id,
            correlation_mode=self._correlation_mode,
            experiment=self._experiment,
            local_api_key=local_api_key,
            **overrides,
        )

    @staticmethod
    def _stop(process: subprocess.Popen[Any]) -> None:
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

    def close(self) -> None:
        self._stopping.set()
        with self._process_lock:
            processes = dict(self._processes)
        if self._closed and all(process.poll() is not None for process in processes.values()):
            return
        self._closed = True
        error: Exception | None = None
        for role in ("preparation", "inference"):
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
    weight_bits: int = 8,
    activation_bits: int = 8,
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
    experiment = model if isinstance(model, Experiment) else None
    pipeline: Pipeline | None = None
    if experiment is not None:
        experiment.resolve()
        pipeline = experiment.pipeline
    elif isinstance(model, Pipeline):
        pipeline = model
    if pipeline is not None:
        kernels = pipeline.components.get("kernels")
        if kernels is not None and kernels.component == "pllm/cpu":
            configured_threads = kernels.params.get("threads")
            if engine_threads is None:
                engine_threads = configured_threads
            elif configured_threads is not None and engine_threads != configured_threads:
                raise ValueError("engine_threads conflicts with the pipeline kernel component")
        model = pipeline.model
    if type(model) is str:
        model = Model(model)
    if type(model) is not Model:
        raise TypeError("model must be a Model, Pipeline, Experiment, or source string")
    if model.kind not in {"huggingface", "safetensors", "vllm", "mlx", "mlx-lm"}:
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
        if type(telemetry_endpoint) is not str or type(telemetry_token) is not str or not telemetry_token:
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
            or endpoint_port is not None and not 1 <= endpoint_port <= 65535
        ):
            raise ValueError("telemetry endpoint must be an HTTP(S) URL without credentials")
    if progress is not None and not callable(progress):
        raise TypeError("progress must be callable")
    return LocalTopology(
        model,
        model_id=resolved_id,
        experiment=experiment,
        engine_threads=engine_threads,
        weight_bits=weight_bits,
        activation_bits=activation_bits,
        correlation_mode=correlation_mode,
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
    )


def serve_local(
    model: Model | Pipeline | Experiment | str,
    *,
    model_id: str | None = None,
    engine_threads: int | None = None,
    weight_bits: int = 8,
    activation_bits: int = 8,
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
