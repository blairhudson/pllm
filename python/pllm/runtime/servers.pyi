from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from pllm.configuration import Experiment, Model, Pipeline
from pllm.runtime.client import OpenAI

class TopologyError(RuntimeError): ...

class RoleStatus:
    role: str
    url: str
    pid: int | None
    running: bool

class LocalTopology:
    @property
    def model_id(self) -> str: ...
    @property
    def inference_url(self) -> str: ...
    @property
    def preparation_url(self) -> str: ...
    @property
    def started(self) -> bool: ...
    @property
    def closed(self) -> bool: ...
    @property
    def requires_preparation(self) -> bool: ...
    @property
    def statuses(self) -> tuple[RoleStatus, ...]: ...
    def start(self) -> LocalTopology: ...
    def is_healthy(self) -> bool: ...
    def client(self, **overrides: Any) -> OpenAI: ...
    def gateway_app(self, *, local_api_key: str, **overrides: Any) -> FastAPI: ...
    def close(self) -> None: ...
    def __enter__(self) -> LocalTopology: ...
    def __exit__(self, _type: Any, _value: Any, _traceback: Any) -> None: ...

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
) -> LocalTopology: ...

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
) -> LocalTopology: ...
