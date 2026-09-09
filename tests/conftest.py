from __future__ import annotations

import os
import socket
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import pytest
import uvicorn

from pllm.runtime import GatewayConfig, create_app
from pllm.runtime.preparation_server import create_preparation_app
from pllm.runtime.transformer_engine import MaskedTransformerEngine


@pytest.fixture(autouse=True)
def _isolate_default_bundle_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Keep ordinary tests from reading or writing the user's persistent cache."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache-home"))


@dataclass
class RunningGateway:
    base_url: str
    api_key: str
    server: uvicorn.Server
    thread: threading.Thread
    audit: list[tuple[str, bytes]] = field(default_factory=list)
    push_api_key: str = ""

    def close(self) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=10)
        if self.thread.is_alive():
            raise RuntimeError("gateway did not stop")


def _free_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def start_gateway(*, bfv: bool = False, backends=(), engines=None, privacy_mode: str = "public") -> RunningGateway:
    api_key = "test-key"
    push_api_key = "correction-push-test-key"
    audit: list[tuple[str, bytes]] = []
    config = GatewayConfig(
        api_keys=(api_key,),
        privacy_mode=privacy_mode,
        allow_insecure_local_correlations=True,
        tenseal_path=os.environ.get("HE_OPENAI_PYDEPS") if bfv else None,
        backends=tuple(backends),
        max_batch_size=32,
        max_batch_wait_ms=1.5,
        provider_push_api_key=push_api_key,
    )
    app = create_app(config, engines=engines, audit_hook=lambda kind, payload: audit.append((kind, payload)))
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error", access_log=False))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(500):
        if server.started:
            break
        if not thread.is_alive():
            raise RuntimeError("gateway failed to start")
        time.sleep(0.01)
    if not server.started:
        raise RuntimeError("gateway start timeout")
    return RunningGateway(
        f"http://127.0.0.1:{port}", api_key, server, thread, audit, push_api_key
    )


def start_preparation(
    engine: MaskedTransformerEngine,
    inference_url: str,
    push_api_key: str,
) -> RunningGateway:
    api_key = "preparation-test-key"
    audit: list[tuple[str, bytes]] = []
    config = GatewayConfig(
        api_keys=(api_key,),
        max_batch_size=32,
        preparation_inference_url=inference_url,
        preparation_push_api_key=push_api_key,
    )
    app = create_preparation_app(
        config,
        engine,
        audit_hook=lambda kind, payload: audit.append((kind, payload)),
    )
    port = _free_port()
    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host="127.0.0.1",
            port=port,
            log_level="error",
            access_log=False,
        )
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(500):
        if server.started:
            break
        if not thread.is_alive():
            raise RuntimeError("preparation service failed to start")
        time.sleep(0.01)
    if not server.started:
        raise RuntimeError("preparation service start timeout")
    return RunningGateway(f"http://127.0.0.1:{port}", api_key, server, thread, audit)


@pytest.fixture
def gateway():
    value = start_gateway()
    try:
        yield value
    finally:
        value.close()
