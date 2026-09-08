from __future__ import annotations

import os
import socket
import threading
import time
from dataclasses import dataclass, field

import pytest
import uvicorn

from pllm.runtime import GatewayConfig, create_app


@dataclass
class RunningGateway:
    base_url: str
    api_key: str
    server: uvicorn.Server
    thread: threading.Thread
    audit: list[tuple[str, bytes]] = field(default_factory=list)

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
    audit: list[tuple[str, bytes]] = []
    config = GatewayConfig(
        api_keys=(api_key,),
        privacy_mode=privacy_mode,
        allow_insecure_local_correlations=True,
        tenseal_path=os.environ.get("HE_OPENAI_PYDEPS") if bfv else None,
        backends=tuple(backends),
        max_batch_size=32,
        max_batch_wait_ms=1.5,
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
    return RunningGateway(f"http://127.0.0.1:{port}", api_key, server, thread, audit)


@pytest.fixture
def gateway():
    value = start_gateway()
    try:
        yield value
    finally:
        value.close()
