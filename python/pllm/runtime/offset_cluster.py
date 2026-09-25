"""Launch two loopback-only offset workers for a bounded research comparison."""

from __future__ import annotations

import os
import secrets
import sys
import time
from pathlib import Path
from typing import Any

import httpx

from pllm.runtime.servers import LocalTopology, _spawn_role_process, _stop_role_process


class LocalOffsetCluster:
    """Supervise child processes; co-location is not non-collusion evidence."""

    def __init__(
        self, checkpoint: Path, *, model_id: str, second_checkpoint: Path | None = None,
    ) -> None:
        self.checkpoint = checkpoint
        self.second_checkpoint = second_checkpoint or checkpoint
        self.model_id = model_id
        self.processes: list[Any] = []
        self.clients: list[httpx.Client] = []
        self.keys: list[str] = []

    def __enter__(self) -> LocalOffsetCluster:
        if self.processes:
            raise RuntimeError("offset worker cluster cannot be started twice")
        try:
            first_port = LocalTopology._free_port(set())
            ports = (first_port, LocalTopology._free_port({first_port}))
            for role, port, source in zip(
                ("worker_a", "worker_b"), ports,
                (self.checkpoint, self.second_checkpoint), strict=True,
            ):
                api_key = secrets.token_hex(32)
                environment = {
                    name: value for name, value in os.environ.items()
                    if name in {
                        "HOME", "PATH", "VIRTUAL_ENV", "PYTHONPATH", "TMPDIR",
                        "LANG", "LC_ALL", "SSL_CERT_FILE", "RUST_LOG",
                    }
                }
                environment["PLLM_OFFSET_WORKER_API_KEY"] = api_key
                worker = _spawn_role_process(
                    [sys.executable, "-m", "pllm.runtime.offset_worker", str(source),
                     "--model-id", self.model_id, "--role", role, "--port", str(port)],
                    environment=environment, discard_output=True,
                )
                self.processes.append(worker)
                self.clients.append(httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=10.0))
                self.keys.append(api_key)
            if self.processes[0].pid == self.processes[1].pid:
                raise RuntimeError("offset worker processes are not distinct")
            for role, worker, client in zip(
                ("worker_a", "worker_b"), self.processes, self.clients, strict=True,
            ):
                deadline = time.monotonic() + 20.0
                while True:
                    if worker.poll() is not None:
                        raise RuntimeError("offset worker exited before health admission")
                    try:
                        response = client.get("/health")
                        if response.status_code == 200 and response.json() == {
                            "status": "ok", "role": role,
                        }:
                            break
                    except httpx.ConnectError:
                        pass
                    if time.monotonic() >= deadline:
                        raise RuntimeError("offset worker did not become ready")
                    time.sleep(0.05)
            return self
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        for client in self.clients:
            client.close()
        for worker in self.processes:
            _stop_role_process(worker)
        self.clients.clear()
        self.processes.clear()
        self.keys.clear()

    def snapshot_process_metrics(self) -> dict[str, dict[str, int | None]]:
        """Sample authenticated worker process CPU and lifetime peak RSS."""
        if len(self.clients) != 2 or len(self.keys) != 2:
            raise RuntimeError("offset workers have not started")
        snapshots: dict[str, dict[str, int | None]] = {}
        for role, client, key in zip(
            ("worker_a", "worker_b"), self.clients, self.keys, strict=True,
        ):
            response = client.get(
                "/v1/offset-reference/metrics",
                headers={"authorization": f"Bearer {key}"},
            )
            if response.status_code != 200:
                raise RuntimeError("offset worker process metrics are unavailable")
            value = response.json()
            if (
                type(value) is not dict
                or set(value) != {"schema", "cpu_ns", "peak_rss_bytes"}
                or value["schema"] != "pllm.offset_worker_process_metrics.v1"
                or type(value["cpu_ns"]) is not int or value["cpu_ns"] < 0
                or (value["peak_rss_bytes"] is not None and (
                    type(value["peak_rss_bytes"]) is not int
                    or value["peak_rss_bytes"] <= 0
                ))
            ):
                raise RuntimeError("offset worker process metrics are malformed")
            snapshots[role] = {
                "cpu_ns": value["cpu_ns"], "peak_rss_bytes": value["peak_rss_bytes"],
            }
        return snapshots

    def __exit__(self, *_exc: object) -> None:
        self.close()


__all__ = ["LocalOffsetCluster"]
