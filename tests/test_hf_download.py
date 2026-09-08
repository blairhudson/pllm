from __future__ import annotations

import hashlib
import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from pllm.runtime.hf_download import _complete_model_cache, _parallel_download


def test_parallel_download_resumes_completed_ranges(tmp_path: Path):
    content = bytes(range(256)) * 1024
    chunk_size = 64 * 1024
    requested_ranges: list[str] = []

    class RangeHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            range_header = self.headers["Range"]
            requested_ranges.append(range_header)
            start_text, end_text = range_header.removeprefix("bytes=").split("-", 1)
            start, end = int(start_text), int(end_text)
            payload = content[start : end + 1]
            self.send_response(206)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Content-Range", f"bytes {start}-{end}/{len(content)}")
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), RangeHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    target = tmp_path / "model.safetensors"
    partial = tmp_path / "model.safetensors.partial"
    state = tmp_path / "model.safetensors.partial.json"
    try:
        with partial.open("wb") as stream:
            stream.truncate(len(content))
        with partial.open("r+b") as stream:
            stream.write(content[: 2 * chunk_size])
        state.write_text(
            json.dumps(
                {
                    "schema": 1,
                    "size": len(content),
                    "sha256": hashlib.sha256(content).hexdigest(),
                    "chunk_size": chunk_size,
                    "complete": [0, 1],
                }
            ),
            encoding="utf-8",
        )

        _parallel_download(
            url=f"http://127.0.0.1:{server.server_port}/model.safetensors",
            target=target,
            size=len(content),
            expected_sha256=hashlib.sha256(content).hexdigest(),
            headers={},
            workers=2,
            source_id="org/model@commit:model.safetensors",
            chunk_size=chunk_size,
            show_progress=False,
        )
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join()

    assert target.read_bytes() == content
    assert requested_ranges == ["bytes=131072-262143"]
    assert not partial.exists()
    assert not state.exists()


def test_complete_model_cache_rejects_same_size_corruption(tmp_path: Path):
    model = tmp_path / "model.safetensors"
    model.write_bytes(b"correct")
    (tmp_path / ".pllm-model.json").write_text(
        json.dumps(
            {
                "repo_id": "org/model",
                "revision": None,
                "files": [
                    {
                        "path": model.name,
                        "size": model.stat().st_size,
                        "sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    assert _complete_model_cache(tmp_path, "org/model", None)
    model.write_bytes(b"corrupt")
    assert not _complete_model_cache(tmp_path, "org/model", None)


def test_parallel_download_checkpoints_inside_request(tmp_path: Path):
    content = bytes(range(256)) * 1024
    chunk_size = 64 * 1024
    requested_ranges: list[str] = []

    class InterruptedRangeHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            range_header = self.headers["Range"]
            requested_ranges.append(range_header)
            start_text, end_text = range_header.removeprefix("bytes=").split("-", 1)
            start, end = int(start_text), int(end_text)
            payload = content[start : end + 1]
            self.send_response(206)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Content-Range", f"bytes {start}-{end}/{len(content)}")
            self.end_headers()
            if len(requested_ranges) == 1:
                self.wfile.write(payload[:chunk_size])
                self.wfile.flush()
                self.connection.shutdown(socket.SHUT_RDWR)
                self.connection.close()
                return
            self.wfile.write(payload)

        def log_message(self, format, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), InterruptedRangeHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    target = tmp_path / "model.safetensors"
    try:
        _parallel_download(
            url=f"http://127.0.0.1:{server.server_port}/model.safetensors",
            target=target,
            size=len(content),
            expected_sha256=hashlib.sha256(content).hexdigest(),
            headers={},
            workers=1,
            source_id="org/model@commit:model.safetensors",
            chunk_size=chunk_size,
            show_progress=False,
        )
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join()

    assert target.read_bytes() == content
    assert requested_ranges == ["bytes=0-262143", "bytes=65536-262143"]
