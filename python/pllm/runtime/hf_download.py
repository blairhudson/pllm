from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

import httpx
from filelock import FileLock
from rich.console import Console
from rich.progress import BarColumn, DownloadColumn, Progress, TextColumn, TransferSpeedColumn

_PARALLEL_THRESHOLD = 8 * 1024 * 1024
_CHUNK_SIZE = 512 * 1024
_DEFAULT_WORKERS = 32
_MANIFEST_NAME = ".pllm-model.json"


class HuggingFaceDownloadError(RuntimeError):
    pass


def _cache_root(cache_dir: str | os.PathLike[str] | None) -> Path:
    if cache_dir is not None:
        return Path(cache_dir).expanduser() / "pllm-models"
    configured = os.getenv("PLLM_HF_MODEL_CACHE")
    if configured:
        return Path(configured).expanduser()
    hub_cache = os.getenv("HF_HUB_CACHE")
    if hub_cache:
        return Path(hub_cache).expanduser() / "pllm-models"
    hf_home = os.getenv("HF_HOME")
    if hf_home:
        return Path(hf_home).expanduser() / "pllm-models"
    return Path.home() / ".cache" / "pllm" / "models"


def _model_cache_path(
    repo_id: str,
    revision: str | None,
    cache_dir: str | os.PathLike[str] | None,
) -> Path:
    identity = f"{repo_id}@{revision or 'main'}"
    readable = re.sub(r"[^A-Za-z0-9._-]+", "--", identity).strip("-.")[:80]
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
    return _cache_root(cache_dir) / f"{readable}-{digest}"


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _complete_model_cache(path: Path, repo_id: str, revision: str | None) -> bool:
    manifest = _read_json(path / _MANIFEST_NAME)
    if manifest is None or manifest.get("repo_id") != repo_id or manifest.get("revision") != revision:
        return False
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        return False
    for entry in files:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            return False
        file_path = path / entry["path"]
        if not file_path.is_file() or file_path.stat().st_size != entry.get("size"):
            return False
        expected_sha256 = entry.get("sha256")
        if isinstance(expected_sha256, str) and _sha256(file_path) != expected_sha256:
            return False
    return True


def cached_huggingface_model(
    repo_id: str,
    *,
    revision: str | None,
    cache_dir: str | os.PathLike[str] | None,
) -> Path | None:
    target = _model_cache_path(repo_id, revision, cache_dir)
    return target.resolve() if _complete_model_cache(target, repo_id, revision) else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _worker_count() -> int:
    raw = os.getenv("PLLM_HF_DOWNLOAD_WORKERS", str(_DEFAULT_WORKERS))
    try:
        workers = int(raw)
    except ValueError as exc:
        raise HuggingFaceDownloadError("PLLM_HF_DOWNLOAD_WORKERS must be an integer") from exc
    if workers < 1 or workers > 64:
        raise HuggingFaceDownloadError("PLLM_HF_DOWNLOAD_WORKERS must be between 1 and 64")
    return workers


def _write_all(stream: Any, data: bytes) -> None:
    view = memoryview(data)
    while view:
        written = stream.write(view)
        if written <= 0:
            raise OSError("failed to write downloaded model chunk")
        view = view[written:]


def _parallel_download(
    *,
    url: str,
    target: Path,
    size: int,
    expected_sha256: str | None,
    headers: dict[str, str],
    workers: int,
    source_id: str,
    chunk_size: int = _CHUNK_SIZE,
    show_progress: bool = True,
) -> None:
    if target.is_file() and target.stat().st_size == size:
        if expected_sha256 is None or _sha256(target) == expected_sha256:
            return

    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(f"{target.name}.partial")
    state_path = target.with_name(f"{target.name}.partial.json")
    identity = {
        "schema": 1,
        "size": size,
        "sha256": expected_sha256,
        "chunk_size": chunk_size,
        "source_id": source_id,
    }
    state = _read_json(state_path)
    if state is not None and state.get("source_id") is None and expected_sha256 is not None:
        state["source_id"] = source_id
        _write_json(state_path, state)
    if state is None or any(state.get(key) != value for key, value in identity.items()):
        state = {**identity, "complete": []}
        with partial.open("wb") as stream:
            stream.truncate(size)
        _write_json(state_path, state)
    elif not partial.is_file() or partial.stat().st_size != size:
        state["complete"] = []
        with partial.open("wb") as stream:
            stream.truncate(size)
        _write_json(state_path, state)

    chunk_count = (size + chunk_size - 1) // chunk_size
    complete = {
        int(index)
        for index in state.get("complete", [])
        if isinstance(index, int) and 0 <= index < chunk_count
    }
    completed_bytes = sum(min(chunk_size, size - index * chunk_size) for index in complete)
    lock = threading.Lock()
    stopped = threading.Event()
    client = httpx.Client(
        follow_redirects=True,
        limits=httpx.Limits(max_connections=workers, max_keepalive_connections=workers),
        timeout=httpx.Timeout(120.0, connect=30.0),
    )
    console = Console(stderr=True)
    console.print(
        f"Downloading {target.name} with {workers} resumable streams "
        f"({completed_bytes / (1024 * 1024):.1f} MiB already complete)"
    )
    progress = Progress(
        TextColumn("{task.description}"),
        BarColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        console=console,
        disable=not show_progress,
    )
    task_id = progress.add_task(target.name, total=size, completed=completed_bytes)

    def download_range(first_index: int, end_index: int) -> None:
        index = first_index
        for attempt in range(4):
            if stopped.is_set():
                raise InterruptedError("download cancelled")
            start = index * chunk_size
            end = min(size, end_index * chunk_size) - 1
            request_headers = {
                **headers,
                "Accept-Encoding": "identity",
                "Range": f"bytes={start}-{end}",
            }
            buffer = bytearray()
            try:
                with (
                    partial.open("r+b", buffering=0) as output,
                    client.stream("GET", url, headers=request_headers) as response,
                ):
                    output.seek(start)
                    response.raise_for_status()
                    if response.status_code != 206:
                        raise HuggingFaceDownloadError(
                            f"server ignored byte range for {target.name}: HTTP {response.status_code}"
                        )
                    expected_range = f"bytes {start}-{end}/{size}"
                    if response.headers.get("Content-Range") != expected_range:
                        raise HuggingFaceDownloadError(
                            f"invalid content range for {target.name} chunks {index}-{end_index - 1}"
                        )
                    for block in response.iter_bytes(min(chunk_size, 1024 * 1024)):
                        if stopped.is_set():
                            raise InterruptedError("download cancelled")
                        buffer.extend(block)
                        while index < end_index:
                            chunk_start = index * chunk_size
                            expected = min(chunk_size, size - chunk_start)
                            if len(buffer) < expected:
                                break
                            chunk = bytes(buffer[:expected])
                            del buffer[:expected]
                            output.seek(chunk_start)
                            _write_all(output, chunk)
                            os.fsync(output.fileno())
                            with lock:
                                complete.add(index)
                                state["complete"] = sorted(complete)
                                _write_json(state_path, state)
                            progress.update(task_id, advance=expected)
                            index += 1
                    if index != end_index or buffer:
                        raise HuggingFaceDownloadError(
                            f"short download for {target.name} chunks {index}-{end_index - 1}"
                        )
                return
            except Exception:
                if stopped.is_set() or attempt == 3:
                    raise
                time.sleep(2**attempt)

    # Keep requests open across several durable chunks. This avoids paying CDN
    # request latency every 512 KiB without weakening interruption recovery.
    request_chunks = max(1, (8 * 1024 * 1024) // chunk_size)
    jobs: list[tuple[int, int]] = []
    for index in range(chunk_count):
        if index in complete:
            continue
        if jobs and jobs[-1][1] == index and index - jobs[-1][0] < request_chunks:
            jobs[-1] = (jobs[-1][0], index + 1)
        else:
            jobs.append((index, index + 1))

    pool = ThreadPoolExecutor(max_workers=max(1, min(workers, len(jobs))))
    futures = [pool.submit(download_range, first, end) for first, end in jobs]
    try:
        with progress:
            for future in as_completed(futures):
                future.result()
    except BaseException:
        stopped.set()
        client.close()
        for future in futures:
            future.cancel()
        raise
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
        client.close()

    if len(complete) != chunk_count:
        raise HuggingFaceDownloadError(f"download did not complete all chunks for {target.name}")
    if expected_sha256 is not None and _sha256(partial) != expected_sha256:
        partial.unlink(missing_ok=True)
        state_path.unlink(missing_ok=True)
        raise HuggingFaceDownloadError(f"SHA-256 mismatch for {target.name}")
    os.replace(partial, target)
    state_path.unlink(missing_ok=True)


def _download_huggingface_model(
    repo_id: str,
    *,
    target: Path,
    revision: str | None,
    token: str | bool | None,
    allow_patterns: tuple[str, ...],
    ignore_patterns: tuple[str, ...],
) -> Path:
    Console(stderr=True).print(f"Resolving Hugging Face model {repo_id}...")
    try:
        from huggingface_hub import HfApi, hf_hub_download, hf_hub_url
        from huggingface_hub.utils import build_hf_headers

        info = HfApi(token=token).model_info(repo_id, revision=revision, files_metadata=True)
    except Exception as exc:
        raise HuggingFaceDownloadError(f"failed to inspect Hugging Face model {repo_id!r}: {exc}") from exc
    commit = info.sha
    if not isinstance(commit, str) or not commit:
        raise HuggingFaceDownloadError(f"Hugging Face returned no commit for {repo_id!r}")

    files: list[dict[str, Any]] = []
    for sibling in info.siblings or []:
        name = sibling.rfilename
        if not any(fnmatch(name, pattern) for pattern in allow_patterns):
            continue
        if any(fnmatch(name, pattern) for pattern in ignore_patterns):
            continue
        if sibling.size is None:
            raise HuggingFaceDownloadError(f"Hugging Face returned no size for {name!r}")
        files.append(
            {
                "path": name,
                "size": int(sibling.size),
                "sha256": None if sibling.lfs is None else sibling.lfs.sha256,
            }
        )
    if not files:
        raise HuggingFaceDownloadError(f"Hugging Face model {repo_id!r} has no supported files")

    target.mkdir(parents=True, exist_ok=True)
    (target / _MANIFEST_NAME).unlink(missing_ok=True)
    workers = _worker_count()
    headers = build_hf_headers(token=token, library_name="pllm")
    for entry in files:
        name = entry["path"]
        destination = (target / name).resolve()
        if not destination.is_relative_to(target.resolve()):
            raise HuggingFaceDownloadError(f"unsafe Hugging Face filename: {name!r}")
        if name.endswith(".safetensors") and entry["size"] >= _PARALLEL_THRESHOLD:
            _parallel_download(
                url=hf_hub_url(repo_id, name, revision=commit),
                target=destination,
                size=entry["size"],
                expected_sha256=entry["sha256"],
                headers=headers,
                workers=workers,
                source_id=f"{repo_id}@{commit}:{name}",
            )
        else:
            hf_hub_download(
                repo_id,
                filename=name,
                revision=commit,
                token=token,
                local_dir=target,
            )

    _write_json(
        target / _MANIFEST_NAME,
        {
            "schema": 1,
            "repo_id": repo_id,
            "revision": revision,
            "commit": commit,
            "files": files,
        },
    )
    return target.resolve()


def download_huggingface_model(
    repo_id: str,
    *,
    revision: str | None,
    token: str | bool | None,
    cache_dir: str | os.PathLike[str] | None,
    allow_patterns: tuple[str, ...],
    ignore_patterns: tuple[str, ...],
) -> Path:
    cached = cached_huggingface_model(repo_id, revision=revision, cache_dir=cache_dir)
    if cached is not None:
        Console(stderr=True).print(f"Using cached Hugging Face model: {cached}")
        return cached

    target = _model_cache_path(repo_id, revision, cache_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(f"{target}.lock"):
        cached = cached_huggingface_model(repo_id, revision=revision, cache_dir=cache_dir)
        if cached is not None:
            Console(stderr=True).print(f"Using cached Hugging Face model: {cached}")
            return cached
        return _download_huggingface_model(
            repo_id,
            target=target,
            revision=revision,
            token=token,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
        )
