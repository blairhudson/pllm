"""Bounded, regenerable disk cache for streamed integer stage weights."""

from __future__ import annotations

import os
import re
import shutil
import time
from pathlib import Path

from filelock import FileLock

DEFAULT_CACHE_MAX_BYTES = 6 * 1024**3
_HEX2 = re.compile(r"[0-9a-f]{2}\Z")
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_STALE_SECONDS = 24 * 60 * 60


def compiled_cache_max_bytes() -> int:
    raw = os.getenv("PLLM_COMPILED_CACHE_MAX_BYTES")
    if raw is None:
        return DEFAULT_CACHE_MAX_BYTES
    try:
        limit = int(raw)
    except ValueError as exc:
        raise ValueError("PLLM_COMPILED_CACHE_MAX_BYTES must be a positive integer") from exc
    if limit < 1:
        raise ValueError("PLLM_COMPILED_CACHE_MAX_BYTES must be a positive integer")
    return limit


def cache_lock(root: Path) -> FileLock:
    root.mkdir(parents=True, exist_ok=True)
    return FileLock(str(root / ".access.lock"))


def trim_compiled_cache(root: Path, *, max_bytes: int, protected: set[Path]) -> int:
    """Prune oldest complete entries; caller holds the cross-process cache lock.

    Never recurse into unknown paths or recently interrupted compilations. A
    protected active model may exceed the cap by itself; its memory-mapped files
    remain available until it unloads.
    """
    if max_bytes < 1:
        raise ValueError("compiled cache limit must be positive")
    if not root.is_dir():
        return 0
    total = 0
    candidates: list[tuple[float, Path, int]] = []
    stale: list[tuple[Path, int]] = []
    now = time.time()
    for prefix in root.iterdir():
        if prefix.is_symlink() or not prefix.is_dir() or not _HEX2.fullmatch(prefix.name):
            continue
        for directory in prefix.iterdir():
            if (
                directory.is_symlink() or not directory.is_dir()
                or not _HEX64.fullmatch(directory.name)
                or directory.name[:2] != prefix.name
            ):
                continue
            files = list(directory.iterdir())
            if any(
                item.is_symlink() or not item.is_file()
                or not item.name.endswith((".i8", ".f32", ".json", ".lock", ".tmp"))
                for item in files
            ):
                continue
            size = sum(item.stat().st_size for item in files)
            total += size
            if directory in protected:
                continue
            committed = (
                any(item.name.endswith(".json") for item in files)
                and any(item.name.endswith(".i8") for item in files)
                and any(item.name.endswith(".scales.f32") for item in files)
                and not any(item.name.endswith(".tmp") for item in files)
            )
            latest = max((item.stat().st_mtime for item in files), default=directory.stat().st_mtime)
            if committed:
                candidates.append((latest, directory, size))
            elif now - latest >= _STALE_SECONDS:
                stale.append((directory, size))
    for directory, size in stale:
        shutil.rmtree(directory)
        total -= size
    for _, directory, size in sorted(candidates, key=lambda row: (row[0], str(row[1]))):
        if total <= max_bytes:
            break
        shutil.rmtree(directory)
        total -= size
    return total
