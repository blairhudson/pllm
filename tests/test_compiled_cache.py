from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path

import pytest

from pllm.runtime.compiled_cache import cache_lock, compiled_cache_max_bytes, trim_compiled_cache
from pllm.runtime.loaders import load_hf_directory
from pllm.runtime.tiny_llama import create_tiny_llama_checkpoint
from pllm.runtime.transformer_engine import MaskedTransformerEngine


def _entry(root: Path, key: str, size: int, *, age: int = 0) -> Path:
    directory = root / key[:2] / key
    directory.mkdir(parents=True)
    for name, data in (
        ("token_lookup.json", "{}"),
        ("token_lookup.i8", "w" * size),
        ("token_lookup.scales.f32", "abcd"),
    ):
        item = directory / name
        item.write_text(data)
        os.utime(item, (time.time() - age, time.time() - age))
    return directory


def test_compiled_cache_prunes_oldest_but_keeps_active_and_unknown_entries(tmp_path: Path) -> None:
    oldest = _entry(tmp_path, "a" * 64, 30, age=90)
    newer = _entry(tmp_path, "b" * 64, 30, age=30)
    active = _entry(tmp_path, "c" * 64, 30)
    foreign = tmp_path / "other-model"
    foreign.mkdir()
    (foreign / "user.bin").write_bytes(b"leave me alone")
    with cache_lock(tmp_path):
        trim_compiled_cache(tmp_path, max_bytes=90, protected={active})
    assert not oldest.exists()
    assert newer.exists() and active.exists()
    assert (foreign / "user.bin").read_bytes() == b"leave me alone"
    with cache_lock(tmp_path):
        trim_compiled_cache(tmp_path, max_bytes=1, protected={active})
    assert not newer.exists()
    assert active.exists()  # active memory maps may exceed the soft cap


def test_compiled_cache_reclaims_stale_partial_but_not_recent_or_symlink(tmp_path: Path) -> None:
    stale = tmp_path / "d0" / ("d0" * 32)
    stale.mkdir(parents=True)
    partial = stale / "token_lookup.i8.tmp"
    partial.write_bytes(b"unused")
    os.utime(partial, (time.time() - 2 * 86400, time.time() - 2 * 86400))
    recent = tmp_path / "e0" / ("e0" * 32)
    recent.mkdir(parents=True)
    (recent / "token_lookup.i8.tmp").write_bytes(b"in progress")
    external = tmp_path / "external"
    external.mkdir()
    (external / "user.bin").write_bytes(b"owned elsewhere")
    (tmp_path / "f0").mkdir()
    (tmp_path / "f0" / ("f0" * 32)).symlink_to(external, target_is_directory=True)
    with cache_lock(tmp_path):
        trim_compiled_cache(tmp_path, max_bytes=100, protected=set())
    assert not stale.exists()
    assert recent.exists()
    assert (external / "user.bin").read_bytes() == b"owned elsewhere"


def test_compiled_cache_budget_env_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    for invalid in ("0", "-1", "NaN"):
        monkeypatch.setenv("PLLM_COMPILED_CACHE_MAX_BYTES", invalid)
        with pytest.raises(ValueError, match="positive integer"):
            compiled_cache_max_bytes()
    monkeypatch.setenv("PLLM_COMPILED_CACHE_MAX_BYTES", "314159")
    assert compiled_cache_max_bytes() == 314159


def test_live_streamed_weights_survive_soft_cap_and_rebuild_after_unload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = create_tiny_llama_checkpoint(tmp_path / "checkpoint", num_hidden_layers=1)
    cache = tmp_path / "compiled"
    monkeypatch.setenv("PLLM_COMPILED_CACHE_MAX_BYTES", "1")
    engine = MaskedTransformerEngine(
        compiled_cache_dir=cache, streaming_threshold_elements=1, threads=1,
    )
    asyncio.run(engine.load(load_hf_directory(root, model_id="cache-budget")))
    first = int(engine.models["cache-budget"].stages["token_lookup"].weight.values[0, 0])
    assert list(cache.rglob("*.i8"))  # an active model is never deleted mid-response
    asyncio.run(engine.unload("cache-budget"))
    assert not list(cache.rglob("*.i8"))
    asyncio.run(engine.load(load_hf_directory(root, model_id="cache-budget")))
    assert int(engine.models["cache-budget"].stages["token_lookup"].weight.values[0, 0]) == first
