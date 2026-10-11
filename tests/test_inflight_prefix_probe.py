"""Tiny lifecycle/ownership/byte checks; no checkpoint loads or timing cohorts."""
import asyncio
import hashlib
import importlib.util
from pathlib import Path
import sys
import os
import json
import subprocess

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("inflight_probe", Path(__file__).resolve().parents[1] / "scripts/probe_inflight_prefix.py")
probe = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = probe
spec.loader.exec_module(probe)
from pllm.runtime.transformer_client import LayerCache, RuntimeSnapshot, _snapshot_state_basis


def result(ids=(1, 2), *, source="b" * 64, pending=False):
    native = "c" * 64
    execution = hashlib.sha256(b"pllm.decoder.execution-basis.v1\0" + native.encode()
        + np.arange(len(ids), dtype="<i8").tobytes() + np.asarray(ids, dtype="<i8").tobytes()).hexdigest()
    basis = _snapshot_state_basis("incremental" if pending else "completed_prefill", len(ids),
        source, native, execution, "response" if pending else None, len(ids))
    keys = np.asarray(ids, np.float32).reshape(-1, 1, 1)
    saved = RuntimeSnapshot(len(ids), [LayerCache(keys.copy(), keys.copy() / 2, len(ids))], {}, source, basis)
    return saved, np.arange(4, dtype=np.float32)


def make(**kwargs):
    client = object()
    return client, probe.InflightPrefixOwner(client, "source", "composition", "b" * 64, **kwargs)


def acquire(owner, client, execute, ids=(1, 2), **kwargs):
    return owner.acquire(client, "source", "composition", "b" * 64, ids, execute, payload_bound=32, **kwargs)


def test_common_prefix_executes_once_branches_independent_and_retired():
    async def run():
        client, owner = make(max_waiters=8)
        gate = asyncio.Event()
        async def execute():
            await gate.wait()
            return result()
        tasks = [asyncio.create_task(acquire(owner, client, execute)) for _ in range(8)]
        await asyncio.sleep(0)
        assert owner.waiters == 8
        gate.set()
        leases = await asyncio.gather(*tasks)
        assert owner.executions == 1 and owner.copy_bytes == 8 * 32
        assert owner.peak_bytes == 2 * 32 + 8 * 64
        leases[0].snapshot.caches[0].key.fill(99)
        leases[0].logits.fill(99)
        assert leases[1].snapshot.caches[0].key[0, 0, 0] == 1
        assert leases[1].logits[0] == 0
        for lease in leases:
            lease.close()
            lease.close()
        assert owner.bytes == owner.waiters == 0 and not owner._nodes
        assert owner.retired == 1
        # Not a completed cache: replay after last lease starts fresh work.
        lease = await acquire(owner, client, execute)
        assert owner.executions == 2
        lease.close()
    asyncio.run(run())


def test_one_cancel_does_not_poison_other_last_cancel_drains():
    async def run():
        client, owner = make()
        gate, drained = asyncio.Event(), asyncio.Event()
        async def execute():
            try:
                await gate.wait()
                return result()
            finally:
                drained.set()
        a = asyncio.create_task(acquire(owner, client, execute))
        b = asyncio.create_task(acquire(owner, client, execute))
        await asyncio.sleep(0)
        a.cancel()
        with pytest.raises(asyncio.CancelledError):
            await a
        assert owner.waiters == 1 and not drained.is_set()
        gate.set()
        lease = await b
        lease.close()
        gate.clear()
        drained.clear()
        c = asyncio.create_task(acquire(owner, client, execute))
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        c.cancel()
        with pytest.raises(asyncio.CancelledError):
            await c
        assert drained.is_set() and owner.cancelled_work == 1
        assert not owner._nodes and owner.bytes == 0
    asyncio.run(run())


def test_failed_producer_broadcast_retires_and_retry_fresh():
    async def run():
        client, owner = make()
        async def fail():
            await asyncio.sleep(0)
            raise RuntimeError("failed branch")
        values = await asyncio.gather(*(acquire(owner, client, fail) for _ in range(3)), return_exceptions=True)
        assert all(isinstance(v, RuntimeError) for v in values)
        assert owner.executions == 1 and owner.bytes == owner.waiters == 0
        async def okay():
            return result()
        lease = await acquire(owner, client, okay)
        assert owner.executions == 2
        lease.close()
    asyncio.run(run())


@pytest.mark.parametrize("bad", ["client", "source", "composition", "binding", "numeric"])
def test_unrelated_ownership_rejected_before_work(bad):
    async def run():
        client, owner = make()
        async def execute():
            raise AssertionError("must not execute")
        args = [client, "source", "composition", "b" * 64, [1, 2], execute]
        kwargs = {"payload_bound": 32}
        if bad == "numeric":
            kwargs["reduction"] = None
        else:
            args[{"client": 0, "source": 1, "composition": 2, "binding": 3}[bad]] = object()
        with pytest.raises(ValueError, match="ownership"):
            await owner.acquire(*args, **kwargs)
        assert owner.executions == owner.bytes == 0
    asyncio.run(run())


@pytest.mark.parametrize("bad", ["pending", "source", "tokens", "nan", "oversize"])
def test_pending_replay_and_substitution_rejected(bad):
    async def run():
        client, owner = make()
        async def execute():
            saved, logits = result((3, 4) if bad == "tokens" else (1, 2),
                source="d" * 64 if bad == "source" else "b" * 64, pending=bad == "pending")
            if bad == "nan":
                saved.caches[0].key.fill(np.nan)
            if bad == "oversize":
                logits = np.zeros(100, np.float32)
            return saved, logits
        with pytest.raises(ValueError, match="provenance"):
            await acquire(owner, client, execute)
        assert owner.bytes == 0 and not owner._nodes
    asyncio.run(run())


def test_mutation_of_producer_after_publication_detected():
    async def run():
        client, owner = make()
        stored = result()
        async def execute():
            return stored
        lease = await acquire(owner, client, execute)
        stored[0].caches[0].key.fill(123)
        with pytest.raises(ValueError, match="substitution"):
            await acquire(owner, client, execute)
        assert lease.snapshot.caches[0].key[0, 0, 0] == 1
        lease.close()
        assert owner.bytes == 0
    asyncio.run(run())


@pytest.mark.parametrize("limit", ["nodes", "waiters", "ram"])
def test_bounds_reject_without_poisoning_owner(limit):
    async def run():
        client, owner = make(max_nodes=1, max_waiters=1 if limit == "waiters" else 2,
                             max_bytes=128 if limit == "ram" else 1024)
        async def execute():
            return result()
        lease = await acquire(owner, client, execute)
        with pytest.raises((RuntimeError, MemoryError)):
            await acquire(owner, client, execute, ids=(3, 4) if limit == "nodes" else (1, 2))
        lease.close()
        assert owner.bytes == owner.waiters == 0
    asyncio.run(run())


def test_no_prefix_match_and_byte_accounting():
    async def run():
        client, owner = make()
        async def first():
            return result()
        async def second():
            return result((3, 4))
        leases = await asyncio.gather(acquire(owner, client, first), acquire(owner, client, second, ids=(3, 4)))
        assert owner.executions == 2
        for lease in leases:
            lease.close()
    asyncio.run(run())
    from types import SimpleNamespace
    stages = {"body": SimpleNamespace(spec=SimpleNamespace(in_features=3, out_features=5))}
    ledger = probe.inflight_stage_ledger({"body": 7}, stages)["body"]
    assert ledger == {"rows": 7, "macs": 105, "clear_f32_input_output_bytes": 224,
        "unframed_i64_input_output_correction_projection_bytes": 728}


def test_branch_failure_releases_only_its_lease_and_erases_last_owner():
    async def run():
        client, owner = make()
        stored = result()
        async def execute():
            return stored
        a, b = await asyncio.gather(acquire(owner, client, execute), acquire(owner, client, execute))
        try:
            a.snapshot.caches[0].key.fill(7)
            raise RuntimeError("private suffix failed")
        except RuntimeError:
            a.close()
        assert owner.waiters == 1 and b.snapshot.caches[0].key[0, 0, 0] == 1
        assert stored[0].caches[0].key[0, 0, 0] == 1
        b.close()
        assert owner.bytes == 0 and not np.any(stored[0].caches[0].key)
    asyncio.run(run())


def test_last_cancellation_reserves_retiring_work_until_cleanup_drained():
    async def run():
        client, owner = make(max_nodes=1, max_bytes=128)
        entered, cleanup = asyncio.Event(), asyncio.Event()
        async def execute():
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                await cleanup.wait()
                raise
        task = asyncio.create_task(acquire(owner, client, execute))
        await entered.wait()
        task.cancel()
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert owner.bytes == 64 and len(owner._retiring) == 1
        with pytest.raises(RuntimeError, match="node bound"):
            await acquire(owner, client, execute)
        cleanup.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert owner.bytes == 0 and not owner._retiring
    asyncio.run(run())


def test_packed_projection_native_rounding_and_per_callback_rows():
    from types import SimpleNamespace
    from pllm import _native
    stage = SimpleNamespace(spec=SimpleNamespace(in_features=3, out_features=2, activation_bits=8),
        output_residue_bits=bytes([3, 6]), seeded_profile=SimpleNamespace(wire_bits=16))
    one = probe.inflight_packed_callback(stage, 1)
    two = probe.inflight_packed_callback(stage, 2)
    assert one["client_inference_tensor_bytes"] == 3
    assert one["inference_client_tensor_bytes"] == 2
    assert one["fresh_preparation_inference_correction_tensor_bytes"] == 2
    assert 2 * one["total_tensor_payload_projection_bytes"] == 14
    assert two["total_tensor_payload_projection_bytes"] == 11
    assert len(_native.offset_pack_rows(bytes(8), bytes([3, 6]), 1)) == 2
    assert len(_native.offset_pack_rows(bytes(16), bytes([3, 6]), 2)) == 3
    assert probe.inflight_packed_callback(stage, 17)["chunks"] == [16, 1]


def test_checkpoint_pin_detects_config_weight_and_namespace_change(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "weights.safetensors").write_bytes(b"public synthetic bytes")
    pin = {"path": str(tmp_path), "files": probe.inflight_checkpoint_files(tmp_path)}
    probe.inflight_verify_checkpoint(pin)
    (tmp_path / "weights.safetensors").write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="checkpoint/config"):
        probe.inflight_verify_checkpoint(pin)


def test_checkpoint_pin_detects_snapshot_symlink_substitution(tmp_path):
    old, new, snapshot = tmp_path / "old", tmp_path / "new", tmp_path / "snapshot"
    old.write_bytes(b"first")
    new.write_bytes(b"second")
    snapshot.mkdir()
    link = snapshot / "config.json"
    link.symlink_to(old)
    pin = {"path": str(snapshot), "files": probe.inflight_checkpoint_files(snapshot)}
    link.unlink()
    link.symlink_to(new)
    with pytest.raises(RuntimeError, match="checkpoint/config"):
        probe.inflight_verify_checkpoint(pin)


def test_fresh_output_rejects_existing_and_removes_failed_attempt(tmp_path):
    path = tmp_path / "evidence.json"
    path.write_text("user work")
    with pytest.raises(FileExistsError):
        with probe.inflight_fresh_output(path):
            raise AssertionError("existing evidence must not be replaced")
    assert path.read_text() == "user work"
    fresh = tmp_path / "new.json"
    with pytest.raises(RuntimeError, match="worker failed"):
        with probe.inflight_fresh_output(fresh) as stream:
            stream.write("partial")
            raise RuntimeError("worker failed")
    assert not fresh.exists()
    with pytest.raises(RuntimeError):
        with probe.inflight_fresh_output(fresh):
            replacement = tmp_path / "replacement.json"
            replacement.write_text("new user work")
            replacement.replace(fresh)
            raise RuntimeError("failed after unrelated replacement")
    assert fresh.read_text() == "new user work"


def test_unexpected_worker_failure_runs_all_owner_cleanup(monkeypatch):
    events = []
    def fail(spec, cleanup):
        for label in ("runtime", "cache", "lease", "snapshot"):
            cleanup.callback(events.append, label)
        raise RuntimeError("unexpected decoder failure")
    monkeypatch.setattr(probe, "inflight_worker_body", fail)
    with pytest.raises(RuntimeError, match="unexpected decoder failure"):
        probe.inflight_worker({"frozen": {}})
    assert events == ["snapshot", "lease", "cache", "runtime"]


def test_historical_geometry_checks_actual_w8_bytes_not_only_descriptor(tmp_path, monkeypatch):
    from types import SimpleNamespace
    weights = np.zeros((2, 3), np.int8)
    digest = hashlib.sha256(weights.tobytes()).hexdigest()
    source = {k: "public-control" for k in ("model", "revision", "checkpoint_digest", "source_lock_digest", "body_fingerprint")}
    geometry = tmp_path / "geometry.json"
    geometry.write_text(json.dumps({"complete": True, "source": source,
        "stages": [{"stage": "body", "shape": [2, 3], "weight_sha256": digest}]}))
    control = tmp_path / "control.json"
    control.write_text(json.dumps({"complete": True, "results": [
        {"execution": {"case": {"name": "prepared/digit8"}, "details": {"report": {
            "runs": [{"model_fingerprint": "public-control"}]}}}} for _ in range(3)]}))
    monkeypatch.setattr(probe, "INFLIGHT_CONTROL", control)
    monkeypatch.setattr(probe, "INFLIGHT_GEOMETRY", geometry)
    fixture = SimpleNamespace(lock=lambda: source, source=SimpleNamespace(path=tmp_path),
        model=SimpleNamespace(stages={"body": SimpleNamespace(spec=SimpleNamespace(role="qkv_proj"),
            weight=SimpleNamespace(values=weights), weight_digest=digest, output_residue_bits=bytes([4, 4]))}))
    assert probe.inflight_control_admission(fixture, {"path": str(tmp_path)})["verified_body_stage_weights"] == 1
    weights.fill(1)
    with pytest.raises(ValueError, match="actual W8 stage weight"):
        probe.inflight_control_admission(fixture, {"path": str(tmp_path)})


def test_unexpected_failure_erases_real_leases_and_completed_cache(monkeypatch):
    from pllm.runtime.prefill_cache import ExactPrefillCache
    client, owner = make()
    cache = ExactPrefillCache(4096, causal_reduction="prefix_f32")
    saved, logits = result()
    assert cache.put("public-prefix", saved, logits)
    async def execute():
        return saved, logits
    lease = asyncio.run(acquire(owner, client, execute))
    def fail(spec, cleanup):
        cleanup.callback(cache.clear)
        cleanup.callback(lease.close)
        raise RuntimeError("unexpected worker error")
    monkeypatch.setattr(probe, "inflight_worker_body", fail)
    with pytest.raises(RuntimeError):
        probe.inflight_worker({"frozen": {}})
    assert cache.entry_count == cache.size_bytes == 0
    assert lease.closed and owner.waiters == owner.bytes == 0
    assert not np.any(saved.caches[0].key) and not np.any(lease.logits)


@pytest.mark.skipif(os.environ.get("PLLM_INFLIGHT_NATIVE") != "1",
                    reason="parent schedules generated native workers serially")
@pytest.mark.parametrize("family", ["qwen2", "qwen3", "llama"])
def test_generated_native_exact_decoder_and_completed_cache_control(tmp_path, family):
    output = tmp_path / "evidence.json"
    subprocess.run([sys.executable, str(Path(probe.__file__)), "--family", family,
        "--output", str(output)], check=True, timeout=240)
    evidence = json.loads(output.read_text())
    assert evidence["completed_cache_incremental_saving"] == 0
    assert evidence["live_provider_or_sdk_claim"] is False
    for modes in evidence["cases"].values():
        for record in modes.values():
            assert all(row.get("exact_logits_kv", row.get("injected_failed_branch", False))
                       for row in record["parity"])
            assert record["owner"]["waiters"] == record["owner"]["bytes"] == 0
        assert modes["completed_cache_hit"]["response_only_stage_ledger"] == modes["inflight_cache_hit"]["response_only_stage_ledger"]
        assert modes["inflight_cache_hit"]["owner"]["executions"] == 0
