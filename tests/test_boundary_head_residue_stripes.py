"""Tiny synthetic tests only; never load a checkpoint or build native code."""
import asyncio
import importlib.util
import sys
import threading
import json
import io
from pathlib import Path

import numpy as np
import pytest

from pllm.runtime.private_boundary_head_codec import DOMAIN, HEADER
from pllm.runtime.private_boundary_protocol import BoundaryError, ROLES

SPEC = importlib.util.spec_from_file_location(
    "head_residue_stripes_probe", Path(__file__).parents[1] / "scripts/probe_boundary_head_residue_stripes.py")
probe = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = probe
SPEC.loader.exec_module(probe)


@pytest.fixture(scope="module")
def fixture():
    return probe.Fixture(vocabulary=17, hidden=8, stripe_rows=6)


@pytest.mark.parametrize("mode", probe.MODES)
def test_tiny_complete_logits_and_real_http_accounting(fixture, mode):
    report = asyncio.run(probe.run_case(fixture, mode))
    assert report["success"], report["error"]
    assert report["remaining_native_tasks"] == 0
    assert report["logits_sha256"] == probe.sha(fixture.logits.tobytes())
    assert len(report["online_tcp_payload_bytes"]) == 4
    if mode == "parallel":
        assert report["installed_operation_counters"]["head"]["calls"] == 1
        assert report["owners"] is None
    else:
        assert report["replay_status"] == [409, 409]
        for role, owner in zip(ROLES, report["owners"]):
            assert owner["attempted"] and owner["retired"] and owner["replay_rejected"]
            assert owner["peak_pending_native"] == 1
            assert owner["peak_output_slots"] <= 2
            body = report["per_role_bodies"][role]
            assert body["generated_reply_body_bytes"] == body["observed_reply_body_bytes"]
            assert report["online_tcp_payload_bytes"][role + "->client"] > body["observed_reply_body_bytes"]
            assert report["online_tcp_payload_bytes"]["client->" + role] > body["attempted_request_body_bytes"]
            assert owner["native_calls"] == (1 if mode == "sliced" else 3)
            assert owner["peak_output_bytes"] <= (17 * 4 if mode == "sliced" else 2 * 6 * 4)


@pytest.mark.parametrize("fault", ["context", "reorder", "malformed", "truncate", "partial", "trailing",
                                   "request_sequence", "request_malformed", "request_oversized",
                                   "request_geometry", "credential"])
def test_failure_never_returns_partial_logits_and_retires_pair(fixture, fault):
    report = asyncio.run(probe.run_case(fixture, "snapshots", fault=fault))
    assert not report["success"]
    assert report["logits_sha256"] is None
    assert report["remaining_native_tasks"] == 0
    assert not report["online_coverage_complete"]
    assert all(o["retired"] for o in report["owners"])
    assert report["replay_status"] == [409, 409]


def test_cancellation_drains_native_before_retirement(fixture):
    entered = threading.Event()
    release = threading.Event()

    def hook(index):
        entered.set()
        if not release.wait(3):
            raise RuntimeError("test native barrier timed out")

    async def scenario():
        # Release only after cancellation has reached the operation; a blocked
        # native call remains owned until its thread exits.
        async def unblock():
            for _ in range(1000):
                if entered.is_set():
                    break
                await asyncio.sleep(.001)
            await asyncio.sleep(.05)
            release.set()
        task = asyncio.create_task(unblock())
        report = await probe.run_case(fixture, "snapshots", cancel_after=.025, native_hook=hook)
        await task
        return report

    report = asyncio.run(scenario())
    assert entered.is_set() and release.is_set()
    assert not report["success"] and report["error"].startswith("CancelledError")
    assert report["remaining_native_tasks"] == 0
    assert all(o["retired"] for o in report["owners"])
    assert sum(o["drained_native_calls"] for o in report["owners"]) >= 1


def test_native_failure_drains_peer_without_partial_success(fixture):
    def hook(index):
        if index == 1:
            raise RuntimeError("synthetic native failure")
    report = asyncio.run(probe.run_case(fixture, "snapshots", native_hook=hook))
    assert not report["success"]
    assert report["remaining_native_tasks"] == 0
    assert all(o["retired"] for o in report["owners"])


def test_one_byte_read_ablation_preserves_complete_output(fixture):
    report = asyncio.run(probe.run_case(fixture, "snapshots", read_chunk_bytes=1))
    assert report["success"], report["error"]
    assert report["candidate_read_chunk_bytes"] == 1


def test_binding_includes_fixed_geometry_tail_session_sequence_and_role(fixture):
    codec = fixture.codecs[0]
    sid = "c" * 32
    args = fixture.codec_args(0, sid)
    raw = np.zeros((1, 6), "<u4").tobytes()
    frames = [codec.encode(raw, role=role, **args) for role in ROLES]
    assert codec.decode(*frames, **args).shape == (1, 6)
    for changes in ({"paired_session": "d" * 32}, {"sequence": 1}, {"epoch": 1}):
        with pytest.raises(BoundaryError):
            codec.decode(*frames, **{**args, **changes})
    with pytest.raises(BoundaryError):
        codec.decode(*frames[::-1], **args)
    changed = probe.Fixture(vocabulary=17, hidden=8, stripe_rows=7, snapshots=False)
    with pytest.raises(BoundaryError):
        changed.codecs[0].decode(*frames, **changed.codec_args(0, sid))
    tail = fixture.codecs[-1]
    with pytest.raises(BoundaryError):
        tail.decode(*frames, **fixture.codec_args(2, sid))


def test_padding_and_signed_bound_fail_closed(fixture):
    # hidden=8 => width=18; tail=5 => 90 bits, six mandatory zero pad bits.
    codec = fixture.codecs[-1]
    args = fixture.codec_args(2, "c" * 32)
    zero = bytes(5 * 4)
    frames = [codec.encode(zero, role=role, **args) for role in ROLES]
    invalid = bytearray(frames[0])
    invalid[-1] |= 128
    # Recompute checksum: padding check must not depend on checksum rejection.
    magic, party, width, binding, _ = HEADER.unpack_from(invalid)
    checksum = bytes.fromhex(probe.sha(DOMAIN + binding + invalid[HEADER.size:]))
    invalid[:HEADER.size] = HEADER.pack(magic, party, width, binding, checksum)
    with pytest.raises(BoundaryError, match="padding"):
        codec.decode(bytes(invalid), frames[1], **args)
    outside = np.asarray([codec.bound + 1, 0, 0, 0, 0], "<u4").tobytes()
    encoded = codec.encode(outside, role=ROLES[0], **args)
    with pytest.raises(BoundaryError, match="signed bound"):
        codec.decode(encoded, frames[1], **args)


@pytest.mark.parametrize("bits", [4, 8])
def test_extreme_i8_and_symmetric_activation_ring_and_tails(bits):
    f = probe.Fixture(vocabulary=7, hidden=3, stripe_rows=4, activation_bits=bits)
    assert f.stripes[-1].shape == (3, 3)
    q = np.asarray([[-((1 << (bits - 1)) - 1), 0, (1 << (bits - 1)) - 1]], np.int64)
    weights = np.asarray([[-128, 127, -128], [127, -128, 127], [-128, -128, 127]], np.int8)
    full = probe.MaskedGEMM(threads=1).compile(weights)
    mask = np.asarray([[0xffffffff, 0x80000000, 1]], "<u4")
    clear = (q % (1 << 32)).astype("<u4")
    a, b = full.wrap32((clear - mask).astype("<u4")), full.wrap32(mask)
    codec = f.codecs[-1]
    args = f.codec_args(1, "c" * 32)
    encoded = [codec.encode(x.tobytes(), role=r, **args) for x, r in zip((a, b), ROLES)]
    np.testing.assert_array_equal(codec.decode(*encoded, **args), q @ weights.astype(np.int64).T)


def test_no_silent_native_row_streaming_claim(fixture):
    assert fixture.setup["additional_snapshot_weight_bytes"] == fixture.weights.nbytes
    assert not hasattr(fixture.full._matrix, "row_range")
    with pytest.raises(ValueError):
        probe.Fixture(vocabulary=2, hidden=1, stripe_rows=3)


def test_cli_resource_rejection_precedes_fixture_allocation(tmp_path, monkeypatch):
    from pllm.runtime.benchmark_memory import HostMemory
    monkeypatch.setattr(probe, "LOCK_PATH", tmp_path / "frontier-probe.lock")
    monkeypatch.setattr(probe, "sample_host", lambda: HostMemory(8 << 30, 1 << 30, 0, 100 << 30))

    def forbidden(*args, **kwargs):
        pytest.fail("resource rejection must precede metadata workers and Fixture allocation")
    monkeypatch.setattr(probe, "Fixture", forbidden)
    monkeypatch.setattr(probe, "provenance", forbidden)
    monkeypatch.setattr(probe.subprocess, "Popen", forbidden)
    target = tmp_path / "reject.json"
    assert probe.cli(["--sampled-qwen", "--sampled-worker", "1", "--output", str(target)]) == 1
    retained = json.loads(target.read_text())
    assert retained["progress"]["failure_code"] == "host_headroom_rejected"
    assert not retained["progress"]["measurement_complete"]


@pytest.mark.parametrize("suffix", ["", ".progress.json", ".runs.jsonl"])
def test_cli_cannot_overwrite_any_output(tmp_path, monkeypatch, suffix):
    target = tmp_path / "existing.json"
    occupied = Path(str(target) + suffix)
    occupied.write_bytes(b"user-owned content")

    def forbidden(*args, **kwargs):
        pytest.fail("occupied output must reject before resource inspection or allocation")
    monkeypatch.setattr(probe, "sample_host", forbidden)
    monkeypatch.setattr(probe, "Fixture", forbidden)
    assert probe.cli(["--output", str(target)]) == 1
    assert occupied.read_bytes() == b"user-owned content"
    if suffix:
        assert not target.exists()


def test_cli_exclusive_reservation_closes_freshness_race(tmp_path, monkeypatch, capsys):
    target = tmp_path / "raced.json"
    original = Path.open

    def raced_open(path, mode="r", *args, **kwargs):
        if path == target and mode == "x+":
            with original(path, "w") as stream:
                stream.write("other process won")
        return original(path, mode, *args, **kwargs)
    monkeypatch.setattr(Path, "open", raced_open)
    assert probe.cli(["--output", str(target)]) == 1
    assert target.read_text() == "other process won"
    assert "output_reservation_race" in capsys.readouterr().out


def test_cli_shared_lock_rejects_before_host_or_fixture(tmp_path, monkeypatch):
    from filelock import FileLock
    lock = tmp_path / "frontier-probe.lock"
    monkeypatch.setattr(probe, "LOCK_PATH", lock)

    def forbidden(*args, **kwargs):
        pytest.fail("busy shared lock must fail before allocation")
    monkeypatch.setattr(probe, "sample_host", forbidden)
    monkeypatch.setattr(probe, "Fixture", forbidden)
    target = tmp_path / "busy.json"
    with FileLock(str(lock), timeout=0):
        assert probe.cli(["--output", str(target)]) == 1
    progress = json.loads(Path(str(target) + ".progress.json").read_text())
    assert progress["failure_code"] == "frontier_cohort_lock_busy"


def test_host_guard_rejects_one_byte_new_swap_and_preserves_full_reserve(monkeypatch):
    from pllm.runtime.benchmark_memory import HostMemory
    samples = iter([HostMemory(32 << 30, 20 << 30, 12, 100 << 30),
                    HostMemory(32 << 30, 19 << 30, 10, 100 << 30),
                    HostMemory(32 << 30, 19 << 30, 11, 100 << 30)])
    monkeypatch.setattr(probe, "sample_host", lambda: next(samples))
    guard = probe.HostGuard(1 << 30)
    guard.check()
    with pytest.raises(probe.RunRejected, match="new_swap_growth"):
        guard.check()
    assert guard.report()["maximum_swap_growth_bytes"] == 1
    monkeypatch.setattr(probe, "sample_host", lambda: HostMemory(32 << 30, 20 << 30, 10, 100 << 30))
    guard = probe.HostGuard(1 << 30)
    monkeypatch.setattr(probe, "sample_host", lambda: HostMemory(32 << 30, 3 << 30, 10, 100 << 30))
    with pytest.raises(probe.RunRejected, match="runtime_reserve_rejected"):
        guard.check()


def test_six_orders_balance_every_position_and_predecessor():
    from collections import Counter
    orders = probe.arm_orders(probe.MODES, 6)
    assert len({tuple(order) for order in orders}) == 6
    for position in range(3):
        assert set(Counter(order[position] for order in orders).values()) == {2}
    assert set(Counter((a, b) for order in orders for a, b in zip(order, order[1:])).values()) == {2}


def test_cli_freezes_sampled_plan_and_retains_exact_orders_without_allocating(tmp_path, monkeypatch):
    from pllm.runtime.benchmark_memory import HostMemory
    monkeypatch.setattr(probe, "LOCK_PATH", tmp_path / "frontier-probe.lock")
    monkeypatch.setattr(probe, "sample_host", lambda: HostMemory(64 << 30, 32 << 30, 0, 100 << 30))
    monkeypatch.setattr(probe, "provenance", lambda: {"native_sha256": "a" * 64, "sources_sha256": "b" * 64})
    captured = []

    class FakeWorker:
        returncode = 0

        def __init__(self, command, **kwargs):
            worker = self

            class Input(io.StringIO):
                def close(self):
                    payload = json.loads(self.getvalue())
                    captured.append(payload)
                    g = payload["args"]
                    orders = probe.arm_orders(g["modes"], g["repeats"])
                    runs = [{"repeat": r, "position": p, "mode": mode, "success": True,
                             "logits_sha256": "c" * 64}
                            for r, order in enumerate(orders) for p, mode in enumerate(order)]
                    report = {"runs": runs, "repeat_orders": orders, "native_sha256": "a" * 64}
                    for run in runs:
                        kwargs["stdout"].write(json.dumps({"kind": "run", "report": run}) + "\n")
                    kwargs["stdout"].write(json.dumps({"kind": "complete", "report": report}) + "\n")
                    kwargs["stdout"].flush()
                    worker.returncode = 0
                    super().close()
            self.stdin = Input()

        def poll(self):
            return self.returncode

    monkeypatch.setattr(probe.subprocess, "Popen", FakeWorker)
    target = tmp_path / "frozen.json"
    assert probe.cli(["--sampled-qwen", "--sampled-worker", "2", "--read-chunk-bytes", "1024",
                      "--output", str(target)]) == 0
    report = json.loads(target.read_text())
    geometry = report["plan"]["geometry"]
    assert (geometry["vocabulary"], geometry["hidden"], geometry["stripe_rows"]) == (151936, 896, 8192)
    assert geometry["read_chunk_bytes"] == 1024 and geometry["repeats"] == 6
    assert geometry["modes"] == list(probe.MODES)
    assert report["plan"]["sampled_worker"] == 2
    assert len(report["runs"]) == report["progress"]["completed_runs"] == 18
    assert report["plan"]["geometry_sha256"] == probe.sha(probe.canonical(captured[0]["args"]))


def test_pressure_abort_drains_owned_process_before_unlock(tmp_path, monkeypatch):
    from filelock import FileLock, Timeout
    from pllm.runtime.benchmark_memory import HostMemory
    lock = tmp_path / "frontier-probe.lock"
    monkeypatch.setattr(probe, "LOCK_PATH", lock)
    samples = iter([HostMemory(64 << 30, 32 << 30, 0, 100 << 30),
                    HostMemory(64 << 30, 32 << 30, 0, 100 << 30),
                    HostMemory(64 << 30, 32 << 30, 1, 100 << 30)])
    monkeypatch.setattr(probe, "sample_host", lambda: next(samples))
    monkeypatch.setattr(probe, "provenance", lambda: {"native_sha256": "a" * 64, "sources_sha256": "b" * 64})
    stopped = []

    class BlockedWorker:
        def __init__(self, *args, **kwargs):
            self.stdin = io.StringIO()
            self.returncode = None

        def poll(self):
            return self.returncode

        def terminate(self):
            with pytest.raises(Timeout):
                FileLock(str(lock), timeout=0).acquire()
            stopped.append(True)
            self.returncode = -15

        def wait(self, timeout=None):
            return self.returncode

    monkeypatch.setattr(probe.subprocess, "Popen", BlockedWorker)
    target = tmp_path / "pressure.json"
    assert probe.cli(["--output", str(target)]) == 1
    assert stopped == [True]
    report = json.loads(target.read_text())
    assert report["progress"]["failure_code"] == "new_swap_growth"
    assert report["host_guard"]["maximum_swap_growth_bytes"] == 1
    with FileLock(str(lock), timeout=0):
        pass
