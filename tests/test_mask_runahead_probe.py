"""Installed native correctness and barrier-controlled one-slot lifecycle tests."""
import importlib.util
import hashlib
import msgpack
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from pllm import _native

from pllm.runtime.preparation_protocol import expand_preparation_mask, expand_output_mask
from pllm.runtime.native import mask_prepared_input, unmask_prepared_output

PATH = Path(__file__).resolve().parents[1] / "scripts/probe_mask_runahead.py"
SPEC = importlib.util.spec_from_file_location("mask_runahead_probe", PATH)
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def owner(inv, lease=None, **kw):
    return probe.MaskRunahead(lease or inv.reserve(inv.capacity), inv.stages,
        chunk_rows=kw.pop("chunk_rows", min(32, inv.capacity)),
        cap_bytes=kw.pop("cap_bytes", 268435456), base_peak_bytes=32 << 20, **kw)


@pytest.mark.parametrize("bits", [16, 24, 32])
def test_native_bytes_tickets_and_modular_numeric_parity(bits):
    sync, future = probe.fixture(bits=bits), probe.fixture(bits=bits)
    baseline = sync.reserve(3)
    run = owner(future)
    try:
        # Actual current stage claimed synchronously before transport dispatch.
        run.lease.take("current", 2)
        baseline.take("current", 2)
        for count, begin in ((2, 0), (1, 2)):
            run.remote_started("current")
            token = run.prefetch("next", count)
            run.remote_finished()
            got, expected = run.consume(token), baseline.take("next", count)
            assert probe.digest(got) == probe.digest(expected)
            req = future.stages["next"].request
            assert got[0].tobytes() == expand_preparation_mask(req)[begin:begin + count].tobytes()
            assert got[1].tobytes() == expand_output_mask(req)[begin:begin + count].tobytes()
            assert got[2] == expected[2] and len(set(got[2])) == count
            # Independent signed integer linear oracle and native mask boundaries.
            x = (np.arange(count * req.in_features, dtype=np.int32).reshape(count, -1) % 9 - 4).astype(np.int8)
            w = np.arange(req.in_features * req.out_features, dtype=np.int64).reshape(
                req.in_features, req.out_features) % 7 - 3
            ring_mask = req.modulus - 1
            masked = mask_prepared_input(x, got[0], bits)
            correction = (got[0].astype(np.int64) @ w - got[1].astype(np.int64)) & ring_mask
            reply = ((masked.astype(np.int64) @ w + correction) & ring_mask).astype(np.uint32)
            restored = unmask_prepared_output(reply, got[1], bits)
            np.testing.assert_array_equal(restored, x.astype(np.int64) @ w)
    finally:
        baseline.close()
        run.close()
        sync.cancel()
        future.cancel()
    assert sync.status() == future.status()
    assert run.waste_bytes == 0
    with pytest.raises(ValueError, match="consumed"):
        future.stages["next"].take_masks(0, 1)


def test_one_slot_next_stage_chunk_and_whole_client_cap():
    inv = probe.fixture(rows=3)
    run = owner(inv, chunk_rows=1)
    try:
        with pytest.raises(RuntimeError, match="remote window"):
            run.prefetch("next", 1)
        run.remote_started("current")
        for stage, count in (("last", 1), ("current", 1), ("next", 2), ("next", 0)):
            with pytest.raises(RuntimeError):
                run.prefetch(stage, count)
        token = run.prefetch("next", 1)
        with pytest.raises(RuntimeError):
            run.prefetch("next", 1)
        run.remote_finished()
        run.consume(token)
        with pytest.raises(RuntimeError):
            run.prefetch("last", 1)
        assert run.charges["copy_overlap_bytes"] == 2 * run.buffer_bound
        assert run.required_bytes == sum(run.charges.values())
    finally:
        run.close()
    lease = probe.fixture().reserve(3)
    with pytest.raises(ValueError, match="cap"):
        owner(lease._owner, lease, cap_bytes=run.required_bytes - 1, chunk_rows=1)
    assert lease._offsets == {}  # cap denial performs no fresh expansion
    lease.close()


@pytest.mark.parametrize("position", ["before", "after"])
@pytest.mark.parametrize("reason", ["completion", "error", "cancellation"])
def test_close_drains_blocked_worker_and_burns_early_stop(position, reason):
    entered, release, closing = threading.Barrier(2), threading.Barrier(2), threading.Barrier(2)

    def block():
        entered.wait(timeout=5)
        release.wait(timeout=5)

    inv = probe.fixture()
    run = owner(inv, on_closing=lambda: closing.wait(timeout=5),
                **{position + "_expand": block})
    run.remote_started("current")
    run.prefetch("next", 1)
    entered.wait(timeout=5)

    def stop():
        if reason == "cancellation":
            run.close()
        else:
            try:
                with run:
                    if reason == "error":
                        raise ValueError("transport failed")
            except ValueError as error:
                assert str(error) == "transport failed"

    with ThreadPoolExecutor(1) as controller:
        stopped = controller.submit(stop)
        closing.wait(timeout=5)
        assert run.state == "closing"
        assert not stopped.done()
        release.wait(timeout=5)
        stopped.result(timeout=5)
    assert run.state == "closed" and run._slot is None
    assert run.expanded_bytes == run.waste_bytes == 4 * (17 + 11)
    assert inv.status()["reserved"] == 0
    assert inv.status()["stage_rows_claimed"] + inv.status()["stage_rows_burned"] == 9
    # All rows, including ones never expanded, are dead after every terminal path.
    for stage in inv.stages.values():
        with pytest.raises(ValueError, match="consumed"):
            stage.take_masks(2, 1)


def test_replay_closes_and_burns_remaining_prefetch():
    inv = probe.fixture()
    run = owner(inv)
    run.remote_started("current")
    token = run.prefetch("next", 1)
    run.remote_finished()
    run.consume(token)
    run.remote_started("next")
    run.prefetch("last", 1)
    run.remote_finished()
    with pytest.raises(RuntimeError, match="replayed"):
        run.consume(token)
    assert run.state == "closed" and run.waste_bytes == 112
    assert inv.status()["reserved"] == 0


def test_expansion_error_drains_and_burns():
    inv = probe.fixture()

    def fail():
        raise ValueError("injected expansion failure")

    run = owner(inv, after_expand=fail)
    run.remote_started("current")
    token = run.prefetch("next", 1)
    run.remote_finished()
    with pytest.raises(ValueError, match="injected"):
        run.consume(token)
    assert run.state == "closed" and run._slot is None
    assert inv.status()["stage_rows_claimed"] == 1
    assert inv.status()["stage_rows_burned"] == 8


def test_real_native_overlap_with_simulated_remote_barriers_only():
    # This proves scheduling overlap, not network/response latency benefit.
    remote_entered, expanded, remote_release = (threading.Barrier(2) for _ in range(3))
    inv = probe.fixture()
    run = owner(inv, after_expand=lambda: expanded.wait(timeout=5))

    def transport_wait():
        remote_entered.wait(timeout=5)
        remote_release.wait(timeout=5)

    with ThreadPoolExecutor(1) as transport:
        response = transport.submit(transport_wait)
        remote_entered.wait(timeout=5)
        run.remote_started("current")
        token = run.prefetch("next", 1)
        expanded.wait(timeout=5)
        assert not response.done()
        remote_release.wait(timeout=5)
        response.result(timeout=5)
    run.remote_finished()
    result = run.consume(token)
    assert result[0].shape == (1, 17)
    run.close()


def test_no_online_preparation_or_new_trust_surface():
    # API receives only previously reserved lease and public order. No refill,
    # Preparation or provider callback can be reached by runahead implementation.
    inv = probe.fixture()
    lease = inv.reserve(3)
    assert inv.available == 0
    run = owner(inv, lease)
    run.remote_started("current")
    token = run.prefetch("next", 1)
    run.remote_finished()
    run.consume(token)
    run.close()
    assert inv.claimed == 3 and inv.available == 0


def test_disjoint_other_lease_survives_close():
    inv = probe.fixture(rows=4)
    first, second = inv.reserve(2), inv.reserve(2)
    run = owner(inv, first, chunk_rows=2)
    run.remote_started("current")
    run.prefetch("next", 1)
    run.close()
    req = inv.stages["next"].request
    r, _, _ = second.take("next", 2)
    assert r.tobytes() == expand_preparation_mask(req)[2:].tobytes()
    second.close()


@pytest.mark.parametrize("bits", [16, 24, 32])
def test_fused_boundary_control_byte_identity_and_direct_cursor_incompatibility(bits):
    inv, control = probe.fixture(bits=bits), probe.fixture(bits=bits)
    run = owner(inv)
    req = inv.stages["next"].request
    widths = bytes([bits] * req.out_features)
    weights = np.full(req.out_features, 0.125, dtype="<f4")
    bias = np.arange(req.out_features, dtype="<f4") / 16
    activation = np.arange(3 * req.in_features, dtype="<f4").reshape(3, -1) / 7 - 2
    layout = _native.PreparedStageLayout("next", req.in_features, widths, bits, bits,
        8, 3, weights.tobytes(), bias.tobytes())
    query = None
    try:
        run.remote_started("current")
        token = run.prefetch("next", 3)
        run.remote_finished()
        r, s, _ = run.consume(token)
        packed, scales = _native.prepared_quantize_pack(activation.tobytes(), r.tobytes(),
            3, req.in_features, 8, bits)
        ticket = "c1" * 16
        query, fused_inputs = layout.prepare(control.stages["next"]._mask_rows,
            0, 3, 3, [ticket], activation.tobytes())
        assert fused_inputs == [packed]
        signed = np.arange(3 * req.out_features, dtype=np.int64).reshape(3, -1) - 15
        masked = ((signed - s.astype(np.int64)) & (req.modulus - 1)).astype(np.uint32)
        # Uniform-width public fixture; independent little-endian bit packing.
        payload = b"".join(int(v).to_bytes(bits // 8, "little") for v in masked.flat)
        frame = b"PLLMPR01" + msgpack.packb([ticket, "next", 3, bits,
            hashlib.sha256(widths).digest(), 0, payload], use_bin_type=True)
        query.receive(frame)
        fused_output, _ = query.finish()
        runahead_output = _native.prepared_dequantize_output(payload, s.tobytes(), widths,
            3, scales, weights.tobytes(), bias.tobytes())
        oracle = signed.astype("<f4") * np.frombuffer(scales, "<f4")[:, None] * weights + bias
        assert memoryview(fused_output).tobytes() == runahead_output == oracle.tobytes()
        # Existing fused owner has no already-expanded-mask entry point: it
        # attempts another native claim, correctly rejects burned rows.
        with pytest.raises(ValueError, match="consumed"):
            layout.prepare(inv.stages["next"]._mask_rows, 0, 3, 3,
                           [ticket], activation.tobytes())
    finally:
        if query is not None:
            query.cancel()
        run.close()
        inv.cancel()
        control.cancel()


def test_inventory_cancel_before_expansion_propagates_and_drains():
    entered, release = threading.Barrier(2), threading.Barrier(2)

    def block():
        entered.wait(timeout=5)
        release.wait(timeout=5)

    inv = probe.fixture()
    run = owner(inv, before_expand=block)
    run.remote_started("current")
    token = run.prefetch("next", 1)
    entered.wait(timeout=5)
    inv.cancel()
    release.wait(timeout=5)
    run.remote_finished()
    with pytest.raises(RuntimeError, match="cancelled"):
        run.consume(token)
    assert run.state == "closed" and run.expanded_bytes == 0
    assert run.waste_bytes == 112  # conservative attempted-buffer waste
    assert inv.status()["reserved"] == 0


def test_concurrent_consume_delivers_handle_once_with_barrier():
    inv = probe.fixture()
    run = owner(inv)
    run.remote_started("current")
    token = run.prefetch("next", 1)
    run.remote_finished()
    both_waiting = threading.Barrier(3)
    original = run._slot

    class GatedFuture:
        def result(self):
            both_waiting.wait(timeout=5)
            return original[4].result()

    run._slot = (*original[:4], GatedFuture())

    def consume():
        try:
            return "delivered", probe.digest(run.consume(token))
        except RuntimeError:
            return "burned", None

    with ThreadPoolExecutor(2) as consumers:
        attempts = [consumers.submit(consume) for _ in range(2)]
        both_waiting.wait(timeout=5)
        outcomes = [attempt.result(timeout=5)[0] for attempt in attempts]
    assert sorted(outcomes) == ["burned", "delivered"]
    assert run.delivered_bytes == 112 and run.waste_bytes == 0
    assert run.state == "closed"


def test_eager_and_verified_controls_cannot_be_prefetched():
    inv = probe.fixture()
    stage = inv.stages["next"]
    # Eager control has no native one-use cursor: reject rather than silently
    # changing its ownership/trust contract.
    stage.input_mask = expand_preparation_mask(stage.request)
    stage.output_mask = expand_output_mask(stage.request)
    stage._mask_rows = None
    lease = inv.reserve(3)
    with pytest.raises(ValueError, match="native unverified"):
        owner(inv, lease)
    lease.close()
    inv = probe.fixture()
    inv.stages["next"].verification = object()
    lease = inv.reserve(3)
    with pytest.raises(ValueError, match="native unverified"):
        owner(inv, lease)
    lease.close()


def test_existing_finish_lifecycle_keeps_online_guard_until_worker_drained():
    from types import SimpleNamespace
    from pllm.runtime.client import RuntimeClient

    entered, release, closing = (threading.Barrier(2) for _ in range(3))

    def block():
        entered.wait(timeout=5)
        release.wait(timeout=5)

    inv = probe.fixture()
    run = owner(inv, before_expand=block, on_closing=lambda: closing.wait(timeout=5))
    run.remote_started("current")
    run.prefetch("next", 1)
    entered.wait(timeout=5)
    refill = []
    # No live HTTP client: invoke actual lifecycle methods against minimal
    # client state, recording (not executing) idle-refill scheduling.
    client = object.__new__(RuntimeClient)
    client._activity_lock = threading.Lock()
    client._transformer_state_lock = threading.Lock()
    client._closing = False
    client._online_active, client._preparation_active = 1, 0
    client.audit = SimpleNamespace(preparation_requests_during_online=0)
    client.background_inventory_refill = True
    client._provider_executor = SimpleNamespace(submit=lambda *args: refill.append(args))
    state = SimpleNamespace(active_prepared_responses=1, retired_inventories=[],
                            prepared_inventory_spare=None, refill_future=None)
    with ThreadPoolExecutor(1) as control:
        finish = control.submit(client._finish_prepared_response, "fixture", state, run)
        closing.wait(timeout=5)
        assert run.state == "closing" and client._online_active == 1
        assert state.active_prepared_responses == 1 and refill == []
        with pytest.raises(Exception, match="preparation cannot run while inference is online"):
            client._begin_preparation()
        release.wait(timeout=5)
        finish.result(timeout=5)
    assert run.state == "closed" and client._online_active == 0
    assert state.active_prepared_responses == 0 and len(refill) == 1
