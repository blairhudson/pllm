"""Small native arithmetic and bounded online row lifecycle controls."""
import importlib.util
import sys
import threading
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

PATH = Path(__file__).resolve().parents[1] / "scripts/probe_masked_decode_microbatch.py"
SPEC = importlib.util.spec_from_file_location("masked_decode_microbatch_probe", PATH)
p = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = p
SPEC.loader.exec_module(p)


class Clock:
    def __init__(self):
        self.value = 0.

    def __call__(self):
        return self.value


def setup(**kwargs):
    clock = Clock()
    w = np.array([[1, -2, 127], [-128, 3, 0]], np.int8)
    owner = p.BatchOwner(w, clock=clock, **kwargs)
    rng = np.random.default_rng(10)
    return owner, clock, rng


def request(owner, rng, sid, ticket="0", *, key=None):
    if sid not in owner.sessions:
        owner.open_session(sid, bytes([int(sid) + 1]) * 32)
    fixture = p.fixture(owner, sid, ticket, rng, key=key)
    k, payload, mac, _, _ = fixture
    reply = owner.submit(sid, ticket, k, payload, mac, expires=10.)
    return reply, fixture


def assert_reply(owner, sid, ticket, reply, fixture):
    key, _, _, x, mask = fixture
    assert reply.done and not reply.error
    np.testing.assert_array_equal((reply.output.astype(np.int64) - mask) % key.modulus,
                                  p.oracle(owner.weights, [x], key.modulus)[0])
    assert reply.mac == p.signature(owner.sessions[sid].secret, sid, ticket, key,
                                    b"reply\0" + reply.output.tobytes())


def test_native_independent_complete_rows_size_and_replay():
    owner, _, rng = setup(batch_size=3)
    assert owner.backend == "rust"
    rows = [request(owner, rng, str(i)) for i in range(3)]
    assert len({f[1] for _, f in rows}) == 3
    assert len({owner.sessions[str(i)].tickets["0"].correction for i in range(3)}) == 3
    assert len(owner.dispatch()) == 3
    for i, (reply, f) in enumerate(rows):
        assert_reply(owner, str(i), "0", reply, f)
        assert owner.sessions[str(i)].tickets["0"].state == "consumed"
        with pytest.raises(ValueError, match="replay"):
            owner.submit(str(i), "0", *f[:3], expires=10.)
    assert owner.metrics["size_dispatch"] == 1


def test_deadline_and_low_load_adaptive():
    owner, clock, rng = setup(batch_size=4, wait_seconds=.25)
    reply, f = request(owner, rng, "0")
    assert not owner.dispatch() and owner.next_wake() == .25
    clock.value = .249
    assert not owner.dispatch()
    clock.value = .25
    owner.dispatch()
    assert reply.queue_seconds == .25
    assert_reply(owner, "0", "0", reply, f)
    assert owner.metrics["deadline_dispatch"] == 1
    adaptive, _, rng = setup(batch_size=4, wait_seconds=.25, adaptive=True)
    reply, f = request(adaptive, rng, "0")
    adaptive.dispatch()
    assert reply.queue_seconds == 0 and adaptive.metrics["idle_dispatch"] == 1


@pytest.mark.parametrize("change", [dict(stage="other"), dict(modulus=65519),
                                    dict(body_fingerprint="other-public-body")])
def test_public_compatible_groups_never_cross(change):
    calls = []
    owner, clock, rng = setup(batch_size=2, wait_seconds=1.,
        kernel=lambda rows, mod: calls.append((len(rows), mod)) or p.oracle(
            np.array([[1, -2, 127], [-128, 3, 0]], np.int8), rows, mod))
    first, _ = request(owner, rng, "0")
    second, _ = request(owner, rng, "1", key=replace(owner.public_key(), **change))
    assert not owner.dispatch()
    clock.value = 1.
    assert owner.dispatch() == [first]
    assert owner.dispatch() == [second]
    assert [n for n, _ in calls] == [1, 1]


@pytest.mark.parametrize("change", [dict(weight_digest="bad"), dict(ring="pow2"),
                                    dict(codec="different"), dict(wire_bits=16),
                                    dict(input_width=4), dict(output_width=3),
                                    dict(model="other"), dict(weight_bits=4)])
def test_unsupported_public_keys_rejected(change):
    owner, _, _ = setup()
    with pytest.raises(ValueError, match="compatibility"):
        owner._validate_key(replace(owner.public_key(), **change))


def test_fairness_one_pending_and_fifo_across_groups():
    owner, clock, rng = setup(batch_size=2, wait_seconds=1.)
    old, _ = request(owner, rng, "0", key=owner.public_key(stage="old"))
    clock.value = .1
    new, _ = request(owner, rng, "1")
    f = p.fixture(owner, "0", "1", rng, key=owner.public_key(stage="old"))
    with pytest.raises(ValueError, match="admission"):
        owner.submit("0", "1", *f[:3], expires=10.)
    clock.value = 1.1
    assert owner.dispatch() == [old]
    assert owner.dispatch() == [new]
    again = owner.submit("0", "1", *f[:3], expires=10.)
    clock.value = 2.1
    assert owner.dispatch() == [again]


def test_wrong_auth_malformed_and_replay_do_not_poison_peer():
    owner, _, rng = setup(batch_size=2, wait_seconds=0.)
    good, f = request(owner, rng, "0")
    owner.open_session("1", b"z" * 32)
    bad = p.fixture(owner, "1", "0", rng)
    with pytest.raises(ValueError, match="authentication"):
        owner.submit("1", "0", *bad[:2], f[2], expires=10.)
    assert owner.sessions["1"].tickets["0"].state == "prepared"
    payload = np.full(3, bad[0].modulus, dtype="<u4").tobytes()
    mac = p.signature(b"z" * 32, "1", "0", bad[0], payload)
    with pytest.raises(ValueError, match="malformed"):
        owner.submit("1", "0", bad[0], payload, mac, expires=10.)
    assert owner.sessions["1"].tickets["0"].state == "burned"
    owner.dispatch()
    assert_reply(owner, "0", "0", good, f)


def test_queue_expiry_cancel_cleanup_and_caps():
    owner, clock, rng = setup(batch_size=4, wait_seconds=1., max_sessions=2,
                              queue_capacity=1, tickets_per_session=1, idle_seconds=20.)
    reply, _ = request(owner, rng, "0")
    owner.open_session("1", b"z" * 32)
    with pytest.raises(ValueError, match="session admission"):
        owner.open_session("2", b"x" * 32)
    with pytest.raises(ValueError, match="ticket admission"):
        p.fixture(owner, "0", "1", rng)
    f = p.fixture(owner, "1", "0", rng)
    with pytest.raises(ValueError, match="row admission"):
        owner.submit("1", "0", *f[:3], expires=10.)
    clock.value = 10.
    owner.cleanup()
    assert reply.error == "row expired" and owner.sessions["0"].tickets["0"].state == "burned"
    queued = owner.submit("1", "0", *f[:3], expires=15.)
    owner.cancel("1")
    assert queued.error == "session cancelled" and not owner.queue
    clock.value = 20.
    owner.cleanup()
    assert not owner.sessions and not owner.inflight


def test_cancel_during_native_work_and_no_reuse():
    entered, release = threading.Event(), threading.Event()
    def blocked(rows, mod):
        entered.set()
        assert release.wait(2)
        return p.oracle(np.array([[1, -2, 127], [-128, 3, 0]], np.int8), rows, mod)
    owner, _, rng = setup(batch_size=2, kernel=blocked)
    one, _ = request(owner, rng, "0")
    two, f = request(owner, rng, "1")
    thread = threading.Thread(target=owner.dispatch)
    thread.start()
    assert entered.wait(2)
    assert all(s.tickets["0"].state == "consumed" for s in owner.sessions.values())
    owner.cancel("0")
    # Recreated identifier uses fresh credentials; old generation cannot write
    # into newly admitted session or alter its pending/ticket ownership.
    owner.open_session("0", b"n" * 32)
    newer, _ = request(owner, rng, "0", ticket="new")
    release.set()
    thread.join(2)
    assert not thread.is_alive() and one.error == "session cancelled" and one.output is None
    assert not newer.done and owner.sessions["0"].pending
    assert_reply(owner, "1", "0", two, f)


def test_compute_failure_burns_consumed_and_expiry_after_compute():
    def broken(rows, mod):
        raise RuntimeError("synthetic native failure")
    owner, _, rng = setup(batch_size=1, kernel=broken)
    reply, f = request(owner, rng, "0")
    owner.dispatch()
    assert reply.error == "kernel failure: RuntimeError"
    assert owner.sessions["0"].tickets["0"].state == "consumed"
    with pytest.raises(ValueError, match="replay"):
        owner.submit("0", "0", *f[:3], expires=10.)
    owner, clock, rng = setup(batch_size=1)
    original = owner.kernel
    def slow(rows, mod):
        result = original(rows, mod)
        clock.value = 10.
        return result
    owner.kernel = slow
    reply, _ = request(owner, rng, "0")
    owner.dispatch()
    assert reply.error == "row expired or session terminal" and reply.output is None


def test_memory_admission_rollback_and_terminal_release():
    owner, _, rng = setup(batch_size=1)
    baseline = owner.charge()
    owner.memory_limit = baseline + 1
    with pytest.raises(ValueError, match="memory admission"):
        owner.open_session("0", b"x" * 32)
    assert not owner.sessions and owner.charge() == baseline
    owner.memory_limit = baseline + 10000
    reply, _ = request(owner, rng, "0")
    owner.close()
    assert reply.done and not owner.sessions and not owner.queue and owner.charge() == baseline


def test_unsigned_near_modulus_and_weight_extremes():
    owner, _, _ = setup(batch_size=2)
    for mod in (2, 65521, 2**31 - 1):
        rows = np.array([[mod - 1, 0, mod - 1], [1, mod - 1, 1]], dtype=np.uint32)
        np.testing.assert_array_equal(owner.matrix.modular(rows, mod), p.oracle(owner.weights, rows, mod))


def test_concurrent_duplicate_enqueue_consumes_exactly_once():
    owner, _, rng = setup(batch_size=1)
    owner.open_session("0", b"x" * 32)
    f = p.fixture(owner, "0", "0", rng)
    gate = threading.Barrier(3)
    outcomes = []
    def submit():
        gate.wait()
        try:
            outcomes.append(owner.submit("0", "0", *f[:3], expires=10.))
        except ValueError as exc:
            outcomes.append(str(exc))
    threads = [threading.Thread(target=submit) for _ in range(2)]
    for thread in threads:
        thread.start()
    gate.wait()
    for thread in threads:
        thread.join(2)
    assert sum(isinstance(x, p.Reply) for x in outcomes) == 1
    assert "ticket replay or unknown" in outcomes
    owner.dispatch()
    assert owner.metrics["rows"] == 1 and not owner.queue


def test_idle_session_expiry_during_compute_suppresses_delivery():
    owner, clock, rng = setup(batch_size=1, idle_seconds=1.)
    original = owner.kernel
    def slow(rows, mod):
        value = original(rows, mod)
        clock.value = 1.
        return value
    owner.kernel = slow
    reply, _ = request(owner, rng, "0")
    owner.dispatch()
    assert reply.error == "session cancelled" and reply.output is None
    assert not owner.sessions and not owner.inflight


def test_authenticated_key_mismatch_and_initial_expiry_burn_only_row():
    owner, clock, rng = setup(batch_size=2, wait_seconds=0.)
    good, f = request(owner, rng, "0")
    owner.open_session("1", b"z" * 32)
    k, payload, _, _, _ = p.fixture(owner, "1", "0", rng)
    alternate = replace(k, stage="other")
    mac = p.signature(b"z" * 32, "1", "0", alternate, payload)
    with pytest.raises(ValueError, match="malformed"):
        owner.submit("1", "0", alternate, payload, mac, expires=10.)
    assert owner.sessions["1"].tickets["0"].state == "burned"
    expired = p.fixture(owner, "1", "1", rng)
    with pytest.raises(ValueError, match="expired"):
        owner.submit("1", "1", *expired[:3], expires=clock())
    assert owner.sessions["1"].tickets["1"].state == "burned"
    owner.dispatch()
    assert_reply(owner, "0", "0", good, f)


def test_constructor_budget_preflight_before_copy_or_native_allocation(monkeypatch):
    import pllm.runtime.native as native
    touched = []
    class NoCopy(np.ndarray):
        def tobytes(self, *args, **kwargs):
            touched.append("copy")
            raise AssertionError("weight copy before admission")
    weights = np.zeros((2, 3), dtype=np.int8).view(NoCopy)
    def forbidden(*args, **kwargs):
        touched.append("allocation")
        raise AssertionError("allocation before admission")
    monkeypatch.setattr(native, "MaskedGEMM", forbidden)
    monkeypatch.setattr(p.np, "frombuffer", forbidden)
    with pytest.raises(ValueError, match="memory admission"):
        p.BatchOwner(weights, memory_bytes=1)
    assert not touched


@pytest.mark.parametrize("policy", [
    {"batch_size": True}, {"batch_size": 1.5}, {"batch_size": 33},
    {"max_sessions": 33}, {"tickets_per_session": 33}, {"queue_capacity": 33},
    {"memory_bytes": True}, {"memory_bytes": 1.5}, {"memory_bytes": (128 << 20) + 1},
    {"wait_seconds": float("nan")}, {"wait_seconds": float("inf")},
    {"wait_seconds": True}, {"wait_seconds": 1.01}, {"wait_seconds": "0.1"},
    {"wait_seconds": 10**400}, {"idle_seconds": 10**400},
    {"idle_seconds": float("inf")}, {"idle_seconds": 0}, {"idle_seconds": 3601},
    {"idle_seconds": True}, {"adaptive": 1}, {"clock": None}, {"kernel": 1},
])
def test_constructor_typed_policy_caps_before_allocation(monkeypatch, policy):
    import pllm.runtime.native as native
    def forbidden(*args, **kwargs):
        raise AssertionError("allocation before policy validation")
    monkeypatch.setattr(native, "MaskedGEMM", forbidden)
    pweights = np.zeros((2, 3), dtype=np.int8)
    monkeypatch.setattr(p.np, "frombuffer", forbidden)
    with pytest.raises(ValueError, match="owner limits"):
        p.BatchOwner(pweights, **policy)


def test_geometry_preflight_rejects_array_like_and_oversize_before_copy(monkeypatch):
    class ArrayLike:
        def __array__(self, *args, **kwargs):
            raise AssertionError("unbounded array-like conversion")
    class NoCopy(np.ndarray):
        def tobytes(self, *args, **kwargs):
            raise AssertionError("unbounded copy")
    oversized = np.zeros((1, 1025), dtype=np.int8).view(NoCopy)
    for weights in (ArrayLike(), oversized, np.zeros((0, 3), np.int8), np.zeros((2, 3), np.int64)):
        with pytest.raises(ValueError, match="bounded int8 ndarray"):
            p.BatchOwner(weights)


def test_native_fallback_rejected_before_weight_copy(monkeypatch):
    import pllm.runtime._native_support as support
    class NoCopy(np.ndarray):
        def tobytes(self, *args, **kwargs):
            raise AssertionError("copy before native availability check")
    monkeypatch.setattr(support, "extension", lambda: None)
    with pytest.raises(ValueError, match="native backend required"):
        p.BatchOwner(np.zeros((2, 3), np.int8).view(NoCopy))


def test_constructor_requires_native_and_disables_build(monkeypatch):
    import pllm.runtime.native as native
    calls = []
    class Fallback:
        native, backend = False, "python-reference"
        def __init__(self, **kwargs):
            calls.append(kwargs)
        def compile(self, weights):
            raise AssertionError("fallback compile forbidden")
    monkeypatch.setattr(native, "MaskedGEMM", Fallback)
    with pytest.raises(ValueError, match="native backend required"):
        p.BatchOwner(np.zeros((2, 3), np.int8))
    assert calls == [{"threads": 1, "build": False}]
