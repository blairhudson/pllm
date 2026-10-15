"""Bounded ownership attacks and installed-wheel native arithmetic oracle."""
from dataclasses import replace
import importlib.util
from pathlib import Path
import sys
import threading

import numpy as np
import pytest

PATH = Path(__file__).resolve().parents[1] / "scripts/probe_boundary_head_runahead.py"
SPEC = importlib.util.spec_from_file_location("boundary_head_runahead_probe", PATH)
p = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = p
SPEC.loader.exec_module(p)


def fixture(a=None, b=None):
    authority = p.Binding("session/epoch/admission", "source-digest", "lm_head", 0)
    replies = []

    def worker(binding, share):
        result = p.Reply(binding, np.asarray([[int(share[0, 0])]], np.uint32))
        replies.append(result)
        return result

    return p.OneAhead(authority, 1, 1, a or worker, b or worker), authority, replies


def test_exact_one_use_and_destruction():
    owner, binding, replies = fixture()
    owner.issue(binding)
    mask = owner.slot[1]
    with pytest.raises(ValueError):
        owner.issue(binding)
    result = owner.consume(binding, np.asarray([[-127]], np.int8))
    assert result.tolist() == [[-127]]
    assert not mask.any() and all(not r.values.any() for r in replies)
    with pytest.raises(ValueError):
        owner.issue(binding)
    owner.close()
    assert owner.stats["peak_futures"] == 1
    assert owner.stats["consumed"] == 1 and owner.stats["unused_b"] == 0


@pytest.mark.parametrize("field,value", [("context", "other"), ("source", "other"),
                                        ("stage", "other"), ("step", 1), ("role", "A")])
def test_substitution_burns_owner(field, value):
    owner, binding, replies = fixture()
    owner.issue(binding)
    mask = owner.slot[1]
    with pytest.raises(ValueError):
        owner.consume(replace(binding, **{field: value}), np.asarray([[1]], np.int8))
    assert owner.closed and owner.slot is None and not mask.any()
    assert owner.stats["unused_b"] == 1
    assert all(not r.values.any() for r in replies)


def test_cancel_after_admission_before_consumption_drains_late_completion():
    entered, release = threading.Event(), threading.Event()
    replies = []

    def delayed(binding, share):
        entered.set()
        assert release.wait(3)
        reply = p.Reply(binding, np.ones((1, 1), np.uint32))
        replies.append(reply)
        return reply

    owner, binding, _ = fixture(b=delayed)
    owner.issue(binding)
    mask = owner.slot[1]
    assert entered.wait(3)
    closer = threading.Thread(target=owner.close)
    closer.start()
    release.set()
    closer.join(3)
    assert not closer.is_alive() and owner.closed and owner.slot is None
    assert not mask.any() and not replies[0].values.any()
    assert owner.stats["issued_b"] == owner.stats["unused_b"] == 1
    assert owner.stats["upload_bytes"] > 4 and owner.stats["download_bytes"] > 4
    with pytest.raises(ValueError):
        owner.issue(binding)


@pytest.mark.parametrize("bad_role", ["A", "B"])
def test_partial_worker_failure_drains_success(bad_role):
    def fail(binding, share):
        raise RuntimeError("worker failed after peer issued")

    owner, binding, replies = fixture(**{bad_role.lower(): fail})
    owner.issue(binding)
    mask = owner.slot[1]
    with pytest.raises(RuntimeError):
        owner.consume(binding, np.asarray([[1]], np.int8))
    assert owner.closed and owner.slot is None and not mask.any()
    assert len(replies) == 1 and not replies[0].values.any()
    assert owner.stats["unused_b"] == 1


def test_bad_reply_binding_and_geometry_destroyed():
    reply = p.Reply(p.Binding("wrong", "wrong", "wrong", 0), np.ones((2, 1), np.uint32))
    owner, binding, _ = fixture(b=lambda *_: reply)
    owner.issue(binding)
    with pytest.raises(ValueError):
        owner.consume(binding, np.asarray([[1]], np.int8))
    assert owner.closed and not reply.values.any()


def test_cancellation_during_a_cannot_deliver():
    entered, release = threading.Event(), threading.Event()

    def slow_a(binding, share):
        entered.set()
        assert release.wait(3)
        return p.Reply(binding, np.ones((1, 1), np.uint32))

    owner, binding, _ = fixture(a=slow_a)
    owner.issue(binding)
    errors = []

    def consume():
        try:
            owner.consume(binding, np.asarray([[1]], np.int8))
        except ValueError as exc:
            errors.append(str(exc))

    caller = threading.Thread(target=consume)
    caller.start()
    assert entered.wait(3)
    closer = threading.Thread(target=owner.close)
    closer.start()
    # Synchronize closed signal without relying on sleep ordering.
    with owner.lock:
        owner.closed = True
    release.set()
    caller.join(3)
    closer.join(3)
    assert not caller.is_alive() and not closer.is_alive()
    assert errors == ["cancelled before delivery"]
    assert owner.stats["consumed"] == 0


def test_fresh_entropy_requested_for_every_issue(monkeypatch):
    calls = []
    original = p.secrets.token_bytes

    def entropy(n):
        calls.append(n)
        return original(n)

    monkeypatch.setattr(p.secrets, "token_bytes", entropy)
    owner, binding, _ = fixture()
    masks = []
    for step in range(3):
        current = replace(binding, step=step)
        owner.issue(current)
        masks.append(owner.slot[1].copy())
        owner.consume(current, np.asarray([[1]], np.int8))
    owner.close()
    assert calls == [4, 4, 4]  # Fresh entropy call, not probabilistic inequality test.


def test_schedule_and_full_geometry_resources():
    model = p.schedule(10, 3, 4, 8)
    assert model["ready_serial_ms"] == 136
    assert model["ready_parallel_ms"] == 112
    assert model["one_ahead_including_unused_tail_ms"] == 104
    assert model["charged_worker_ms"] == 56
    assert model["gain_vs_ready_parallel_ms"] == 8
    assert p.schedule(10, 3, 4, 8, unused=1)["charged_worker_ms"] == 60
    assert p.resources(896, 151936)["retained_b_result_bytes"] == 607744


def test_native_full_logits_and_worker_replay():
    from pllm.runtime.native import MaskedGEMM
    weights = np.asarray([[127, -127], [-1, 1], [0, 0]], np.int8)
    binding = p.Binding("session", "source", "head", 0)
    worker = p.NativeRole(MaskedGEMM(threads=1).compile(weights), binding, "B")
    share = np.asarray([[2**32 - 1, 2**31]], np.uint32)
    reply = worker(binding, share)
    expected = (share.astype(np.uint64) @ weights.astype(np.int64).T.astype(np.uint64)).astype(np.uint32)
    assert np.array_equal(reply.values, expected)
    with pytest.raises(ValueError):
        worker(binding, share)
    result = p.run(vocabulary=17, hidden=8, steps=2)
    assert result["exact_full_logits"]
    ledger = result["measured_synthetic"]["one_ahead"][0]["ledger"]
    assert ledger["issued_b"] == 2 and ledger["unused_b"] == 0
    parallel = result["measured_synthetic"]["ready_parallel"][0]["ledger"]
    assert parallel["issued_b"] == 2 and parallel["consumed"] == 2
    assert parallel["upload_bytes"] == ledger["upload_bytes"]
    assert parallel["download_bytes"] == ledger["download_bytes"]
    assert not result["measured_ablations"] and not result["modeled_ablations"]
    for field in ("probe_source", "native_build", "installed_native_wrapper", "installed_quantization"):
        assert len(result["provenance"][field]["sha256"]) == 64
        assert result["provenance"][field]["bytes"] > 0


@pytest.mark.parametrize("owner_class", [p.OneAhead, p.ReadyParallel])
def test_extreme_zero_logits_sampler_and_rng_exact(owner_class):
    from types import SimpleNamespace
    from pllm.runtime.native import MaskedGEMM
    from pllm.runtime.quantization import dequantize_matmul
    from pllm.runtime.transformer_client import MaskedTransformerClientRuntime
    weights = np.asarray([[127, -127], [-127, 127], [0, 0], [1, -1]], np.int8)
    binding = p.Binding("session", "source", "head", 0)
    kernel = MaskedGEMM(threads=1)
    a, b = (p.NativeRole(kernel.compile(weights.copy()), binding, role) for role in ("A", "B"))
    owner = owner_class(binding, 2, 4, a, b)
    actual_rng = SimpleNamespace(rng=np.random.default_rng(19))
    expected_rng = SimpleNamespace(rng=np.random.default_rng(19))
    try:
        for step, row in enumerate(([127, -127], [0, 0], [-128, 127])):
            q = np.asarray([row], np.int8)
            current = replace(binding, step=step)
            if isinstance(owner, p.ReadyParallel):
                result = owner.execute(current, q)
            else:
                owner.issue(current)
                result = owner.consume(current, q)
            expected = q.astype(np.int64) @ weights.astype(np.int64).T
            scale_x = np.asarray([[.25]], np.float32)
            scale_w = np.asarray([.1, .2, .3, .4], np.float32)
            logits = dequantize_matmul(result, scale_x, scale_w).reshape(-1)
            reference = dequantize_matmul(expected, scale_x, scale_w).reshape(-1)
            assert logits.tobytes() == reference.tobytes()
            for temperature in (0, .7):
                assert MaskedTransformerClientRuntime.sample(actual_rng, logits, temperature=temperature, top_p=.9) == MaskedTransformerClientRuntime.sample(expected_rng, reference, temperature=temperature, top_p=.9)
                assert actual_rng.rng.bit_generator.state == expected_rng.rng.bit_generator.state
    finally:
        owner.close()


@pytest.mark.parametrize("window", [0, 1, 10, 100])
def test_equal_independent_service_has_no_gain(window):
    model = p.schedule(window, 10, 10, 8)
    assert model["gain_vs_ready_parallel_ms"] == 0
    assert model["issued_b"] == 8 and model["unused_b"] == 0
    asymmetric = p.schedule(window, 5, 15, 8)
    assert asymmetric["gain_vs_ready_parallel_ms"] == 8 * min(window, 10)
    assert p.schedule(window, 15, 5, 8)["gain_vs_ready_parallel_ms"] == 0


def test_unused_tail_is_explicit_charged_ablation():
    result = p.run(vocabulary=17, hidden=8, steps=2, unused_tail=True)
    primary = result["measured_synthetic"]["one_ahead"][0]["ledger"]
    ablation = result["measured_ablations"]["one_ahead_unused_tail"][0]["ledger"]
    assert primary["issued_b"] == 2 and primary["unused_b"] == 0
    assert ablation["issued_b"] == 3 and ablation["unused_b"] == 1
    assert ablation["upload_bytes"] > primary["upload_bytes"]
    assert ablation["download_bytes"] > primary["download_bytes"]
    assert all(m["case"] == "unused_tail_ablation" for m in result["modeled_ablations"])


@pytest.mark.parametrize("bad_role", ["A", "B"])
def test_ready_parallel_failure_uses_same_terminal_cleanup(bad_role):
    def fail(binding, share):
        raise RuntimeError("worker failed")

    base, binding, replies = fixture(**{bad_role.lower(): fail})
    owner = p.ReadyParallel(binding, 1, 1, base.a, base.b)
    base.close()
    with pytest.raises(RuntimeError):
        owner.execute(binding, np.asarray([[1]], np.int8))
    assert owner.closed and owner.slot is None
    assert all(not reply.values.any() for reply in replies)
