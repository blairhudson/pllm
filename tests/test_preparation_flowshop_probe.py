"""Small stdlib scheduling/lifecycle checks; no checkpoint or native benchmarks."""
import importlib.util
import itertools
from pathlib import Path
import secrets
import struct
import sys
import threading
import time

import pytest

PATH = Path(__file__).resolve().parents[1] / "scripts/probe_preparation_flowshop.py"
spec = importlib.util.spec_from_file_location("preparation_flowshop_probe", PATH)
p = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = p
spec.loader.exec_module(p)
MIB = 1 << 20


def jobs(times=((1, 1), (2, 1), (2, 2)), work=None):
    return [p.Job(str(i), c, u, MIB if work is None else work[i])
            for i, (c, u) in enumerate(times)]


def bindings(js):
    session = secrets.token_hex(16)
    return [p.Binding(session, secrets.token_hex(16), j.stage, "test-plan", "test-weight",
                      2, 3, 2, "signed-i64-test-only", secrets.token_hex(32)) for j in js]


def fixture_compute(binding, stop):
    # Different loop orientation from handwritten fixture oracle below.
    weights = ((1, 2), (3, 4), (-2, 5))
    masks, outputs = ((2, -1, 3), (-1, 2, 0)), ((1, 2), (4, -9))
    values = []
    for row, output in zip(masks, outputs):
        correction = [-s for s in output]
        for x, weight in zip(row, weights):
            for column, w in enumerate(weight):
                correction[column] += x * w
        values.extend(correction)
    return p.Material(binding, bytearray(struct.pack(">4q", *values)))


def local(js=None, **kwargs):
    js = js or jobs()
    bs = bindings(js)
    ledger = p.LocalLedger(bs)
    result = p.run_local(js, bs, kwargs.pop("order", p.priority(js)),
                         compute=kwargs.pop("compute", fixture_compute), ledger=ledger,
                         width=kwargs.pop("width", 2), budget=kwargs.pop("budget", 2 * MIB), **kwargs)
    return result, ledger


def test_independent_tick_oracle_exhausts_integer_tiny_schedules():
    checked = 0
    for times in itertools.product((1, 2), repeat=6):
        js = jobs([(times[2 * i], times[2 * i + 1]) for i in range(3)])
        for width, budget in ((1, 16 * MIB), (2, 2 * MIB), (3, 2 * MIB)):
            for order in itertools.permutations(range(3)):
                model = p.simulate(js, order, width=width, budget=budget)
                assert model["modeled_seconds"] == p.tick_cost(js, order, width=width, budget=budget)
                assert model["peak_work_bytes"] <= budget and model["peak_stage_slots"] <= width
                checked += 1
    assert checked == 1152


def test_heterogeneous_work_head_blocking_matches_independent_oracle():
    js = jobs([(3, 2), (1, 4), (2, 1)], [2 * MIB, MIB, 2 * MIB])
    for order in itertools.permutations(range(3)):
        result = p.simulate(js, order, width=4, budget=3 * MIB)
        assert result["modeled_seconds"] == p.tick_cost(js, order, width=4, budget=3 * MIB)
    oracle = p.exhaustive(js, width=4, budget=3 * MIB)
    assert oracle["enumerated"] == 6
    assert oracle["best"]["modeled_seconds"] == min(
        p.tick_cost(js, o, width=4, budget=3 * MIB) for o in itertools.permutations(range(3)))


def test_gain_loss_and_no_universal_johnson_claim():
    for times, source_cost, heuristic_cost in (
            ([(1, 1), (2, 1), (2, 2)], 6, 5),
            ([(1, 1), (2, 1), (1, 1)], 4, 5),
            ([(8, 1), (1, 8), (2, 4), (7, 1)], 18, 21)):
        js = jobs(times)
        assert p.simulate(js, range(len(js)), width=2, budget=2 * MIB)["modeled_seconds"] == source_cost
        assert p.simulate(js, p.priority(js), width=2, budget=2 * MIB)["modeled_seconds"] == heuristic_cost


def test_deadline_fairness_constraints_filter_not_silently_weaken():
    js = jobs([(1, 1), (2, 1), (1, 1)])
    js[1] = p.Job("1", 2, 1, MIB, deadline=3)
    heuristic = p.simulate(js, p.priority(js), width=2, budget=2 * MIB)
    assert heuristic["deadline_misses"] == ["1"]
    assert p.exhaustive(js, width=2, budget=2 * MIB)["best"]["modeled_seconds"] == 4
    assert p.exhaustive(js, width=2, budget=2 * MIB, max_wait=0)["best"] is None
    assert p.exhaustive(js, width=2, budget=2 * MIB, max_wait=2)["best"] is not None


def test_invalid_admission_and_bounded_oracle():
    for js, order in ((jobs(), [0, 0, 2]), (jobs(work=[3 * MIB, MIB, MIB]), [0, 1, 2]),
                      ([p.Job("x", float("nan"), 1, MIB)], [0])):
        with pytest.raises(ValueError):
            p.simulate(js, order, width=2, budget=2 * MIB)
    with pytest.raises(ValueError, match="eight"):
        p.exhaustive(jobs([(1, 1)] * 9))
    calls = []
    with pytest.raises(ValueError):
        local(jobs(work=[3 * MIB, MIB, MIB]), compute=lambda *x: calls.append(x))
    assert not calls


def test_real_local_fake_transport_numeric_and_all_ack_before_seal():
    js, seen = jobs(), []
    bs = bindings(js)
    ledger = p.LocalLedger(bs)
    def push(material, seq):
        # Independent hand-derived integer results, no simulator arithmetic.
        assert struct.unpack(">4q", material.payload) == (-8, 13, 1, 15)
        seen.append(material.binding.stage)
        assert ledger.state == "OPEN"
        return ledger.push(material, seq)
    result = p.run_local(js, bs, [2, 0, 1], compute=fixture_compute, ledger=ledger,
                         push=push, width=2, budget=2 * MIB)
    assert result["state"] == "READY" and seen == ledger.acked == result["accepted_order"]
    assert result["peak_work_bytes"] <= 2 * MIB
    assert all(m.burned and not m.payload for m in ledger.materials)


def test_slots_and_bytes_remain_charged_while_push_waits_for_ack():
    js = jobs([(1, 1)] * 4)
    bs, issued = bindings(js), []
    ledger = p.LocalLedger(bs)
    def compute(binding, stop):
        issued.append(binding.stage)
        return fixture_compute(binding, stop)
    def push(material, seq):
        # At most two total jobs before first ACK, despite completed computes.
        if seq == 0:
            assert len(issued) <= 2
        return ledger.push(material, seq)
    result = p.run_local(js, bs, range(4), compute=compute, ledger=ledger,
                         push=push, width=4, budget=2 * MIB)
    assert result["peak_stage_slots"] == 2


@pytest.mark.parametrize("field", ["session", "attempt", "stage", "plan", "weight_digest",
                                  "rows", "in_features", "out_features", "ring", "seed_commitment"])
def test_binding_substitution_rejected(field):
    from dataclasses import replace
    js = jobs([(1, 1)])
    b = bindings(js)[0]
    ledger = p.LocalLedger([b])
    ledger.reserve(b)
    value = getattr(b, field)
    wrong = replace(b, **{field: value + 1 if isinstance(value, int) else value + "-wrong"})
    with pytest.raises(ValueError, match="binding"):
        ledger.push(fixture_compute(wrong, None), 0)


def test_replay_reordered_ack_premature_and_wrong_order_seal():
    bs = bindings(jobs([(1, 1)] * 2))
    ledger = p.LocalLedger(bs)
    for binding in bs:
        ledger.reserve(binding)
    a, b = [fixture_compute(binding, None) for binding in bs]
    aa = ledger.push(a, 0)
    with pytest.raises(ValueError):
        ledger.push(a, 1)
    with pytest.raises(ValueError):
        ledger.seal([a.binding.stage])
    bb = ledger.push(b, 1)
    with pytest.raises(ValueError):
        ledger.acknowledge(b, 1, bb)
    with pytest.raises(ValueError):
        ledger.acknowledge(a, 0, bb)
    ledger.acknowledge(a, 0, aa)
    with pytest.raises(ValueError):
        ledger.acknowledge(a, 0, aa)
    ledger.acknowledge(b, 1, bb)
    with pytest.raises(ValueError):
        ledger.seal(list(reversed(ledger.accepted)))
    ledger.seal(ledger.accepted.copy())
    with pytest.raises(ValueError):
        ledger.push(a, 2)


def test_one_use_reserved_before_compute_and_not_reusable_after_failure():
    b = bindings(jobs([(1, 1)]))[0]
    ledger = p.LocalLedger([b])
    material = fixture_compute(b, None)
    with pytest.raises(ValueError):
        ledger.push(material, 0)
    ledger.reserve(b)
    with pytest.raises(ValueError, match="reservation"):
        ledger.reserve(b)
    ledger.abort()
    with pytest.raises(ValueError, match="reservation"):
        ledger.reserve(b)


@pytest.mark.parametrize("fault", ["compute", "push", "ack", "cancel", "timeout", "substitution"])
def test_failure_drains_workers_and_burns_all_material(fault):
    js = jobs([(1, 1)] * 3)
    bs, materials = bindings(js), []
    ledger = p.LocalLedger(bs)
    other_started = threading.Event()
    stopped = threading.Event()
    def compute(binding, stop):
        material = fixture_compute(binding, stop)
        if binding.stage == "0" and fault == "substitution":
            material.binding = bs[1]
        if binding.stage == "1":
            other_started.set()
            assert stop.wait(2), "abort must precede join"
            stopped.set()
        else:
            assert other_started.wait(2)
        if binding.stage == "0" and fault == "compute":
            # Failed callback owns and burns material before throwing.
            material.burn()
            materials.append(material)
            raise RuntimeError("compute failed")
        materials.append(material)
        return material
    cancel = threading.Event()
    def push(material, seq):
        if fault == "push":
            raise RuntimeError("push failed")
        ack = ledger.push(material, seq)
        if fault == "cancel":
            cancel.set()
        if fault == "timeout":
            time.sleep(0.025)
        return b"bad" if fault == "ack" else ack
    with pytest.raises((RuntimeError, ValueError)):
        p.run_local(js, bs, range(3), compute=compute, ledger=ledger, push=push,
                    cancelled=cancel.is_set, width=2, budget=2 * MIB,
                    timeout=0.02 if fault == "timeout" else 3)
    assert ledger.state == "BURNED" and stopped.is_set()
    assert len(materials) == 2 and all(m.burned and not m.payload for m in materials)
    assert not any(t.name.startswith("flowshop-local") for t in threading.enumerate())


def test_pre_cancelled_issues_nothing_and_burns_inventory():
    js, called = jobs(), []
    bs = bindings(js)
    ledger = p.LocalLedger(bs)
    with pytest.raises(RuntimeError, match="cancelled"):
        p.run_local(js, bs, range(3), compute=lambda *x: called.append(x), ledger=ledger,
                    cancelled=lambda: True)
    assert not called and ledger.state == "BURNED"


def test_failure_after_prior_ack_burns_prior_and_late_inventory():
    js, made = jobs([(1, 1)] * 2), []
    bs = bindings(js)
    ledger = p.LocalLedger(bs)
    def compute(binding, stop):
        material = fixture_compute(binding, stop)
        made.append(material)
        if binding.stage == "1":
            material.burn()
            raise ValueError("later failure")
        return material
    with pytest.raises(ValueError, match="later failure"):
        p.run_local(js, bs, range(2), compute=compute, ledger=ledger, width=1)
    assert ledger.acked == ["0"] and ledger.state == "BURNED"
    assert all(m.burned and not m.payload for m in made)


@pytest.mark.parametrize("constraint", ["deadline", "wait"])
def test_actual_deadline_and_wait_limits_burn_without_seal(constraint):
    js = jobs([(1, 1)] * 2)
    if constraint == "deadline":
        js[0] = p.Job("0", 1, 1, MIB, deadline=0.000000001)
    bs = bindings(js)
    ledger = p.LocalLedger(bs)
    with pytest.raises(RuntimeError, match="expired"):
        p.run_local(js, bs, range(2), compute=fixture_compute, ledger=ledger, width=1,
                    max_wait=0 if constraint == "wait" else None)
    assert ledger.state == "BURNED"


def test_duplicate_stage_attempt_and_mixed_session_rejected():
    from dataclasses import replace
    bs = bindings(jobs([(1, 1)] * 2))
    for duplicate in (replace(bs[1], stage=bs[0].stage),
                      replace(bs[1], attempt=bs[0].attempt),
                      replace(bs[1], session="other")):
        with pytest.raises(ValueError):
            p.LocalLedger([bs[0], duplicate])


def test_oversize_returned_payload_burned_on_failure():
    js = jobs([(1, 1)])
    bs, made = bindings(js), []
    ledger = p.LocalLedger(bs)
    def compute(binding, stop):
        material = p.Material(binding, bytearray(MIB + 1))
        made.append(material)
        return material
    with pytest.raises(ValueError, match="exceeds declared work"):
        p.run_local(js, bs, [0], compute=compute, ledger=ledger, width=1)
    assert ledger.state == "BURNED" and made[0].burned


def test_source_derived_shapes_and_work_match_existing_formula():
    js, provenance = p.source_jobs(16, 10_000_000, 0.001)
    assert [v["shape_out_in"] for v in provenance] == [[1152, 896], [9728, 896], [896, 4864]]
    assert len(js) == 6
    assert [j.work for j in js[:3]] == [p.window.preparation_job_work_bytes(
        rows=16, in_features=i, out_features=o) for o, i in ((1152, 896), (9728, 896), (896, 4864))]
    big, _ = p.source_jobs(150, 10_000_000, 0.001)
    with pytest.raises(ValueError, match="oversize"):
        p.simulate(big, range(6), budget=16 * MIB)
