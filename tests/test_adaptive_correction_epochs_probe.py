"""Independent bounded scheduling and source-evidence regression controls."""
import ast
from itertools import product

import pytest

from scripts import probe_adaptive_correction_epochs as probe


@pytest.mark.parametrize("cap", range(1, 65))
def test_every_stop_and_fixed_or_growing_policy(cap):
    policies = [probe.schedule(cap, e) for e in range(1, cap + 1)]
    policies += [probe.schedule(cap, e, geometric=True) for e in (1, 2, 4, 8)]
    for parts, output in product(policies, range(cap + 1)):
        row = probe.account(parts, prompt=36, output=output, cap=cap)
        # Independent per-selection coverage enumeration, including un-emitted EOS.
        needed = list(range(output + 1 if output < cap else cap))
        epochs = []
        offset = 0
        for size in parts:
            positions = list(range(offset, offset + size))
            if set(positions).intersection(needed):
                epochs.append(positions)
            offset += size
        reserved = 36 * 93 + 3 + 96 * (sum(map(len, epochs)) - 1)
        consumed = 36 * 93 + 3 + 96 * (len(needed) - 1)
        assert row["issued_rows"] == reserved
        assert row["claimed_rows"] == consumed
        assert row["burned_rows"] == reserved - consumed
        assert row["pauses"] == len(epochs) - 1
        assert row["logical_correction_frames"] == len(epochs) * 96
        assert row["max_executed_history_bound"] == 36 + cap - 1
        assert row["all_link_tcp_stream_bytes"] is None


def partitions(cap, minimum=1):
    if not cap:
        yield ()
    else:
        for size in range(1, cap + 1):
            if size < minimum and size != cap:
                continue
            for tail in partitions(cap - size, size):
                yield (size, *tail)


@pytest.mark.parametrize("cap", range(1, 10))
@pytest.mark.parametrize("penalty", [0, 96, 384, 1536])
def test_dp_matches_independent_exhaustive_partitions(cap, penalty):
    for counts in probe.priors(cap).values():
        parts = probe.optimize(cap, counts, penalty)
        probe.validate(parts, cap)
        actual = probe.expectation(parts, counts, cap=cap, penalty=penalty)
        all_scores = [probe.expectation(p, counts, cap=cap, penalty=penalty)
                      for p in partitions(cap)]
        assert actual["objective_row_equivalents"] == min(
            row["objective_row_equivalents"] for row in all_scores)


@pytest.mark.parametrize("cap", [1, 2, 4, 8, 16, 32, 64])
def test_exact_toy_state_rng_pending_and_eos_across_sessions(cap):
    for stop in range(cap + 1):
        eos = stop if stop < cap else None
        control = probe.toy_oracle((cap,), cap=cap, eos_after=eos)
        for start in (1, 2, 4, 8):
            parts = probe.schedule(cap, start, geometric=True)
            assert probe.toy_oracle(parts, cap=cap, eos_after=eos) == control
        assert (control["pending"] is None) == (stop < cap)


def test_boundary_eos_is_not_free_and_full_cap_no_domination():
    parts = (4, 8, 16, 32, 4)
    assert probe.account(parts, prompt=36, output=3, cap=64)["pauses"] == 0
    assert probe.account(parts, prompt=36, output=4, cap=64)["pauses"] == 1
    control = probe.account((64,), prompt=36, output=64, cap=64)
    growing = probe.account(parts, prompt=36, output=64, cap=64)
    assert growing["issued_rows"] == control["issued_rows"]
    assert growing["pauses"] > control["pauses"]
    assert growing["logical_correction_frames"] > control["logical_correction_frames"]


def test_idle_only_burn_cancellation_and_ack_failure():
    state = probe.Lifecycle()
    state.prepare(10)
    state.admit()
    state.claim(3)
    with pytest.raises(ValueError):
        state.prepare(5)
    state.retire()
    state.prepare(5)
    with pytest.raises(ValueError, match="acknowledgement"):
        state.admit(acknowledged=False)
    state.prepare(2)
    state.retire()  # cancelled before reservation -> discard, never reuse
    assert (state.issued, state.claimed, state.burned, state.discarded) == (17, 3, 12, 2)
    assert state.session == 2
    state.prepare(7)
    state.admit()
    state.retire(confirmed=False)
    with pytest.raises(ValueError):
        state.prepare(1)


@pytest.mark.parametrize("cap,start", [(0, 1), (65, 1), (8, 0), (8, 65), (True, 4)])
def test_invalid_public_bounds(cap, start):
    with pytest.raises(ValueError):
        probe.schedule(cap, start)


def test_schedule_and_prior_validation():
    for parts, cap in [((4, 2, 8), 14), ((0, 8), 8), ((4,), 8)]:
        with pytest.raises(ValueError):
            probe.validate(parts, cap)
    for counts in ({}, {-1: 1}, {9: 1}, {3: 0}, {3: True}):
        with pytest.raises(ValueError):
            probe.optimize(8, counts, 96)


def test_retained_completed_controls_all_replicates():
    controls = probe.measured_controls()
    assert len(controls) == 12
    for cell in controls:
        if cell["workload"] == "early":
            assert cell["claimed_rows"] == 3639
            assert cell["issued_rows"] == (3639 if cell["variant"] == "epochs" else 9399)
        else:
            assert cell["issued_rows"] == cell["claimed_rows"] == 5697


def test_isolated_source_row_rule_not_heavy_runtime_import():
    path = probe.ROOT / "python/pllm/runtime/semantic_stages.py"
    tree = ast.parse(path.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == "prepared_phase_rows")
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    rule = namespace["prepared_phase_rows"]
    steps = [{"executor": "remote_stage", "operation_ids": [str(i)],
              "terminal_row_only": i >= 93} for i in range(96)]
    schedule = {phase: {"steps": steps} for phase in ("prefill", "decode")}
    bindings = {f"{phase}:{i}": (str(i), "linear")
                for phase in schedule for i in range(96)}
    first = rule(schedule, bindings, 36, 3)
    assert sum(first["prefill"].values()) == 3351
    assert sum(first["decode"].values()) == 288
    continuation = rule(schedule, bindings, 1, 7, continuation=True)
    assert sum(continuation["prefill"].values()) + sum(continuation["decode"].values()) == 768
    # Fixture stage ownership is explicit; this does not generate a native graph.


def test_exact_known_payload_formula_and_unknown_real_widths():
    assert probe.payload_bytes({"a": 3}) is None
    assert probe.payload_bytes({"a": 3, "b": 2}, {"a": 9, "b": 13}) == 8
    with pytest.raises(ValueError):
        probe.payload_bytes({"a": 3}, {"b": 13})


@pytest.mark.parametrize("cap", [8, 16, 32, 64])
def test_optimized_schedules_all_stops_and_toy_parity(cap):
    for counts, penalty in product(probe.priors(cap).values(), (0, 96, 384, 1536)):
        parts = probe.optimize(cap, counts, penalty)
        for stop in range(cap + 1):
            row = probe.account(parts, prompt=36, output=stop, cap=cap)
            assert row["issued_rows"] == row["claimed_rows"] + row["burned_rows"]
            eos = stop if stop < cap else None
            assert probe.toy_oracle(parts, cap=cap, eos_after=eos) == probe.toy_oracle(
                (cap,), cap=cap, eos_after=eos)


def test_expected_conditional_improvement_not_universal_win():
    counts = {3: 80, 64: 20}
    growing = probe.expectation((4, 8, 16, 32, 4), counts, cap=64, penalty=384)
    fixed = [probe.expectation(probe.schedule(64, e), counts, cap=64, penalty=384)
             for e in range(1, 65)]
    assert growing["objective_row_equivalents"] < min(
        row["objective_row_equivalents"] for row in fixed)
    assert growing["full_cap_pauses"] == 4
    assert probe.optimize(64, {64: 100}, 384) == (64,)
