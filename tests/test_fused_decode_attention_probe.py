"""Independent bitwise and exclusive-storage controls for decode scratch reuse."""
import importlib.util
from pathlib import Path
import weakref
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest

PATH = Path(__file__).resolve().parents[1] / "scripts/probe_fused_decode_attention.py"
SPEC = importlib.util.spec_from_file_location("fused_decode_probe", PATH)
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


@pytest.mark.parametrize("profile", probe.PROFILES)
@pytest.mark.parametrize("layout", ("c", "sequence_major", "reverse", "fortran", "capacity_view"))
@pytest.mark.parametrize("group", (1, 2, 4))
@pytest.mark.parametrize("position,extreme", ((0, False), (8, False), (22, False), (8, True)))
def test_original_bits(profile, layout, group, position, extreme):
    args = probe.fixture(layout=layout, group=group, position=position, extreme=extreme)
    before = [a.copy() for a in args]
    expected = probe.oracle(*args, group, profile)
    assert probe.bits_equal(probe.baseline(*args, group, profile), expected)
    assert probe.bits_equal(probe.fused(*args, group, profile), expected)
    assert all(probe.bits_equal(a, b) for a, b in zip(args, before))


@pytest.mark.parametrize("profile", probe.PROFILES)
@pytest.mark.parametrize("width", (1, 7, 16, 32, 64, 65, 128))
@pytest.mark.parametrize("capacity", (1, 17, 64))
@pytest.mark.parametrize("layout", ("sequence_major", "reverse", "fortran"))
def test_reduction_shape_boundaries(profile, width, capacity, layout):
    args = probe.fixture(width=width, capacity=capacity, position=capacity // 2,
                         layout=layout, group=3, heads=2)
    expected = probe.oracle(*args, 3, profile)
    assert probe.bits_equal(probe.baseline(*args, 3, profile), expected)
    assert probe.bits_equal(probe.fused(*args, 3, profile), expected)


@pytest.mark.parametrize("profile", probe.PROFILES)
def test_exclusive_lifetimes(profile):
    args = probe.fixture(layout="reverse")
    refs, addresses = [], []

    def observe(stage, array):
        if stage == "probabilities" and profile == "prefix_f32":
            assert all(ref() is None for ref in refs)
        refs.append(weakref.ref(array))
        addresses.append(array.__array_interface__["data"][0])
        assert not any(np.shares_memory(array, a) for a in args)

    output = probe.fused(*args, 2, profile, observe=observe)
    assert all(ref() is None for ref in refs)
    assert addresses[0] == addresses[1]
    if profile == "dense":
        assert len(set(addresses)) == 1
    assert not any(np.shares_memory(output, a) for a in args)


def test_full_owner_not_logical_view():
    args = probe.fixture(layout="capacity_view")
    ledger = probe.accounting(*args, 2, "prefix_f32", "fused")
    assert ledger["input_unique_owner_bytes"] > ledger["logical_operand_bytes"]
    assert ledger["complete_output_bytes"] == args[0].nbytes
    assert ledger["full_score_bytes"] == 4 * args[0].shape[1] * args[1].shape[2]
    assert not ledger["is_whole_client_admission"]


def test_fixed_capacity_original_mask_contract():
    q, k, _, p = probe.fixture()
    scores = probe.semantic_attention_scores(q, k, group=2, layout=probe.LAYOUT)
    attrs = {"kind": "full_causal", "absolute_positions_input": "input.positions",
             "padding_mask_input": "input.attention_mask",
             "valid_lengths_input": "input.sequence_lengths",
             "maximum_position_embeddings": 23, "key_domain": "fixed_capacity",
             "cache_validity": "valid_lengths_fixed_capacity"}
    masked = probe.mask_causal_scores(scores, p, attrs, padding_mask=np.ones((1, 1), np.bool_),
                                     valid_lengths=np.array([9], np.int64))
    assert probe.bits_equal(masked, probe.mask_causal_scores(scores, p, {}))
    with pytest.raises(ValueError):
        probe.mask_causal_scores(scores, p, attrs, padding_mask=np.ones((1, 1), np.bool_),
                                valid_lengths=np.array([24], np.int64))


def test_no_profile_swap_or_stride_copy_permission():
    # Original multiquery counterexamples still exclude generic transformations.
    path = PATH.with_name("probe_exact_attention_streaming.py")
    spec = importlib.util.spec_from_file_location("original_streaming_probe", path)
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    assert all(r["changed_words"] > 0 for r in original.counterexamples())


def test_bounds_and_installed_identity():
    args = probe.fixture()
    for profile in ("bf16", "legacy", "unknown"):
        with pytest.raises(ValueError):
            probe.fused(*args, 2, profile)
    with pytest.raises(ValueError):
        probe.fused(np.repeat(args[0], 2, axis=2), *args[1:], 2, "dense")
    with pytest.raises(ValueError):
        probe.fixture(capacity=4097)
    assert "site-packages/pllm" in probe.identity()["pllm"]


def timing_geometry(**changes):
    return {"capacity": 1024, "position": 287, "width": 64,
            "heads": 2, "group": 7, "layout": "sequence_major"} | changes


@pytest.mark.parametrize("iterations", (0, -1, 1025, True, 1.5))
def test_iteration_rejection_before_allocation(monkeypatch, iterations):
    monkeypatch.setattr(probe, "fixture", lambda **kwargs: pytest.fail("fixture allocated"))
    with pytest.raises(ValueError, match="iterations"):
        probe.measure(timing_geometry(), "dense", iterations, 3)


def test_work_and_memory_rejection():
    with pytest.raises(ValueError, match="product-work"):
        probe.measurement_plan(timing_geometry(), "dense", 1024, 1)
    # Individual work passes, cohort cap rejects.
    with pytest.raises(ValueError, match="product-work"):
        probe.measurement_plan(timing_geometry(), "dense", 256, 5)
    with pytest.raises(ValueError, match="memory"):
        probe.measurement_plan(timing_geometry(capacity=4096, width=256, heads=16, group=8),
                               "dense", 1, 1)
    plan = probe.measurement_plan(timing_geometry(), "dense", 256, 3)
    assert plan["products_per_op"] == 2 * 14 * 1024 * 64
    assert plan["products_per_child"] == plan["products_per_op"] * 257
    assert plan["operations_per_arm"] == 771


@pytest.mark.parametrize("changes", ({"capacity": 4097}, {"position": 1024}, {"width": 257},
                                     {"heads": 17}, {"group": 9}, {"layout": "reverse"},
                                     {"capacity": True}))
def test_public_geometry_rejected_before_admission(monkeypatch, changes):
    monkeypatch.setattr(probe, "fixture", lambda **kwargs: pytest.fail("fixture allocated"))

    def runner(*args, **kwargs):
        pytest.fail("invalid geometry reached admission")

    with pytest.raises(ValueError, match="geometry"):
        probe.measure(timing_geometry(**changes), "dense", 256, 3, runner=runner)


def test_no_parent_fixture_before_admission(monkeypatch):
    monkeypatch.setattr(probe, "fixture", lambda **kwargs: pytest.fail("parent allocated fixture"))

    @contextmanager
    def rejected(*args, **kwargs):
        assert kwargs["memory_bytes"] == probe.measurement_plan(
            timing_geometry(), "prefix_f32", 256, 3)["memory_admission_bytes"]
        raise RuntimeError("admission denied")
        yield

    with pytest.raises(RuntimeError, match="admission denied"):
        probe.measure(timing_geometry(), "prefix_f32", 256, 3, runner=rejected)


@pytest.mark.parametrize("key", ("build", "probe_sha256", "operators", "related_sha256"))
def test_frozen_identity_mismatch(key):
    frozen = probe.identity()
    changed = frozen | {key: "different"}
    with pytest.raises(RuntimeError, match="identity mismatch"):
        probe.require_identity(changed, frozen)


def test_child_identity_rejected_before_fixture(monkeypatch):
    monkeypatch.setattr(probe, "fixture", lambda **kwargs: pytest.fail("child allocated fixture"))
    with pytest.raises(RuntimeError, match="identity mismatch"):
        probe.worker({"fixture": timing_geometry(), "profile": "dense", "arm": "fused",
                      "iterations": 1, "expected_identity": {}})


def test_parent_checks_each_child_and_records_swap(monkeypatch):
    monkeypatch.setattr(probe, "fixture", lambda **kwargs: pytest.fail("parent allocated fixture"))
    calls = []

    @contextmanager
    def runner(*args, **kwargs):
        def run(payload):
            calls.append(payload)
            return {"identity": payload["expected_identity"], "payload": payload,
                    "charged_operations": 257, "charged_float_products": 132636672,
                    "output_sha256": "same"}
        guard = SimpleNamespace(check=lambda: None, sample=lambda: SimpleNamespace(swap_used=12),
                                maximum_swap_growth_bytes=2, minimum_available_bytes=2**30, error=None)
        yield run, guard, SimpleNamespace(swap_used=10)

    result = probe.measure(timing_geometry(), "prefix_f32", 256, 3, runner=runner)
    assert len(calls) == 6
    assert result["charged_operations_per_arm"] == {"baseline": 771, "fused": 771}
    assert result["watchdog"]["baseline_swap_used_bytes"] == 10
    assert result["watchdog"]["final_swap_used_bytes"] == 12
    assert result["gain"] is None

    @contextmanager
    def mismatched(*args, **kwargs):
        yield lambda payload: {"identity": {}}, None, None

    with pytest.raises(RuntimeError, match="identity mismatch"):
        probe.measure(timing_geometry(), "prefix_f32", 256, 3, runner=mismatched)


def test_worker_cold_and_repeated_fields_with_fake_clock(monkeypatch):
    args = probe.fixture()
    calls = []
    monkeypatch.setattr(probe, "fixture", lambda **kwargs: args)
    monkeypatch.setattr(probe, "identity", lambda: {"frozen": True})

    def operation(*args):
        calls.append(1)
        return np.zeros_like(args[0])

    monkeypatch.setattr(probe, "fused", operation)
    cpu_clock = iter((0., 1., 10., 12., 20., 26.))
    wall_clock = iter((0., 1.5, 10., 13., 20., 29.))
    monkeypatch.setattr(probe.time, "process_time", lambda: next(cpu_clock))
    monkeypatch.setattr(probe.time, "perf_counter", lambda: next(wall_clock))
    result = probe.worker({"fixture": timing_geometry(), "profile": "dense", "arm": "fused",
                           "iterations": 3, "expected_identity": {"frozen": True}})
    assert len(calls) == 4
    assert result["fixture_setup"]["cpu_seconds"] == 1
    assert result["fixture_setup"]["wall_seconds"] == 1.5
    assert result["cold"]["cpu_seconds"] == 2
    assert result["cold"]["wall_seconds"] == 3
    assert result["repeated_same_fixture"]["cpu_seconds_total"] == 6
    assert result["repeated_same_fixture"]["wall_seconds_total"] == 9
    assert result["repeated_same_fixture"]["cpu_seconds_per_op"] == 2
    assert result["repeated_same_fixture"]["wall_seconds_per_op"] == 3
    assert result["cold"]["output_sha256"] == result["output_sha256"]
