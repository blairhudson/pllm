"""Independent, bounded FP64 parity and rejection tests; no timing/checkpoints."""
import importlib.util
from pathlib import Path
import platform

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location("exact_f64_probe", Path(__file__).resolve().parents[1]
                                            / "scripts/probe_exact_f64_corrections.py")
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)
BACKENDS = ["numpy"] + (["accelerate"] if platform.system() == "Darwin"
                        and platform.machine() == "arm64" else [])


@pytest.mark.parametrize("bits", [16, 24, 32])
def test_whole_domain_inclusive_bounds(bits):
    p = probe.bounds(bits)
    assert p["universal_width"] * 128 * p["mask"] <= 2**53
    assert (p["universal_width"] + 1) * 128 * p["mask"] > 2**53
    assert p["source_l1_limit"] * p["mask"] <= 2**53
    assert p["product_min"] == -128 * p["mask"]
    assert p["product_max"] == 127 * p["mask"]
    if bits == 32:
        assert p["universal_width"] == 16384
        assert p["float_subtraction_universal_width"] == 16383


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("bits", [16, 24, 32])
@pytest.mark.parametrize("cols", [1, 7, 1023, 1024, 16384, 16385])
def test_random_extremes_boundaries_and_subtraction(backend, bits, cols):
    rng = np.random.default_rng(5005 + cols)
    mask = (1 << bits) - 1
    w = rng.integers(-128, 128, (5, cols), dtype=np.int8)
    w[0] = -128
    w[1] = 127
    w[2] = np.resize(np.array([-128, 127, 1, 0], np.int8), cols)
    x = rng.integers(0, mask + 1, (3, cols), dtype=np.uint32)
    x[0] = mask
    x[1] = np.resize(np.array([0, 127, 128, 255, 256, mask, 1 << (bits - 1)], np.uint32), cols)
    s = np.array([[mask, 0, 1, mask, 0]] * 3, np.uint32)
    expected = probe.integer_oracle(w, x, bits, s)
    actual = probe.project(w, x, bits, s, tile=2, backend=backend)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("backend", BACKENDS)
def test_source_certificate_expands_width_and_rejects_stale_claim(backend):
    cols = 20001
    w = np.ones((2, cols), np.int8)
    x = np.full((1, cols), 0xffffffff, np.uint32)
    with pytest.raises(ValueError, match="unsound width"):
        probe.project(w, x, 32, block=cols, backend=backend)
    assert probe.source_bound(w, 32) <= 2**53
    np.testing.assert_array_equal(probe.project(w, x, 32, block=cols, certified=True, backend=backend),
                                  probe.integer_oracle(w, x, 32))
    w[:] = -128
    with pytest.raises(ValueError, match="source certificate"):
        probe.project(w, x, 32, block=cols, certified=True, backend=backend)


@pytest.mark.parametrize("backend", BACKENDS)
def test_all_zero_long_certificate(backend):
    w = np.zeros((1, 20000), np.int8)
    x = np.full((1, 20000), 0xffffffff, np.uint32)
    s = np.array([[0xffffffff]], np.uint32)
    assert int(probe.project(w, x, 32, s, block=20000, certified=True, backend=backend)[0, 0]) == 1


def test_float_subtraction_counterexample():
    # Dot itself exact at universal width; odd output-mask subtraction is not.
    dot = -128 * 16384 * (2**32 - 1)
    mask = 2**32 - 1
    assert int(np.float64(dot)) == dot
    assert abs(dot - mask) > 2**53
    assert int(np.float64(dot) - np.float64(mask)) != dot - mask
    w = np.full((1, 16384), -128, np.int8)
    x = np.full((1, 16384), mask, np.uint32)
    s = np.full((1, 1), mask, np.uint32)
    assert int(probe.project(w, x, 32, s)[0, 0]) == ((dot - mask) & mask)


def test_width_16385_can_lose_unit_even_without_subtraction():
    # First 16384 terms use extrema; last term crosses exact range by one.
    exact = -128 * 16384 * (2**32 - 1) - (2**21 + 1)
    assert exact == -(2**53 + 1)
    assert int(np.float64(exact)) != exact
    w = np.full((1, 16385), -128, np.int8)
    w[0, -1] = -1
    x = np.full((1, 16385), 0xffffffff, np.uint32)
    x[0, -1] = 2**21 + 1
    with pytest.raises(ValueError, match="unsound width"):
        probe.project(w, x, 32, block=16385)
    np.testing.assert_array_equal(probe.project(w, x, 32), probe.integer_oracle(w, x, 32))


def test_unsound_canceling_total_is_not_certificate():
    # Net signed sum zero does not certify arbitrary intermediate sub-sums.
    w = np.resize(np.array([-128, 127, 1], np.int8), (1, 30000))
    assert int(w.astype(np.int64).sum()) == 0
    x = np.full((1, 30000), 0xffffffff, np.uint32)
    with pytest.raises(ValueError, match="source certificate"):
        probe.project(w, x, 32, block=30000, certified=True)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf, 1.5, float(2**53 + 2)])
def test_checked_conversion_rejects(bad):
    with pytest.raises(ValueError):
        probe.checked_integer(np.array([[bad]], np.float64), 2**53)


def test_i64_not_i32_conversion_and_float_checker_limitation():
    values = np.array([[-2**53, 2**53, -128 * (2**32 - 1)]], np.float64)
    np.testing.assert_array_equal(probe.checked_integer(values, 2**53), values.astype(np.int64))
    # Integrality check cannot distinguish an integer-valued wrong result.
    assert probe.checked_integer(np.array([[float(2**53 + 1)]], np.float64), 2**53)[0, 0] == 2**53


@pytest.mark.parametrize("fault", ["weight_dtype", "input_dtype", "ring", "outside_ring",
                                   "shape", "mask_dtype", "mask_shape", "mask_range", "block", "tile"])
def test_reject_invalid_domain(fault):
    w = np.ones((2, 3), np.int8)
    x = np.zeros((1, 3), np.uint32)
    s = np.zeros((1, 2), np.uint32)
    bits, kwargs = 16, {}
    if fault == "weight_dtype": w = w.astype(np.int16)
    if fault == "input_dtype": x = x.astype(np.int64)
    if fault == "ring": bits = 17
    if fault == "outside_ring": x[0, 0] = 65536
    if fault == "shape": x = x[:, :2]
    if fault == "mask_dtype": s = s.astype(np.int64)
    if fault == "mask_shape": s = s[:, :1]
    if fault == "mask_range": s[0, 0] = 65536
    if fault == "block": kwargs["block"] = 0
    if fault == "tile": kwargs["tile"] = 1025
    with pytest.raises(ValueError):
        probe.project(w, x, bits, s, **kwargs)


def test_oracle_budget_workspace_and_costs(monkeypatch):
    w = np.zeros((1024, 4096), np.int8)
    x = np.zeros((1, 4096), np.uint32)
    with pytest.raises(ValueError, match="oracle MAC"):
        probe.integer_oracle(w, x, 32)
    monkeypatch.setattr(probe, "WORKSPACE", 16)
    with pytest.raises(ValueError, match="workspace"):
        probe.project(np.zeros((2, 3), np.int8), np.zeros((1, 3), np.uint32), 32)
    cost = probe.cost(256, 4096, 32, 32)
    assert cost["dgemm_calls"] == 1
    assert cost["digit8_sgemm_calls"] == 20
    assert cost["f64_input_conversion_elements"] == 32 * 4096
    assert cost["f64_weight_conversion_read_write_bytes"] == 9 * 256 * 4096
    assert cost["candidate_rust_workspace_bytes"] > cost["digit8_workspace_bytes"]
    with pytest.raises(ValueError, match="unsound cost"):
        probe.cost(1, 20000, 1, 32, block=20000)


def test_noncontiguous_operands_are_copied_correctly():
    w = np.arange(30, dtype=np.int8).reshape(3, 10)[:, ::2]
    x = np.arange(10, dtype=np.uint32).reshape(1, 10)[:, ::2]
    np.testing.assert_array_equal(probe.project(w, x, 24), probe.integer_oracle(w, x, 24))


@pytest.mark.skipif("accelerate" not in BACKENDS, reason="Apple Silicon fixture only")
def test_ffi_rejects_wrong_types_before_pointer_access():
    fixture = probe.accelerate_module()
    with pytest.raises(ValueError):
        fixture.product(np.zeros((1, 2), np.float32), np.zeros((1, 2), np.float64))
