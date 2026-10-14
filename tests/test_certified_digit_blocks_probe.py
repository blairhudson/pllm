"""Adversarial certificate and independent modular-oracle tests."""
import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

PATH = Path(__file__).resolve().parents[1] / "scripts/probe_certified_digit_blocks.py"
SPEC = importlib.util.spec_from_file_location("certified_digit_blocks_probe", PATH)
p = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(p)


def test_minus128_strict_boundary_and_not_127_bound():
    w = np.full((2, 2048), -128, dtype=np.int8)
    c = p.plan(w, 1, 32)
    assert [r[3] - r[2] for r in c["records"]] == [1023, 1023, 2]
    assert c["records"][0][4] * 128 == 16760832
    assert 1024 * 128 * 128 == p.LIMIT  # Exact endpoint, but native rejects it.
    assert 1024 * 127 * 128 < p.LIMIT  # Wrong magnitude would admit unsafe native block.


def test_l1_last_admissible_integer_and_next_column():
    w = np.array([[-128] * 1023 + [-127, -1]], dtype=np.int8)
    c = p.plan(w, 1, 16)
    assert c["records"][0][2:] == [0, 1024, 131071]
    x = np.full((1, 1025), 128, dtype=np.uint32)
    np.testing.assert_array_equal(p.execute(w, x, 16, c), p.integer_oracle(w, x, 16))


def test_cancellation_cannot_replace_absolute_certificate():
    w = np.tile(np.array([-128, 127], dtype=np.int8), (1, 2048))
    assert abs(int(w.astype(np.int64).sum())) * 128 < p.LIMIT
    c = p.plan(w, 1, 32)
    assert max(r[3] - r[2] for r in c["records"]) == 1028
    assert len(c["records"]) == 4


def test_all_u16_balanced_digits_and_higher_ring_carries():
    # Exhaust low-two-digit carry transitions; independently reconstruct modulo.
    x = np.arange(65536, dtype=np.uint64)
    for bits in (16, 24, 32):
        values = x if bits == 16 else x + np.uint64((1 << bits) - 65536)
        carry = 0
        reconstructed = np.zeros_like(values, dtype=np.int64)
        for shift in range(0, bits, 8):
            digit = (((values + np.uint64(carry)) >> np.uint64(shift)) + 128) & 255
            digit = digit.astype(np.int64) - 128
            assert np.all((digit >= -128) & (digit <= 127))
            if shift < 16:
                assert digit.min() == -128 and digit.max() == 127
            reconstructed += digit << shift
            carry += 128 << shift
        np.testing.assert_array_equal(reconstructed & ((1 << bits) - 1), values)


def test_every_subset_partial_sum_for_extreme_digits():
    # Covers subset/reordering partials; final cancellation alone is irrelevant.
    weights = [-128, 127, 0, -1, 1, -128, 127]
    bound = 128 * sum(abs(w) for w in weights)
    for digits in ([ -128] * 7, [127] * 7, [-128, 127, -128, 127, -128, 127, -128]):
        products = [w * d for w, d in zip(weights, digits)]
        for subset in range(1 << len(products)):
            value = sum(v for i, v in enumerate(products) if subset & (1 << i))
            assert abs(value) <= bound < p.LIMIT
            assert int(np.float32(value)) == value


def test_per_output_not_average_and_output_tiles():
    w = np.ones((5, 2050), dtype=np.int8)
    w[2] = -128
    c = p.plan(w, 1, 32, tile_rows=2)
    widths = [[e - s for f, _, s, e, _ in c["records"] if f == first] for first in (0, 2, 4)]
    assert widths == [[2050], [1023, 1023, 4], [2050]]
    assert c["cost"]["weight_conversion_elements"] == w.size
    assert c["cost"]["dense_products"] == 4 * w.size


@pytest.mark.parametrize("bits", [16, 24, 32])
@pytest.mark.parametrize("tile_rows,max_block", [(1, 19), (3, 8192)])
def test_extreme_carries_wrap_and_centered_subtraction(bits, tile_rows, max_block):
    rng = np.random.default_rng(8675309)
    w = rng.integers(-128, 128, (4, 2053), dtype=np.int16).astype(np.int8)
    w[0] = -128
    w[1] = 127
    w[2] = 0
    mask = (1 << bits) - 1
    values = np.array([0, 127, 128, 255, 256, 32768, 0xffffffff, 0x7fffffff, 0x80808080], dtype=np.uint32) & mask
    x = np.tile(np.resize(values, w.shape[1]), (3, 1))
    x[1] = 128
    x[2] = mask
    r = np.array([[0, mask, 1 << (bits - 1), 1],
                  [mask, 0, 1, 1 << (bits - 1)], [1, 2, 3, mask]], dtype=np.uint32)
    c = p.plan(w, 3, bits, tile_rows=tile_rows, max_block=max_block)
    actual = p.execute(w, x, bits, c, r)
    expected = p.integer_oracle(w, x, bits, r)
    np.testing.assert_array_equal(actual, expected)
    centered_expected = np.array([[int(v) if int(v) < 1 << (bits - 1) else int(v) - (1 << bits)
                                   for v in row] for row in expected], dtype=np.int64)
    np.testing.assert_array_equal(p.centered(actual, bits), centered_expected)


def test_zero_weights_larger_block_still_workspace_bounded():
    w = np.zeros((1024, 16384), dtype=np.int8)
    c = p.plan(w, 256, 32)
    assert c["width_cap"] < w.shape[1]
    assert c["cost"]["workspace_bytes"] <= p.WORKSPACE
    assert c["cost"]["sgemm_calls"] == 8
    assert c["cost"]["baseline_sgemm_calls"] == 68
    assert c["cost"]["mask_digit_conversion_elements"] == c["cost"]["baseline_mask_digit_conversion_elements"]


@pytest.mark.parametrize("mutation", ["source", "bound", "gap", "cost", "domain"])
def test_modified_certificate_rejected(mutation):
    w = np.ones((2, 30), dtype=np.int8)
    c = copy.deepcopy(p.plan(w, 1, 16))
    if mutation == "source":
        w[0, 0] = 2
    elif mutation == "bound":
        c["records"][0][4] = 0
    elif mutation == "gap":
        c["records"][0][2] = 1
    elif mutation == "cost":
        c["cost"]["workspace_bytes"] = 0
    else:
        c["bits"] = 32
    with pytest.raises(ValueError):
        p.execute(w, np.zeros((1, 30), dtype=np.uint32), 16, c)


def test_invalid_geometry_and_noncanonical_inputs():
    w = np.ones((1, 2), dtype=np.int8)
    for batch, bits in [(0, 16), (4097, 16), (True, 16), (1, 17)]:
        with pytest.raises(ValueError):
            p.plan(w, batch, bits)
    with pytest.raises(ValueError):
        p.plan(w.astype(np.uint8), 1, 16)
    with pytest.raises(ValueError):
        p.plan(w, 1, 16, max_block=0)
    c = p.plan(w, 1, 16)
    with pytest.raises(ValueError):
        p.execute(w, np.array([[65536, 0]], dtype=np.uint32), 16, c)
    with pytest.raises(ValueError):
        p.execute(w, np.zeros((1, 2), dtype=np.uint32), 16, c,
                  np.array([[65536]], dtype=np.uint32))


def test_scan_admission_denied_before_import_or_value_reads():
    with pytest.raises(ValueError, match="admission"):
        p.scan({})


def test_native_admission_denied_without_loading_or_compilation():
    for function in (p.native_identity, p.compile_worker, p.native_check_worker):
        with pytest.raises(ValueError, match="admission"):
            function({})


def test_ctypes_transfer_wrapper_with_mock_abi_only(monkeypatch):
    """Exercises Python ownership/transfers, NOT compiled C or Accelerate."""
    import ctypes as ct

    class Function:
        def __init__(self, call):
            self.call = call

        def __call__(self, *args):
            return self.call(*args)

    class FakeLibrary:
        def __init__(self):
            self.destroyed = 0
            for name in ("create", "stats", "records", "execute", "destroy"):
                setattr(self, "cdb_" + name, Function(getattr(self, name)))

        def create(self, source, length, rows, cols, batch, bits, certified, result):
            self.weights = np.ctypeslib.as_array(source).copy().reshape(rows, cols)
            self.plan = p.plan(self.weights, batch, bits)
            self.batch, self.bits = batch, bits
            ct.cast(result, ct.POINTER(ct.c_void_p)).contents.value = 123
            return 0

        def stats(self, handle, values, count):
            cost = self.plan["cost"]
            maximum = max(e - s for _, _, s, e, _ in self.plan["records"])
            for i, v in enumerate([cost["sgemm_calls"], maximum, cost["workspace_bytes"],
                                   len(self.plan["records"]), cost["certificate_reserved_bytes"],
                                   cost["dense_products"], self.weights.size, self.weights.size]):
                values[i] = v
            return 0

        def records(self, handle, values, count):
            for i, v in enumerate([v for record in self.plan["records"] for v in record + [0]]):
                values[i] = v
            return 0

        def execute(self, handle, inputs, input_len, mask, mask_len, output, output_len):
            x = np.ctypeslib.as_array(inputs).reshape(self.batch, self.weights.shape[1])
            r = np.ctypeslib.as_array(mask).reshape(self.batch, self.weights.shape[0])
            result = p.integer_oracle(self.weights, x, self.bits, r)
            np.ctypeslib.as_array(output)[:] = result.ravel()
            return 0

        def destroy(self, handle):
            self.destroyed += 1

    library = FakeLibrary()
    monkeypatch.setattr(ct, "CDLL", lambda path: library)
    w = np.array([[-128, 127, 1], [1, -1, 0]], dtype=np.int8)
    x = np.array([[128, 255, 65535]], dtype=np.uint32)
    r = np.array([[65535, 32768]], dtype=np.uint32)
    expected = p.integer_oracle(w, x, 16, r)
    fixture = p.NativeFixture("unused-mock-library", w, 1, 16, True)
    w[:] = 0  # Source snapshot must not follow caller mutation.
    actual = fixture(x, r)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(x, [[128, 255, 65535]])
    fixture.close()
    fixture.close()
    assert library.destroyed == 1
    np.testing.assert_array_equal(actual, expected)  # Returned copy survives close.
    with pytest.raises(ValueError, match="closed"):
        fixture(x, r)


def matched_mock_runs():
    sample = {"iteration": 0, "warmup": True, "input_sha256": "x", "output_mask_sha256": "r",
              "output_sha256": "y", "all_outputs_checked": 896}
    result = []
    for arm in ("fixed", "certified", "installed_integer", "installed_digit8"):
        result.append({"replicate": 0, "revision": "rev", "source_lock_sha256": "lock",
                       "stages": [{"key": "down", "certificate": {"weight_sha256": "w"},
                                   "scales_f32_sha256": "s", "native_measurement": {
                                       "arm": arm, "batch": 1, "native_identity": {"library_sha256": "lib"},
                                       "installed_extension_sha256": "sdk", "samples": [copy.deepcopy(sample)]}}]})
    return result


@pytest.mark.parametrize("mutation", [None, "output", "source", "library", "missing", "duplicate"])
def test_matched_native_cohort_identity_gate(mutation):
    runs = matched_mock_runs()
    if mutation == "output":
        runs[1]["stages"][0]["native_measurement"]["samples"][0]["output_sha256"] = "different"
    elif mutation == "source":
        runs[1]["stages"][0]["certificate"]["weight_sha256"] = "different"
    elif mutation == "library":
        runs[1]["stages"][0]["native_measurement"]["native_identity"]["library_sha256"] = "different"
    elif mutation == "missing":
        runs.pop()
    elif mutation == "duplicate":
        runs.append(copy.deepcopy(runs[0]))
    if mutation is None:
        p.validate_matched_native_runs(runs)
    else:
        with pytest.raises(ValueError):
            p.validate_matched_native_runs(runs)


def test_synthetic_replay_retains_exact_outputs_native_unexecuted():
    import json

    previous = json.loads((PATH.parents[1] / "docs/evidence/certified-digit-blocks-synthetic.json").read_text())
    report = p.synthetic()
    assert report["fixtures"] == previous["fixtures"]
    assert report["count_only_shapes"] == previous["count_only_shapes"]
    assert report["oracle_output_values"] == 90
    assert report["native_fixture"]["compiled_or_executed_by_worker"] is False
    assert report["native_timing"] is None and report["no_checkpoint_scan"] is True
    assert report["source_sha256"][str(p.FIXTURE_SOURCE.relative_to(p.ROOT))] == p.file_hash(p.FIXTURE_SOURCE)
    assert report == previous  # Exact reproducible artifact, including source identities.


@pytest.mark.parametrize("arguments", [
    ["--compile-native", "--native-library", "unused.dylib"],
    ["--native-check", "--native-library", "unused.dylib", "--admit-native-run"],
    ["--native-timing", "--native-library", "unused.dylib", "--admit-native-run",
     "--native-build-record", "unused.json"],
])
def test_native_cli_fails_closed_before_external_work(arguments, tmp_path):
    import subprocess
    import sys

    result = subprocess.run([sys.executable, str(PATH), "--output", str(tmp_path / "report.json"), *arguments],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 2
    assert "error:" in result.stderr
    assert not (tmp_path / "report.json").exists()
