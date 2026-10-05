"""Independent numerical, finite-domain privacy and malformed-codec witnesses."""
import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from network_algebra_hypotheses import (  # noqa: E402
    annihilator_widths, displacement_rank_bound, gf2_rank, moment_bypass,
    radix_pack, radix_size, radix_unpack, recurrence_leakage_witness, row_widths,
    signed_row_dictionary,
)


def test_annihilator_reduction_preserves_prepared_integer_result():
    from pllm import _native
    rng = np.random.default_rng(1984)
    w = rng.integers(-30, 31, (17, 12), dtype=np.int8)
    w[:, 0], w[:, 1], w[:, 2] = 0, w[:, 1] * 2, w[:, 2] * 4
    output_bits, input_bits = row_widths(w), annihilator_widths(w)
    assert input_bits[0] == 1 and input_bits[2] < output_bits.max()
    x = rng.integers(-127, 128, (5, 12), dtype=np.int64)
    r = rng.integers(0, 2**32, x.shape, dtype=np.int64)
    s = rng.integers(0, 2**32, (5, 17), dtype=np.int64)
    a = ((x - r) % 2**32).astype("<u4")
    encoded = _native.offset_pack_rows(a.tobytes(), input_bits.tobytes(), 5)
    decoded = np.frombuffer(_native.unpack_residue_rows(encoded, input_bits.tobytes(), 5), "<u4").reshape(x.shape)
    q = 1 << output_bits.astype(np.int64)
    result = (decoded.astype(np.int64) @ w.astype(np.int64).T + (r @ w.astype(np.int64).T - s) + s) % q
    centered = np.where(result >= q // 2, result - q, result)
    np.testing.assert_array_equal(centered, x @ w.astype(np.int64).T)
    # A maliciously smaller width is not generally safe.
    bad = a.astype(np.int64) % (1 << np.maximum(1, input_bits.astype(int) - 1))
    assert np.any((bad @ w.astype(np.int64).T + r @ w.astype(np.int64).T) % q != result)


def test_projected_uniform_mask_has_identical_small_domain_transcripts():
    # Input masks remain uniform in each published quotient, for every input.
    histograms = []
    for x in itertools.product(range(4), repeat=2):
        counts = np.zeros((4, 2), int)
        for r in itertools.product(range(8), repeat=2):
            counts[(x[0] - r[0]) % 4, (x[1] - r[1]) % 2] += 1
        histograms.append(counts)
    assert all(np.array_equal(histograms[0], h) for h in histograms)
    assert np.unique(histograms[0]).tolist() == [8]


@pytest.mark.parametrize("modulus", [3, 129, 65521, 8388617, 2**32])
def test_radix_roundtrip_and_canonical_bounds(modulus):
    rng = np.random.default_rng(7)
    for count in (1, 15, 16, 17, 33):
        values = rng.integers(0, modulus, count, dtype=np.uint32)
        encoded = radix_pack(values, modulus)
        assert len(encoded) == radix_size(count, modulus)
        np.testing.assert_array_equal(radix_unpack(encoded, count, modulus), values)
        for bad in (encoded[:-1], encoded + b"\0"):
            with pytest.raises(ValueError, match="length"):
                radix_unpack(bad, count, modulus)
    with pytest.raises(ValueError, match="noncanonical"):
        radix_pack(np.array([modulus], np.uint64), modulus)
    if modulus != 2**32:
        with pytest.raises(ValueError, match="padding"):
            radix_unpack(b"\xff" * radix_size(1, modulus), 1, modulus)


@pytest.mark.parametrize("groups", [1, 4, 16])
def test_moment_bypass_preserves_extremes_and_integer_matrix_outputs(groups):
    rng = np.random.default_rng(218)
    w = rng.integers(-128, 128, (47, 64), dtype=np.int8)
    w[0], w[1], w[2] = -128, 127, 0
    residual, center = moment_bypass(w, groups)
    x = rng.integers(-127, 128, (8, 64), dtype=np.int64)
    sums = x.reshape(len(x), groups, -1).sum(axis=2)
    actual = x @ residual.astype(np.int64).T + sums @ center.astype(np.int64).T
    np.testing.assert_array_equal(actual, x @ w.astype(np.int64).T)
    assert np.all(row_widths(residual) <= row_widths(w))


def test_row_dictionary_handles_signs_zero_and_minus128_without_wrapping():
    w = np.array([[1, 2], [-1, -2], [0, 0], [1, 2], [-128, 1], [127, -1]], np.int8)
    selected, mapping, signs = signed_row_dictionary(w)
    assert len(selected) == 3
    restored = np.zeros(w.shape, np.int16)
    for j in range(len(w)):
        if mapping[j] >= 0:
            restored[j] = w[selected[mapping[j]]].astype(np.int16) * signs[j]
    np.testing.assert_array_equal(restored, w)


def test_displacement_bound_and_mask_ratchet_counterexample():
    assert gf2_rank(np.eye(8, dtype=np.int8)) == 8
    assert gf2_rank(np.ones((8, 8), np.int8)) == 1
    toeplitz = (np.arange(12)[:, None] - np.arange(10)[None, :]).astype(np.int8)
    assert displacement_rank_bound(toeplitz) == (0, 9)
    assert recurrence_leakage_witness()["exposed_private_relations"] == 31
