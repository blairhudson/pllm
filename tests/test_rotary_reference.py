"""Independent finite ideal-math checks. Authored tests; no native/libm binding.

Literal anchors are mathematical expectations, never observed-bit exceptions.
Full-grid mutation tests check comparator sensitivity, not native agreement.
"""

from fractions import Fraction

import pytest

from rotary_reference import (
    coefficient_q18, f32_dyadic, positive_power_bits, reference_pairs,
    round_f32, trig_bits,
)


def _assert_rows(expected, actual):
    # Materialization is test-only. Check length and every coordinate/stage;
    # never let zip silently forgive a missing final coordinate.
    expected, actual = tuple(expected), tuple(actual)
    assert len(actual) == len(expected), "coordinate count"
    fields = ("position", "pair", "exponent", "positive power", "inverse",
              "angle", "cosine", "sine", "cos_q18", "sin_q18")
    for want, got in zip(expected, actual):
        assert type(got) is tuple and len(got) == 10
        for field, a, b in zip(fields, want, got):
            assert a == b, (field, want[:2], a, b)


@pytest.fixture(scope="module")
def grid():
    # Independent declared geometry arithmetic, not config identity evidence.
    hidden, heads, kv_heads = 896, 14, 2
    assert hidden % heads == 0 and heads % kv_heads == 0
    width = hidden // heads
    assert width == 64 and width % 2 == 0
    return tuple(reference_pairs(1000000, width, 4))


def test_complete_geometry_and_literal_anchors(grid):
    assert len(grid) == 160
    assert tuple(row[:2] for row in grid) == tuple(
        (p, j) for p in range(5) for j in range(32))
    partners = tuple((j, j + 32) for j in range(32))
    assert sorted(lane for pair in partners for lane in pair) == list(range(64))
    assert sum(len(row[6:8]) for row in grid) == 320
    assert all(type(row) is tuple for row in grid)
    for row in grid[:32]:
        assert row[5:] == (0x00000000, 0x3F800000, 0x00000000, 262144, 0)
    expected_q18 = ((262144, 0), (141637, 220586), (-109090, 238366),
                    (-259520, 36993), (-171348, -198391))
    for p, expected in enumerate(expected_q18):
        row = grid[p * 32]
        assert row[2:5] == (0x00000000, 0x3F800000, 0x3F800000)
        assert row[8:] == expected
    assert grid[48] == (1, 16, 0x3F000000, 0x447A0000, 0x3A83126F,
                        0x3A83126F, 0x3F7FFFF8, 0x3A83126E, 262143, 262)


def test_exact_tail_and_nondefault_theta(grid):
    tail = tuple(reference_pairs(16, 8, 4))[-1]
    assert tail == (4, 3, 0x3F400000, 0x41000000, 0x3E000000,
                    0x3F000000, 0x3F60A940, 0x3EF57744, 230053, 125678)
    other = tuple(reference_pairs(16, 64, 4))
    assert other[48][3:8] != grid[48][3:8]
    assert other[-1][3:8] != grid[-1][3:8]
    # Position zero cannot identify theta from coefficient bits.
    assert tuple(row[5:] for row in other[:32]) == tuple(row[5:] for row in grid[:32])


def test_32_positive_power_endpoint_goldens():
    # All 32 independently known powers for theta=1, not a copied table.
    rows = tuple(reference_pairs(1, 64, 0))
    assert tuple(row[3] for row in rows) == (0x3F800000,) * 32
    assert tuple(row[4] for row in rows) == (0x3F800000,) * 32
    assert positive_power_bits(16, 0, 1) == 0x3F800000
    assert positive_power_bits(16, 1, 4) == 0x40000000
    assert positive_power_bits(16, 1, 2) == 0x40800000
    assert positive_power_bits(16, 3, 4) == 0x41000000
    assert positive_power_bits(16, 1, 1) == 0x41800000


def test_separate_reciprocal_then_product_not_fused():
    row = tuple(reference_pairs(49, 4, 3))[-1]
    assert row[:6] == (3, 1, 0x3F000000, 0x40E00000, 0x3E124925, 0x3EDB6DB8)
    # Independently specified fused rational rounds one ULP lower. This case
    # really distinguishes product staging, rather than claiming coincidence.
    assert round_f32(Fraction(3, 7)) == 0x3EDB6DB7
    assert row[5] != 0x3EDB6DB7


@pytest.mark.parametrize("value,bits", [
    (Fraction(1) + Fraction(1, 1 << 24), 0x3F800000),
    (Fraction(1) + Fraction(3, 1 << 24), 0x3F800002),
    (-Fraction(1) - Fraction(1, 1 << 24), 0xBF800000),
    (Fraction(1, 1 << 150), 0x00000000),
    (-Fraction(1, 1 << 150), 0x80000000),
    (Fraction(3, 1 << 150), 0x00000002),
    (Fraction((1 << 24) - 1, 1 << 150), 0x00800000),
    (Fraction(1, 1 << 149), 0x00000001),
    (Fraction(1, 1 << 126), 0x00800000),
    (Fraction(1), 0x3F800000),
    (Fraction(-4), 0xC0800000),
])
def test_rounding_literal_midpoints_and_boundaries(value, bits):
    assert round_f32(value) == bits


@pytest.mark.parametrize("bits,value", [
    (0x00000000, Fraction(0)), (0x80000000, Fraction(0)),
    (0x00000001, Fraction(1, 1 << 149)),
    (0x807FFFFF, -Fraction((1 << 23) - 1, 1 << 149)),
    (0x00800000, Fraction(1, 1 << 126)),
    (0x3F000000, Fraction(1, 2)), (0xC0800000, Fraction(-4)),
])
def test_decode_literal_dyadics(bits, value):
    assert f32_dyadic(bits) == value


def test_signed_zero_and_explicit_refusal():
    assert round_f32(0, negative_zero=True) == 0x80000000
    assert trig_bits(0x80000000) == (0x3F800000, 0x80000000)
    assert trig_bits(0) == (0x3F800000, 0)
    with pytest.raises(ArithmeticError, match="unresolved"):
        trig_bits(0x40800000, term_limit=1)
    for bits in (0xBF800000, 0x40800001, 0x7F800000, 0x7FC00000):
        with pytest.raises(ValueError):
            trig_bits(bits)
    with pytest.raises(ValueError, match="bound"):
        round_f32(1 << 16384)
    with pytest.raises(ValueError, match="overflow"):
        round_f32(1 << 128)
    with pytest.raises(ValueError):
        round_f32(True)


@pytest.mark.parametrize("theta,width,last", [
    (True, 64, 4), (1000000, True, 4), (1000000, 64, False),
    (0, 64, 4), ((1 << 24) + 1, 64, 4), (Fraction(16), 64, 4),
    (16, 1, 4), (16, 6, 4), (16, 256, 4), (16, 64, -1), (16, 64, 5),
])
def test_public_validation_is_eager(theta, width, last):
    with pytest.raises(ValueError):
        reference_pairs(theta, width, last)  # No next()/iteration needed.


def test_upper_domain_is_not_narrowed():
    rows = tuple(reference_pairs(1 << 24, 128, 4))
    assert len(rows) == 320 and rows[-1][:2] == (4, 63)


def test_all_source_conversions_and_negative_truncation(grid):
    # Sole use of existing source oracle: converter under test, not arithmetic
    # authority. No NumPy operations or expected-value helpers are used here.
    from replicated_decoder_oracle import source_f32

    raw = b"".join(bits.to_bytes(4, "little") for row in grid for bits in row[6:8])
    converted = tuple(int(value) for value in source_f32(raw))
    # Independent integer decoding, not coefficient_q18/f32_dyadic under test.
    expected_values = []
    for row in grid:
        for bits in row[6:8]:
            exponent = (bits >> 23) & 255
            mantissa = bits & 0x7FFFFF
            if exponent:
                mantissa += 1 << 23
            shift = exponent - 132 if exponent else -131
            magnitude = mantissa << shift if shift >= 0 else mantissa >> -shift
            expected_values.append(-magnitude if bits & 0x80000000 else magnitude)
    expected = tuple(expected_values)
    assert len(converted) == 320 and converted == expected
    assert tuple(value for row in grid for value in row[8:10]) == expected
    assert coefficient_q18(0x3A83126E) == 262
    assert coefficient_q18(0x3A83126D) == 262
    assert coefficient_q18(0xBF800000) == -262144
    # cos(2) is negative nonintegral after scaling: floor differs by one.
    assert grid[64][8] == -109090
    mutated = list(converted)
    mutated[128] = -109091
    assert tuple(mutated) != expected


@pytest.mark.parametrize("mutation", [
    "missing_tail", "swap_heads", "swap_frequency", "swap_cos_sin",
    "reuse_position", "width_two", "half_exponent", "negative_power",
    "same_q18_ulp", "negative_floor",
])
def test_entire_coordinate_comparator_rejects_mutations(grid, mutation):
    rows = list(grid)
    index = 48
    row = list(rows[index])
    stage = None
    if mutation == "missing_tail":
        rows.pop()
    elif mutation == "swap_heads":
        # Equal-width head payload swap, retain destination coordinates.
        rows = [row[:2] + grid[(row[0] * 32 + (31 - row[1]))][2:] for row in grid]
    elif mutation == "swap_frequency":
        rows[32], rows[63] = rows[63], rows[32]
    elif mutation == "swap_cos_sin":
        row[6], row[7] = row[7], row[6]
        row[8], row[9] = row[9], row[8]
        rows[index] = tuple(row)
    elif mutation == "reuse_position":
        rows[128:] = [(4, j) + grid[96 + j][2:] for j in range(32)]
    elif mutation == "width_two":
        rows = [entry[:2] + grid[entry[0] * 32][2:] for entry in grid]
    elif mutation == "half_exponent":
        # First differing stage is exponent, before downstream coincidences.
        assert row[2] == 0x3F000000
        row[2] = 0x3E800000
        stage = "exponent"
        rows[index] = tuple(row)
    elif mutation == "negative_power":
        # theta**(-1/2) rounds to .001, not positive-power-stage 1000.
        # Do not claim that reciprocal and fused-negative outputs differ here.
        assert row[3] == 0x447A0000
        row[3] = 0x3A83126F
        stage = "positive power"
        rows[index] = tuple(row)
    elif mutation == "same_q18_ulp":
        assert row[7] == 0x3A83126E and row[9] == 262
        row[7] = 0x3A83126D
        assert coefficient_q18(row[7]) == 262
        stage = "sine"
        rows[index] = tuple(row)
    elif mutation == "negative_floor":
        row = list(rows[64])
        row[8] = -109091
        rows[64] = tuple(row)
        stage = "cos_q18"
    with pytest.raises(AssertionError) as failure:
        _assert_rows(grid, rows)
    if stage:
        assert stage in str(failure.value)


def test_comparator_accepts_complete_immutable_grid(grid):
    _assert_rows(grid, grid)
