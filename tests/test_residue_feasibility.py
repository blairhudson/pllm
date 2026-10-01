"""Adversarial offline CRT checks; no claim of private reconstruction."""

from fractions import Fraction
import importlib.util
from itertools import product
from pathlib import Path
import sys

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "residue_probe", Path(__file__).resolve().parents[1] / "scripts/probe_residue_feasibility.py"
)
assert _SPEC is not None and _SPEC.loader is not None
_PROBE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _PROBE
_SPEC.loader.exec_module(_PROBE)
Basis, arithmetic_probe = _PROBE.Basis, _PROBE.arithmetic_probe
linear_residues, round_even = _PROBE.linear_residues, _PROBE.round_even


@pytest.mark.parametrize("moduli", [(), (3, 3), (6, 9), (1,), (True, 5), (65537,), [3, 5]])
def test_invalid_basis_rejected(moduli):
    with pytest.raises(ValueError):
        Basis(moduli)


def test_exact_signed_endpoints_and_ambiguous_interval():
    basis = Basis((16, 17))
    for x in range(-127, 128):
        assert basis.lift(basis.encode(x), -127, 127) == x
    for x in (-136, 136):
        with pytest.raises(ValueError, match="ambiguous"):
            basis.lift(basis.encode(x), -136, 136)
    # One residue outside a valid admitted domain must not silently nearest-lift.
    with pytest.raises(ValueError, match="outside"):
        basis.lift(basis.encode(128), -127, 127)


@pytest.mark.parametrize("residues", [(3, 0), (-1, 0), (0,), (0, 0, 0), (True, 0)])
def test_malformed_residue_rejected(residues):
    with pytest.raises(ValueError):
        Basis((3, 5)).canonical(residues)


def test_rounding_matches_independent_fraction_oracle_with_negative_ties():
    for n, d in product(range(-257, 258), range(1, 18)):
        assert round_even(n, d) == round(Fraction(n, d))
    assert -1 // 2 == -1 and round_even(-1, 2) == 0
    assert round_even(-3, 2) == -2 and round_even(-5, 2) == -2
    with pytest.raises(ValueError):
        round_even(1, 0)


def test_mixed_radix_different_oracle_exhaustive():
    basis = Basis((5, 7, 11))
    for x in range(basis.modulus):
        a, b, c = basis.mixed_radix(basis.encode(x))
        assert a + 5 * b + 35 * c == x


def test_public_linear_shared_parity_all_small_shares():
    basis = Basis((5, 7, 11))
    for x in range(-7, 8):
        for a in range(basis.modulus):
            b = (x - a) % basis.modulus
            ra = linear_residues(basis, (-5,), (a,), bias=3)
            rb = linear_residues(basis, (-5,), (b,))
            out = tuple((u + v) % m for u, v, m in zip(ra, rb, basis.moduli, strict=True))
            assert basis.lift(out, -38, 38) == -5 * x + 3


def test_base_extension_is_not_sharewise_even_with_crt():
    source, target = Basis((3, 5)), Basis((17,))
    for x, a in ((1, 14), (-1, 14)):
        b = (x - a) % source.modulus
        naive = (source.canonical(source.encode(a)) + source.canonical(source.encode(b))) % 17
        assert naive != target.canonical(target.encode(x))


def test_cross_ring_rounding_and_channel_sign_counterexamples():
    assert pow(2, -1, 5) != round_even(1, 2)
    assert sum(round_even(x, 2) for x in (1, 1)) != round_even(2, 2)
    assert 3 * round_even(1, 2) != round_even(3, 2)
    assert -2 % 3 == 1 % 3 and (-2 < 0) != (1 < 0)
    assert sum(x * x for x in (1, 1)) % 15 != (1 + 1) ** 2 % 15
    # Denominator not invertible in channel; rational representation still works
    # with a public numerator/denominator pair, not modular inverse arithmetic.
    with pytest.raises(ValueError):
        pow(2, -1, 16)


def test_rns_truncation_theorem_needs_exact_cross_ring_remainder():
    p, q = 5, 17
    for x in range(p * q):
        assert (pow(p, -1, q) * (x % q - x % p)) % q == x // p
    # Algorithm 3 of Frederiksen et al.: even a noiseless pair rbar=r
    # introduces a secret wrap correction; it is not exact base extension.
    x, r = 4, 3
    opened = (x + r) % p
    assert (opened - r) % q == (x - p) % q != x % q


def test_bounded_probe_carries_quadratic_and_no_free_private_reconstruction():
    report = arithmetic_probe()
    assert report["linear_exact_cases"] == 2025
    assert report["rational_quadratic_exact_cases"] == 225
    assert report["delayed_linear_quadratic_linear_exact_cases"] == 225
    assert report["narrow_share_signed_carry_cases"] == 20480
    assert set(report["secret_carry_histogram_offline_only"]) == {0, 1, 2}
    assert report["private_reconstruction_implemented"] is False
    assert report["private_quadratic_multiplication_implemented"] is False
