"""Exact conditional ideal-RAA ensemble distance union bounds.

Integer q supports algebraic evaluation; field interpretation requires a prime
power. The current seeded generator's ensemble provenance is unproved. Results
are union bounds, not actual probabilities, security claims, or native funding
evidence. The candidate budget bounds neither CPU time nor RSS: arbitrary-size
integer arithmetic and library scratch are not accounted for by this schedule.

Only O(D) rational entries are retained, with no global count/binomial cache.
Importing this module performs no evaluations, argument parsing, or output.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from math import comb
import json


MAX_TERMS = 1_000_000


def _integer(name: str, value: int, minimum: int, maximum: int | None = None) -> None:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer (not bool)")
    if value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"{name} is outside its allowed range")


@dataclass(frozen=True)
class Profile:
    k: int
    r: int
    q: int

    def __post_init__(self) -> None:
        _integer("k", self.k, 1)
        _integer("r", self.r, 1)
        _integer("q", self.q, 2)

    @property
    def n(self) -> int:
        return self.k * self.r


OFFICIAL_PROFILES = (Profile(64, 8, 2013265921), Profile(128, 8, 2013265921))


@dataclass(frozen=True)
class Prefix:
    d: int
    coefficient: Fraction
    cumulative: Fraction
    bound: Fraction


@dataclass(frozen=True)
class Evaluation:
    profile: Profile
    prefixes: tuple[Prefix, ...]
    charged_terms: int
    stop_reason: str


class BudgetExceeded(Exception):
    """The next candidate or assembly would exceed the permitted schedule."""


def _dimensions(n: int, q: int, **weights: int) -> None:
    _integer("n", n, 0)
    _integer("q", q, 2)
    for name, value in weights.items():
        _integer(name, value, 0, n)


def _choose(n: int, k: int) -> int:
    return comb(n, k) if 0 <= k <= n else 0


def _count(
    n: int, h: int, l: int, q: int, charge: Callable[[], None] | None = None
) -> int:
    """Shared kernel; charge every endpoint candidate, even when invalid."""
    if l == 0:
        return int(h == 0)
    a, b = q - 1, q - 2
    total = 0
    for m in range(1, l + 1):
        for endpoint in (0, 1):
            if charge is not None:
                charge()
            j = h - 2 * m + endpoint
            # Guard before powers, including q=2: 0**0=1, 0**j=0 for j>0.
            if not 0 <= j <= l - m:
                continue
            gaps = _choose(n - l, m - endpoint)
            if gaps == 0:
                continue
            total += (
                _choose(l - 1, m - 1)
                * gaps
                * a**m
                * _choose(l - m, j)
                * b**j
            )
    return total


def accumulator_count(n: int, h: int, l: int, q: int) -> int:
    """Count accumulator outputs of weight l for input weight h."""
    _dimensions(n, q, h=h, l=l)
    return _count(n, h, l, q)


def _transition(
    n: int, h: int, l: int, q: int, charge: Callable[[], None] | None = None
) -> Fraction:
    numerator = _count(n, h, l, q, charge)
    if charge is not None:
        charge()
    return Fraction(numerator, comb(n, h) * (q - 1) ** h)


def transition(n: int, h: int, l: int, q: int) -> Fraction:
    """Exact conditional transition to weight l."""
    _dimensions(n, q, h=h, l=l)
    return _transition(n, h, l, q)


def transition_tail(n: int, h: int, D: int, q: int) -> Fraction:
    """Exact probability of output weight at most D, including zero."""
    _dimensions(n, q, h=h, D=D)
    return sum((_transition(n, h, l, q) for l in range(D + 1)), Fraction(0))


def _profile(profile: Profile) -> None:
    if not isinstance(profile, Profile):
        raise TypeError("profile must be a Profile")


def schedule_terms(profile: Profile, D: int) -> int:
    """Full unsaturated schedule through D; excludes no zero-valued work."""
    _profile(profile)
    _integer("D", D, 0, profile.n)
    n, k, r = profile.n, profile.k, profile.r
    mixture = sum(
        min(k, 2 * l // r) * (2 * l + 1)
        for l in range(1, min(n, 2 * D) + 1)
    )
    outer = sum(min(n, 2 * d) * (2 * d + 1) for d in range(1, D + 1))
    return mixture + outer


def profile_evaluate(
    profile: Profile, *, term_cap: int = MAX_TERMS, max_d: int | None = None
) -> Evaluation:
    """Stream complete coefficients, retaining partial effort on budget stop.

    Each transition costs two endpoint candidates per output run plus one
    assembly. Mixtures are computed once per new l; outer transitions run even
    when their mixture is zero. Saturation takes priority over the d limit.
    """
    _profile(profile)
    _integer("term_cap", term_cap, 0, MAX_TERMS)
    if max_d is None:
        max_d = profile.n
    _integer("max_d", max_d, 0, profile.n)
    charged = 0

    def charge() -> None:
        nonlocal charged
        if charged >= term_cap:
            raise BudgetExceeded
        charged += 1

    n, k, r, q = profile.n, profile.k, profile.r, profile.q
    mixtures = [Fraction(0)]
    prefixes: list[Prefix] = []
    cumulative = Fraction(0)
    for d in range(1, max_d + 1):
        try:
            for l in range(len(mixtures), min(n, 2 * d) + 1):
                mixture = Fraction(0)
                for w in range(1, min(k, 2 * l // r) + 1):
                    value = _transition(n, r * w, l, q, charge)
                    mixture += comb(k, w) * (q - 1) ** (w - 1) * value
                mixtures.append(mixture)
            coefficient = Fraction(0)
            for l in range(1, min(n, 2 * d) + 1):
                value = _transition(n, l, d, q, charge)
                coefficient += mixtures[l] * value
        except BudgetExceeded:
            return Evaluation(profile, tuple(prefixes), charged, "budget")
        cumulative += coefficient
        bound = min(Fraction(1), cumulative)
        prefixes.append(Prefix(d, coefficient, cumulative, bound))
        if bound == 1:
            return Evaluation(profile, tuple(prefixes), charged, "saturation")
    return Evaluation(profile, tuple(prefixes), charged, "limit")


def _fraction_to_dict(value: Fraction) -> dict:
    return {
        "numerator_hex": hex(value.numerator),
        "denominator_hex": hex(value.denominator),
    }


def evaluation_to_dict(result: Evaluation) -> dict:
    """Encode exact rationals without decimal integer conversion limits."""
    if not isinstance(result, Evaluation):
        raise TypeError("result must be an Evaluation")
    profile = result.profile
    return {
        "profile": {"k": profile.k, "r": profile.r, "q": profile.q, "n": profile.n},
        "prefixes": [
            {
                "d": prefix.d,
                "coefficient": _fraction_to_dict(prefix.coefficient),
                "cumulative": _fraction_to_dict(prefix.cumulative),
                "bound": _fraction_to_dict(prefix.bound),
            }
            for prefix in result.prefixes
        ],
        "charged_terms": result.charged_terms,
        "stop_reason": result.stop_reason,
        "last_complete_d": result.prefixes[-1].d if result.prefixes else 0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("512", "1024", "all"), default="all")
    args = parser.parse_args(argv)
    profiles = (
        OFFICIAL_PROFILES
        if args.profile == "all"
        else tuple(p for p in OFFICIAL_PROFILES if p.n == int(args.profile))
    )
    output = {
        "scope": "conditional_ideal_raa_ensemble",
        "evaluations": [evaluation_to_dict(profile_evaluate(p)) for p in profiles],
    }
    print(json.dumps(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
