"""Bounded exact-real reciprocal contract; no CKKS/backend or model semantics."""
from dataclasses import dataclass
from fractions import Fraction

from benchmarks.research.thor_literal_reconciliation import (
    Budget, MAX_MEMORY_BYTES, MAX_OUTPUT_BYTES, StudyLimit, peak_rss_bytes,
)

MAX_INPUT_BITS = 16
MAX_STEPS = 8
MAX_INTEGER_BITS = 65536
MAX_WORK = 1 << 62


class InvalidReciprocal(ValueError):
    """Invalid public domain/count, rejected before division/exponentiation."""


@dataclass(frozen=True)
class Quote:
    integer_bits: int
    work: int
    storage: int
    output: int


class RationalBudget(Budget):
    """Reuse the ordinary single-process research guard; exhaustion is sticky."""

    def __init__(self):
        super().__init__(evaluations=66)
        self.work = self.output = 0
        self.poisoned = False

    def fail(self, reason):
        self.poisoned = True
        raise StudyLimit(reason)

    def check(self):
        if self.poisoned:
            raise StudyLimit("poisoned rational campaign")
        try:
            super().check()
        except StudyLimit:
            self.poisoned = True
            raise

    def reserve(self, quote):
        self.check()
        try:
            observed = peak_rss_bytes()
        except StudyLimit:
            self.fail("RSS telemetry unavailable")
        if (quote.integer_bits > MAX_INTEGER_BITS or self.work + quote.work > MAX_WORK
                or self.output + quote.output > MAX_OUTPUT_BYTES - 8192
                # Retained row objects, JSON fragments, final str/UTF-8 copies.
                or observed + quote.storage + 8 * MAX_OUTPUT_BYTES > MAX_MEMORY_BYTES):
            self.fail("rational work/storage/output preflight exhausted")
        self.work += quote.work
        self.output += quote.output

    def count(self):
        try:
            super().count()
        except StudyLimit:
            self.poisoned = True
            raise


def preflight(a, b, x, m):
    """No division/pow before bounded types, input bits and public domain checks.

    u numerator/denominator bit lengths are bounded by cross-product sums.
    Squaring doubles these each step; the product's exponents sum to 2**m-1.
    Three times that bound covers unreduced add/multiply temporaries, reciprocal
    and residual/error oracle cross-products. School arithmetic plus Euclidean
    GCD is conservatively charged cubically, with 64 bit-work units per call.
    128 simultaneous integer/object slots bound this straight-line scalar code;
    output quote covers four numerator/denominator pairs plus record metadata.
    """
    if type(m) is not int or any(type(v) is not Fraction for v in (a, b, x)):
        raise InvalidReciprocal("exact Fraction inputs and integer count required")
    components = tuple((v.numerator, v.denominator) for v in (a, b, x))
    if any(type(n) is not int for pair in components for n in pair):
        raise InvalidReciprocal("exact built-in integer Fraction components required")
    if any(denominator <= 0 for _, denominator in components):
        raise InvalidReciprocal("strictly positive Fraction denominators required")
    bits = [int.bit_length(n) for pair in components for n in pair]
    if max(bits) > MAX_INPUT_BITS or m > MAX_STEPS:
        raise StudyLimit("input/count preflight exhausted")
    if a <= 0 or b <= 0 or a > b or not a <= x <= b or m < 0:
        raise InvalidReciprocal("require 0<a<=x<=b and m>=0")
    n = bits[4] + bits[3]
    d = bits[5] + bits[2]
    fraction_bits = (n + d + 2) * (1 << m) + 4 * max(bits) + 16
    integer_bits = 3 * fraction_bits
    calls = 32 * m + 128
    return Quote(integer_bits, 64 * calls * integer_bits**3,
                 128 * (integer_bits // 8 + 256), 4 * 2 * (fraction_bits + 3) // 3 + 2048)


def reciprocal(a, b, x, m, budget):
    try:
        quote = preflight(a, b, x, m)
    except StudyLimit:
        budget.fail("input/count preflight exhausted")
    budget.reserve(quote)
    u = x / b
    r, e = Fraction(1), 1 - u
    for _ in range(m):
        budget.check()
        r, e = r * (1 + e), e * e  # RHS uses BOTH old values.
    budget.check()
    return r / b
