"""D36 integer mass bounds, conditional on genuine E/max provenance.

No nExp evaluation/table generation, masked execution, source authentication or
ideal-softmax error criterion. This checks all integer partitions allowed by
0<=E_i<=4096 and some E_i=4096; sharp witnesses need not be attainable nExp rows.

Upper proof: D=64Z+t, 0<=t<=63, Z>=64. Therefore
  S=sum floor(E_i*R/4096) <= D*R/4096 <= 64D/Z
    =4096+64t/Z <=4159.
Also R<=4096, so each product<=2^24 and each P_i<=4096.

Lower proof: write 262144=Z*R+u, 0<=u<Z. Since D>=64Z,
floor(D*R/4096)>=floor(4096-u/64). For k=2, Z<=128 gives
at least4094; for k=3, Z<=192 gives at least4093. Choose one
E=4096 existentially (no protocol winner opening). Its product floor is the
integer R. The remaining two floors lose at most one, hence S>=4092 for k=3.
For k=2 only one term remains and there is no additional floor loss.

All partitions: fix E0=4096, T=D-4096, x+y=T, 0<=x,y<=4096.
Their mass is floor(D*R/4096)-[(xR mod4096)>(TR mod4096)].
Two Euclidean floor sums count that indicator on the entire feasible x interval
in logarithmic work. Symmetry/inclusion-exclusion restores every ordered triple,
including multiple maxima, without enumerating millions of partitions.
"""

import json

SCALE = 4096


def _floor_sum(n: int, modulus: int, a: int, b: int) -> int:
    """sum((a*i+b)//modulus for i in range(n)), Euclidean reduction."""
    if min(n, a, b) < 0 or modulus <= 0:
        raise ValueError("nonnegative floor-sum arguments and positive modulus required")
    total = 0
    while True:
        quotient, a = divmod(a, modulus)
        total += n * (n - 1) // 2 * quotient
        quotient, b = divmod(b, modulus)
        total += n * quotient
        top = a * n + b
        if top < modulus:
            return total
        n, b = divmod(top, modulus)
        modulus, a = a, modulus


def _loss_count(t: int, reciprocal: int) -> tuple[int, int]:
    lower, upper = max(0, t - SCALE), min(SCALE, t)
    count = upper - lower + 1
    residue = t * reciprocal % SCALE
    start = lower * reciprocal
    loss = _floor_sum(count, SCALE, reciprocal, start + SCALE - 1 - residue)
    loss -= _floor_sum(count, SCALE, reciprocal, start)
    return count, loss


def intrinsic_bounds(k: int) -> tuple[int, int]:
    """Plain arithmetic metadata, not a private/source/range certificate."""
    if type(k) is not int or k not in (1, 2, 3):
        raise ValueError("active count must be 1, 2 or 3")
    return {1: (4096, 4096), 2: (4094, 4159), 3: (4092, 4159)}[k]


def mass_classes(k: int, denominator: int) -> dict[int, int]:
    """Exact mass->multiplicity for ALL ordered k-tuples with this D/max4096."""
    intrinsic_bounds(k)
    if type(denominator) is not int or not SCALE <= denominator <= k * SCALE:
        raise ValueError("denominator outside genuine max-relation envelope")
    z = denominator // 64
    reciprocal = 262144 // z
    base = denominator * reciprocal // SCALE
    if k == 1:
        return {SCALE: 1}
    if k == 2:
        return {base: 1 if denominator == 2 * SCALE else 2}
    count, loss = _loss_count(denominator - SCALE, reciprocal)
    # Three choices of fixed maximum. Pair intersections have no floor loss;
    # each has exactly one remaining term if 8192<=D<=12288.
    pairs = int(denominator >= 2 * SCALE)
    triple = int(denominator == 3 * SCALE)
    classes = {base: 3 * (count - loss) - 3 * pairs + triple, base - 1: 3 * loss}
    assert 0 <= loss <= count and all(n >= 0 for n in classes.values())
    return {mass: n for mass, n in classes.items() if n}


def _witness(k: int, denominator: int, wanted: int) -> list[int]:
    """Search only one extremal D (at most4097 pairs), never the whole grid."""
    reciprocal = 262144 // (denominator // 64)
    t = denominator - SCALE
    for x in range(max(0, t - SCALE), min(SCALE, t) + 1):
        entries = [SCALE, x, t - x] if k == 3 else [SCALE, t]
        if sum(e * reciprocal // SCALE for e in entries) == wanted:
            return entries
        if k == 2:
            break
    raise AssertionError("positive mass multiplicity had no witness")


def verify_contract() -> dict:
    counts = {1: 1, 2: 0, 3: 0}
    extrema = {2: [10**9, -1], 3: [10**9, -1]}
    extremal_d = {2: [None, None], 3: [None, None]}
    base_lower, base_upper = 10**9, -1
    for d in range(SCALE, 3 * SCALE + 1):
        z, reciprocal = d // 64, 262144 // (d // 64)
        assert 64 <= z <= 192 and z != 0
        assert 1365 <= reciprocal <= SCALE
        assert SCALE * reciprocal <= 1 << 24 < 1 << 48
        base = d * reciprocal // SCALE
        base_lower, base_upper = min(base_lower, base), max(base_upper, base)
        for k in (2, 3):
            if d > k * SCALE:
                continue
            classes = mass_classes(k, d)
            counts[k] += sum(classes.values())
            lo, hi = min(classes), max(classes)
            lower, upper = intrinsic_bounds(k)
            assert lower <= lo <= hi <= upper
            if lo < extrema[k][0]:
                extrema[k][0], extremal_d[k][0] = lo, d
            if hi > extrema[k][1]:
                extrema[k][1], extremal_d[k][1] = hi, d
    assert (base_lower, base_upper) == (4093, 4159)
    assert counts == {k: (SCALE + 1) ** k - SCALE**k for k in (1, 2, 3)}
    assert all(tuple(extrema[k]) == intrinsic_bounds(k) for k in (2, 3))
    witnesses = {
        str(k): [_witness(k, extremal_d[k][i], extrema[k][i]) for i in (0, 1)]
        for k in (2, 3)
    }
    return {
        "denominators_checked": 8193,
        "ordered_partitions_covered": counts,
        "mass_bounds": {k: intrinsic_bounds(k) for k in (1, 2, 3)},
        "witnesses_under_numeric_premises_only": witnesses,
        "max_entry": 4096,
        "max_final_product": 1 << 24,
        "scope": "conditional integer arithmetic; no nExp, masked/source authority or ideal-softmax criterion",
    }


if __name__ == "__main__":
    print(json.dumps(verify_contract(), indent=2, sort_keys=True))
