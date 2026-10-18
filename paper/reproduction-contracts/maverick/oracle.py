#!/usr/bin/env python3
"""Independent, bounded scalar oracle for Maverick v1 Protocols 2/3.

Standard library only. No PLLM kernels, author code, model, native build, entropy,
or security admission. All explicit challenges are TEST INPUTS. See README.md.
"""

from collections import Counter
from dataclasses import dataclass
from decimal import Decimal, localcontext, ROUND_CEILING
from fractions import Fraction
from math import comb, isqrt


BABYBEAR = 2_013_265_921
PROPOSED_Q31_PLUS = 2_147_483_659  # Arithmetic candidate, not an approved profile.


def is_prime32(q):
    return (type(q) is int and 2 <= q < 2**32
            and all(q % d for d in range(2, isqrt(q) + 1)))


def canonical(values, q, length=None):
    values = tuple(values)
    if length is not None and len(values) != length:
        raise ValueError("length mismatch")
    if any(type(x) is not int or not 0 <= x < q for x in values):
        raise ValueError("noncanonical field element")
    return values


def centered(x, q):
    canonical([x], q)
    return x if x <= q // 2 else x - q


def transpose(a):
    if not a or not a[0] or any(len(r) != len(a[0]) for r in a):
        raise ValueError("empty or ragged matrix")
    return [list(c) for c in zip(*a)]


def matmul(a, b, q):
    """Unbounded integer dot products followed by modular reduction."""
    bt = transpose(b)
    if not a or any(len(r) != len(b) for r in a):
        raise ValueError("matrix shape mismatch")
    return [[sum(x * y for x, y in zip(row, col)) % q for col in bt]
            for row in a]


def matvec(a, x, q):
    return [r[0] for r in matmul(a, [[v] for v in x], q)]


def dot(a, b, q):
    if len(a) != len(b):
        raise ValueError("dot shape mismatch")
    return sum(x * y for x, y in zip(a, b)) % q


def sparse_vector(length, entries, q, weight=None):
    entries = tuple(entries)
    if not entries or (weight is not None and len(entries) != weight):
        raise ValueError("wrong sparse weight")
    out = [0] * length
    for position, value in entries:
        if (type(position) is not int or not 0 <= position < length
                or type(value) is not int or not 0 < value < q):
            raise ValueError("invalid sparse entry")
        if out[position]:
            raise ValueError("duplicate sparse position")
        out[position] = value
    return out


def draw_below(upper, words, bits=32):
    """Rejection rule using caller-supplied uniform words; no RNG fallback."""
    if not 0 < upper <= 2**bits:
        raise ValueError("invalid sampling domain")
    limit = 2**bits - 2**bits % upper
    for word in words:
        if type(word) is not int or not 0 <= word < 2**bits:
            raise ValueError("invalid random word")
        if word < limit:
            return word % upper
    raise ValueError("randomness exhausted")


def partial_shuffle_support(length, weight, offsets):
    """Offsets are independent uniform draws in [0, length-i)."""
    if not 0 < weight <= length or len(offsets) != weight:
        raise ValueError("invalid support parameters")
    pool = list(range(length))
    for i, offset in enumerate(offsets):
        if not 0 <= offset < length - i:
            raise ValueError("support offset out of range")
        j = i + offset
        pool[i], pool[j] = pool[j], pool[i]
    return tuple(pool[:weight])


@dataclass(frozen=True)
class Raa:
    """M Appendix B: Pi[i,j]=1 iff pi[j]=i (zero-based here)."""

    q: int
    n: int
    r: int
    pi1: tuple
    pi2: tuple
    v1: tuple
    v2: tuple

    def __post_init__(self):
        if (not is_prime32(self.q) or type(self.n) is not int or type(self.r) is not int
                or not 1 <= self.n <= 128 or not 1 <= self.r <= 8):
            raise ValueError("oracle field/dimension bound")
        for name in ("pi1", "pi2", "v1", "v2"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        for pi in (self.pi1, self.pi2):
            if any(type(i) is not int for i in pi) or sorted(pi) != list(range(self.N)):
                raise ValueError("not a permutation")
        for v in (self.v1, self.v2):
            canonical(v, self.q, self.N)
            if 0 in v:
                raise ValueError("zero code scale")

    @property
    def N(self):
        return self.n * self.r

    def encode(self, w):
        """G^T w: gather, scale, prefix, twice. Never dense at large N."""
        w = canonical(w, self.q, self.n)
        a = [x for x in w for _ in range(self.r)]
        for pi, v in ((self.pi1, self.v1), (self.pi2, self.v2)):
            out, total = [], 0
            for j in range(self.N):
                total = (total + a[pi[j]] * v[j]) % self.q
                out.append(total)
            a = out
        return a

    def syndrome(self, e):
        """G e: suffix, scale, scatter, twice; sum repeat blocks."""
        a = list(canonical(e, self.q, self.N))
        for pi, v in ((self.pi2, self.v2), (self.pi1, self.v1)):
            out, total = [0] * self.N, 0
            for j in range(self.N - 1, -1, -1):
                total = (total + a[j]) % self.q
                out[pi[j]] = total * v[j] % self.q
            a = out
        return [sum(a[i:i + self.r]) % self.q for i in range(0, self.N, self.r)]

    def dense(self):
        """Independent literal product R Pi1 V1 A Pi2 V2 A, tiny only."""
        if self.N > 64:
            raise ValueError("dense oracle is restricted to N<=64")
        a = [[int(i <= j) for j in range(self.N)] for i in range(self.N)]
        g = [[int(j // self.r == i) for j in range(self.N)] for i in range(self.n)]
        for pi, v in ((self.pi1, self.v1), (self.pi2, self.v2)):
            p = [[int(pi[j] == i) for j in range(self.N)] for i in range(self.N)]
            diagonal = [[v[i] if i == j else 0 for j in range(self.N)]
                        for i in range(self.N)]
            for factor in (p, diagonal, a):
                g = matmul(g, factor, self.q)
        return g

    def record(self):
        return {k: getattr(self, k) for k in ("q", "n", "r", "pi1", "pi2", "v1", "v2")}


def prepare(matrix, gx, gy):
    if gx.q != gy.q or len(matrix) != gy.n:
        raise ValueError("publication binding mismatch")
    m = [list(canonical(row, gx.q, gx.n)) for row in matrix]
    return matmul(m, gx.dense(), gx.q), matmul(transpose(gy.dense()), m, gx.q)


def validate_publication(matrix, gx, gy, p, projection):
    """Full transparent-preprocessing equality, not a sampled spot check."""
    expected_p, expected_q = prepare(matrix, gx, gy)
    if len(p) != gy.n or len(projection) != gy.N:
        return False
    try:
        p = [list(canonical(row, gx.q, gx.N)) for row in p]
        projection = [list(canonical(row, gx.q, gx.n)) for row in projection]
    except ValueError:
        return False
    return p == expected_p and projection == expected_q


def protocol3(matrix, x, gx, gy, ex_entries, ey_entries):
    q = gx.q
    x = canonical(x, q, gx.n)
    p, projection = prepare(matrix, gx, gy)
    ex = sparse_vector(gx.N, ex_entries, q)
    # Dense algebra deliberately independent of fast syndrome and encoder.
    mask = matvec(gx.dense(), ex, q)
    masked_x = [(a + b) % q for a, b in zip(x, mask)]
    correction = matvec(p, ex, q)
    masked_y = matvec(matrix, masked_x, q)
    z = matvec(transpose(gy.dense()), masked_y, q)
    checks = []
    for entries in ey_entries:
        e = sparse_vector(gy.N, entries, q)
        checks.append([dot(matvec(transpose(projection), e, q), masked_x, q),
                       dot(e, z, q)])
    return {"P": p, "Q": projection, "mask": mask, "masked_x": masked_x,
            "correction": correction, "masked_y": masked_y, "z": z,
            "checks": checks, "y": [(a - b) % q for a, b in zip(masked_y, correction)]}


def batch_publication(matrices, primary, secondary):
    if (not matrices or primary.q != secondary.q
            or secondary.n != len(matrices) * len(matrices[0][0])):
        raise ValueError("batch dimensions")
    n = len(matrices[0][0])
    for m in matrices:
        if len(m) != primary.n:
            raise ValueError("batch shape mismatch")
        for row in m:
            canonical(row, primary.q, n)
    qs = [matmul(transpose(primary.dense()), m, primary.q) for m in matrices]
    stack = [row for projection in qs for row in transpose(projection)]
    aux = matmul(transpose(stack), secondary.dense(), primary.q)
    return stack, aux


def batch_checks(inputs, outputs, primary, secondary, aux, es, eps, responses):
    """Protocol 2 equations on already frozen vectors; checks both layers."""
    q = primary.q
    z = [matvec(transpose(primary.dense()), y, q) for y in outputs]
    secondary_checks, original_checks = [], []
    for e, response in zip(es, responses):
        for ep in eps:
            secondary_checks.append([dot(e, matvec(aux, ep, q), q),
                                     dot(matvec(secondary.dense(), ep, q), response, q)])
        at = 0
        for x, encoded in zip(inputs, z):
            original_checks.append([dot(response[at:at + len(x)], x, q), dot(e, encoded, q)])
            at += len(x)
    return {"secondary": secondary_checks, "original": original_checks}


class FrozenBatch:
    """Test-only state oracle: two fixation barriers, terminal accept/retire.

    No secure RNG or erasure claims. Explicit challenges must never be a public
    production API. Request state is represented only by an acceptance marker.
    """

    def __init__(self, matrices, primary, secondary, t=31, ell=2, tp=31, ellp=2):
        if not 0 < t <= primary.N or not 0 < tp <= secondary.N or min(ell, ellp) < 1:
            raise ValueError("batch security parameters")
        self.primary, self.secondary = primary, secondary
        self.t, self.ell, self.tp, self.ellp = t, ell, tp, ellp
        self.count, self.n = len(matrices), len(matrices[0][0])
        self.stack, self.aux = batch_publication(matrices, primary, secondary)
        self.phase, self.accepted_state = "collecting", None
        self.inputs = self.outputs = self.es = self.responses = ()

    def retire(self):
        self.phase, self.accepted_state = "retired", None
        self.inputs = self.outputs = self.es = self.responses = ()

    def require(self, phase):
        if self.phase != phase:
            self.retire()
            raise ValueError("invalid batch phase")

    def fix_originals(self, inputs, outputs):
        self.require("collecting")
        try:
            if len(inputs) != self.count or len(outputs) != self.count:
                raise ValueError("omitted claim")
            self.inputs = tuple(canonical(x, self.primary.q, self.n) for x in inputs)
            self.outputs = tuple(canonical(y, self.primary.q, self.primary.n) for y in outputs)
        except ValueError:
            self.retire()
            raise
        self.phase = "originals_fixed"

    def reveal_primary(self, entries):
        self.require("originals_fixed")
        try:
            if len(entries) != self.ell:
                raise ValueError("primary count")
            self.es = tuple(tuple(sparse_vector(self.primary.N, e, self.primary.q, self.t))
                            for e in entries)
        except ValueError:
            self.retire()
            raise
        self.phase = "primary_revealed"
        return self.es

    def fix_auxiliary(self, responses):
        self.require("primary_revealed")
        try:
            if len(responses) != self.ell:
                raise ValueError("auxiliary count")
            self.responses = tuple(canonical(u, self.primary.q, self.secondary.n) for u in responses)
        except ValueError:
            self.retire()
            raise
        self.phase = "auxiliary_fixed"

    def finish(self, entries):
        self.require("auxiliary_fixed")
        try:
            if len(entries) != self.ellp:
                raise ValueError("secondary count")
            eps = [sparse_vector(self.secondary.N, e, self.primary.q, self.tp) for e in entries]
            checks = batch_checks(self.inputs, self.outputs, self.primary, self.secondary,
                                  self.aux, self.es, eps, self.responses)
            accepted = all(a == b for pairs in checks.values() for a, b in pairs)
        except ValueError:
            self.retire()
            raise
        self.retire()
        if accepted:
            self.phase, self.accepted_state = "accepted", "verified-logits-window"
        return accepted, checks


def lemma_bound(q, delta=Fraction(1, 2), t=31, ell=2):
    return ((1 - delta)**t + Fraction(1, q - 1))**ell


def exact_sparse_pass(q, N, weight, t):
    """Exact single-check probability for any fixed encoded error of this weight.

    Condition on K overlapping support positions. For independent nonzero field
    terms, a_0=1 and a_(k+1)=(1-a_k)/(q-1). Mix by the hypergeometric law.
    """
    if not 0 <= weight <= N or not 0 <= t <= N:
        raise ValueError("invalid probability parameters")
    a, result = Fraction(1), Fraction(0)
    for k in range(t + 1):
        if k <= weight and 0 <= t - k <= N - weight:
            result += Fraction(comb(weight, k) * comb(N - weight, t - k), comb(N, t)) * a
        a = (1 - a) / (q - 1)
    return result


def privacy_weight(n, r=8, security=128):
    """M Eq.(1), delta=1/2. High precision ceiling; not a security estimator."""
    with localcontext() as ctx:
        ctx.prec = 80
        value = 2 * (Decimal(security) * Decimal(2).ln() - Decimal(r * n).ln())
        return int(value.to_integral_value(rounding=ROUND_CEILING))


def distance_dimension_two(g, q):
    """Exact instance certificate, ALL messages in F_q^2, without q^2 work.

    Every nonzero column vanishes for exactly one projective message direction.
    Normalize that column; its multiplicity is the count of such zeros. Zero
    columns vanish for every message. Thus d=N-zero_columns-max_multiplicity.
    """
    if len(g) != 2:
        raise ValueError("certificate only supports dimension two")
    columns = transpose(g)
    for row in g:
        canonical(row, q)
    zeros, counts = 0, Counter()
    for a, b in columns:
        if a == b == 0:
            zeros += 1
        else:
            direction = (1, b * pow(a, -1, q) % q) if a else (0, 1)
            counts[direction] += 1
    distance = len(columns) - zeros - max(counts.values(), default=0)
    return {"N": len(columns), "zero_columns": zeros,
            "largest_direction_multiplicity": max(counts.values(), default=0),
            "minimum_distance": distance}
