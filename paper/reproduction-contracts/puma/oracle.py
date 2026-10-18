"""Offline public-fixture arithmetic/interval specification, NOT an MPC runtime.

Only standard-library integers are used. No native/SPU/Python-package import.
Checking clear fixture values here never authorizes opening private runtime X/Y.
"""
from dataclasses import dataclass
import hashlib
import json

NUMERIC_PROFILE = "puma.source-derived-q18.v1"
PROFILE = "puma.spu-a42c261.mul-aa-truncapr18.ring64.quarter-gap.replicated3.v1"
TRUNC_ONLY_PROFILE = "puma.spu-a42c261.truncapr.ring64.quarter-gap.replicated3.v1"
PRSS_PROFILE = "puma.spu-a42c261.yacl-5418371.prss.le64.aes128-ctr-call-reset.v1"
MAX_LANES = 1024
MAX_OPERATIONS = 4096
MAX_AES_WORK = (3 * MAX_LANES + 3) * MAX_OPERATIONS
U64_MAX = (1 << 64) - 1
Q18 = 1 << 18


class Rejected(ValueError):
    """Public descriptor, capacity, or clear-fixture precondition failed."""


def integer(value, lower, upper):
    if type(value) is not int or not lower <= value <= upper:
        raise Rejected("integer bounds")
    return value


def parameters(width, shift):
    integer(width, 3, 64)
    integer(shift, 1, width - 2)
    return 1 << width, 1 << (width - 2), 1 << shift


@dataclass(frozen=True)
class Interval:
    """Closed integer interval. Wide results fit signed 128 bits."""
    lo: int
    hi: int

    def __post_init__(self):
        integer(self.lo, -(1 << 127), (1 << 127) - 1)
        integer(self.hi, self.lo, (1 << 127) - 1)


def product_hull(x, y, width=64):
    """Exact extrema over the full Cartesian integer box, before any modulus."""
    integer(width, 3, 64)
    for interval in (x, y):
        integer(interval.lo, -(1 << (width - 1)), (1 << (width - 1)) - 1)
        integer(interval.hi, interval.lo, (1 << (width - 1)) - 1)
    corners = (x.lo*y.lo, x.lo*y.hi, x.hi*y.lo, x.hi*y.hi)
    return Interval(min(corners), max(corners))


def admit_product(x, y, width=64):
    """IFF every independent-box product satisfies the source quarter-gap."""
    hull = product_hull(x, y, width)
    gap = 1 << (width - 2)
    if hull.lo < -gap or hull.hi >= gap:
        raise Rejected("unwrapped product outside quarter-gap")
    return hull


def rounded_hull(product, width=64, shift=18):
    _, gap, unit = parameters(width, shift)
    if product.lo < -gap or product.hi >= gap:
        raise Rejected("unwrapped product outside quarter-gap")
    return Interval(product.lo // unit, -((-product.hi) // unit))


def trunc_source(value, mask, width=64, shift=18):
    """Source high-bit formula, separate from the literal signed oracle."""
    modulus, gap, unit = parameters(width, shift)
    integer(value, -gap, gap - 1)  # BEFORE reduction: never admit a wrapped value.
    integer(mask, 0, modulus - 1)
    c = (value + gap + mask) % modulus
    high = modulus // 2
    bit = (c // high) ^ (mask // high)
    return (c % high) // unit - (mask % high) // unit + bit*(high // unit) - gap // unit


def literal_signed(value, mask, width=64, shift=18):
    """Independent integer floor of the signed value plus low-mask residue."""
    modulus, gap, unit = parameters(width, shift)
    integer(value, -gap, gap - 1)
    integer(mask, 0, modulus - 1)
    return (value + mask % unit) // unit


def distribution(value, width=64, shift=18):
    _, gap, unit = parameters(width, shift)
    integer(value, -gap, gap - 1)
    floor, remainder = divmod(value, unit)
    return floor, floor + bool(remainder), remainder, unit


def multiply_batch(xs, ys, x_bounds, y_bounds, masks):
    """Bounded clear fixture batch; admit ALL public boxes before evaluation.

    Membership checks validate test fixtures, not private protocol input values.
    Returned values are Q18 integers, conditional on the supplied public masks.
    """
    n = integer(len(xs), 1, MAX_LANES)
    if any(len(v) != n for v in (ys, x_bounds, y_bounds, masks)):
        raise Rejected("shape")
    for xb, yb in zip(x_bounds, y_bounds):
        admit_product(xb, yb)
    for x, y, xb, yb, mask in zip(xs, ys, x_bounds, y_bounds, masks):
        integer(x, xb.lo, xb.hi)
        integer(y, yb.lo, yb.hi)
        integer(mask, 0, U64_MAX)
    # Only after the whole batch is admitted. No wrapped product is substituted.
    return tuple(trunc_source(x*y, r) for x, y, r in zip(xs, ys, masks))


def local_mul(x_pair, y_pair, own_edge, next_edge, width=64):
    """One role's exact MulAA equation; caller supplies only adjacent views."""
    integer(width, 3, 64)
    modulus = 1 << width
    if len(x_pair) != 2 or len(y_pair) != 2:
        raise Rejected("local replicated shape")
    for word in (*x_pair, *y_pair, own_edge, next_edge):
        integer(word, 0, modulus - 1)
    a, b = x_pair
    c, d = y_pair
    return (a*c + a*d + b*c + own_edge - next_edge) % modulus


def local_trunc_piece(c, rb, rc, output_mask, is_a, width=64, shift=18):
    """Local evaluator equation given its authorized masked opening/correlation."""
    modulus, gap, unit = parameters(width, shift)
    for word in (c, rb, rc, output_mask):
        integer(word, 0, modulus - 1)
    if type(is_a) is not bool:
        raise Rejected("role")
    msb = c // (modulus // 2)
    bit_share = rb + (msb if is_a else 0) - 2*msb*rb
    value = bit_share*(modulus // 2 // unit) - rc - output_mask
    if is_a:
        value += (c % (modulus // 2)) // unit - gap // unit
    return value % modulus


@dataclass(frozen=True)
class Cursor:
    """Public schedule counters only; never seed custody or a restartable runtime."""
    public: int = 0
    pair: int = 0
    private: tuple = (0, 0, 0)
    work: tuple = (0, 0, 0)
    sequence: int = 0

    def __post_init__(self):
        integer(self.public, 0, U64_MAX)
        integer(self.pair, 0, U64_MAX)
        integer(self.sequence, 0, MAX_OPERATIONS)
        if type(self.private) is not tuple or type(self.work) is not tuple or len(self.private) != 3 or len(self.work) != 3:
            raise Rejected("cursor shape")
        for value in self.private:
            integer(value, 0, U64_MAX)
        for value in self.work:
            integer(value, 0, MAX_AES_WORK)


def zero_share_schedule(lanes, pair_start):
    """Source s_i-s_(i+1) zero sharing: ONE Both fill, not a hidden extra fill.

    MulAA uses this zero term directly. A standalone randomized P2A zero share
    additionally rotates it, but is not part of the admitted compound schedule.
    """
    integer(lanes, 1, MAX_LANES)
    integer(pair_start, 0, U64_MAX)
    q = (lanes + 1) // 2
    integer(pair_start + q, 0, U64_MAX)
    return {"start": pair_start, "end": pair_start + q, "control": "Both",
            "words_per_stream": lanes, "aes_per_role": 2*q}


def schedule(cursor, lanes, raw_pivot):
    """Closed sequential MulAA -> TruncAPr(18), from validated shared counters.

    No extra kernel, batching split, fork, P2A debug mask or cursor reset is
    inferred. This pure offline calculation has NO issuance/retirement authority.
    """
    integer(lanes, 1, MAX_LANES)
    integer(cursor.sequence, 0, MAX_OPERATIONS - 1)
    integer(raw_pivot, 0, U64_MAX - 2)  # source's top two words alias roles
    q = (lanes + 1) // 2
    a = raw_pivot % 3
    b, h = (a + 1) % 3, (a + 2) % 3
    public_end = integer(cursor.public + 1, 0, U64_MAX)
    pair_end = integer(cursor.pair + 3*q, 0, U64_MAX)
    private = tuple(integer(v + (lanes if i == h else 0), 0, U64_MAX)
                    for i, v in enumerate(cursor.private))
    work_delta = tuple((6*q + lanes + 1) if i == h else (4*q + 1) for i in range(3))
    work = tuple(integer(v + delta, 0, MAX_AES_WORK) for v, delta in zip(cursor.work, work_delta))
    controls = ["Both" if i == h else "First" if i == a else "Second" for i in range(3)]
    plan = {"profile": PROFILE, "numeric_profile": NUMERIC_PROFILE, "prss_profile": PRSS_PROFILE,
            "sequence": cursor.sequence, "lanes": lanes, "shift": 18,
            "roles": {"a": a, "b": b, "h": h}, "public_pivot_start": cursor.public,
            "raw_pivot_hex": f"{raw_pivot:016x}",
            "mul_zero": zero_share_schedule(lanes, cursor.pair),
            "trunc_mask_start": cursor.pair + q, "trunc_output_start": cursor.pair + 2*q,
            "trunc_controls": controls, "helper_private_start": cursor.private[h],
            "helper_private_words": 2*lanes, "pair_end": pair_end,
            "aes_work_delta": list(work_delta),
            "mul_routes": [[i, (i + 2) % 3] for i in range(3)],
            "directed_frames": 9, "wire_payload_words": 11*lanes}
    return plan, Cursor(public_end, pair_end, private, work, cursor.sequence + 1)


def canonical_bytes(value):
    """Offline vector serialization only, NOT a proposed authenticated wire codec."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def digest(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()
