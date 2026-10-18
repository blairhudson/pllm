"""Bounded independent arithmetic specifications; not protected implementations.

See shared-profile-resolution.md. Standard library only. No upstream imports,
model execution, protocol randomness, or production defaults. Candidate SPU
semantics remain distinct from paper-derived SIGMA/FuseFSS semantics.
"""

from collections import Counter
from decimal import Decimal, localcontext
from fractions import Fraction
from hashlib import sha256
from math import isqrt
import json
import struct


def signed(x, bits):
    x %= 1 << bits
    return x - (1 << bits) if x >= 1 << (bits - 1) else x


def encode(value, fractional, policy):
    """Exact rational input; callers must name rounding, including tie policy."""
    x = Fraction(value) * (1 << fractional)
    lo = x.numerator // x.denominator
    if policy == "floor":
        return lo
    if policy == "toward_zero":
        return int(x)
    if policy == "nearest_even":
        rem = x - lo
        return lo + int(rem > Fraction(1, 2) or (rem == Fraction(1, 2) and lo % 2))
    if policy == "nearest_up":
        return (2 * x.numerator + x.denominator) // (2 * x.denominator)
    raise ValueError("explicit supported rounding policy required")


def encode_f32(value, fractional):
    """SPU candidate: C++ float literal, then scaled signed integer cast."""
    f32 = struct.unpack("<f", struct.pack("<f", float(value)))[0]
    return encode(Fraction(f32), fractional, "toward_zero")


def puma_trunc(x, bits, drop, mask):
    """Reconstructed TruncAPr formula, not an implementation of its protocol."""
    if not 1 <= drop <= bits - 2 or not -(1 << (bits - 2)) <= x < 1 << (bits - 2):
        raise ValueError("TruncAPr requires public shift and quarter-ring gap")
    n = 1 << bits
    half = n // 2
    mask %= n
    c = (x + n // 4 + mask) % n
    high = (c // half) ^ (mask // half)
    return signed((c % half >> drop) - (mask % half >> drop)
                  + (high << (bits - 1 - drop)) - (1 << (bits - 2 - drop)), bits)


def _puma_mul(x, y, masks):
    # Admission, not wrap-and-hope: this candidate subprofile proves a gap.
    return puma_trunc(x * y, 64, 18, next(masks))


def puma_reciprocal(x, masks):
    """Positive secret reciprocal, pinned HAL schedule, two iterations, Q18."""
    if not 0 < x < 1 << 36:
        raise ValueError("positive reciprocal bit-reversal domain required")
    f = 18
    one = 1 << f
    factor = 1 << (2 * f - 1 - (x.bit_length() - 1))
    c = _puma_mul(x, factor, masks)
    w = _puma_mul(encode_f32("2.9142", f) - 2 * c, factor, masks)
    e = one - _puma_mul(x, w, masks)
    r = _puma_mul(w, one + e, masks)
    e = _puma_mul(e, e, masks)
    return _puma_mul(r, one + e, masks)


def puma_rsqrt(x, masks):
    """Pinned full-accuracy HAL schedule. No invented Newton refinement."""
    if not 0 < x < 1 << 35:
        raise ValueError("positive rsqrt bit-reversal domain required")
    f = 18
    j = x.bit_length()  # z = 2**j, one above highest set bit
    u = _puma_mul(x, 1 << (2 * f - 1 - j), masks)
    coeffs = [encode_f32(c, f) for c in
              ("-15.47994394", "38.4714796", "-49.86605845", "26.02942339")]
    power = u
    acc = power * coeffs[0]
    for coeff in coeffs[1:]:
        power = _puma_mul(power, u, masks)
        acc += power * coeff
    r = puma_trunc(acc, 64, f, next(masks)) + encode_f32("4.14285016", f)
    comp = (724 if j % 2 == 0 else 512) * (1 << (f - 1 - j // 2))
    return _puma_mul(r, comp, masks)


def puma_rmsnorm(x, gamma, masks):
    """D10 epsilon-one, uncentered Q18 row; clear independent test oracle only."""
    if not 1 <= len(x) <= 128 or len(gamma) != len(x):
        raise ValueError("bounded RMSNorm row and matching gamma required")
    tape = iter(masks)
    squares = [_puma_mul(v, v, tape) for v in x]
    variance = _puma_mul(sum(squares), encode_f32(Fraction(1, len(x)), 18), tape) + 1
    inverse = puma_rsqrt(variance, tape)
    normalized = [_puma_mul(v, inverse, tape) for v in x]
    return [_puma_mul(v, g, tape) for v, g in zip(normalized, gamma)]


def bumblebee_trunc(x, share0, wrap_share0, bits, drop):
    """Pinned general signed path including party-0 parity correction."""
    if not 1 <= drop <= bits - 2:
        raise ValueError("bounded oracle keeps at least two bits")
    n = 1 << bits
    a = (share0 + n // 2) % n
    b = (x - share0) % n
    wrap = (a + b) // n
    z0 = ((a >> drop) - (wrap_share0 << (bits - drop))
          - (1 << (bits - drop - 1))) % n
    z1 = ((b >> drop) - ((wrap - wrap_share0) << (bits - drop))) % n
    return signed(z0 + z1 + (z0 & 1), bits)


def bumblebee_less(x, threshold, share0, bits, drop):
    """MSB after *local* low-bit dropping of the two difference shares."""
    n = 1 << bits
    a = (share0 - threshold) % n
    b = (x - share0) % n
    return int(signed((a >> drop) + (b >> drop), bits - drop) < 0)


def positive_extension(x, share0, bits, target):
    """Reconstruction of pinned M=quarter-ring known-sign extension."""
    if target <= bits or not -(1 << (bits - 2)) <= x < 1 << (bits - 2):
        raise ValueError("positive-offset extension range is not established")
    n = 1 << bits
    # Rank 0 adds M, then signed extension adds half a ring.
    a = (share0 + n // 4 + n // 2) % n
    b = (x - share0) % n
    # x + M has MSB 0; after sign flip it has MSB 1, hence AND wrap.
    wrap = (a >> (bits - 1)) & (b >> (bits - 1))
    return signed(a + b - wrap * n - n // 2 - n // 4, target)


def lift(x, bits, odd_modulus):
    """Pinned modulus-up on canonical representatives: nearest/ties-up."""
    n = 1 << bits
    return ((odd_modulus * (x % n) + n // 2) // n) % odd_modulus


def down(x, bits, odd_modulus):
    """Nearest division by odd modulus has no exact half ties."""
    n = 1 << bits
    return ((2 * n * (x % odd_modulus) + odd_modulus) // (2 * odd_modulus)) % n


def prime_u64(n):
    """Deterministic Miller-Rabin for bounded public parameter arithmetic."""
    if n < 2 or n >= 1 << 64:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0:
            return n == p
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in (2, 325, 9375, 28178, 450775, 9780504, 1795265022):
        if a % n == 0:
            continue
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def coefficient_moduli(widths, degree=8192):
    """Independent SEAL-4.1.2 parameter-factory specification, inert numbers."""
    primes = {}
    for width, count in Counter(widths).items():
        value = ((1 << width) - 1) // (2 * degree) * (2 * degree) + 1
        primes[width] = []
        while len(primes[width]) < count:
            if prime_u64(value):
                primes[width].append(value)
            value -= 2 * degree
    return [primes[width].pop() for width in widths]


def fuse_partition(boundaries, mask):
    """Protocol 3 standard intervals and payload ordinals, including padding."""
    n = boundaries[-1]
    if boundaries[0] != 0 or n & (n - 1) or any(a >= b for a, b in zip(boundaries, boundaries[1:])):
        raise ValueError("canonical full-domain power-of-two partition required")
    m = len(boundaries) - 1
    starts = sorted(((a + mask) % n, i) for i, a in enumerate(boundaries[:-1]))
    b = [v for v, _ in starts]
    order = [i for _, i in starts]
    if b[0] == 0:
        b.append(n)
        if m < n:
            j = next(j for j in range(m) if b[j + 1] - b[j] >= 2)
            b.insert(j + 1, b[j] + 1)
            order.insert(j + 1, order[j])
    else:
        b = [0] + b + [n]
        order = [order[-1]] + order
    return b, order


def fuse_predicate(xhat, mask, threshold, bits):
    n = 1 << bits
    if not 0 <= threshold <= n:
        raise ValueError("sentinels are 0 and N")
    r = mask % n
    theta = (r + threshold) % n
    return int(xhat % n < theta) ^ int(xhat % n < r) ^ int(r + threshold >= n)


def sigma_tables(precision=80):
    """Paper LUT equations; Decimal series avoids host-libm table dependence."""
    with localcontext() as ctx:
        ctx.prec = precision
        pi = Decimal("3.141592653589793238462643383279502884197169399375105820974944592307816406286208998628034825342117067982148")
        root = (Decimal(2) / pi).sqrt()
        gelu = []
        for i in range(256):
            x = Decimal(i) / 64
            term, total, k = x, x, 0
            while abs(term) > Decimal(10) ** (10 - precision):
                k += 1
                term *= -x * x / (2 * k)
                total += term / (2 * k + 1)
            gelu.append(int(x * (1 - root * total) * 2048))
        silu = [int(Decimal(i) / 64 / (1 + (Decimal(i) / 64).exp()) * 4096)
                for i in range(1024)]
        exp0 = [int((-Decimal(i) / 4096).exp() * 4096) for i in range(256)]
        exp1 = [int((-Decimal(i) / 16).exp() * 4096) for i in range(256)]
    return {"gelu": gelu, "silu": silu, "exp0": exp0, "exp1": exp1}


def sigma_activation(x, table):
    y = x // 64  # early signed TR, BEFORE absolute value
    index = min(abs(y), len(table) - 1)
    return (x if y >= 0 else 0) - table[index]


def sigma_softmax(row, tables):
    if not row or len(row) > 1024:
        raise ValueError("bounded source workload required")
    maximum = max(row)
    exps = []
    for x in row:
        c = min(maximum - x, 65535)
        exps.append(tables["exp0"][c % 256] * tables["exp1"][c // 256] // 4096)
    denominator_q6 = sum(exps) // 64
    inverse = (1 << 18) // denominator_q6
    return [e * inverse // 4096 for e in exps]


def sigma_rsqrt_table_entry(index, width, fractional=12):
    m, e = divmod(index, 64)
    if not 0 <= index < 8192 or width <= 0:
        raise ValueError("13-bit index and positive width required")
    # floor(sqrt(2**(4*f) * width / ((128+m)*2**e/128)))
    return isqrt(((1 << (4 * fractional + 7)) * width) // ((128 + m) << e))


def sigma_norm(row, gamma=None, beta=None):
    """Figure 12, unwrapped admitted integers, f=12. Includes s+1 guard."""
    k = len(row)
    if not 1 <= k <= 4096:
        raise ValueError("bounded norm width required")
    gamma = [4096] * k if gamma is None else gamma
    beta = [0] * k if beta is None else beta
    if len(gamma) != k or len(beta) != k:
        raise ValueError("affine width mismatch")
    mean = sum(row) * (4096 // k) // 4096
    centered = [x - mean for x in row]
    s = sum(x * x for x in centered) + 1
    e = s.bit_length() - 1
    if e >= 50:
        raise ValueError("GPT2 ring50 norm overflow")
    m = ((s << 7) >> e) - 128
    inverse = sigma_rsqrt_table_entry((m << 6) | e, k)
    return [((x * inverse // 4096) * g // 4096) + b
            for x, g, b in zip(centered, gamma, beta)]


def vectors():
    from itertools import repeat

    tables = sigma_tables()
    samples = [-16385, -16384, -4097, -4096, -65, -64, -63, -1, 0, 1, 63, 64, 65, 4096, 16384]
    return {
        "schema": "pllm.shared_profile_scalar_vectors.v1",
        "scope": "arithmetic only; candidate profiles are not adoption approvals",
        "sigma_table_sha256_u64le": {name: sha256(b"".join(struct.pack("<Q", x) for x in table)).hexdigest()
                                      for name, table in tables.items()},
        "sigma_activation_inputs_q12": samples,
        "sigma_gelu_q12": [sigma_activation(x, tables["gelu"]) for x in samples],
        "sigma_silu_q12": [sigma_activation(x, tables["silu"]) for x in samples],
        "sigma_softmax_q12": [sigma_softmax(x, tables) for x in
                              ([0, 0], [-81920, -81920], [0, -4096], [0, -65536], [0, 0, 0])],
        "sigma_norm_q12": [sigma_norm([-4096, -4096, 4096, 4096]), sigma_norm([4096] * 4),
                           sigma_norm([4096] * 768)[:4]],
        "puma_hal_f32_q18": {x: encode_f32(x, 18) for x in
                            ("2.9142", "-15.47994394", "38.4714796", "-49.86605845", "26.02942339", "4.14285016")},
        "puma_zero_mask_tape_inputs_q18": [65536, 131072, 262144, 524288, 1048576],
        "puma_zero_mask_tape_reciprocal_q18": [puma_reciprocal(x, repeat(0)) for x in (65536, 131072, 262144, 524288, 1048576)],
        "puma_zero_mask_tape_rsqrt_q18": [puma_rsqrt(x, repeat(0)) for x in (65536, 131072, 262144, 524288, 1048576)],
        "fuse_partition_0_3_8_masks_0_2_5": [fuse_partition([0, 3, 8], r) for r in (0, 2, 5)],
        "bumblebee_lift_4bit_t257": [lift(x, 4, 257) for x in (0, 1, 7, 8, 15)],
        "bumblebee_source_olt_key_moduli": coefficient_moduli([59, 55, 49, 55]),
        "bumblebee_source_bole_cipher_moduli": coefficient_moduli([60, 59, 40]),
        "bumblebee_source_bole_plain_moduli_descending": sorted(coefficient_moduli([47, 47, 47]), reverse=True),
    }


if __name__ == "__main__":
    print(json.dumps(vectors(), indent=2, sort_keys=True))
