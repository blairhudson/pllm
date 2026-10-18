"""Independent clear mathematics, not MPC, an RNG, or a production fallback.

No imports from the compiler or from author implementations. Table floors are
certified with rational enclosures, rather than inferred from host libm output.
"""

from fractions import Fraction
from math import isqrt


def _atan_bounds(inverse, precision):
    x = Fraction(1, inverse)
    term = total = x
    k = 0
    tolerance = Fraction(1, 1 << precision)
    while True:
        following = -term * x * x * Fraction(2 * k + 1, 2 * k + 3)
        if abs(following) < tolerance:
            return min(total, total + following), max(total, total + following)
        total += following
        term = following
        k += 1


def _sqrt_lower(value, precision):
    scale = 1 << precision
    return Fraction(isqrt((value.numerator * scale * scale) // value.denominator), scale)


def gelu_table(precision=128):
    """256 exact floor entries; raise rather than guess any unresolved floor.

    Machin's identity encloses pi using alternating rational arctangent series.
    Integral_0^x exp(-t*t/2) dt uses its separate alternating rational series.
    Once its terms decrease, the next omitted term bounds the remainder.
    Integer square roots enclose sqrt(2/pi). Both endpoints must have one floor.
    """
    if type(precision) is not int or not 80 <= precision <= 256:
        raise ValueError("bounded rational precision required")
    a_lo, a_hi = _atan_bounds(5, precision + 16)
    b_lo, b_hi = _atan_bounds(239, precision + 16)
    pi_lo, pi_hi = 16 * a_lo - 4 * b_hi, 16 * a_hi - 4 * b_lo
    root_lo = _sqrt_lower(2 / pi_hi, precision)
    root_hi = _sqrt_lower(2 / pi_lo, precision) + Fraction(1, 1 << precision)
    tolerance = Fraction(1, 1 << precision)
    result = [0]
    for i in range(1, 256):
        x = Fraction(i, 64)
        term = total = x
        k = 0
        while True:
            following = -term * x * x * Fraction(2 * k + 1, 2 * (k + 1) * (2 * k + 3))
            # x < 4; k >= 16 makes all remaining magnitudes decreasing.
            if k >= 16 and abs(following) < tolerance:
                lo, hi = min(total, total + following), max(total, total + following)
                break
            total += following
            term = following
            k += 1
        lower = 2048 * x * (1 - root_hi * hi)
        upper = 2048 * x * (1 - root_lo * lo)
        floor_lo = lower.numerator // lower.denominator
        floor_hi = upper.numerator // upper.denominator
        if floor_lo != floor_hi or floor_lo < 0:
            raise ArithmeticError("unresolved GELU table floor")
        result.append(floor_lo)
    return tuple(result)


def gelu_signed(x, table):
    reduced = x // 64
    return (x if reduced >= 0 else 0) - table[min(abs(reduced), 255)]


def _bit(expression, x, bits):
    tag, *args = expression
    modulus = 1 << bits
    if tag == "const":
        return args[0]
    if tag == "lt":
        return int(x < args[0])
    if tag == "low_lt":
        return int(x % (1 << args[0]) < args[1])
    if tag == "slt":
        signed = x if x < modulus // 2 else x - modulus
        return int(signed < args[0])
    if tag == "msb":
        return ((x + args[0]) % modulus) // (modulus // 2)
    if tag == "not":
        return 1 - _bit(args[0], x, bits)
    a, b = (_bit(arg, x, bits) for arg in args)
    if tag == "xor":
        return int(a != b)
    if tag == "and":
        return int(a == 1 and b == 1)
    if tag == "or":
        return int(a == 1 or b == 1)
    raise ValueError("unknown reference expression")


def interpret_unfused(descriptor, word):
    """Direct interval scan and expanded affine formula; no masked rewrite.

    Caller supplies a validated descriptor. This oracle does not share lookup,
    Horner, normalization, query evaluation, or Boolean VM code with the compiler.
    """
    modulus = 1 << descriptor.word.bits
    if type(word) is not int or not 0 <= word < modulus:
        raise ValueError("canonical input required")
    for i, (lower, upper) in enumerate(zip(descriptor.boundaries, descriptor.boundaries[1:])):
        if lower <= word < upper:
            piece = descriptor.pieces[i]
            result = (piece.constant + piece.linear * word) % modulus
            return result, tuple(_bit(expr, word, descriptor.word.bits) for expr in piece.booleans)
    raise ValueError("incomplete descriptor")
