"""Finite ideal-math comparator, NOT a Rust libm or native parity oracle.

Each individual operation is correctly rounded to binary32. No observed-bit
exceptions, tolerances, production exports, or process-wide frequency cache.
The public bounds are test bounds, not execution admission policy.
"""

from fractions import Fraction


_MAX_BITS = 16384
_MAX_FINITE = 0x7F7FFFFF


def _integer(value, low, high, name):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name}: exact non-bool integer in [{low}, {high}] required")


def _rational(value):
    if type(value) not in (int, Fraction):
        raise ValueError("int or Fraction required")
    # Check existing operands before any multiplication, power, or search.
    if (abs(value.numerator).bit_length() > _MAX_BITS
            or value.denominator.bit_length() > _MAX_BITS):
        raise ValueError("rational operand exceeds finite arithmetic bound")
    return Fraction(value)


def f32_dyadic(bits):
    """Decode finite binary32 exactly; zero sign lives in bits, not Fraction."""
    _integer(bits, 0, 0xFFFFFFFF, "bits")
    exponent = (bits >> 23) & 255
    if exponent == 255:
        raise ValueError("finite binary32 required")
    significand = bits & 0x7FFFFF
    shift = -149
    if exponent:
        significand |= 1 << 23
        shift = exponent - 150
    value = (Fraction(significand << shift) if shift >= 0
             else Fraction(significand, 1 << -shift))
    return -value if bits >> 31 else value


def round_f32(value, *, negative_zero=False):
    """Exact nearest-even finite rounding, including subnormals and signed zero.

    Overflow is deliberately refused, not implemented: callers in this finite
    comparator never need it. Input bit lengths are bounded before arithmetic.
    """
    value = _rational(value)
    if type(negative_zero) is not bool:
        raise ValueError("negative_zero must be bool")
    sign = 0x80000000 if value < 0 or (not value and negative_zero) else 0
    value = abs(value)
    if value > f32_dyadic(_MAX_FINITE):
        raise ValueError("finite rounding overflow")
    if not value:
        return sign
    n, d = value.numerator, value.denominator
    exponent = n.bit_length() - d.bit_length()
    below_power = n < d << exponent if exponent >= 0 else n << -exponent < d
    if below_power:
        exponent -= 1
    shift = max(exponent - 23, -149)
    scaled_n, scaled_d = (n, d << shift) if shift >= 0 else (n << -shift, d)
    q, remainder = divmod(scaled_n, scaled_d)
    if 2 * remainder > scaled_d or (2 * remainder == scaled_d and q & 1):
        q += 1
    if q == 0:
        return sign
    if shift == -149 and q < 1 << 23:
        return sign | q
    if q == 1 << 24:
        q >>= 1
        shift += 1
    return sign | ((shift + 150) << 23) | (q - (1 << 23))


def positive_power_bits(theta, numerator, denominator):
    """Round theta**(numerator/denominator) by exact algebraic comparisons.

    At most 31 adjacent-encoding bisections; reduced degree <=64. Largest
    cross-products have fewer than 10000 bits on this admitted domain.
    """
    _integer(theta, 1, 1 << 24, "theta")
    _integer(denominator, 1, 64, "denominator")
    _integer(numerator, 0, denominator, "numerator")
    exponent = Fraction(numerator, denominator)
    a, b = exponent.numerator, exponent.denominator
    target = theta ** a

    def compare(candidate):
        candidate = f32_dyadic(candidate)
        return candidate.numerator ** b - target * candidate.denominator ** b

    low, high = 0x3F800000, round_f32(theta)
    while low < high:
        middle = (low + high + 1) // 2
        if compare(middle) <= 0:
            low = middle
        else:
            high = middle - 1
    if compare(low) == 0:
        return low
    upper = low + 1
    midpoint = (f32_dyadic(low) + f32_dyadic(upper)) / 2
    decision = midpoint.numerator ** b - target * midpoint.denominator ** b
    if decision > 0:
        return low
    if decision < 0:
        return upper
    return low if not low & 1 else upper


def _trig_round(argument, sine, term_limit):
    """Alternating rational enclosure; refuse unresolved 64-term intervals."""
    term = argument if sine else Fraction(1)
    total = term
    square = argument * argument
    for k in range(1, term_limit):
        degree = 2 * k + (1 if sine else 0)
        following = -term * square / ((degree - 1) * degree)
        next_total = total + following
        # Once decreasing, all later magnitude ratios decrease as well.
        if abs(following) <= abs(term):
            left, right = sorted((total, next_total))
            low_bits, high_bits = round_f32(left), round_f32(right)
            if low_bits == high_bits:
                return low_bits
        term, total = following, next_total
    raise ArithmeticError("Taylor enclosure unresolved within term bound")


def trig_bits(angle_bits, *, term_limit=64):
    """Return (cosine, sine) bits for exact binary32 arguments in [-0, 4]."""
    _integer(term_limit, 1, 64, "term_limit")
    argument = f32_dyadic(angle_bits)
    if not 0 <= argument <= 4:
        raise ValueError("trig argument must be in [0, 4]")
    if not argument:
        return 0x3F800000, angle_bits
    return (_trig_round(argument, False, term_limit),
            _trig_round(argument, True, term_limit))


def coefficient_q18(bits):
    """Truncate the exact dyadic coefficient toward zero, never floor."""
    value = f32_dyadic(bits)
    if not -1 <= value <= 1:
        raise ValueError("coefficient outside [-1, 1]")
    return int(value * (1 << 18))


def reference_pairs(theta_u64, width, last_position):
    """Yield immutable (p,j,e,power,inverse,angle,cos,sin,cos_q18,sin_q18).

    Validation is eager even though rows stream. Frequency work is bounded to
    this invocation (<=64 triples); no registry or global observations.
    """
    _integer(theta_u64, 1, 1 << 24, "theta_u64")
    _integer(width, 2, 128, "width")
    if width & (width - 1):
        raise ValueError("width must be a power of two")
    _integer(last_position, 0, 4, "last_position")

    def rows():
        frequencies = []
        for j in range(width // 2):
            exponent = Fraction(2 * j, width)
            power = positive_power_bits(theta_u64, exponent.numerator, exponent.denominator)
            inverse = round_f32(1 / f32_dyadic(power))
            frequencies.append((round_f32(exponent), power, inverse))
        for p in range(last_position + 1):
            for j, (exponent, power, inverse) in enumerate(frequencies):
                angle = round_f32(p * f32_dyadic(inverse))
                cosine, sine = trig_bits(angle)
                yield (p, j, exponent, power, inverse, angle, cosine, sine,
                       coefficient_q18(cosine), coefficient_q18(sine))

    return rows()
