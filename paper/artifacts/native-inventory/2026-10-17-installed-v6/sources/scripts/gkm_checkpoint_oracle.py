"""Independent scalar IEEE-bit oracle for the GKM source-weight admission probe.

No NumPy, native kernels, floating arithmetic, or candidate decoder is used here.
The rational source value and nearest-even integer are computed with Python ints.
"""

from __future__ import annotations


def scalar_bits(raw: bytes, dtype: str) -> dict:
    formats = {"BF16": (2, 8, 7, 127), "F16": (2, 5, 10, 15), "F32": (4, 8, 23, 127)}
    size, exponent_bits, fraction_bits, bias = formats[dtype]
    if len(raw) != size:
        raise ValueError("oracle requires exactly one encoded value")
    bits = int.from_bytes(raw, "little")
    sign = bits >> (8 * size - 1)
    exponent = (bits >> fraction_bits) & ((1 << exponent_bits) - 1)
    fraction = bits & ((1 << fraction_bits) - 1)
    if exponent == (1 << exponent_bits) - 1:
        return {"finite": False, "kind": "nan" if fraction else "infinity", "bits": bits}
    significand = fraction if exponent == 0 else (1 << fraction_bits) + fraction
    power = (1 if exponent == 0 else exponent) - bias - fraction_bits
    numerator = significand << max(power, 0)
    denominator = 1 << max(-power, 0)
    quotient, remainder = divmod(numerator << 11, denominator)
    if 2 * remainder > denominator or (2 * remainder == denominator and quotient & 1):
        quotient += 1
    if dtype == "BF16":
        f32_bits = bits << 16
    elif dtype == "F32":
        f32_bits = bits
    elif not significand:
        f32_bits = sign << 31
    else:
        top = significand.bit_length() - 1
        f32_bits = ((sign << 31) | ((top + power + 127) << 23)
                    | ((significand << (23 - top)) & ((1 << 23) - 1)))
    return {"finite": True, "bits": bits, "numerator": -numerator if sign else numerator,
            "denominator": denominator, "rounded": -quotient if sign else quotient,
            "float32_bits": f32_bits, "negative_zero": bool(sign and not significand)}
