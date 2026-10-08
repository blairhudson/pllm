"""Fixed public-range encrypted softmax recipes, with no intermediate decryption.

NEXUS Eq. (4) supplies the repeated-squaring exponential. THOR Appendix C supplies
the polynomial and Algorithm 3 supplies square/normalize. Public shift/range,
delta1=1, delta2=2, and unit-factor Goldschmidt iterations are PLLM adaptations.
"""
from __future__ import annotations

BOUND = 0.125
DEPTH = 14
THOR_EXP = (
    9.9999e-1, 9.9999e-1, 5.0002e-1, 1.6667e-1, 4.1653e-2, 8.3316e-3,
    1.3920e-3, 1.9871e-4, 2.4471e-5, 2.7286e-6, 2.9466e-7, 2.6438e-8,
    1.4896e-9, 1.2104e-10, 2.0777e-11, 1.3376e-12,
)


def row_sum(a, x, width):
    # Repeated rows occupy all slots. Sequential one-step rotation needs only
    # one evaluation key and never reads a data-dependent address.
    result, shifted = x, x
    for _ in range(1, width):
        shifted = a.rotate(shifted, 1)
        result = a.add(result, shifted)
    return result


def double_to(a, x, factor):
    for _ in range(factor.bit_length() - 1):
        x = a.add(x, x)
    return x


def reciprocal(a, u, iterations):
    # 1/u ≈ (1+e)(1+e²)..., e=1-u. Residual is e^(2^iterations).
    # Public range analysis places u in (0,2), independent of private values.
    error = a.add_plain(a.negate(u), 1.0)
    result = a.add_plain(error, 1.0)
    for _ in range(1, iterations):
        error = a.multiply(error, error)
        result = a.multiply(result, a.add_plain(error, 1.0))
    return result


def normalize(a, x, width, iterations, *, squared_probabilities=False):
    scaled = double_to(a, x, width) if squared_probabilities else a.multiply_plain(x, 1 / width)
    inverse = reciprocal(a, row_sum(a, scaled, width), iterations)
    return a.multiply(scaled, inverse)


def polynomial(a, x):
    # Balanced powers, not Horner's fifteen dependent ciphertext products.
    powers = {1: x}
    terms = [a.multiply_plain(x, THOR_EXP[1])]
    for k, coefficient in enumerate(THOR_EXP[2:], 2):
        low = k // 2
        powers[k] = a.multiply(powers[low], powers[k - low])
        terms.append(a.multiply_plain(powers[k], coefficient))
    # Retire cached powers before allocating reduction copies and sums; this
    # keeps the complete polynomial within the admitted live-ciphertext bound.
    powers.clear()
    result = terms[0]
    for term in terms[1:]:
        result = a.add(result, term)
    return a.add_plain(result, THOR_EXP[0])


def evaluate(a, x, method, width):
    if method == "repeated_squaring":
        exponential = a.add_plain(a.multiply_plain(x, 1 / 256), 1 - BOUND / 256)
        for _ in range(8):
            exponential = a.multiply(exponential, exponential)
        return normalize(a, exponential, width, 3)
    if method == "square_normalize":
        exponential = polynomial(a, a.multiply_plain(x, 0.5))
        probabilities = normalize(a, exponential, width, 2)
        squared = a.multiply(probabilities, probabilities)
        return normalize(a, squared, width, 2, squared_probabilities=True)
    raise ValueError("unknown encrypted softmax recipe")
