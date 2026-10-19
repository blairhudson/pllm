"""Clear, bounded D8 gate contracts. No upstream imports or protected execution."""

from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from hashlib import sha256
from math import isqrt
from pathlib import Path
import importlib.util
import json
import struct
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def load_local(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Only independently PLLM-authored mathematics/encoding; never author source.
math_ref = load_local("gate_set_shared_math", ROOT / "shared_profile_oracle.py")
scalar = load_local("gate_set_scalar", HERE.parent / "reference.py")

N = 1 << 50
HALF = N // 2
PROFILE = scalar.PROFILE


def integer(x, lo, hi):
    if type(x) is not int or not lo <= x <= hi:
        raise ValueError("exact integer outside declared domain")
    return x


def signed_code(x):
    return integer(x, -HALF, HALF - 1)


def no_wrap(x):
    return signed_code(x)


def shift(word, bits, drop, kind):
    integer(bits, 1, 50)
    integer(drop, 0, bits - 1)
    integer(word, 0, (1 << bits) - 1)
    if kind not in ("TR", "ARS"):
        raise ValueError("explicit TR/ARS required")
    x = word if word < 1 << (bits - 1) else word - (1 << bits)
    output_bits = bits - drop if kind == "TR" else bits
    return {"bits": output_bits, "word": (x // (1 << drop)) % (1 << output_bits)}


def compare(word, bits, threshold, kind="unsigned", low_bits=None):
    integer(bits, 1, 50)
    integer(word, 0, (1 << bits) - 1)
    if kind == "signed":
        integer(threshold, -(1 << (bits - 1)), 1 << (bits - 1))
        x = word if word < 1 << (bits - 1) else word - (1 << bits)
    elif kind in ("unsigned", "low"):
        width = integer(low_bits, 1, bits) if kind == "low" else bits
        integer(threshold, 0, 1 << width)
        x = word % (1 << width)
    else:
        raise ValueError("explicit comparison domain required")
    if kind != "low" and low_bits is not None:
        raise ValueError("unexpected low-bit view")
    return B(int(x < threshold))


@dataclass(frozen=True)
class B:
    value: int

    def __post_init__(self):
        integer(self.value, 0, 1)


@dataclass(frozen=True)
class A:
    word: int
    bits: int = 50
    fractional: int = 0

    def __post_init__(self):
        integer(self.bits, 1, 50)
        integer(self.fractional, 0, self.bits - 1)
        integer(self.word, 0, (1 << self.bits) - 1)


def b2a(bit, bits=50):
    if type(bit) is not B:
        raise ValueError("B2A requires B, not an arithmetic word")
    return A(bit.value, bits, 0)


def b2a_xor_shares(left, right, bits=50):
    if type(left) is not B or type(right) is not B:
        raise ValueError("Boolean XOR shares required")
    integer(bits, 1, 50)
    # Algebra only. Native realization needs fresh correlation/product, not opens.
    return A((left.value + right.value - 2 * left.value * right.value) % (1 << bits), bits, 0)


def phi_select(bit, yes, no):
    if type(bit) is not B or type(yes) is not A or type(no) is not A:
        raise ValueError("typed Phi select requires B,A,A")
    if (yes.bits, yes.fractional) != (no.bits, no.fractional):
        raise ValueError("Phi scale/width mismatch")
    gain = b2a(bit, yes.bits)
    return A((no.word + gain.word * (yes.word - no.word)) % (1 << yes.bits), yes.bits, yes.fractional)


def external(value):
    return scalar.encode_external(value, scalar.ArithmeticType(50, 12))


@lru_cache(maxsize=1)
def tables():
    # Existing paper-equation generator; dual precision is regression evidence,
    # exact rational exp certificates below independently settle all new floors.
    first, second = math_ref.sigma_tables(80), math_ref.sigma_tables(96)
    if first != second:
        raise ArithmeticError("unresolved table floor")
    return {key: tuple(value) for key, value in first.items()}


def digest(values):
    return sha256(b"".join(struct.pack("<Q", x) for x in values)).hexdigest()


def exp_bounds(x, precision=120):
    """Rational positive Taylor sum with geometric upper bound on omitted tail."""
    if type(x) is not Fraction or not 0 <= x <= 16:
        raise ValueError("bounded rational exponential argument required")
    integer(precision, 80, 256)
    total = term = Fraction(1)
    for n in range(1, 512):
        term *= x / n
        total += term
        following = term * x / (n + 1)
        ratio = x / (n + 2)
        if ratio < 1:
            tail = following / (1 - ratio)
            if tail < Fraction(1, 1 << precision):
                return total, total + tail
    raise ArithmeticError("exponential certificate budget")


def certified_floor(lo, hi):
    a, b = lo.numerator // lo.denominator, hi.numerator // hi.denominator
    if a != b:
        raise ArithmeticError("unresolved floor certificate")
    return a


def certify_tables():
    result = {"silu": [], "exp0": [], "exp1": []}
    for name, count, divisor in (("silu", 1024, 64), ("exp0", 256, 4096), ("exp1", 256, 16)):
        for i in range(count):
            lo, hi = exp_bounds(Fraction(i, divisor))
            if name == "silu":
                numerator = 64 * i
                value = certified_floor(numerator / (1 + hi), numerator / (1 + lo))
            else:
                value = certified_floor(4096 / hi, 4096 / lo)
            result[name].append(value)
    return result


def activation(x, name="silu"):
    signed_code(x)
    if name not in ("silu", "gelu"):
        raise ValueError("unknown activation")
    y = x // 64
    t = tables()[name][min(abs(y), len(tables()[name]) - 1)]
    # Nonidentity activation Phi proposal: B2A(D)*original_X - T[J].
    selected = phi_select(B(int(y >= 0)), A(x % N, 50, 12), A(0, 50, 12))
    return {"floor": y, "index": min(abs(y), len(tables()[name]) - 1),
            "output": (selected.word - t) % N}


def nexp(c):
    integer(c, 0, 65535)
    a, b = tables()["exp0"][c % 256], tables()["exp1"][c // 256]
    return a * b // 4096


def reciprocal_q(k):
    integer(k, 1, 1024)
    return 6 + k.bit_length()  # ceil(log2(k+1)), including power-of-two k


def reciprocal_entry(index, k):
    q = reciprocal_q(k)
    integer(index, 0, (1 << q) - 1)
    return 0 if index == 0 else (1 << 18) // index


def reciprocal(z, k):
    reciprocal_q(k)
    integer(z, 64, 64 * k)
    return reciprocal_entry(z, k)


def softmax(row, active=None):
    if type(row) is not list or not 1 <= len(row) <= 1024:
        raise ValueError("bounded row required")
    for x in row:
        signed_code(x)
    active = [1] * len(row) if active is None else active
    if type(active) is not list or len(active) != len(row):
        raise ValueError("mask shape")
    for bit in active:
        integer(bit, 0, 1)
    if not any(active):
        raise ValueError("at least one active causal entry required")
    maximum = max(x for x, bit in zip(row, active) if bit)
    exps = []
    for x, bit in zip(row, active):
        # Sufficient signed50 subtraction obligation, before clipping.
        difference = no_wrap(maximum - x) if bit else 0
        exps.append(nexp(min(difference, 65535)) if bit else 0)
    z = sum(exps) // 64
    r = reciprocal(z, sum(active))
    return {"exp": exps, "denominator": z, "reciprocal": r,
            "output": [e * r // 4096 for e in exps]}


def norm_index(s):
    integer(s, 1, N - 1)
    e = s.bit_length() - 1
    m = ((s << 7) >> e) - 128
    return {"exponent": e, "mantissa": m, "index": 64 * m + e}


def rsqrt_entry(index, k):
    integer(index, 0, 8191)
    integer(k, 1, 4096)
    return math_ref.sigma_rsqrt_table_entry(index, k)


def norm(row, gamma=None, beta=None, kind="layer"):
    if type(row) is not list or not 1 <= len(row) <= 4096 or kind not in ("layer", "rms"):
        raise ValueError("bounded LayerNorm/RMSNorm row required")
    k = len(row)
    gamma = [4096] * k if gamma is None else gamma
    beta = [0] * k if beta is None else beta
    if type(gamma) is not list or type(beta) is not list or len(gamma) != k or len(beta) != k:
        raise ValueError("affine shape")
    for x in row + gamma + beta:
        signed_code(x)
    total = no_wrap(sum(row))
    mean = no_wrap(total * (4096 // k)) // 4096 if kind == "layer" else 0
    centered = [no_wrap(x - mean) for x in row]
    # Conservative sufficient signed50 admission, stricter than unsigned index.
    s = no_wrap(sum(x * x for x in centered) + 1)
    index = norm_index(s)
    inverse = rsqrt_entry(index["index"], k)
    q = [no_wrap(x * inverse) // 4096 for x in centered]
    output = [no_wrap(no_wrap(x * g) // 4096 + b) for x, g, b in zip(q, gamma, beta)]
    return {"mean": mean, "sum_squares_plus_one": s, **index,
            "inverse": inverse, "output": output}


def canonical(obj):
    return (json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("ascii")


def cases():
    cases = []
    def add(gate, args, expected):
        cases.append({"id": f"{gate}-{len(cases):04d}", "gate": gate, "args": args, "expected": expected})
    samples = sorted({-(1 << 37), (1 << 37) - 1, -65473, -65472, -65471, 65471, 65472, 65473,
                      -16385, -4097, -4096, -129, -128, -127, -65, -64, -63, -1, 0, 1, 63, 64, 65, 4096})
    for x in samples:
        add("silu", [x], activation(x))
    for c in (0, 1, 255, 256, 257, 4095, 4096, 65534, 65535):
        add("nexp", [c], nexp(c))
    for k in (1, 2, 3, 768, 1024):
        for z in sorted({64, 65 if k > 1 else 64, 64 * k - 1 if k > 1 else 64, 64 * k}):
            add("reciprocal", [z, k], reciprocal(z, k))
    for row, active in (([0], None), ([0, 0], None), ([0, 0, 0], None), ([0, -4096], None),
                        ([0, -65536], None), ([0, 99999, -4096], [1, 0, 1])):
        add("softmax", [row, active], softmax(row, active))
    for kind, row, g, b in (("layer", [4096] * 4, None, None), ("layer", [4096] * 768, None, None),
                            ("layer", [-4096, -4096, 4096, 4096], None, None),
                            ("rms", [0] * 4, None, None), ("rms", [4096] * 4, None, None),
                            ("layer", [-65, 64], [-4096, 8192], [1, -1])):
        add("norm", [row, g, b, kind], norm(row, g, b, kind))
    for s in sorted({1, N - 1, *(v for e in range(1, 50) for v in ((1 << e) - 1, 1 << e, (1 << e) + 1))}):
        add("norm-index", [s], norm_index(s))
    for bits in (8, 38, 50):
        for x in (-65, -64, -1, 0, 1, 63, 64):
            for kind in ("TR", "ARS"):
                args = [x % (1 << bits), bits, 6, kind]
                add("shift", args, shift(*args))
    for l in (0, 1):
        for r in (0, 1):
            add("b2a", [l, r, 50], b2a_xor_shares(B(l), B(r)).word)
    for word in (0, 127, 128, 255):
        for kind, threshold, low_bits in (("unsigned", 128, None), ("signed", 0, None), ("low", 4, 3)):
            args = [word, 8, threshold, kind, low_bits]
            add("compare", args, compare(*args).value)
    for bit in (0, 1):
        add("phi-select", [bit, N - 1, 4096], phi_select(B(bit), A(N - 1, 50, 12), A(4096, 50, 12)).word)
    for numerator in (-5, -3, -1, 1, 3, 5):
        add("external", [numerator, 8192], external(Fraction(numerator, 8192)))
    return cases


def fixture():
    values = {name: list(table) for name, table in tables().items() if name != "gelu"}
    for k in (4, 768):
        values[f"rsqrt-k{k}"] = [rsqrt_entry(i, k) for i in range(8192)]
    for k in (1, 2, 3, 768, 1024):
        values[f"reciprocal-k{k}"] = [reciprocal_entry(i, k) for i in range(1 << reciprocal_q(k))]
    hashes = {name: digest(table) for name, table in values.items()}
    return {"schema": "pllm.fusefss.gate-set.v1", "profile": PROFILE,
            "source_lock_sha256": sha256((HERE / "source-lock.json").read_bytes()).hexdigest(),
            "tables": values, "sha256_u64le": hashes, "cases": cases()}


if __name__ == "__main__":
    sys.stdout.buffer.write(canonical(fixture()))
