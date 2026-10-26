"""Finite, noiseless integer/phase oracle; NOT encrypted comparison or adaptation.

Primary source: https://github.com/zhouzhangwalker/HE3DB/tree/
1d8d7827baf32b4094b7547aeeaa9faf59253b26 (all paths relative to that pin).
* src/HEDB/comparison/extract_msb.cpp: ExtractMSB5/9/10, ImExtractMSB5..33,
  HomMSB dispatch (lines 5..252). Width 10 takes TWO lvl1 peels. Mixed dispatch
  calls ExtractMSB9 even when its remaining width is 5; preserve that call.
* src/HEDB/comparison/tfhepp_utils.cpp: LOG_to_ARI (19..27), MSB and identity
  overloads (51..129): same-level offsets Q/64 and Q/128; mixed offset Q/64.
* src/HEDB/comparison/tfhepp_utils.h: private mu_polygen stores -mu;
  gpolygen stores 2**scale_bits * floor(i / 2**(nbit-plain_bits)).
* src/HEDB/comparison/HomCompare.h: comparisons (14..64) request p+1 bits;
  GE negates logic MSB, then optionally calls LOG_to_ARI.
* thirdparty/TFHEpp/include/params/128bit.hpp (7..50): lvl0 uint32/n=672/k=1;
  lvl1 uint32/N=1024/k=1 and lvl2 uint64/N=2048/k=1, both logic mu=Q/8.
* thirdparty/TFHEpp/include/gatebootstrapping.hpp: BlindRotate (21..37),
  body floor, componentwise rounded masks with unsigned wrap before shifts.

Only shared signed/encode helpers are imported, by repository offline-file
pattern. No upstream code, dependencies, polynomial arrays, keys, or tensors.
Scalar table access replaces a polynomial only in the stated phase idealization.
Exact mixed scale conversion requires zero error AND a divisible lattice. Actual
KS/PBS noise distribution, failure probability, and security remain UNKNOWN;
they are not implicitly zero outside this explicitly noiseless experiment.
Error bounds below are conditional on circular cell margins at EACH lookup;
a small final error cannot certify intermediate cells. Odd q (including an
unknown 53-bit ROSETTA profile) lacks these exact dyadic scales. This does not
recover ROSETTA parameters, implement HE, or establish any security claim.

Run: python3 -B benchmarks/research/torus_comparison.py
"""

from dataclasses import dataclass
from fractions import Fraction
import importlib.util
import json
from pathlib import Path


_path = Path(__file__).resolve().parents[2] / "paper/reproduction-contracts/shared_profile_oracle.py"
_spec = importlib.util.spec_from_file_location("torus_shared_profile", _path)
_shared = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_shared)
signed, encode = _shared.signed, _shared.encode

CASE_LIMIT = 20_000
PBS_LIMIT = 13


def _int(*values):
    if any(type(v) is not int for v in values):
        raise TypeError("plain integer inputs required")


def domain(width, bits):
    _int(width, bits)
    if width not in (32, 64) or not 1 <= bits <= (10 if width == 32 else 33):
        raise ValueError("supported signed lattice: w32 b1..10; w64 b1..33")
    return 1 << width


def word(value, width):
    _int(value, width)
    if width not in (32, 64) or not 0 <= value < 1 << width:
        raise ValueError("canonical uint32/uint64 word required")
    return value


def lattice(d, width, bits):
    q = domain(width, bits)
    _int(d)
    if not -(1 << (bits - 1)) <= d < 1 << (bits - 1):
        raise ValueError("unwrapped signed difference does not fit")
    return encode(d, width - bits, "floor") % q


def profile(width):
    if width not in (32, 64):
        raise ValueError("only pinned dyadic profiles supported")
    return (1024, 4, 64) if width == 32 else (2048, 5, 128)


def table_value(index, width, *, scale_bits=None, constant=None):
    """Lazy negacyclic lookup; no polynomial materialization."""
    n, t, _ = profile(width)
    _int(index)
    if not 0 <= index < 2 * n or (scale_bits is None) == (constant is None):
        raise ValueError("index and exactly one table kind required")
    i = index % n
    if constant is not None:
        _int(constant)
        value = constant
    else:
        _int(scale_bits)
        if not 0 <= scale_bits < width:
            raise ValueError("scale exponent outside output word")
        value = (1 << scale_bits) * (i // (n >> t))
    return (value if index < n else -value) % (1 << width)


def phase_index(c, width, *, offset=0):
    word(c, width)
    _int(offset)
    n, _, _ = profile(width)
    return 2 * n * ((c + offset) % (1 << width)) // (1 << width)


def direct_msb(c, width, *, arithmetic=False):
    """Same-level MSB only: logic -Q/8,+Q/8; arithmetic 0,Q/2."""
    q = 1 << width
    _, _, divisor = profile(width)
    mu = q // (4 if arithmetic else 8)
    out = table_value(phase_index(c, width, offset=q // divisor), width, constant=-mu)
    return (out + (mu if arithmetic else 0)) % q


def identity(c, width, scale_bits):
    _, _, divisor = profile(width)
    return table_value(phase_index(c, width, offset=(1 << width) // divisor),
                       width, scale_bits=scale_bits)


def exact_scale32(c, *, error):
    """Lattice-only idealization of mixed-width conversion, NOT a real KS."""
    word(c, 64)
    _int(error)
    if error != 0 or c % (1 << 32):
        raise ValueError("exact conversion requires zero error and divisible lattice")
    return c >> 32


@dataclass
class Work:
    pbs: int = 0
    peels: int = 0
    conversions: int = 0

    def lookup(self):
        self.pbs += 1
        if self.pbs > PBS_LIMIT:
            raise ValueError("scalar ideal PBS budget exceeded")


def extract(c, width, bits, *, arithmetic=False, check_peel=None):
    """Literal dispatch, ideal scalar tables. Returns lvl1-width output + counts.

    check_peel observes (w,b,t,input,low,residual); its independent expectation
    must be derived from original Python input, never from this recurrence.
    """
    q = domain(width, bits)
    word(c, width)
    if c % (1 << (width - bits)):
        raise ValueError("exact signed lattice required")
    work = Work()

    def peel(value, w, b):
        modulus = 1 << w
        _, t, _ = profile(w)
        shifted = (value << (b - t - 1)) % modulus
        work.lookup()
        high = direct_msb(shifted, w, arithmetic=True)
        work.lookup()
        low = identity((shifted - high) % modulus, w, w - b)
        residual = (value - low) % modulus
        work.peels += 1
        if check_peel is not None:
            check_peel(w, b, t, value, low, residual)
        return residual, b - t

    # ImExtractMSB14..33 peel in lvl2, then force ExtractMSB9 in lvl1.
    force9 = width == 64 and bits > 9
    if width == 64:
        while bits > 9:
            c, bits = peel(c, 64, bits)
        c = exact_scale32(c, error=0)  # Explicit noiseless premise, not measured KS.
        work.conversions += 1
        width = 32
    # ExtractMSB10 calls ExtractMSB9 after one peel, even at remaining b=6.
    if bits == 10:
        c, bits = peel(c, 32, bits)
        force9 = True
    if bits > 5 or force9:
        c, bits = peel(c, 32, bits)
    work.lookup()
    return direct_msb(c, width, arithmetic=arithmetic), work


def logic_ge(a, b, width, bits):
    _int(a, b)
    c = lattice(a - b, width, bits)  # Admission on UNWRAPPED difference.
    sign_word, work = extract(c, width, bits)
    ge_word = (-sign_word) % (1 << 32)
    return ge_word == (1 << 29), work


def literal_log_to_ari(logic_word):
    """UNQUALIFIED LITERAL DIAGNOSTIC: centered converter representation.

    Private mu_polygen(-Q/4) stores +Q/4; add original Q/8. Thus true maps
    to 3Q/8, false to -Q/8. Not a 0/Q/2 GE contract; no upstream bug claim.
    Whole-callsite normalization/completeness remains unknown.
    """
    q = 1 << 32
    word(logic_word, 32)
    if logic_word not in (q // 8, 7 * q // 8):
        raise ValueError("exact logic encoding required")
    return (table_value(phase_index(logic_word, 32), 32, constant=q // 4)
            + q // 8) % q


def blind_rotation(body, mask_secret_pairs, *, domain_bits, n):
    """Componentwise index and exact lifted rounding error, at most 672 masks.

    Actual pinned lvl01/lvl02 domain is lvl0 uint32 (672 masks) for BOTH N.
    domain_bits is explicit for algebra controls, not a fabricated lvl2 input.
    Body floor error in (-1,0]; mask round error in (-1/2,1/2]. Unsigned wrap
    happens BEFORE shifts. Lifted error agrees modulo 2N across word wrap.
    """
    word(body, domain_bits)
    if n not in (1024, 2048):
        raise ValueError("pinned target N required")
    q = 1 << domain_bits
    h = q // (2 * n)
    index = body // h
    error = -Fraction(body % h, h)
    for count, (a, secret) in enumerate(mask_secret_pairs, 1):
        if count > 672:
            raise ValueError("bounded lvl0 mask count exceeded")
        word(a, domain_bits)
        _int(secret)
        if secret not in (0, 1):
            raise ValueError("binary secret control required")
        index -= secret * (((a + h // 2) % q) // h)
        error -= secret * (Fraction((a + h // 2) // h) - Fraction(a, h))
    return index % (2 * n), error


def circular_margin(c, width, *, offset=0):
    """Conservative distance to either boundary of the current phase-index cell."""
    word(c, width)
    n, _, _ = profile(width)
    h = (1 << width) // (2 * n)
    residue = (c + offset) % h
    return min(residue, h - residue)


def same_cell_bound(c, error, width, *, offset=0):
    _int(error)
    margin = circular_margin(c, width, offset=offset)
    return error == 0 or abs(signed(error, width)) < margin


def error_recurrence(e_c, e_h, e_l, shift, width):
    """Exact modular identities only; certifying cells is a separate obligation.

    eU=2**shift eC; e(U-H)=2**shift eC-eH; eNext=eC-eL.
    No unconditional noise bounds or probability model follow from these.
    """
    _int(e_c, e_h, e_l, shift, width)
    profile(width)
    if not 0 <= shift < width:
        raise ValueError("bounded shift required")
    return (signed(e_c << shift, width), signed((e_c << shift) - e_h, width),
            signed(e_c - e_l, width))


def endpoint_cases(bits):
    """Streaming low-five-digit transitions in endpoint/zero blocks, +/-1."""
    lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    yield lo
    yield hi
    yield -1
    yield 0
    yield 1
    for block in (lo, -32, 0, (hi // 32) * 32):
        for transition in (block, block + 32):
            for neighbor in (-1, 0, 1):
                d = transition + neighbor
                if lo <= d <= hi:
                    yield d


class Counterexample(AssertionError):
    pass


def run():
    """Streaming bounded sweep; stop immediately on a recurrence contradiction."""
    counts = {"cases": 0, "signed_cases": 0, "ge_cases": 0, "peel_checks": 0,
              "pbs": 0, "max_pbs": 0, "max_peels": 0, "conversions": 0}

    def account(work):
        counts["cases"] += 1
        if counts["cases"] > CASE_LIMIT:
            raise ValueError("finite case cap exceeded")
        counts["pbs"] += work.pbs
        counts["max_pbs"] = max(counts["max_pbs"], work.pbs)
        counts["max_peels"] = max(counts["max_peels"], work.peels)
        counts["conversions"] += work.conversions

    def check_signed(d, w, b):
        dropped = 0

        def check_peel(pw, pb, t, c, low, residual):
            nonlocal dropped
            # Independent Euclidean digit decomposition of ORIGINAL d.
            stage_d = d // (1 << dropped)
            delta = 1 << (pw - pb)
            digit = stage_d % (1 << t)
            q = 1 << pw
            expected_low = digit * delta
            expected_residual = (stage_d - digit) * delta % q
            if (c, low, residual) != (stage_d * delta % q, expected_low, expected_residual):
                raise Counterexample(f"low digit: w={w} b={b} d={d} stage={pw}/{pb} "
                                     f"low={low} expected={expected_low} residual={residual} "
                                     f"expected_residual={expected_residual}")
            dropped += t
            counts["peel_checks"] += 1

        result, work = extract(lattice(d, w, b), w, b, check_peel=check_peel)
        expected = (1 << 29) if d < 0 else 7 * (1 << 29)
        if result != expected:
            raise Counterexample(f"sign: w={w} b={b} d={d} got={result} expected={expected}")
        counts["signed_cases"] += 1
        account(work)

    for w in (32, 64):
        for b in range(1, 11):
            for d in range(-(1 << (b - 1)), 1 << (b - 1)):
                check_signed(d, w, b)
    for a in range(64):
        for b in range(64):
            result, work = logic_ge(a, b, 32, 7)
            if result != (a >= b):
                raise Counterexample(f"GE: a={a} b={b} got={result}")
            counts["ge_cases"] += 1
            account(work)
    for b in range(11, 34):
        for d in endpoint_cases(b):
            check_signed(d, 64, b)
    h = (1 << 32) // 2048
    component, error = blind_rotation(h // 2, ((h // 2, 1),), domain_bits=32, n=1024)
    if component != 2047 or error != -1:
        raise Counterexample("component-rounding negative control failed")
    return {"scope": "noiseless integer/phase idealization ONLY", "counts": counts,
            "recurrence_counterexamples": 0,
            "blind_rotation_control": {"domain_bits": 32, "n": 1024,
                                       "phase_only_index": 0, "component_index": component},
            "literal_centered_converter_diagnostic": {
                "label": "UNQUALIFIED LITERAL DIAGNOSTIC; whole-callsite completeness unknown",
                "true": literal_log_to_ari(1 << 29),
                "false": literal_log_to_ari(7 << 29),
                "claimed_zero_half_contract": "NOT asserted for LOG_to_ARI/GE"},
            "real_ks_pbs_noise_and_security": "unknown; not modeled"}


if __name__ == "__main__":
    try:
        print(json.dumps(run(), indent=2, sort_keys=True))
    except Counterexample as exc:
        print(json.dumps({"counterexample": str(exc), "status": "STOP: independent review required"}))
        raise SystemExit(1)
