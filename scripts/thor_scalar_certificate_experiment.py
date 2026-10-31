"""Pinned THOR scalar experiment; ideal real payloads, not a CKKS certificate.

Provenance (read-only, never imported or vendored): crypto-starlab/THOR,
b513c06aaabccc04b560721e35e9a98008428c25, src/thor/nonlinear/layernorm.py
lines 5..12, 21..27, 65..75, 104..130. Domain and identity: repository
docs/research/thor-multiblock-indexing-audit.md:481..526,557..624.

Only source_trace uses NumPy, lazily. It preserves source literal types,
evaluation grouping, unsorted root index and both evaluations of 3/kn.
All subsequent arithmetic uses exact ratios of those actual float values.
No tensors, upstream imports, model coverage, encrypted error or tolerance
adoption. CERTIFIED_BOUND means an enclosure, not a tolerance acceptance.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from fractions import Fraction
import math
import sys


PIN = "b513c06aaabccc04b560721e35e9a98008428c25"
PROFILES = ((.15, 10, 1), (.2, 150, 2), (.75, 2500, 2))
N = 768
BITS = 192
MAX_STEPS = 256
EPS = Fraction(1, 2**40)
U_STAR = 1 + EPS
Ratio = tuple[int, int]
Interval = tuple[Fraction, Fraction]


class Refused(ValueError):
    """Invalid source arithmetic or unfinished trace; never convergence."""

    def __init__(self, message, trace=None):
        super().__init__(message)
        self.trace = trace


def float_ratio(value) -> Ratio:
    """Keep actual binary floating value, not its decimal spelling."""
    try:
        if isinstance(value, complex) or not math.isfinite(value):
            raise Refused("complex or nonfinite source scalar")
        return value.as_integer_ratio()
    except Refused:
        raise
    except (AttributeError, OverflowError, ValueError, TypeError) as exc:
        raise Refused("source scalar has no finite integer ratio") from exc


def rational(value) -> Fraction:
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value)
    raise Refused("exact helper requires int or Fraction")


def interval(lower, upper) -> Interval:
    lower, upper = rational(lower), rational(upper)
    if lower > upper:
        raise Refused("reversed or empty interval")
    return lower, upper


def positive_coefficients(p, q, h):
    p, q, h = map(rational, (p, q, h))
    if p <= 0 or q <= 0 or h <= 0:
        raise Refused("p, q and h must be positive")
    return p, q, h


@dataclass
class Step:
    counter: int
    roots: tuple[Ratio, Ratio]
    root_index: int
    ratios: dict[str, Ratio]

    def Q(self, name):
        return Fraction(*self.ratios[name])


@dataclass
class Trace:
    profile: tuple
    numpy_version: str
    ratios: dict[str, Ratio]
    steps: list[Step] = field(default_factory=list)
    completed: bool = False

    def Q(self, name):
        return Fraction(*self.ratios[name])

    @property
    def terminal(self):
        return self.steps[-1].Q("post") if self.steps else self.Q("en0")


def source_trace(profile, safety_limit=256) -> Trace:
    """Transcribe only scalar source execution; refuse rather than substitute."""
    if type(safety_limit) is not int or not 1 <= safety_limit <= MAX_STEPS:
        raise Refused("safety_limit must be an integer in [1, 256]")
    if profile not in PROFILES:
        raise Refused("only the three pinned profiles are admitted")
    try:
        import numpy as np
    except ImportError as exc:
        raise Refused("NumPy required for source trace; no replacement roots") from exc

    # Preserve Python int/float literals and source operation order.
    a, M, g = PROFILES[PROFILES.index(profile)]
    var_e = 10**(-5)
    en = a/M
    C = (M*1.05+var_e)*N**2
    b = 1/C**(1/2)
    d = var_e/C
    raw_b = b
    if g == 2:
        # Scalar ufunc division matches one float64 element of source mask/2.
        b = np.float64(b)/2
    alpha = .001
    stop = 1-alpha
    ratios = {name: float_ratio(value) for name, value in (
        ("a", a), ("M", M), ("g", g), ("var_e", var_e), ("C", C),
        ("raw_b", raw_b), ("b", b), ("d", d), ("en0", en), ("stop", stop))}
    trace = Trace((a, M, g), np.__version__, ratios)
    try:
        while True:
            # Validate before comparison: NaN must never become a false exit.
            float_ratio(en)
            if en <= 0:
                raise Refused("nonpositive scalar state")
            if not en < stop:
                trace.completed = True
                return trace
            if len(trace.steps) >= safety_limit:
                raise Refused("safety limit: unfinished source trace")
            pre = en
            roots = np.roots([1-en**3,6*en**2-6,9-9*en])
            if len(roots) != 2 or np.iscomplexobj(roots):
                raise Refused("complex or invalid root pair")
            root_ratios = tuple(float_ratio(root) for root in roots)
            if any(root <= 0 for root in roots):
                raise Refused("nonpositive root")
            kn = roots[1]  # UNSORTED second returned root, never sort/select by size.
            p = kn**(3/2)/2
            h = 3/kn
            q = kn**3/4
            h_again = 3/kn
            for value in (p, q, h, h_again):
                float_ratio(value)
            if p <= 0 or q <= 0 or h <= 0 or h != h_again:
                raise Refused("invalid coefficients or unequal 3/kn evaluations")
            en = kn*en*(3-kn*en)**2/4
            float_ratio(en)
            if en <= 0:
                raise Refused("nonpositive post-state")
            trace.steps.append(Step(len(trace.steps) + 1, root_ratios, 1,
                                    {name: float_ratio(value) for name, value in (
                                        ("pre", pre), ("post", en), ("k", kn),
                                        ("p", p), ("q", q), ("h", h),
                                        ("h_again", h_again), ("stop", stop))}))
    except Refused as exc:
        exc.trace = trace
        raise
    except (ArithmeticError, ValueError, TypeError, RuntimeWarning,
            np.linalg.LinAlgError) as exc:
        raise Refused(f"source arithmetic refused: {exc}", trace) from exc


def domain(trace) -> Interval:
    """D(v)=768² Q(b)² g² v + Q(d), including actual float endpoints."""
    scale = N**2 * trace.Q("b")**2 * trace.Q("g")**2
    return interval(scale * trace.Q("a") + trace.Q("d"),
                    scale * trace.Q("M") + trace.Q("d"))


def F(q, h, x):
    q, h, x = map(rational, (q, h, x))
    return q*x*(h-x)**2


def cubic_range(q, h, lower, upper) -> Interval:
    """Exact extrema: endpoints, contained h/3 (maximum), h (zero)."""
    q, h = rational(q), rational(h)
    if q <= 0 or h <= 0:
        raise Refused("cubic requires positive q and h")
    lower, upper = interval(lower, upper)
    points = [lower, upper]
    points.extend(x for x in (h/3, h) if lower <= x <= upper)
    values = [F(q, h, x) for x in points]
    return min(values), max(values)


def product_interval(left, right) -> Interval:
    left, right = interval(*left), interval(*right)
    values = [x*y for x in left for y in right]
    return min(values), max(values)


def outward(bounds, bits=BITS) -> Interval:
    lower, upper = interval(*bounds)
    if type(bits) is not int or not 0 <= bits <= BITS:
        raise Refused("rounding bits must be in [0, 192]")
    grid = 1 << bits
    lo = (lower.numerator * grid) // lower.denominator
    hi = -((-upper.numerator * grid) // upper.denominator)
    return Fraction(lo, grid), Fraction(hi, grid)


def sqrt_bracket(value, bits=BITS) -> Interval:
    value = rational(value)
    if value < 0:
        raise Refused("negative square root")
    if type(bits) is not int or not 0 <= bits <= BITS:
        raise Refused("sqrt bits must be in [0, 192]")
    grid = 1 << bits
    s = math.isqrt((value.numerator << (2*bits)) // value.denominator)
    lower = Fraction(s, grid)
    upper = lower if lower*lower == value else Fraction(s+1, grid)
    return lower, upper


def exact_step(A, R, P, p, q, h):
    """Point oracle interface; simultaneous old-A recurrence, q need not equal p²."""
    A, R, P = map(rational, (A, R, P))
    p, q, h = positive_coefficients(p, q, h)
    if P <= 0:
        raise Refused("nonpositive P")
    return F(q, h, A), p*R*(h-A), P*q/p**2


def direct_enclosure(D, coefficients):
    """Conservative interval route; no sufficient-theorem premise assumed.

    Zero factors permit rho>=0 and the square-identity route, but never prove
    strict positivity. If any factor sign is unproved, sqrt(D)*R preserves sign.
    Rounding only follows exact extrema/products; old A drives both updates.
    """
    D = interval(*D)
    if D[0] <= 0:
        raise Refused("D must be strictly positive")
    A, R, P = D, (Fraction(1), Fraction(1)), Fraction(1)
    nonnegative, strictly_positive = True, True
    for p, q, h in coefficients:
        p, q, h = positive_coefficients(p, q, h)
        old_A = A
        factor = (p*(h-old_A[1]), p*(h-old_A[0]))
        nonnegative = nonnegative and factor[0] >= 0
        strictly_positive = strictly_positive and factor[0] > 0
        R = outward(product_interval(R, factor))
        A = outward(cubic_range(q, h, *old_A))
        P *= q/p**2
    if nonnegative:
        rho = (sqrt_bracket(A[0]/P)[0], sqrt_bracket(A[1]/P)[1])
        route = "direct-square-identity"
    else:
        root_D = (sqrt_bracket(D[0])[0], sqrt_bracket(D[1])[1])
        rho = outward(product_interval(root_D, R))
        route = "direct-signed-R"
    error = max(abs(rho[0]-1), abs(rho[1]-1))
    return {"A": A, "R": R, "P": P, "rho": rho, "error": error,
            "nonnegative_factors": nonnegative,
            "strictly_positive_factors": strictly_positive, "route": route,
            "status": "CERTIFIED_BOUND" if rho[0] >= 0 else "INCONCLUSIVE"}


def check_sufficient(trace):
    """Report every premise independently; all success alone licenses theorem.

    Consistent with audit's ideal recurrence and sign/product identity:
    let c_j=.95*(1-eps)^j. For 0<c<=1 and 0<e<1<h,
    F(c*e)>=c*F(e). Induct on [c_j*e_j,U*] subset (0,h): its endpoint
    minimum, using the checked rounded-post bridge, gives the next lower
    bound c_{j+1}*e_{j+1}. The maximum on [0,h] is 4*q*h**3/27<=U*,
    giving the next upper bound. Thus c_j*e_j <= A_j <= U* and every
    p*(h-A) factor is strictly positive.
    m<=32 and the actual stop give
      rho² >= .95*(1-eps)^32*(.999-eps)/(1+32eps) > .974²,
      rho² <= (1+eps)/(1-32eps) < 1.001².
    Hence .974<rho<1.001 and |rho-1|<.026. These are conditional
    mathematical conclusions, NOT an adopted application tolerance.
    """
    lower, upper = domain(trace)
    e0, stop = trace.Q("en0"), trace.Q("stop")
    premises = {}

    def record(name, condition):
        premises[name] = bool(condition)

    record("initial_lower_0.95e0", Fraction(95, 100)*e0 <= lower)
    record("initial_shortfall_retained", lower < e0)
    record("domain_positive", 0 < lower)
    record("domain_ordered", lower <= upper)
    record("domain_upper", upper <= U_STAR)
    record("terminating_trace", trace.completed and trace.terminal >= stop)
    record("m_at_most_32", len(trace.steps) <= 32)
    record("actual_stop_lower", stop >= Fraction(999, 1000)-EPS)
    P = Fraction(1)
    previous = e0
    for index, step in enumerate(trace.steps, 1):
        e, nxt = step.Q("pre"), step.Q("post")
        p, q, h = (step.Q(name) for name in ("p", "q", "h"))
        prefix = f"step_{index}."
        record(prefix+"counter", step.counter == index)
        record(prefix+"state_chain", e == previous)
        record(prefix+"pre_positive", e > 0)
        record(prefix+"pre_below_one", e < 1)
        record(prefix+"pre_continues", e < stop)
        record(prefix+"post_positive", nxt > 0)
        record(prefix+"p_positive", p > 0)
        record(prefix+"q_positive", q > 0)
        record(prefix+"h_above_upper", h > U_STAR)
        record(prefix+"repeated_h_equal", h == step.Q("h_again"))
        record(prefix+"root_index", step.root_index == 1 and
               step.Q("k") == Fraction(*step.roots[1]))
        record(prefix+"stop_operand", step.Q("stop") == stop)
        record(prefix+"cubic_maximum", 4*q*h**3/27 <= U_STAR)
        record(prefix+"endpoint_minimum",
               min(F(q, h, e), F(q, h, U_STAR)) >= (1-EPS)*nxt)
        if p == 0:
            P = None
        elif P is not None:
            P *= q/p**2
        previous = nxt
    record("product_drift_lower", P is not None and P >= 1-32*EPS)
    record("product_drift_upper", P is not None and P <= 1+32*EPS)
    # Exact arithmetic also checks the numerical implications of this theorem.
    record("theorem_lower_arithmetic",
           Fraction(95, 100)*(1-EPS)**32*(Fraction(999, 1000)-EPS)
           / (1+32*EPS) > Fraction(974, 1000)**2)
    record("theorem_upper_arithmetic",
           U_STAR/(1-32*EPS) < Fraction(1001, 1000)**2)
    failed = [name for name, passed in premises.items() if not passed]
    return {"premises": premises, "failed": failed, "success": not failed,
            "P": P, "D_minus_en0": lower-e0}


def analyze(trace):
    if not trace.completed:
        raise Refused("unfinished trace cannot certify convergence", trace)
    sufficient = check_sufficient(trace)
    if sufficient["success"]:
        return {"status": "CERTIFIED_BOUND", "route": "conditional-theorem",
                "D": domain(trace), "P": sufficient["P"],
                "rho": (Fraction(974, 1000), Fraction(1001, 1000)),
                "rho_strict": True, "error": Fraction(26, 1000),
                "error_strict": True, "nonnegative_factors": True,
                "strictly_positive_factors": True, "sufficient": sufficient}
    result = direct_enclosure(domain(trace),
                              ((step.Q("p"), step.Q("q"), step.Q("h"))
                               for step in trace.steps))
    result.update(D=domain(trace), sufficient=sufficient,
                  rho_strict=False, error_strict=False)
    return result


def decimal_outward(value, places=12, upper=False):
    """Explicit decimal floor/ceiling of exact rational, not nearest formatting."""
    value = rational(value)
    if type(places) is not int or not 0 <= places <= 30:
        raise Refused("decimal places must be in [0, 30]")
    scale = 10**places
    num = value.numerator*scale
    units = -((-num)//value.denominator) if upper else num//value.denominator
    sign = "-" if units < 0 else ""
    whole, tail = divmod(abs(units), scale)
    return f"{sign}{whole}.{tail:0{places}d}" if places else f"{sign}{whole}"


def show_interval(bounds):
    lower, upper = bounds
    return (f"[{lower},{upper}] "
            f"decimal_outward=[{decimal_outward(lower)},"
            f"{decimal_outward(upper, upper=True)}]")


def print_trace(trace):
    print(f"  initial_float_ratios={trace.ratios}")
    for step in trace.steps:
        print(f"  counter={step.counter} roots_unsorted={step.roots} "
              f"index={step.root_index} float_ratios={step.ratios}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--safety-limit", type=int, choices=range(1, MAX_STEPS+1),
                        default=MAX_STEPS, metavar="1..256")
    args = parser.parse_args(argv)
    print(f"THOR source={PIN} Python={sys.version.split()[0]} n={N}; "
          "ideal scalar bounds only; no adopted tolerance or CKKS claim")
    print("profile | iterations | stopstate | failedpremises | route | D | P | "
          "sign | rho | error | status")
    refused = False
    for profile in PROFILES:
        try:
            trace = source_trace(profile, args.safety_limit)
            result = analyze(trace)
        except Refused as exc:
            refused = True
            partial = exc.trace
            iterations = len(partial.steps) if partial else "unavailable"
            terminal = partial.terminal if partial else "unavailable"
            print(f"{profile} | {iterations} | {terminal} | unavailable | "
                  f"none | unavailable | unavailable | unproved | unavailable | "
                  f"unavailable | REFUSED: {exc}")
            if args.trace and partial:
                print_trace(partial)
            continue
        sufficient = result["sufficient"]
        sign = ("strict-positive" if result["strictly_positive_factors"] else
                "nonnegative" if result["nonnegative_factors"] else "unproved")
        strict = "strict" if result["rho_strict"] else "closed"
        error_op = "<" if result["error_strict"] else "<="
        print(f"{profile} | {len(trace.steps)} | {trace.terminal} "
              f">= {trace.Q('stop')} | {','.join(sufficient['failed']) or 'none'} | "
              f"{result['route']} | {show_interval(result['D'])} | {result['P']} | "
              f"{sign} | {strict} {show_interval(result['rho'])} | "
              f"{error_op}{result['error']} "
              f"decimal_upper={decimal_outward(result['error'], upper=True)} | "
              f"{result['status']}")
        print(f"  NumPy={trace.numpy_version} D(a)-Q(en0)={sufficient['D_minus_en0']}")
        for name, passed in sufficient["premises"].items():
            print(f"  premise {name}={passed}")
        if args.trace:
            print_trace(trace)
    return 2 if refused else 0


if __name__ == "__main__":
    raise SystemExit(main())
