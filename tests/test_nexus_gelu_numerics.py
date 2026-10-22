"""Independent finite-algebra checks for the optional, research-only experiment."""

from fractions import Fraction as F
import builtins
import importlib.util
import json
from pathlib import Path

import pytest


PATH = Path(__file__).parents[1] / "examples/benchmarks/nexus_gelu_numerics.py"
SPEC = importlib.util.spec_from_file_location("nexus_gelu_numerics", PATH)
n = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(n)


@pytest.fixture
def arithmetic():
    pytest.importorskip("mpmath", minversion="1.3.0")
    return n.Arithmetic(60)


def contains(a, interval, point):
    lo, hi = a.ends(interval)
    assert lo <= point <= hi


def test_printed_exact_coefficients_and_independent_finite_powers():
    assert n.P == tuple(
        map(
            F,
            (
                "-0.5054031199708174",
                "-0.42226581151983866",
                "-0.11807612951181953",
                "-0.011034134030615728",
            ),
        )
    )
    assert n.Q == tuple(
        map(
            F,
            (
                "0.008526321541038084",
                "0.5",
                "0.3603292692789629",
                "0",
                "-0.037688200365904236",
                "0",
                "0.0018067462606141187",
            ),
        )
    )
    assert n.SHORT_P == tuple(
        map(
            F,
            (
                "-0.5054031199708174",
                "-0.4222658115198386",
                "-0.1180761295118195",
                "-0.0110341340306157",
            ),
        )
    )
    assert n.SHORT_Q == tuple(
        map(
            F,
            (
                "0.0085263215410380",
                "0.5",
                "0.3603292692789629",
                "0",
                "-0.037688200365904",
                "0",
                "0.0018067462606141",
            ),
        )
    )
    assert n.SHORT_P[1] != n.P[1]
    for x in (F(0), F(1), F(-1), F(1, 4)):
        f = (315 * x - 420 * x**3 + 378 * x**5 - 180 * x**7 + 35 * x**9) / 128
        g = (5850 * x - 34974 * x**3 + 97015 * x**5 - 113492 * x**7 + 46623 * x**9) / 1024
        assert n.polynomial(x, n.F4) == f
        assert n.polynomial(x, n.G4) == g
    assert n.sign_stages(F(0)) == (0, 0, 0, 0)
    assert n.polynomial(F(1), n.G4) == F(511, 512)
    assert n.polynomial(F(-1), n.G4) == -F(511, 512)
    assert n.polynomial(F(1), n.F4) == 1
    assert n.polynomial(F(-1), n.F4) == -1


def test_threshold_ties_and_branch_oracle():
    assert n.piecewise(F(-4)) == 0
    for x, coefficients in ((F(-39, 20), n.P), (F(3), n.Q)):
        expected = sum(c * x**i for i, c in enumerate(coefficients))
        assert n.piecewise(x) == expected
    assert n.piecewise(F(4)) == 4
    assert n.piecewise(F(-5)) == 0


def test_threshold_comparator_is_half_not_endpoint_oracle(arithmetic):
    a = arithmetic
    for i, t in enumerate(n.THRESHOLDS):
        result = n.blend(a.point(t), a.point)
        assert result["h"][i] == a.point(F(1, 2))
        assert abs(sum(result["b"]) - 1) < a.mp.mpf("1e-55")
        assert result["y"] != n.piecewise(a.point(t), a.point)


def test_difference_scales_and_sampling():
    assert n.difference_scales() == (12, F(199, 20), 11)
    assert n.difference_scales((0, 2), (F(1),)) == (1,)
    with pytest.raises(ValueError):
        n.difference_scales((1, 1))
    points = n.sample_points(2)
    assert set(n.DOMAIN + n.THRESHOLDS).issubset(points)
    for t, d in zip(n.THRESHOLDS, n.SCALES):
        for k in range(25):
            for s in (-1, 1):
                x = t + s * d / 2**k
                if -8 <= x <= 8:
                    assert x in points


def test_two_to_minus_twenty_slope_and_amplification(arithmetic):
    a = arithmetic
    z = a.point(F(1, 2**20))
    sign = n.sign_stages(z, a.point)[-1]
    assert 0 < sign < a.point(F(1, 1000))
    derivative = F(315, 128) ** 2 * F(5850, 1024) ** 2
    assert abs(sign / z - a.point(derivative)) < a.mp.mpf("0.0001")
    # An inactive high-degree Q at -8 magnifies even a small comparator error.
    x = a.point(-8)
    q = n.polynomial(x, n.Q, a.point)
    assert abs(q) > 100
    assert abs(a.point(F(1, 1000)) * q) > a.point(F(1, 10))
    for x in (-8, F(-39, 20), F(3), F(7, 2)):
        px = a.point(x)
        data = n.blend(px, a.point)
        ideal = [int(F(x) > t) for t in n.THRESHOLDS]
        terms = [
            (h - bit) * factor
            for h, bit, factor in zip(
                data["h"], ideal, (data["p"], data["q"] - data["p"], px - data["q"])
            )
        ]
        error = data["y"] - n.piecewise(px, a.point)
        assert abs(error - sum(terms)) < a.mp.mpf("1e-50")
        assert abs(error) <= sum(map(abs, terms)) + a.mp.mpf("1e-50")


def test_outward_rational_operations_and_binary_serialization(arithmetic):
    a = arithmetic
    third = a.exact(F(1, 3))
    for interval, exact in (
        (third + third, F(2, 3)),
        (third * third, F(1, 9)),
        (-third, F(-1, 3)),
        (third / 7, F(1, 21)),
    ):
        encoded = a.encode(interval)

        def decode(bound):
            return F(int(bound["mantissa"])) * F(2) ** bound["exponent2"]

        assert decode(encoded["lower"]) <= exact <= decode(encoded["upper"])


@pytest.mark.parametrize("direction", [-1, 1])
def test_signed_band_point_encloses_independent_exact_composition(arithmetic, direction):
    def f(x):
        return (315 * x - 420 * x**3 + 378 * x**5 - 180 * x**7 + 35 * x**9) / 128

    def g(x):
        return (5850 * x - 34974 * x**3 + 97015 * x**5 - 113492 * x**7 + 46623 * x**9) / 1024

    z = F(direction, 2**20)
    expected = f(f(g(g(z))))
    a = arithmetic
    encoded = a.encode(n.sign_stages(a.exact(z), a.exact)[-1])

    def decode(bound):
        return F(int(bound["mantissa"])) * F(2) ** bound["exponent2"]

    assert decode(encoded["lower"]) <= expected <= decode(encoded["upper"])
    assert F(999, 1000) < abs(expected - direction) < 1


def test_interval_polynomials_contain_independent_exact_fraction_cases(arithmetic):
    a = arithmetic
    for coefficients in (n.P, n.Q, n.F4, n.G4):
        enclosure = n.polynomial(a.interval(F(-1, 4), F(1, 2)), coefficients, a.exact)
        encoded = a.encode(enclosure)

        def decode(bound):
            return F(int(bound["mantissa"])) * F(2) ** bound["exponent2"]

        for x in (F(-1, 4), F(0), F(1, 3), F(1, 2)):
            exact = sum(c * x**i for i, c in enumerate(coefficients))
            assert decode(encoded["lower"]) <= exact <= decode(encoded["upper"])


def test_missing_optional_dependency_is_explicit(monkeypatch):
    original = builtins.__import__

    def without_mpmath(name, *args, **kwargs):
        if name == "mpmath":
            raise ImportError("test missing optional experiment dependency")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_mpmath)
    with pytest.raises(RuntimeError, match="optional experiment dependency mpmath==1.3.0"):
        n.Arithmetic()


def test_interval_stages_and_outputs_contain_independent_points(arithmetic):
    a = arithmetic
    for lo, hi in (
        (F(-4), F(-399, 100)),
        (F(-2), F(-19, 10)),
        (F(299, 100), F(301, 100)),
        (F(7), F(8)),
    ):
        metrics = n.cell_metrics(a, lo, hi)
        for rational in (lo, (lo + hi) / 2, hi):
            x = a.point(rational)
            data = n.blend(x, a.point)
            contains(a, metrics["output"], data["y"])
            contains(a, metrics["output_error_piecewise"], abs(data["y"] - n.piecewise(x, a.point)))
            for i, stages in enumerate(data["stages"]):
                for j, value in enumerate(stages):
                    contains(a, metrics[f"stage_range_{i}_{j}"], value)


def test_transcendental_enclosures(arithmetic):
    a = arithmetic
    for rational in (F(-8), F(-4), F(-1, 3), F(0), F(1, 3), F(4), F(8)):
        # Higher precision independent erf reference; interval uses Taylor remainder.
        with a.mp.workdps(100):
            x = a.point(rational)
            reference = (1 + a.mp.erf(x / a.mp.sqrt(2))) / 2
            contains(a, a.cdf_endpoint(rational), reference)
    lo, hi = F(-1, 3), F(2, 3)
    intervals = a.targets(a.interval(lo, hi), lo, hi)
    for x in (lo, F(0), hi):
        for enclosure, reference in zip(intervals, a.targets(a.point(x))):
            contains(a, enclosure, reference)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(dps=39),
        dict(dps=201),
        dict(grid=1),
        dict(grid=4098),
        dict(max_cells=9),
        dict(max_cells=4097),
        dict(max_depth=-1),
        dict(max_depth=25),
        dict(max_work=9),
        dict(max_work=8193),
        dict(grid=True),
    ],
)
def test_resource_validation(kwargs):
    with pytest.raises(ValueError):
        n.run(**kwargs)


def test_caps_and_unresolved_no_sample_promotion(arithmetic):
    result = n.enclose_domain(arithmetic, max_cells=10, max_depth=0, max_work=10)
    assert result["leaf_cells"] == result["cell_evaluations"] == 10
    assert result["maximum_leaf_depth"] == 0
    assert result["stage_domain_check"] == "unresolved"
    assert result["unresolved_stage_cells"] > 0
    assert "no error tolerance approved" in result["error_acceptance"]
    assert "sampled" not in result
    json.dumps(result, allow_nan=False)


def test_adaptive_refinement_obeys_all_caps(arithmetic):
    result = n.enclose_domain(arithmetic, max_cells=12, max_depth=1, max_work=14)
    assert result["leaf_cells"] == 12
    assert result["cell_evaluations"] == 14
    assert result["maximum_leaf_depth"] == 1
    assert result["stopped_at"]["cell_cap"]
    assert result["stopped_at"]["work_cap"]


def test_tiny_cli_writes_one_standard_json_result(tmp_path):
    pytest.importorskip("mpmath")
    path = tmp_path / "numeric.json"
    n.main(
        [
            "--output",
            str(path),
            "--dps",
            "40",
            "--grid",
            "2",
            "--max-cells",
            "10",
            "--max-depth",
            "0",
            "--max-work",
            "10",
        ]
    )
    result = json.loads(path.read_text())
    assert list(tmp_path.iterdir()) == [path]
    assert result["difference_scales_exact"] == ["12", "199/20", "11"]
    assert result["sampled"]["status"] == "high_precision_estimates_only_not_uniform_bounds"
    assert len(result["sampled"]["normalized_2^-20_cases"]) == 6
    assert result["enclosed"]["stage_domain_check"] == "unresolved"


def test_cli_requires_output_and_rejects_invalid_args(tmp_path):
    with pytest.raises(SystemExit):
        n.main([])
    path = tmp_path / "unused.json"
    with pytest.raises(SystemExit):
        n.main(["--output", str(path), "--max-work", "0"])
    assert not path.exists()
