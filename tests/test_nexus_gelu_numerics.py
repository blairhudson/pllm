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


@pytest.mark.parametrize("index", range(3))
def test_one_sided_threshold_oracle_and_closed_cell_enclosure(arithmetic, index):
    a = arithmetic
    t = n.THRESHOLDS[index]
    epsilon = F(1, 2**24)
    left = (F(0), n.polynomial(t - epsilon, n.P), n.polynomial(t - epsilon, n.Q))[index]
    right = (n.polynomial(t + epsilon, n.P), n.polynomial(t + epsilon, n.Q), t + epsilon)[index]
    assert n.piecewise(t - epsilon) == left
    assert n.piecewise(t + epsilon) == right
    for lo, hi in ((t - epsilon, t), (t, t), (t, t + epsilon)):
        metrics = n.cell_metrics(a, lo, hi)
        for rational in (lo, hi):
            x = a.point(rational)
            result = n.blend(x, a.point)
            contains(a, metrics["output_error_piecewise"], abs(result["y"] - n.piecewise(x, a.point)))
            for i, value in enumerate(result["b"]):
                contains(a, metrics[f"selector_{i}"], value)


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
    for t, d in zip(n.THRESHOLDS, n.SCALES):
        # Original-unit 2^-20 is not normalized separation 2^-20.
        assert (t + n.BAND - t) / d != n.BAND
        assert (t + d * n.BAND - t) / d == n.BAND
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
@pytest.mark.parametrize("multiple", [1, 2])
def test_signed_band_point_encloses_independent_exact_composition(arithmetic, direction, multiple):
    def f(x):
        return (315 * x - 420 * x**3 + 378 * x**5 - 180 * x**7 + 35 * x**9) / 128

    def g(x):
        return (5850 * x - 34974 * x**3 + 97015 * x**5 - 113492 * x**7 + 46623 * x**9) / 1024

    z = F(direction * multiple, 2**20)
    expected = f(f(g(g(z))))
    a = arithmetic
    encoded = a.encode(n.sign_stages(a.exact(z), a.exact)[-1])

    def decode(bound):
        return F(int(bound["mantissa"])) * F(2) ** bound["exponent2"]

    assert decode(encoded["lower"]) <= expected <= decode(encoded["upper"])
    assert F(999, 1000) < abs(expected - direction) < 1


def test_strict_outside_band_and_exact_jump_blockers(arithmetic):
    a = arithmetic
    result = n.numeric_blockers(a)
    witnesses = result["sign_point_witnesses"]
    assert len(witnesses) == 4

    def decode(bound):
        return F(int(bound["mantissa"])) * F(2) ** bound["exponent2"]

    for witness in witnesses:
        z = F(witness["z_exact"])
        assert witness["strictly_outside_normalized_band"] == (abs(z) > n.BAND)
        assert F(999, 1000) < decode(witness["sign_error_enclosure"]["lower"])
        assert decode(witness["sign_error_enclosure"]["upper"]) < 1
        assert decode(witness["comparator_error_enclosure"]["lower"]) > F(499, 1000)
        for t, d, x in zip(n.THRESHOLDS, n.SCALES, witness["x_by_threshold_exact"]):
            assert (F(x) - t) / d == z
            assert n.DOMAIN[0] <= F(x) <= n.DOMAIN[1]
    assert sum(w["strictly_outside_normalized_band"] for w in witnesses) == 2
    epsilon = F(1, 2**24)
    for jump in result["oracle_jumps"]:
        t = F(jump["threshold_exact"])
        left, right = map(F, (jump["oracle_value_and_left_limit_exact"], jump["oracle_right_limit_exact"]))
        assert n.piecewise(t) == left
        assert abs(n.piecewise(t + epsilon) - right) < F(1, 10**6)
        assert F(jump["jump_exact"]) == right - left
        assert F(jump["continuous_approximation_uniform_error_floor_exact"]) == abs(right - left) / 2
    floor = F(result["continuous_oracle_uniform_error_floor_exact"])
    assert F("0.0079") < floor < F("0.0080")


def test_negative_selector_witnesses_are_not_roundoff(arithmetic):
    a = arithmetic
    for witness in n.numeric_blockers(a)["negative_selector_point_witnesses"]:
        x, index = F(witness["x_exact"]), witness["selector_index"]
        data = n.blend(a.exact(x), a.exact)
        lo, hi = a.ends(data["b"][index])
        assert lo <= hi < a.point(F(-1, 10**12))
        contains(a, sum(data["b"]), a.point(1))
        # No individual comparator needs to overshoot for adjacent differences to do so.
        for h in data["h"]:
            hlo, hhi = a.ends(h)
            assert 0 <= hlo <= hhi <= 1


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
    assert len(result["numeric_blockers"]["sign_point_witnesses"]) == 4
    assert result["enclosed"]["stage_domain_check"] == "unresolved"


def test_cli_requires_output_and_rejects_invalid_args(tmp_path):
    with pytest.raises(SystemExit):
        n.main([])
    path = tmp_path / "unused.json"
    with pytest.raises(SystemExit):
        n.main(["--output", str(path), "--max-work", "0"])
    assert not path.exists()


@pytest.mark.parametrize("schedule", [(2, 2), (6, 2), (10, 2), (10, 3)])
def test_approved_schedules_enclose_independent_high_precision_powers(arithmetic, schedule):
    a = arithmetic
    for z in (F(-1, 2**20), F(1, 2**20), F(1, 16), F(1)):
        with a.mp.workdps(100):
            value = a.point(z)
            expected = []
            for stage in range(sum(schedule)):
                x = value
                if stage < schedule[0]:
                    value = (5850*x - 34974*x**3 + 97015*x**5 - 113492*x**7 + 46623*x**9) / 1024
                else:
                    value = (315*x - 420*x**3 + 378*x**5 - 180*x**7 + 35*x**9) / 128
                expected.append(value)
        stages = n.sign_stages(a.exact(z), a.exact, schedule)
        assert len(stages) == sum(schedule)
        for enclosure, point in zip(stages, expected):
            contains(a, enclosure, point)
    with pytest.raises(ValueError, match="four approved"):
        n.sign_stages(F(0), schedule=(3, 3))


def test_follow_up_sample_sets_and_group_units(arithmetic):
    normalized = n.normalized_sample_points()
    assert len(normalized) <= 307
    assert len(n.sample_points(257)) == 381
    assert set(n.SEPARATIONS).issubset(normalized)
    assert {-2*n.BAND, 2*n.BAND}.issubset(normalized)
    report = n.sampled_report(arithmetic, 2, n.BASELINE, n.SEPARATIONS)
    for epsilon in n.SEPARATIONS:
        groups = report["separated_and_within_band"]
        assert groups[f"separated_{epsilon}"]["count"] + groups[f"within_band_{epsilon}"]["count"] == report["count"]
        for t, d in zip(n.THRESHOLDS, n.SCALES):
            assert n.separated_cell(t+d*epsilon, t+d*epsilon, epsilon)
            assert not n.separated_cell(t, t, epsilon)


def test_arithmetic_cost_matches_executed_horner_dag():
    class Node:
        counts = {"constant_products": 0, "variable_products": 0,
                  "additions_and_subtractions": 0, "divisions_by_public_constants": 0}

        def __init__(self, constant=False, depth=0):
            self.constant, self.depth = constant, depth

        def __mul__(self, other):
            other = other if isinstance(other, Node) else Node(True)
            key = "constant_products" if self.constant or other.constant else "variable_products"
            self.counts[key] += 1
            return Node(self.constant and other.constant, max(self.depth, other.depth)+1)

        def __add__(self, other):
            other = other if isinstance(other, Node) else Node(True)
            self.counts["additions_and_subtractions"] += 1
            return Node(self.constant and other.constant, max(self.depth, other.depth))

        __sub__ = __add__
        __rsub__ = __add__

        def __truediv__(self, other):
            assert not isinstance(other, Node) or other.constant
            self.counts["divisions_by_public_constants"] += 1
            return Node(self.constant, self.depth)

    for schedule in n.FOLLOW_UP_SCHEDULES:
        Node.counts = dict.fromkeys(Node.counts, 0)
        result = n.blend(Node(), lambda _: Node(True), schedule=schedule)
        cost = n.arithmetic_cost(schedule)["one_blend"]
        for name, count in Node.counts.items():
            assert cost[name] == count
        assert cost["multiplication_dependency_depth"] == result["y"].depth


def test_follow_up_interval_work_is_charged_before_evaluation(arithmetic):
    a = arithmetic
    attempted = []

    def spend():
        attempted.append(1)

    def evaluate(lo, hi):
        assert len(attempted) > 0
        return {"error": a.interval(0, hi-lo)}

    leaves, work = n.adaptive_partition(
        a, [(F(-1), F(1))], evaluate, "error", 3, 1, 5, spend)
    assert work == len(attempted) == 3
    assert len(leaves) == 2
    assert all(cell[4] <= 1 for cell in leaves)
    with pytest.raises(ValueError, match="initial partition"):
        n.adaptive_partition(a, [(F(0), F(1))]*3, evaluate, "error", 2, 1, 5)
    with pytest.raises(n.StudyLimit):
        n.adaptive_partition(a, [(F(0), F(1))], evaluate, "error", 2, 1, 5,
                             lambda: (_ for _ in ()).throw(n.StudyLimit("cap")))


def test_hard_wall_alarm_interrupts_and_restores_handler():
    import signal
    import time

    before = signal.getsignal(signal.SIGALRM)
    with pytest.raises(n.StudyLimit, match="wall-time"):
        with n.wall_limit(0.02):
            time.sleep(1)
    assert signal.getsignal(signal.SIGALRM) == before
    assert signal.getitimer(signal.ITIMER_REAL)[0] == 0
    with pytest.raises(n.StudyLimit):
        with n.wall_limit(0.02):
            with n.wall_limit(1):
                time.sleep(1)
    assert signal.getsignal(signal.SIGALRM) == before
    assert signal.getitimer(signal.ITIMER_REAL)[0] == 0


def test_follow_up_incomplete_is_not_success_and_cli_rejects_budget_changes(monkeypatch, tmp_path):
    def timeout(*_args):
        raise n.StudyLimit("test candidate time cap")

    monkeypatch.setattr(n, "sign_sampled_report", timeout)
    result = n.run_follow_up()
    assert result["status"] == "incomplete"
    assert len(result["candidates"]) == 4
    assert all(c["status"] == "incomplete" for c in result["candidates"])
    assert result["attempted_cell_evaluations"] == 0
    assert result["continuous_oracle_uniform_error_floor_exact"] == str(F("0.00793176970057680015"))
    path = tmp_path / "follow-up.json"
    with pytest.raises(SystemExit) as exc:
        n.main(["--follow-up", "--output", str(path)])
    assert exc.value.code == 1
    assert json.loads(path.read_text())["status"] == "incomplete"
    with pytest.raises(SystemExit):
        n.main(["--follow-up", "--dps", "200", "--output", str(path)])


def test_nonfinite_interval_cannot_serialize_as_finite_binary_endpoint(arithmetic):
    with pytest.raises(ArithmeticError, match="nonfinite"):
        arithmetic.encode(arithmetic.iv.mpf([0, "inf"]))


def test_follow_up_completed_work_does_not_certify_unresolved_intervals(monkeypatch):
    monkeypatch.setattr(n, "sign_sampled_report", lambda *_: {"count": 0})
    monkeypatch.setattr(n, "sampled_report", lambda *_: {"count": 0})

    def enclosure(_a, _schedule, _epsilon, spend):
        spend()
        return {"error_acceptance": "unresolved", "cell_evaluations": 1}

    monkeypatch.setattr(n, "follow_up_enclosure", enclosure)
    result = n.run_follow_up()
    assert result["status"] == "completed_bounded_study_not_certified"
    assert result["attempted_cell_evaluations"] == 16
    assert all(c["gelu_enclosed"]["error_acceptance"] == "unresolved"
               for c in result["candidates"])


def test_full_domain_separated_cell_selection_keeps_boundaries_exact(arithmetic, monkeypatch):
    # Exercise real initial cuts/classification without re-running the full study.
    a = arithmetic
    original = n.adaptive_partition

    def initial_only(a, intervals, evaluate, priority, _cells, _depth, _work, spend):
        return original(a, intervals, evaluate, priority, 64, 0, 128, spend)

    monkeypatch.setattr(n, "adaptive_partition", initial_only)
    attempted = []
    report = n.follow_up_enclosure(a, n.BASELINE, None, lambda: attempted.append(1))
    assert report["cell_evaluations"] == len(attempted) == 22
    assert report["maximum_leaf_depth"] == 0
    for epsilon in n.SEPARATIONS:
        assert report["separated_subsets"][str(epsilon)]["selected_cells"] > 0
        assert report["separated_subsets"][str(epsilon)]["error_acceptance"].startswith("unresolved")


def test_total_deadline_writes_only_small_incomplete_report(monkeypatch, tmp_path):
    def total_timeout(result):
        result.update(candidates=[{"schedule": [2, 2], "status": "incomplete"}],
                      attempted_cell_evaluations=7, large_partial_payload="x"*10000)
        raise n.StudyLimit("test total deadline")

    monkeypatch.setattr(n, "run_follow_up", total_timeout)
    output = tmp_path / "timeout.json"
    with pytest.raises(SystemExit) as exc:
        n.main(["--follow-up", "--output", str(output)])
    assert exc.value.code == 1
    result = json.loads(output.read_text())
    assert result["status"] == "incomplete"
    assert result["attempted_cell_evaluations"] == 7
    assert "large_partial_payload" not in result


def test_shared_study_counter_never_exceeds_total_cap(monkeypatch):
    monkeypatch.setattr(n, "sign_sampled_report", lambda *_: {"count": 0})
    monkeypatch.setattr(n, "sampled_report", lambda *_: {"count": 0})

    def excessive_work(_a, _schedule, _epsilon, spend):
        # Synthetic counter operations only, not numerical cell evaluations.
        for _ in range(700):
            spend()
        return {}

    monkeypatch.setattr(n, "follow_up_enclosure", excessive_work)
    result = n.run_follow_up()
    assert result["status"] == "incomplete"
    assert result["attempted_cell_evaluations"] == 2048
    assert all(c["status"] == "incomplete" for c in result["candidates"])
