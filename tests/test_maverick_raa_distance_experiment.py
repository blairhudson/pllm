"""Independent exact-distance experiment tests; authored source, not yet run.

Small finite-field enumerations are test oracles, not security claims. CLI tests
replace evaluation so neither official million-term curve runs here.
"""

from collections import Counter, defaultdict
from dataclasses import FrozenInstanceError
from fractions import Fraction
import importlib.util
from itertools import permutations, product
import json
from math import comb
from pathlib import Path
import sys

import pytest


@pytest.fixture
def experiment():
    path = Path(__file__).resolve().parents[1] / "scripts/maverick_raa_distance_experiment.py"
    spec = importlib.util.spec_from_file_location("maverick_raa_distance_under_test", path)
    module = importlib.util.module_from_spec(spec)
    # Dataclasses resolve annotations through the defining module registry.
    previous = sys.modules.get(spec.name)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        if previous is None:
            sys.modules.pop(spec.name, None)
        else:
            sys.modules[spec.name] = previous


def weight(vector):
    return sum(value != 0 for value in vector)


def accumulate(vector, q):
    total = 0
    output = []
    for value in vector:
        total = (total + value) % q
        output.append(total)
    return tuple(output)


def enumerated_counts(n, q):
    counts = Counter()
    for vector in product(range(q), repeat=n):
        counts[weight(vector), weight(accumulate(vector, q))] += 1
    return counts


def state_counts(n, q):
    """Track input/output weights and whether the previous sum is zero.

    From a nonzero sum, one nonzero increment cancels it; q-2 do not.
    This recurrence is independent of any positive-run counting formula.
    """
    states = {(0, 0, False): 1}
    for _ in range(n):
        following = defaultdict(int)
        for (h, l, nonzero), count in states.items():
            following[h, l + nonzero, nonzero] += count
            if nonzero:
                following[h + 1, l, False] += count
                following[h + 1, l + 1, True] += count * (q - 2)
            else:
                following[h + 1, l + 1, True] += count * (q - 1)
        states = following
    counts = Counter()
    for (h, l, _), count in states.items():
        counts[h, l] += count
    return counts


def schedule_events(profile, maximum_d):
    """Literal work events: new intermediate coefficients, then output assembly."""
    built_l = 0
    for d in range(1, maximum_d + 1):
        for l in range(built_l + 1, min(profile.n, 2 * d) + 1):
            for w in range(1, min(profile.k, 2 * l // profile.r) + 1):
                for term in range(2 * l):
                    yield d, "intermediate_run", l, w, term
                yield d, "intermediate_assembly", l, w, 0
        built_l = min(profile.n, 2 * d)
        for l in range(1, min(profile.n, 2 * d) + 1):
            for term in range(2 * d):
                yield d, "output_run", l, 0, term
            yield d, "output_assembly", l, 0, 0


def ensemble_coefficients(profile):
    """Enumerate actual independent stage permutations and nonzero scales.

    Count canonical projective message classes, not probability of their union.
    Only tiny prime fields are used, so modular addition is field addition.
    """
    assert profile.n <= 3 and profile.q in (2, 3)
    transforms = tuple(product(permutations(range(profile.n)),
                               product(range(1, profile.q), repeat=profile.n)))
    counts = Counter()
    for message in product(range(profile.q), repeat=profile.k):
        first = next((value for value in message if value), None)
        if first != 1:
            continue
        repeated = tuple(value for value in message for _ in range(profile.r))
        for permutation1, scales1 in transforms:
            middle = accumulate(tuple(repeated[i] * scale % profile.q
                                      for i, scale in zip(permutation1, scales1)), profile.q)
            for permutation2, scales2 in transforms:
                output = accumulate(tuple(middle[i] * scale % profile.q
                                          for i, scale in zip(permutation2, scales2)), profile.q)
                counts[weight(output)] += 1
    denominator = len(transforms) ** 2
    return {d: Fraction(counts[d], denominator) for d in range(1, profile.n + 1)}


@pytest.mark.parametrize("q,maximum_n", [(2, 5), (3, 5), (5, 4)])
def test_accumulator_against_all_vectors(experiment, q, maximum_n):
    for n in range(maximum_n + 1):
        expected = enumerated_counts(n, q)
        for h in range(n + 1):
            row = 0
            denominator = comb(n, h) * (q - 1) ** h
            for l in range(n + 1):
                count = experiment.accumulator_count(n, h, l, q)
                assert count == expected[h, l], (n, h, l, q)
                assert type(count) is int
                probability = experiment.transition(n, h, l, q)
                assert isinstance(probability, Fraction)
                assert probability == Fraction(count, denominator)
                if h > 2 * l:
                    assert count == 0
                row += count
                assert experiment.transition_tail(n, h, l, q) == Fraction(row, denominator)
            assert row == denominator


@pytest.mark.parametrize("n,q", [(16, 2), (16, 3), (32, 5)])
def test_accumulator_against_two_state_dp(experiment, n, q):
    expected = state_counts(n, q)
    for h in range(n + 1):
        assert sum(expected[h, l] for l in range(n + 1)) == comb(n, h) * (q - 1) ** h
        for l in range(n + 1):
            assert experiment.accumulator_count(n, h, l, q) == expected[h, l]


def test_binary_zero_exponent_and_inclusive_zero_tail(experiment):
    assert experiment.accumulator_count(0, 0, 0, 2) == 1
    assert experiment.transition_tail(0, 0, 0, 2) == Fraction(1)
    assert experiment.accumulator_count(2, 2, 1, 2) == 1
    assert experiment.transition_tail(2, 2, 0, 2) == 0
    assert experiment.transition_tail(2, 2, 1, 2) == 1
    assert experiment.transition_tail(2, 0, 0, 2) == 1


@pytest.mark.parametrize("k,r,q", [(1, 1, 2), (2, 1, 3), (1, 2, 3), (3, 1, 2), (1, 3, 3)])
def test_prefixes_against_actual_tiny_ensemble(experiment, k, r, q):
    profile = experiment.Profile(k, r, q)
    coefficients = ensemble_coefficients(profile)
    result = experiment.profile_evaluate(profile, max_d=profile.n)
    cumulative = Fraction(0)
    expected = []
    for d in range(1, profile.n + 1):
        cumulative += coefficients[d]
        expected.append((d, coefficients[d], cumulative, min(Fraction(1), cumulative)))
        if cumulative >= 1:
            break
    assert [(p.d, p.coefficient, p.cumulative, p.bound) for p in result.prefixes] == expected
    assert result.stop_reason == ("saturation" if cumulative >= 1 else "limit")
    assert result.charged_terms == len(tuple(schedule_events(profile, expected[-1][0])))


def test_public_records_and_official_profiles(experiment):
    assert experiment.MAX_TERMS == 1_000_000
    assert [(p.k, p.r, p.q, p.n) for p in experiment.OFFICIAL_PROFILES] == [
        (64, 8, 2013265921, 512), (128, 8, 2013265921, 1024)]
    profile = experiment.Profile(1, 1, 2)
    prefix = experiment.Prefix(1, Fraction(1), Fraction(1), Fraction(1))
    evaluation = experiment.Evaluation(profile, (prefix,), 6, "saturation")
    for record, field, value in [(profile, "k", 2), (prefix, "d", 2),
                                 (evaluation, "charged_terms", 0)]:
        with pytest.raises(FrozenInstanceError):
            setattr(record, field, value)


def test_profile_validation(experiment):
    for index in range(3):
        for invalid in (True, False, 1.0, "2", None, Fraction(2)):
            args = [1, 1, 2]
            args[index] = invalid
            with pytest.raises(TypeError):
                experiment.Profile(*args)
    for args in [(0, 1, 2), (1, 0, 2), (1, 1, 1), (-1, 1, 2)]:
        with pytest.raises(ValueError):
            experiment.Profile(*args)


def test_accumulator_helper_validation(experiment):
    for function in (experiment.accumulator_count, experiment.transition, experiment.transition_tail):
        for index in range(4):
            for invalid in (True, False, 2.0, "2", None, Fraction(2)):
                args = [2, 1, 1, 3]
                args[index] = invalid
                with pytest.raises(TypeError):
                    function(*args)
        for args in [(-1, 0, 0, 2), (2, -1, 1, 2), (2, 3, 1, 2),
                     (2, 1, -1, 2), (2, 1, 3, 2), (2, 1, 1, 1)]:
            with pytest.raises(ValueError):
                function(*args)


def test_schedule_and_evaluator_validation(experiment):
    profile = experiment.Profile(2, 2, 3)
    for invalid in (True, False, 1.0, "1", Fraction(1)):
        with pytest.raises(TypeError):
            experiment.schedule_terms(profile, invalid)
        with pytest.raises(TypeError):
            experiment.profile_evaluate(profile, max_d=invalid)
    with pytest.raises(TypeError):
        experiment.schedule_terms(profile, None)
    for invalid in (-1, profile.n + 1):
        with pytest.raises(ValueError):
            experiment.schedule_terms(profile, invalid)
        with pytest.raises(ValueError):
            experiment.profile_evaluate(profile, max_d=invalid)
    for invalid in (True, False, 1.0, "1", None, Fraction(1)):
        with pytest.raises(TypeError):
            experiment.profile_evaluate(profile, term_cap=invalid)
    for invalid in (-1, experiment.MAX_TERMS + 1):
        with pytest.raises(ValueError):
            experiment.profile_evaluate(profile, term_cap=invalid)
    assert issubclass(experiment.BudgetExceeded, Exception)


@pytest.mark.parametrize("k,r,q", [(1, 1, 2), (2, 2, 3), (3, 2, 5)])
def test_schedule_is_literal_work_event_count(experiment, k, r, q):
    profile = experiment.Profile(k, r, q)
    for d in range(profile.n + 1):
        assert experiment.schedule_terms(profile, d) == len(tuple(schedule_events(profile, d)))


def test_every_budget_edge_retains_work_but_publishes_only_complete_prefixes(experiment):
    profile = experiment.Profile(2, 2, 3)
    full = experiment.profile_evaluate(profile, max_d=profile.n)
    assert full.prefixes
    costs = {p.d: len(tuple(schedule_events(profile, p.d))) for p in full.prefixes}
    final_cost = costs[full.prefixes[-1].d]
    for cap in range(final_cost + 1):
        result = experiment.profile_evaluate(profile, term_cap=cap, max_d=profile.n)
        expected = tuple(p for p in full.prefixes if costs[p.d] <= cap)
        assert result.prefixes == expected, cap
        # Refusal must not restart accounting or expose a partial coefficient.
        assert result.charged_terms == cap
        assert result.stop_reason == (full.stop_reason if cap == final_cost else "budget")


def test_stop_priority_and_empty_limits(experiment):
    profile = experiment.Profile(1, 1, 2)
    for cap in (0, 1, 5):
        result = experiment.profile_evaluate(profile, term_cap=cap)
        assert result.prefixes == ()
        assert result.charged_terms == cap
        assert result.stop_reason == "budget"
    result = experiment.profile_evaluate(profile, term_cap=6, max_d=1)
    assert len(result.prefixes) == 1
    assert result.charged_terms == 6
    assert result.stop_reason == "saturation"
    for cap in (0, 6, experiment.MAX_TERMS):
        result = experiment.profile_evaluate(profile, term_cap=cap, max_d=0)
        assert result.prefixes == ()
        assert result.charged_terms == 0
        assert result.stop_reason == "limit"


def test_official_first_two_coefficients_and_exact_budget_boundary(experiment):
    for profile in experiment.OFFICIAL_PROFILES:
        assert experiment.schedule_terms(profile, 2) == 35
        result = experiment.profile_evaluate(profile, term_cap=35, max_d=2)
        assert result.stop_reason == "limit"
        assert result.charged_terms == 35
        assert [p.d for p in result.prefixes] == [1, 2]
        first, second = result.prefixes
        assert first.coefficient == first.cumulative == first.bound == 0
        expected = Fraction(420 * profile.k,
                            comb(profile.n, 4) * comb(profile.n, 2) * (profile.q - 1) ** 6)
        assert second.coefficient == second.cumulative == second.bound == expected
        budget = experiment.profile_evaluate(profile, term_cap=35)
        assert budget.prefixes == result.prefixes
        assert budget.charged_terms == 35
        assert budget.stop_reason == "budget"
        partial = experiment.profile_evaluate(profile, term_cap=34, max_d=2)
        assert partial.prefixes == (first,)
        assert partial.charged_terms == 34
        assert partial.stop_reason == "budget"


def decoded_fraction(value):
    assert set(value) == {"numerator_hex", "denominator_hex"}
    assert all(isinstance(component, str) for component in value.values())
    numerator = int(value["numerator_hex"], 16)
    denominator = int(value["denominator_hex"], 16)
    return Fraction(numerator, denominator)


def test_json_exact_hex_round_trip_without_decimal_limit_changes(experiment):
    profile = experiment.Profile(1, 1, 2)
    huge = Fraction((1 << 20000) + 1, (1 << 19999) + 3)
    prefix = experiment.Prefix(1, huge, huge, Fraction(1))
    result = experiment.Evaluation(profile, (prefix,), 6, "saturation")
    before = sys.get_int_max_str_digits()
    encoded = experiment.evaluation_to_dict(result)
    recovered = json.loads(json.dumps(encoded))
    assert sys.get_int_max_str_digits() == before
    assert set(recovered) == {"profile", "prefixes", "charged_terms", "stop_reason", "last_complete_d"}
    assert recovered["profile"] == {"k": 1, "r": 1, "q": 2, "n": 1}
    assert recovered["charged_terms"] == 6
    assert recovered["stop_reason"] == "saturation"
    assert recovered["last_complete_d"] == 1
    assert len(recovered["prefixes"]) == 1
    serialized = recovered["prefixes"][0]
    assert set(serialized) == {"d", "coefficient", "cumulative", "bound"}
    assert serialized["d"] == 1
    for field in ("coefficient", "cumulative", "bound"):
        assert decoded_fraction(serialized[field]) == getattr(prefix, field)
    empty = experiment.evaluation_to_dict(experiment.Evaluation(profile, (), 0, "budget"))
    assert empty["prefixes"] == []
    assert empty["last_complete_d"] == 0


@pytest.mark.parametrize("argv,indices", [([], (0, 1)), (["--profile", "all"], (0, 1)),
                                          (["--profile", "512"], (0,)),
                                          (["--profile", "1024"], (1,))])
def test_cli_dispatch_without_real_evaluation(experiment, monkeypatch, capsys, argv, indices):
    calls = []

    def fake_evaluate(profile, **kwargs):
        calls.append((profile, kwargs))
        return experiment.Evaluation(profile, (), 0, "budget")

    monkeypatch.setattr(experiment, "profile_evaluate", fake_evaluate)
    assert experiment.main(argv) == 0
    output = capsys.readouterr()
    assert output.err == ""
    document = json.loads(output.out)
    assert set(document) == {"scope", "evaluations"}
    assert document["scope"] == "conditional_ideal_raa_ensemble"
    selected = [experiment.OFFICIAL_PROFILES[index] for index in indices]
    assert [profile for profile, _ in calls] == selected
    for _, kwargs in calls:
        assert set(kwargs) <= {"term_cap", "max_d"}
        assert kwargs.get("term_cap", experiment.MAX_TERMS) == experiment.MAX_TERMS
        assert kwargs.get("max_d") is None
    assert document["evaluations"] == [experiment.evaluation_to_dict(
        experiment.Evaluation(profile, (), 0, "budget")) for profile in selected]


@pytest.mark.parametrize("argv", [["--profile", "256"], ["--term-cap", "0"], ["--max-d", "0"]])
def test_cli_rejects_noncontract_options_without_evaluating(experiment, monkeypatch, capsys, argv):
    def unexpected_evaluation(*args, **kwargs):
        pytest.fail("invalid CLI invoked evaluation")

    monkeypatch.setattr(experiment, "profile_evaluate", unexpected_evaluation)
    with pytest.raises(SystemExit) as error:
        experiment.main(argv)
    assert error.value.code == 2
    assert capsys.readouterr().out == ""
