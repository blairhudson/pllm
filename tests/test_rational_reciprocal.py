"""Normal tests; resource observations belong to separate pure CLI, not pytest."""
from fractions import Fraction as F

import pytest

from benchmarks.research import rational_reciprocal as shared
from benchmarks.research.reciprocal_exact import encode_report, records, study


@pytest.fixture
def budget(monkeypatch):
    # Pytest/plugins are not the 64-MiB single-process study.
    monkeypatch.setattr(shared.Budget, "check", lambda self: None)
    monkeypatch.setattr(shared, "peak_rss_bytes", lambda: 1 << 20)
    return shared.RationalBudget()


@pytest.mark.parametrize("identity,a,b,x,m,invalid", list(records()))
def test_exact_records(identity, a, b, x, m, invalid, budget):
    if invalid:
        with pytest.raises(shared.InvalidReciprocal):
            shared.reciprocal(a, b, x, m, budget)
        assert budget.work == 0
        return
    y = shared.reciprocal(a, b, x, m, budget)
    power = 2**m
    residual = ((b - x) / b)**power
    expected = (1 - residual) / x
    assert y == expected
    assert abs(1 - x*y) == residual <= ((b-a)/b)**power
    assert abs(1/x-y) == residual/x


def test_campaign_and_duplicate_identity(budget):
    result = study(budget)
    assert result["status"] == "completed"
    assert (result["completed_valid"], result["rejected_invalid"]) == (60, 6)
    assert len({r[0] for r in records()}) == 66
    assert result["state_cells"] == 0 and not result["ckks_qualified"]


@pytest.mark.parametrize("kind", ["bits", "steps", "work", "storage", "output"])
def test_poison_before_arithmetic(kind, budget, monkeypatch):
    a, m = F(1, 2), 2
    if kind == "bits":
        a = F(1, 1 << 20)
    elif kind == "steps":
        m = 9
    elif kind == "work":
        budget.work = shared.MAX_WORK
    elif kind == "storage":
        monkeypatch.setattr(shared, "peak_rss_bytes", lambda: 64 << 20)
    else:
        budget.output = shared.MAX_OUTPUT_BYTES
    with pytest.raises(shared.StudyLimit):
        shared.reciprocal(a, F(1), F(1), m, budget)
    assert budget.poisoned
    with pytest.raises(shared.StudyLimit):
        shared.reciprocal(F(1), F(1), F(1), 0, budget)


def test_exhaustion_no_partial_success(budget):
    budget.limit = 3
    report = study(budget)
    assert report["status"] == "failed" and "records" not in report
    assert report["attempted"] == report["completed_valid"] == 3
    assert budget.poisoned


def test_late_serialization_guard_preserves_counts(budget):
    report = study(budget)
    budget.poisoned = True
    output = encode_report(report, budget)
    assert '"status": "failed"' in output and '"records"' not in output
    assert report["completed_valid"] == 60 and report["rejected_invalid"] == 6


class HookObject:
    def trap(self, *args):
        raise AssertionError("unadmitted Fraction component hook called")

    __abs__ = bit_length = __lt__ = __le__ = __gt__ = __ge__ = trap
    __eq__ = __ne__ = __truediv__ = __rtruediv__ = trap
    __mul__ = __rmul__ = __bool__ = trap


class HookInt(int):
    __abs__ = bit_length = __lt__ = __le__ = __gt__ = __ge__ = HookObject.trap
    __eq__ = __ne__ = __truediv__ = __rtruediv__ = HookObject.trap
    __mul__ = __rmul__ = __bool__ = HookObject.trap


@pytest.mark.parametrize("position", range(3))
@pytest.mark.parametrize("field", ("_numerator", "_denominator"))
@pytest.mark.parametrize("component", (HookObject(), HookInt(1)))
def test_fraction_component_hooks_rejected_before_admission(position, field, component, budget):
    values = [F(1), F(1), F(1)]
    setattr(values[position], field, component)
    with pytest.raises(shared.InvalidReciprocal, match="built-in integer"):
        shared.reciprocal(*values, 1, budget)
    assert budget.work == budget.output == 0


@pytest.mark.parametrize("position", range(3))
@pytest.mark.parametrize("denominator", (0, -1))
def test_malformed_denominator_rejected_before_division(position, denominator, budget):
    values = [F(1), F(1), F(1)]
    values[position]._denominator = denominator
    with pytest.raises(shared.InvalidReciprocal, match="positive Fraction denominators"):
        shared.reciprocal(*values, 1, budget)
    assert budget.work == budget.output == 0
