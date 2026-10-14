"""Evidence admission controls; no timing or native compilation."""
import importlib.util
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

PATH = Path(__file__).resolve().parents[1] / "scripts/benchmark_tier_objectives.py"
SPEC = importlib.util.spec_from_file_location("tier_objectives_benchmark", PATH)
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


@pytest.mark.parametrize("replicates", [True, 0, 4, 1.5])
def test_rejects_unbounded_replicates_before_identity(monkeypatch, replicates):
    monkeypatch.setattr(probe, "identity", lambda *_: pytest.fail("identity before admission"))
    with pytest.raises(ValueError):
        probe.measure("mask", replicates)


def test_worker_rejects_changed_geometry_before_import(monkeypatch):
    monkeypatch.setattr(probe, "load_probe", lambda *_: pytest.fail("allocated before admission"))
    with pytest.raises(ValueError):
        probe.worker({"case": dict(kind="mask", rows=1 << 40), "identity": {}})


@pytest.mark.parametrize("failure", ["identity", "case", "pressure"])
def test_parent_rejects_changed_evidence(monkeypatch, failure):
    monkeypatch.setattr(probe, "identity", lambda *_: {"frozen": True})

    @contextmanager
    def runner(*_, **__):
        def run(payload):
            return {"case": {} if failure == "case" else payload["case"],
                    "identity": {} if failure == "identity" else payload["identity"]}
        guard = SimpleNamespace(check=lambda: None, error="pressure" if failure == "pressure" else None)
        yield run, guard, SimpleNamespace(swap_used=0)

    with pytest.raises(RuntimeError):
        probe.measure("mask", 1, runner=runner)


def test_alternating_fixed_cases_and_unknown_serving_gain(monkeypatch):
    monkeypatch.setattr(probe, "identity", lambda *_: {"frozen": True})
    observed = []

    @contextmanager
    def runner(*_, **kwargs):
        assert kwargs["memory_bytes"] == 384 << 20
        def run(payload):
            observed.append(payload["case"])
            return payload
        guard = SimpleNamespace(check=lambda: None, error=None,
                                maximum_swap_growth_bytes=0, minimum_available_bytes=1 << 30)
        yield run, guard, SimpleNamespace(swap_used=0)

    result = probe.measure("mask", 2, runner=runner)
    plan = probe.cases("mask")
    assert observed == plan + list(reversed(plan))
    assert result["serving_gain"] is None
    assert result["whole_client_tier_qualified"] is False
