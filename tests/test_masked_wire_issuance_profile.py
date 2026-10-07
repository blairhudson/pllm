"""Keep measured primitive costs tied to checked outputs and complete payloads."""
import copy
import json
from pathlib import Path

import pytest


@pytest.fixture
def profile(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root))
    from benchmarks.research.masked_wire_issuance import validate

    report = json.loads((root / "docs/evidence/research-masked-wire-issuance.json").read_text())
    return report, validate


def test_retained_issuance_profile_checks_outputs_and_all_material(profile):
    report, validate = profile
    validate(report)
    assert report["checked_integer_outputs"] == 1926
    assert "instrumentation" in report["timing_scope"]
    assert "no secret bytes retained" in report["material_scope"]


@pytest.mark.parametrize("corruption", ["output", "material", "peer_frame", "duplicate_case", "double_count"])
def test_issuance_profile_rejects_incomplete_or_miscounted_evidence(profile, corruption):
    original, validate = profile
    report = copy.deepcopy(original)
    first = report["cases"][0]
    if corruption == "output":
        first["complete"]["output_sha256"] = "0" * 64
    elif corruption == "material":
        first["microprofile"]["components"].pop("product")
    elif corruption == "peer_frame":
        first["complete"]["phase_frame_bytes"][0][0] -= 1
    elif corruption == "duplicate_case":
        report["cases"][-1] = copy.deepcopy(first)
    else:
        first["complete"]["export_ns"] = first["complete"]["wall_ns"]
    with pytest.raises(AssertionError):
        validate(report)


def test_whole_response_dispatch_factories_replay_saved_observations(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root))
    from benchmarks.research.paper_cohorts import extract

    path = root / "docs/evidence/research-nonlinear-dispatch-qwen25.json"
    replay = extract(path, [f"benchmarks.research.nonlinear_dispatch:{name}"
                            for name in ("serial", "overlap")])
    assert replay["checks"]["passed"]
    assert replay["candidates"] == json.loads(path.read_text())["candidates"]
    outputs = set()
    for candidate in replay["candidates"]:
        result = candidate["report"]
        assert result["configuration"]["client_process_isolated"]
        assert result["memory_guard"]["maximum_host_swap_growth_bytes"] == 0
        run, = result["runs"]
        assert run["status"] == "completed"
        assert run["tokens"]["input_tokens"] == 30
        assert run["tokens"]["output_tokens"] == 2
        outputs.add(run["generation"]["output_text_digest"])
    assert len(outputs) == 1
