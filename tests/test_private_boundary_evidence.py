"""Historical boundary evidence stays source-bound and cannot qualify partial RAM."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from scripts.benchmark_coverage import external_private_boundary
from scripts.summarize_private_boundary_sdk import summarize

ROOT = Path(__file__).resolve().parents[1]


def load(tokens):
    raw = (ROOT / f"docs/evidence/private-boundary-qwen-{tokens}-sdk.json").read_bytes()
    control = (ROOT / "docs/evidence/private-boundary-qwen-control.json").read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest(), json.loads(control), hashlib.sha256(control).hexdigest()


@pytest.mark.parametrize("tokens", [48, 150])
def test_summaries_replay_original_cohorts_without_loading_models(tokens, monkeypatch):
    from benchmarks.research.private_boundary import recorded_contract
    import pllm.model_loader

    def forbidden(*args, **kwargs):
        raise AssertionError("read-only evidence must not load model sources")

    monkeypatch.setattr(pllm.model_loader, "resolve_model", forbidden)
    document, source_hash, control, control_hash = load(tokens)
    summary = summarize(document, source_hash, control, control_hash)
    assert summary == json.loads((ROOT / f"docs/evidence/private-boundary-qwen-{tokens}-sdk.summary.json").read_bytes())
    for variant, group in summary["groups"].items():
        assert group["sample_count"] == 3
        assert group["maximum_client_subprocess_peak_rss_bytes"] > 0
        assert group["whole_client_peak_rss_bytes"] is None
        assert group["whole_client_memory_tier_qualified"] is False
        contract = recorded_contract(f"private-boundary-qwen-{tokens}-sdk", variant)
        assert contract["experiment"] == group["experiment"]
        assert contract["requires_original_measurement_driver"] is True
    linked = external_private_boundary(document, source_hash, None, control, control_hash)
    assert linked["status"] == "validated_historical"
    assert linked["configurations"]["coupled"]["client_samples"] == 3
    assert external_private_boundary(document, source_hash, "f" * 64, control, control_hash)["status"] == "stale_build"
    assert external_private_boundary(document, source_hash, document["build"]["expected_digest"],
                                     control, control_hash)["status"] == "validated_current"


@pytest.mark.parametrize("change", [
    lambda d: d.update(complete=False),
    lambda d: d.update(tiny=True),
    lambda d: d["results"].pop(),
    lambda d: d["attempts"].pop(),
    lambda d: d["results"][1].update(replicate=1),
    lambda d: d["memory_admission"]["48/coupled"]["topology"]["estimate"].update(pipeline_digest="0" * 64),
    lambda d: d["results"][0]["wan"].update(client_stream_bytes=0),
    lambda d: d["results"][0]["wan"]["role_parties"].update(boundary_c="boundary_c"),
    lambda d: d["results"][0]["client"]["responses"][0]["trajectory"][0].update(logits_and_kv="0" * 64),
    lambda d: d["results"][0]["client"]["audit"].update(correction_ack_bytes=0),
    lambda d: d["results"][1]["client"]["cleanup"].update(boundary_transports_retired_and_owned_http_closed=False),
    lambda d: d["swap"].update(maximum_growth_bytes=1),
    lambda d: d["verification"].update(maximum_client_rss_across_replicates_bytes=1),
    lambda d: d["driver_implementation"].update(other="0" * 64),
    lambda d: d["sampler_fixture"].update(diagnostic_cost_charged_equally=False),
])
def test_changed_or_incomplete_boundary_evidence_cannot_be_promoted(change):
    document, source_hash, control, control_hash = load(48)
    changed = deepcopy(document)
    change(changed)
    with pytest.raises((ValueError, RuntimeError)):
        summarize(changed, source_hash, control, control_hash)


def test_changed_control_cannot_substitute_for_measured_pipeline():
    document, source_hash, control, control_hash = load(48)
    control["pipeline"]["components"]["kernels"]["params"]["threads"] = 2
    with pytest.raises(ValueError, match="native admission"):
        summarize(document, source_hash, control, control_hash)


def test_installed_boundary_components_have_ordinary_response_targets():
    from examples.benchmarks.private_boundary import suite
    from pllm.metrics import benchmark_capabilities

    selected = suite()
    assert all(case.build.expected_digest is not None for case in selected.cases)
    assert len({case.build.expected_digest for case in selected.cases}) == 1
    targets = [case.target.to_spec() for case in selected.cases]
    capabilities = {item.name: item for item in benchmark_capabilities()}
    for identity in ("pllm/private-lookup-full-head/v1", "pllm/prepared-body-two-online-boundary/v1"):
        assert not capabilities[identity].matches(targets[0])
        assert capabilities[identity].matches(targets[1])
