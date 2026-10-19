"""Deferred coordinated-wheel gates: generated source, real final acceptance.

target/puma-sdk-integration-env-v1/bin/python -I -m pytest -o pythonpath=. tests/test_quality_outcomes_sdk.py -q
No native acceptance mocks, checkpoint downloads, or memory-guard bypasses.
"""
import json

import pytest

from pllm import _native
from pllm.configuration import Deployment, ExecutionBudget, Experiment
from pllm.profiles import OpenWeightVerifiableCpu
from pllm.runtime.reference_benchmark import run_reference_benchmark
from pllm.runtime.quality_outcomes import validate_report, require_complete_for_ranking, write_report, read_report
from test_replicated_policy_sdk import combined_experiment, observe_publishers, assert_publishers_retired

pytestmark = pytest.mark.rust


def test_generated_shared_puma_and_gkm_retain_same_logical_cohort(tmp_path, monkeypatch):
    from pllm.runtime import servers
    # Must fail on a stale wheel, never infer rejection from generic ValueError.
    assert issubclass(_native.NumericDomainRejected, ValueError)
    puma = combined_experiment(tmp_path / "source", rejecting=True).with_params(
        budget__requests=3, budget__max_new_tokens=1)
    gkm = Experiment("verified-shared-cohort", OpenWeightVerifiableCpu(puma.pipeline.model),
        Deployment.local(root=str(tmp_path / "gkm-roles")),
        ExecutionBudget(requests=3, max_input_tokens=8, max_new_tokens=1))
    owners, topologies = observe_publishers(monkeypatch), []
    original = servers.build_roles
    def roles(*args, **kwargs):
        topology = original(*args, **kwargs)
        topologies.append(topology)
        return topology
    monkeypatch.setattr(servers, "build_roles", roles)
    report = run_reference_benchmark([puma, gkm], ["hello", "world", "hello"],
        outcomes="record", cohort_key=b"q" * 32, cohort_key_id="generated-shared",
        reference_storage="streamed")
    puma_row, verified_row = report["candidates"]
    assert puma_row["coverage"] == {"total": 3, "accepted": 2, "rejected": 1, "failed": 0}
    assert verified_row["coverage"] == {"total": 3, "accepted": 3, "rejected": 0, "failed": 0}
    assert puma_row["top1_agreement"] is None
    assert puma_row["accepted_agreement"]["sample_count"] == 2
    assert puma_row["accepted_agreement"]["top1_agreement"] == 1
    assert [row["position"] for row in puma_row["outcomes"]] == [0, 1, 2]
    assert puma_row["outcomes"][1]["code"] == "numeric_domain_rejected"
    encoded = json.dumps(report, allow_nan=False)
    assert '"hello"' not in encoded and '"world"' not in encoded
    assert "protected range checks" not in encoded and '"selected"' not in encoded
    assert '"logits"' not in encoded and '"token_ids"' not in encoded
    assert_publishers_retired(owners)
    assert len(owners) == 3 and all(owner.finish_calls == 1 for owner in owners)
    assert len(topologies) == 2 and all(not status.running for topology in topologies for status in topology.statuses)
    validate_report(report)
    output = tmp_path / "quality-v2.json"
    write_report(report, output)
    assert read_report(output.read_bytes()) == report
    with pytest.raises(ValueError, match="partial"):
        require_complete_for_ranking(report)


def test_generated_numeric_rejection_still_aborts_default_v1(tmp_path, monkeypatch):
    experiment = combined_experiment(tmp_path / "source", rejecting=True)
    owners = observe_publishers(monkeypatch)
    with pytest.raises(_native.NumericDomainRejected):
        run_reference_benchmark([experiment], ["world"], reference_storage="streamed")
    assert_publishers_retired(owners)


@pytest.mark.parametrize("substitution", ["source", "selection"])
def test_generated_admission_substitution_is_failed_never_numeric_rejection(tmp_path, monkeypatch, substitution):
    from pllm.runtime import replicated_transport as transport, servers
    from pllm.runtime.delegated_protocol import pack, unpack
    assert issubclass(_native.NumericDomainRejected, ValueError)
    experiment = combined_experiment(tmp_path / "source", rejecting=True)
    owners, topologies = observe_publishers(monkeypatch), []
    original_post, original_roles = transport._Peers.post, servers.build_roles
    def post(self, index, action, payload=b"", **kwargs):
        if action == "admit":
            value = unpack(payload, limit=transport.ADMISSION_LIMIT)
            manifest = json.loads(value["manifest"])
            if substitution == "source":
                manifest["source"][0] ^= 1
            else:
                manifest["execution"]["selection"] = "expanded"
            value["manifest"] = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
            payload = pack(value)
        return original_post(self, index, action, payload, **kwargs)
    def roles(*args, **kwargs):
        topology = original_roles(*args, **kwargs)
        topologies.append(topology)
        return topology
    monkeypatch.setattr(transport._Peers, "post", post)
    monkeypatch.setattr(servers, "build_roles", roles)
    report = run_reference_benchmark([experiment], ["hello"], outcomes="record", cohort_key=b"q" * 32,
        cohort_key_id="substitution", reference_storage="streamed")
    assert report["terminal_state"] == "execution_failed"
    assert report["candidates"][0]["coverage"] == {"total": 1, "accepted": 0, "rejected": 0, "failed": 1}
    assert all(item["code"] != "numeric_domain_rejected" for item in report["candidates"][0]["outcomes"])
    assert_publishers_retired(owners)
    assert all(owner.finish_calls == 0 for owner in owners)
    assert topologies and all(not status.running for topology in topologies for status in topology.statuses)
    validate_report(report)


def test_generated_quality_cancellation_drains_before_sanitized_partial_report(tmp_path, monkeypatch):
    from pllm.runtime import replicated_transport as transport, servers
    from pllm.runtime.quality_outcomes import QualityRunCancelled
    assert issubclass(_native.NumericDomainRejected, ValueError)
    experiment = combined_experiment(tmp_path / "source", rejecting=True)
    owners, topologies, interrupted = observe_publishers(monkeypatch), [], []
    original_post, original_roles = transport._Peers.post, servers.build_roles
    def post(self, index, action, payload=b"", **kwargs):
        result = original_post(self, index, action, payload, **kwargs)
        if action == "finish" and not interrupted:
            interrupted.append(True)
            raise KeyboardInterrupt()
        return result
    def roles(*args, **kwargs):
        topology = original_roles(*args, **kwargs)
        topologies.append(topology)
        return topology
    monkeypatch.setattr(transport._Peers, "post", post)
    monkeypatch.setattr(servers, "build_roles", roles)
    with pytest.raises(QualityRunCancelled) as cancelled:
        run_reference_benchmark([experiment], ["hello", "world"], outcomes="record", cohort_key=b"q" * 32,
            cohort_key_id="cancelled", reference_storage="streamed")
    report = cancelled.value.report
    assert interrupted and report["terminal_state"] == "cancelled"
    assert report["candidates"][0]["coverage"] == {"total": 2, "accepted": 0, "rejected": 0, "failed": 2}
    assert [row["attempted"] for row in report["candidates"][0]["outcomes"]] == [True, False]
    assert_publishers_retired(owners)
    assert all(owner.finish_calls == 0 for owner in owners)
    assert topologies and all(not status.running for topology in topologies for status in topology.statuses)
    assert '"world"' not in json.dumps(report) and '"hello"' not in json.dumps(report)
    validate_report(report)
