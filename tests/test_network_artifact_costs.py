"""Public artifact locality changes cost ordering, never native role admission."""
import asyncio
from dataclasses import replace
import hashlib
import json

import pytest

from pllm import Deployment, ExecutionBudget, Experiment, Model, lower_model
from pllm.compiler import plan
from pllm.deployment import NetworkError
from pllm.model_loader import resolve_model
from pllm.profiles import MaskedLinearCpu
from pllm.protocols import ClientBundleTransport
from pllm.roles import ClientLinearRoles
from pllm.runtime.bundle_artifacts import export_bundle
from pllm.runtime.tiny_llama import create_tiny_llama_checkpoint
from pllm.runtime.transformer_engine import MaskedTransformerEngine
from pllm.search import ArtifactCostEvidence, PlanningPolicy, PlanningRequest
from test_network_planning import snapshot

pytestmark = pytest.mark.rust


@pytest.fixture
def inputs(tmp_path):
    root = create_tiny_llama_checkpoint(tmp_path / "model", hidden_size=128,
        intermediate_size=512, num_hidden_layers=1, num_attention_heads=4,
        num_key_value_heads=2, head_dim=32)
    model = Model.path(str(root), model_id="locality-cost")
    source = resolve_model(model)
    graph = lower_model(json.loads((root / "config.json").read_bytes()), batch=1,
                        max_input_tokens=3, max_new_tokens=2)
    candidates, evidence = [], []
    for roles in ((), ("attention_output", "qkv_projection")):
        pipeline = MaskedLinearCpu(model, delivery=ClientBundleTransport("artifacts"),
            **({"placement": ClientLinearRoles(roles)} if roles else {}))
        experiment = Experiment("attention" if roles else "baseline", pipeline,
            Deployment.local(root="local://locality"), ExecutionBudget(1, 3, 2))
        engine = MaskedTransformerEngine(threads=1, weight_bits=8, activation_bits=8,
                                         client_linear_roles=roles)
        asyncio.run(engine.load(source.manifest))
        payload = engine.client_bundle("locality-cost")
        artifacts = export_bundle(payload)
        evidence.append(ArtifactCostEvidence.from_manifest(experiment, graph, source.source_lock_digest,
            artifacts.manifest, bundle_digest=hashlib.sha256(payload).hexdigest(), bundle_bytes=len(payload)))
        candidates.append(experiment)
    return PlanningRequest(graph, tuple(candidates), PlanningPolicy("client", 100, 2, 1000,
        ("horizon_accounted_body_bytes",), minimum_remote_mac_fraction=0.5),
        source.source_lock_digest, artifact_evidence=tuple(evidence))


def test_horizon_amortizes_public_objects_but_never_fresh_preparation(inputs):
    cold = plan(inputs, snapshot=snapshot())
    warm_request = replace(inputs, policy=replace(inputs.policy, reuse_horizon=128))
    warm = plan(warm_request, snapshot=snapshot())
    assert cold.status == warm.status == "feasible"
    assert cold.experiment.name == "baseline"
    assert warm.experiment.name == "attention"
    for result, horizon in ((cold, 1), (warm, 128)):
        costs = result.to_spec()["selection"]["costs"]
        assert costs["horizon_accounted_body_bytes"] == costs["artifact_miss_bytes"] + horizon * (
            costs["artifact_manifest_bytes"] + costs["total_arithmetic_body_bytes"])
        assert costs["preprocessing_all_link_body_bytes"] > 0
        assert costs["full_wire_bytes"] is None
        assert PlanningRequest.from_spec(result.request.to_spec()) == result.request


def test_residency_deduplicates_public_payload_but_manifest_still_charged(inputs):
    evidence = tuple(replace(item, resident_keys=tuple(key for key, _ in item.objects))
                     for item in inputs.artifact_evidence)
    request = replace(inputs, artifact_evidence=evidence)
    result = plan(request, snapshot=snapshot())
    costs = result.to_spec()["selection"]["costs"]
    assert result.status == "feasible" and result.experiment.name == "attention"
    assert costs["artifact_miss_bytes"] == 0
    assert costs["horizon_accounted_body_bytes"] > costs["total_arithmetic_body_bytes"]
    assert costs["role_weight_estimates"]["client"] > 0


def test_missing_artifact_evidence_unknown_and_wrong_context_rejected(inputs):
    result = plan(replace(inputs, artifact_evidence=()), snapshot=snapshot())
    assert result.status == "infeasible"
    assert any("UNKNOWN_REQUIRED_COST" in row["reason"] for row in result.to_spec()["rejections"])
    original = inputs.artifact_evidence[0]
    for overrides in ({"source_lock_digest": "0" * 64}, {"model_plan_digest": "e" * 64},
                      {"configuration_digest": "f" * 64}):
        with pytest.raises(NetworkError):
            replace(inputs, artifact_evidence=(replace(original, **overrides),))
    with pytest.raises(NetworkError):
        replace(original, resident_keys=("a" * 64,))
    with pytest.raises(NetworkError):
        replace(inputs.policy, reuse_horizon=True)
    with pytest.raises(NetworkError):
        replace(original, origin="measured")
