"""Compiler-admitted load-time choices, resource limits and no speculative tuning."""

from __future__ import annotations

import json
import time
from dataclasses import replace

import pytest

from pllm import Deployment, ExecutionBudget, Experiment, Model, OpenAI
from pllm.compiler import CompilationError, plan, plan_on_load
from pllm.deployment import NetworkSnapshot, NetworkSpec, PartyOffer, open_execution
from pllm.kernels import AppleMetal, Cpu
from pllm.plan import PlanningResult
from pllm.profiles import MaskedLinearCpu, TwoOnlineOffsetCpu, VerifiedMaskedLinearCpu
from pllm.quantization import SymmetricPerRow
from pllm.runtime.tiny_llama import create_tiny_llama_checkpoint
from pllm.search import GridSearch, PlanningPolicy, SearchSpace, optimization_space

pytestmark = pytest.mark.rust


@pytest.fixture
def loaded(tmp_path):
    root = create_tiny_llama_checkpoint(
        tmp_path / "model",
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
    )
    model = Model.path(str(root), model_id="automatic-source")
    experiment = Experiment(
        "automatic",
        MaskedLinearCpu(
            model,
            kernels=Cpu(threads=1),
            quantization=SymmetricPerRow(causal_reduction="prefix_f32"),
        ),
        Deployment.local(root="local://automatic"),
        ExecutionBudget(1, 32, 2),
    )
    now = time.time_ns() // 1_000_000
    offers = tuple(
        PartyOffer(
            party,
            party,
            "b" * 64,
            now + 300_000,
            ("trusted_client",)
            if party == "client"
            else ("trusted_preparation", "masked_linear_provider", "public_linear_provider"),
            1 << 30,
            1 << 30,
            4,
            0,
            ("*",),
            ("*",),
            devices=("cpu", "metal"),
        )
        for party in ("client", "a", "b")
    )
    snapshot = NetworkSnapshot("automatic", offers, (), "fixture", now, now + 300_000)
    policy = PlanningPolicy(
        "client",
        now,
        64,
        1024,
        ("online_all_link_body_bytes",),
        minimum_remote_mac_fraction=0.5,
    )
    return root, experiment, snapshot, policy


def test_automatic_space_is_bounded_and_preserves_privacy_numerics(loaded):
    _, base, _, _ = loaded
    space = optimization_space(
        base, allow_client_weights=True, prefix_cache_bytes=1 << 20, metal_min_rows=2
    )
    rows = GridSearch("automatic", space).candidates()
    assert 1 < len(rows) <= 64
    for row in rows:
        assert row.experiment.pipeline.model == base.pipeline.model
        assert row.experiment.budget == base.budget
        for key in ("quantization", "linear", "preparation", "inference"):
            assert row.experiment.pipeline.components[key] == base.pipeline.components[key]
        row.experiment.resolve()
    assert any(
        "placement" in row.experiment.pipeline.components
        and row.experiment.pipeline.components["kernels"].component
        == AppleMetal.descriptor.component
        for row in rows
    )
    with pytest.raises(ValueError, match="bound"):
        optimization_space(base, allow_client_weights=True, metal_min_rows=2, max_candidates=1)
    for row in GridSearch("conservative", optimization_space(base)).candidates():
        assert "placement" not in row.experiment.pipeline.components
        assert "cache" not in row.experiment.pipeline.components


def test_preserves_verifier_and_two_worker_topology(loaded):
    _, base, _, _ = loaded
    verified = replace(base, pipeline=VerifiedMaskedLinearCpu(base.pipeline.model))
    for row in GridSearch("verified", optimization_space(verified, metal_min_rows=2)).candidates():
        assert (
            row.experiment.pipeline.components["verification"]
            == verified.pipeline.components["verification"]
        )
        assert "cache" not in row.experiment.pipeline.components
    offset = replace(base, pipeline=TwoOnlineOffsetCpu(base.pipeline.model))
    rows = GridSearch("offset", optimization_space(offset)).candidates()
    assert len(rows) == 16
    for row in rows:
        assert row.experiment.resolve().client_runtime == "compiled_offset_v1"


def test_proposals_include_cross_capability_combinations_and_require_weight_permission(loaded):
    _, base, _, _ = loaded
    space = optimization_space(base, allow_client_weights=True, client_prefix_layers=(1,),
                               prefix_cache_bytes=1 << 20, metal_min_rows=2, max_candidates=128)
    rows = GridSearch("unions", space).candidates()
    assert len(rows) == 128
    assert any(row.experiment.resolve().client_prefix_layers == 1
               and row.experiment.resolve().client_linear_roles
               and row.experiment.resolve().bundle_compression == "artifacts-zlib"
               and row.experiment.resolve().prefix_cache_bytes
               and row.experiment.pipeline.components["kernels"] == AppleMetal(min_rows=2)
               for row in rows)
    with pytest.raises(ValueError, match="permission"):
        optimization_space(base, client_prefix_layers=(1,))
    for factory in (VerifiedMaskedLinearCpu, TwoOnlineOffsetCpu):
        selected = replace(base, pipeline=factory(base.pipeline.model,
            quantization=SymmetricPerRow(causal_reduction="prefix_f32")))
        rows = GridSearch("combined", optimization_space(selected, prefix_cache_bytes=1 << 20)).candidates()
        assert any(row.experiment.resolve().prefix_cache_bytes
                   and row.experiment.resolve().bundle_compression == "artifacts-zlib" for row in rows)


def test_offset_costing_prices_seeded_and_bounded_residue_transport(loaded):
    _, base, snapshot, policy = loaded
    offset = replace(base, pipeline=TwoOnlineOffsetCpu(base.pipeline.model,
        quantization=SymmetricPerRow(causal_reduction="prefix_f32")))
    selected = plan_on_load(offset, snapshot=snapshot, policy=policy)
    assert selected.status == "feasible"
    assert selected.experiment.pipeline.components["linear"].params == {
        "input_encoding": "seeded", "output_encoding": "row_residues"}
    assert "delivery" not in selected.experiment.pipeline.components
    ordinary = plan(replace(selected.request, candidates=(offset,)), snapshot=snapshot)
    assert selected.costs["online_all_link_body_bytes"] < ordinary.costs["online_all_link_body_bytes"]


def test_load_selects_priced_ownership_not_unpriced_metal_or_transport(loaded):
    _, base, snapshot, policy = loaded
    space = optimization_space(
        base, allow_client_weights=True, prefix_cache_bytes=1 << 20, metal_min_rows=2
    )
    result = plan_on_load(base, snapshot=snapshot, policy=policy, space=space)
    assert result.status == "feasible" and result.exhaustive
    assert result.experiment.resolve().client_linear_roles
    # Equal byte estimates cannot imply faster GPU/compression/cache behavior.
    selected = result.experiment.pipeline.components
    assert selected["kernels"] == base.pipeline.components["kernels"]
    assert not {"inventory", "delivery", "cache"} & selected.keys()
    assert result.request.source_lock_digest is not None
    assert result.costs["origin"] == "estimate"
    assert PlanningResult.from_spec(result.to_spec()).digest == result.digest
    ordinary = plan(
        replace(
            result.request,
            candidates=(base,),
            policy=replace(
                result.request.policy, incumbent_configuration_digest=base.configuration_digest()
            ),
        ),
        snapshot=snapshot,
    )
    assert ordinary.costs["online_all_link_body_bytes"] > result.costs["online_all_link_body_bytes"]
    limited = replace(policy, max_client_weight_bytes=ordinary.costs["client_weight_bytes"])
    constrained = plan_on_load(base, snapshot=snapshot, policy=limited, space=space)
    assert constrained.experiment == base


def test_unknown_cpu_and_cache_memory_are_not_free(loaded):
    _, base, snapshot, policy = loaded
    result = plan_on_load(base, snapshot=snapshot, policy=replace(policy, max_client_cpu_ns=1))
    assert result.status == "infeasible"
    assert any("UNKNOWN_REQUIRED_COST" in row["reason"] for row in result.to_spec()["rejections"])
    cached = next(
        row.experiment
        for row in GridSearch(
            "cache", optimization_space(base, prefix_cache_bytes=1 << 20)
        ).candidates()
        if "cache" in row.experiment.pipeline.components
    )
    cold = plan_on_load(base, snapshot=snapshot, policy=policy)
    candidate = plan(
        replace(
            cold.request,
            candidates=(cached,),
            policy=replace(cold.request.policy, incumbent_configuration_digest=None),
        ),
        snapshot=snapshot,
    )
    assert candidate.costs["role_memory_estimates"]["client"] == cold.costs[
        "role_memory_estimates"
    ]["client"] + (1 << 20)


def test_source_and_contract_changes_fail_before_planning(loaded, monkeypatch):
    root, base, snapshot, policy = loaded
    altered = base.pipeline.with_params(
        quantization=SymmetricPerRow(weight_bits=4, activation_bits=4)
    )
    with pytest.raises(ValueError, match="numeric or privacy"):
        plan_on_load(
            base,
            snapshot=snapshot,
            policy=policy,
            space=SearchSpace(base, {"pipeline": (altered,)}),
        )
    from pllm import model_loader

    original = model_loader.resolve_model

    def changed(*args, **kwargs):
        resolved = original(*args, **kwargs)
        config = json.loads((root / "config.json").read_bytes())
        config["rope_theta"] = 12000
        (root / "config.json").write_text(json.dumps(config))
        return resolved

    monkeypatch.setattr(model_loader, "resolve_model", changed)
    with pytest.raises(CompilationError, match="changed after"):
        plan_on_load(base, snapshot=snapshot, policy=policy)


@pytest.mark.integration
def test_selected_load_plan_runs_through_existing_role_and_sdk_path(loaded):
    _, base, snapshot, policy = loaded
    selected = plan_on_load(
        base,
        snapshot=snapshot,
        policy=policy,
        space=optimization_space(base, allow_client_weights=True, metal_min_rows=2),
    )
    with open_execution(selected, network=NetworkSpec("automatic", snapshot)) as execution:
        with OpenAI(execution=execution) as client:
            response = client.responses.create(input="Hi", max_output_tokens=2, temperature=0.0)
            assert response.usage.output_tokens == 2
            assert client.privacy_audit.inference_stage_calls > 0
            assert client.privacy_audit.plaintext_prompt_bytes_sent == 0
            assert client.privacy_audit.plaintext_token_ids_sent == 0


def test_load_time_never_reintroduces_an_excluded_incumbent(loaded, monkeypatch):
    _, base, snapshot, policy = loaded
    from pllm import model_loader

    def forbidden(*args, **kwargs):
        raise AssertionError("constraints must be checked before source resolution")

    monkeypatch.setattr(model_loader, "resolve_model", forbidden)
    space = SearchSpace(base, {"pipeline__kernels__threads": (2,)})
    with pytest.raises(ValueError, match="constraints are never bypassed"):
        plan_on_load(base, snapshot=snapshot, policy=policy, space=space)
