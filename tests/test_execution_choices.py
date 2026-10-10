"""Fine-grained planning, exact installed combinations and real lease execution."""
from dataclasses import replace
import time

import httpx
import pytest

from pllm import Deployment, OpenAI
from pllm.compiler import plan, plan_on_load
from pllm.deployment import ClientResources, NetworkError, PartySpec, discover, open_execution
from pllm.deployment.resources import _capacity
from pllm.deployment.selection import SelectedClientFactory
from pllm.preparation import ModelAwareCorrections, PreparedInventory
from pllm.protocols import ClientBundleTransport, MaskedLinear, TwoOnlineOffsetLinear
from pllm.quantization import SymmetricPerRow
from pllm.schedulers import MlpPrefillTiling
from pllm.search import GridSearch, optimization_space
from pllm.search.optimizations import compatible_execution_choices
from test_load_time_optimization import loaded  # noqa: F401
from test_network_execution import http_hosts, setup  # noqa: F401
from test_network_prepared import prepared  # noqa: F401

pytestmark = pytest.mark.rust


def test_explicit_component_axes_preserve_interactions_and_admit_each_candidate(loaded):
    _, base, snapshot, policy = loaded
    axes = {
        "linear": (MaskedLinear(request_encoding="stage_packed", output_encoding="row_residues"),),
        "delivery": (ClientBundleTransport("artifacts", storage="paged"),),
        "inventory": (PreparedInventory("request-sized", rows=1, refill="on-demand", allocation="demand"),),
        "scheduler": (MlpPrefillTiling(tile_rows=16),),
    }
    space = optimization_space(base, component_choices=axes, max_candidates=16)
    rows = GridSearch("fine", space).candidates()
    assert len(rows) == 16
    assert base.pipeline in [row.experiment.pipeline for row in rows]
    assert all(compatible_execution_choices(base.pipeline, row.experiment.pipeline) for row in rows)
    result = plan_on_load(base, snapshot=snapshot, policy=policy, space=space)
    assert result.status == "feasible" and result.exhaustive
    assert result.experiment.pipeline.components["linear"].params["request_encoding"] == "stage_packed"
    ordinary = plan(replace(result.request, candidates=(base,)), snapshot=snapshot)
    assert result.costs["online_all_link_body_bytes"] < ordinary.costs["online_all_link_body_bytes"]
    with pytest.raises(ValueError, match="bound"):
        optimization_space(base, component_choices=axes, max_candidates=15)
    with pytest.raises(ValueError, match="numeric"):
        optimization_space(base, component_choices={"quantization": (SymmetricPerRow(weight_bits=4),)})


def test_prepared_geometry_prices_seed_direction_demand_and_terminal_rows(loaded):
    _, base, snapshot, policy = loaded
    variants = (base, replace(base, pipeline=base.pipeline.with_params(
        linear=MaskedLinear(prefill_pruning="terminal"), inventory=PreparedInventory(
            "request-sized", rows=1, allocation="demand", refill="on-demand"))))
    results = [plan_on_load(candidate, snapshot=snapshot, policy=replace(policy,
        candidate_selection="pinned")) for candidate in variants]
    assert all(row.status == "feasible" for row in results)
    full, lean = [row.costs for row in results]
    assert lean["online_all_link_body_bytes"] < full["online_all_link_body_bytes"]
    assert lean["preprocessing_all_link_body_bytes"] < full["preprocessing_all_link_body_bytes"]
    assert lean["role_linear_macs"]["preparation"] < full["role_linear_macs"]["preparation"]
    links = results[1].costs["role_links"]
    assert not any(row["source_role"] == "preparation" and row["target_role"] == "client" for row in links)
    assert sum(row["bytes"] for row in links if row["source_role"] == "client"
               and row["target_role"] == "preparation") == 32 * 8  # four remote groups per layer


def test_zero_hysteresis_uses_secondary_objectives_but_positive_margin_holds(loaded):
    _, base, snapshot, policy = loaded
    space = optimization_space(base, component_choices={"linear": (
        MaskedLinear(request_encoding="stage_packed", output_encoding="row_residues"),)})
    policy = replace(policy, objectives=("prefill_dependent_exchanges", "online_all_link_body_bytes"))
    selected = plan_on_load(base, snapshot=snapshot, policy=policy, space=space)
    assert selected.experiment.pipeline != base.pipeline
    held = plan_on_load(base, snapshot=snapshot, policy=replace(policy,
        minimum_switch_improvement_fraction=0.05,
        incumbent_configuration_digest=base.configuration_digest()), space=space)
    assert held.experiment.pipeline == base.pipeline


def test_host_alternatives_preserve_loaded_owners(prepared):
    spec = prepared[2][0]
    pipeline = spec.experiment.pipeline.with_params(
        delivery=ClientBundleTransport("artifacts", storage="paged", batch_objects=64),
        inventory=PreparedInventory("request-sized", rows=1, stage_window=2, refill="on-demand"),
        linear=MaskedLinear(prefill_chunk_rows=4, request_encoding="compact"))
    promoted = replace(spec, alternatives=(pipeline,))
    assert PartySpec.from_spec(promoted.to_spec()) == promoted
    # An Inference owner does not allocate Preparation's weight storage. A
    # Preparation host must actually preload its advertised storage contract.
    paged = pipeline.with_params(preparation=ModelAwareCorrections(storage="paged"))
    assert replace(spec, alternatives=(paged,)).alternatives == (paged,)
    with pytest.raises(NetworkError, match="shared engine"):
        replace(prepared[2][1], alternatives=(paged,))
    with pytest.raises(NetworkError, match="shared engine"):
        replace(spec, alternatives=(pipeline.with_params(linear=MaskedLinear(output_encoding="row_residues")),))


@pytest.mark.integration
def test_automatic_prepared_selects_paging_inventory_and_codec_together(prepared, tmp_path, monkeypatch):
    network, credentials, specs, graph, lock = prepared
    full = specs[0].experiment
    lean = full.pipeline.with_params(
        delivery=ClientBundleTransport("artifacts", storage="paged", batch_objects=64),
        inventory=PreparedInventory("request-sized", rows=1, refill="on-demand", allocation="demand", stage_window=2),
        linear=MaskedLinear(request_encoding="compact", prefill_chunk_rows=4))
    offered = (network, credentials, tuple(replace(spec, alternatives=(lean,)) for spec in specs), graph, lock)
    for name, secret in credentials.items():
        monkeypatch.setenv(name, secret)
    monkeypatch.setenv("PLLM_CLIENT_BUNDLE_CACHE_DIR", str(tmp_path / "bundles"))
    monkeypatch.setattr(_capacity, "observe", lambda: ClientResources(
        time.time_ns() // 1_000_000, 8 << 30, 128 << 20, 512 << 20))
    experiment = replace(full, deployment=Deployment.network(
        network_id=network.network_id, network_spec_digest=network.digest))
    with http_hosts(offered, tmp_path) as (_, _, children):
        outputs = []
        for policy, expected in ((SelectedClientFactory.default_policy(candidate_selection="pinned"), full.pipeline),
                                 (None, lean)):
            with OpenAI(network=network, experiment=experiment, planning=policy) as client:
                response = client.responses.create(input="Hi", max_output_tokens=2, temperature=0)
                outputs.append(response.output_text)
                chosen = client.planning(response.id).experiment.pipeline
                assert chosen.to_spec() == expected.to_spec(), "\n".join(
                    row["reason"] for row in client.planning(response.id).to_spec()["rejections"])
                assert client.privacy_audit.to_dict()["plaintext_token_ids_sent"] == 0
            assert not _capacity._owners
            assert all(row.active_sessions == 0 for row in discover(network, credentials=credentials).offers)
        assert outputs[0] == outputs[1]
        assert all(child.poll() is None for child in children)


@pytest.mark.integration
def test_installed_residue_layout_and_paged_preparation_are_selected_and_bound(prepared, tmp_path, monkeypatch):
    network, credentials, specs, graph, lock = prepared
    base = specs[0].experiment
    packed = base.pipeline.with_params(
        preparation=ModelAwareCorrections(storage="paged"),
        linear=MaskedLinear(request_encoding="stage_packed", output_encoding="row_residues"),
        delivery=ClientBundleTransport("artifacts", storage="paged", compression="zlib"),
        inventory=PreparedInventory("request-sized", rows=1, allocation="demand", refill="on-demand"))
    configured = (network, credentials, tuple(replace(spec,
        experiment=replace(spec.experiment, pipeline=packed)) for spec in specs), graph, lock)
    for name, secret in credentials.items():
        monkeypatch.setenv(name, secret)
    monkeypatch.setattr(_capacity, "observe", lambda: ClientResources(
        time.time_ns() // 1_000_000, 8 << 30, 128 << 20, 512 << 20))
    experiment = replace(base, deployment=Deployment.network(
        network_id=network.network_id, network_spec_digest=network.digest))
    with http_hosts(configured, tmp_path):
        factory = SelectedClientFactory(network, experiment=experiment)
        result = factory.select()
        assert result.experiment.pipeline.to_spec() == packed.to_spec()
        with open_execution(result, network=network, credentials=credentials) as lease:
            url, bearer = lease._topology._connections["inference"]
            wrong = httpx.post(url + "/v1/runtime/sessions",
                headers={"authorization": f"Bearer {bearer}"}, json={
                    "model": experiment.resolve().model, "decoder_plan": {"max_input_tokens": 1},
                    "max_output_tokens": 2, "prepared_request_encoding": "raw"})
            assert wrong.status_code == 409
            assert wrong.json() == {"error": "lease protocol rejected"}
            with OpenAI(execution=lease) as client:
                response = client.responses.create(input="Hi", max_output_tokens=2, temperature=0)
                assert response.usage.output_tokens == 2
                assert client.privacy_audit.to_dict()["plaintext_token_ids_sent"] == 0
        assert all(row.active_sessions == 0 for row in discover(network, credentials=credentials).offers)


@pytest.mark.integration
def test_offset_installed_codec_choice_binds_its_own_session(setup, tmp_path, monkeypatch):
    network, credentials, specs, graph, lock = setup
    full = specs[0].experiment
    lean = full.pipeline.with_params(linear=TwoOnlineOffsetLinear(
        input_encoding="seeded", output_encoding="row_residues", dispatch="seed_first"),
        delivery=ClientBundleTransport("artifacts", compression="zlib", batch_objects=64))
    configured = (network, credentials, tuple(replace(spec, alternatives=(lean,)) for spec in specs), graph, lock)
    for name, secret in credentials.items():
        monkeypatch.setenv(name, secret)
    monkeypatch.setattr(_capacity, "observe", lambda: ClientResources(
        time.time_ns() // 1_000_000, 8 << 30, 128 << 20, 512 << 20))
    experiment = replace(full, deployment=Deployment.network(
        network_id=network.network_id, network_spec_digest=network.digest))
    with http_hosts(configured, tmp_path):
        with OpenAI(network=network, experiment=experiment) as client:
            response = client.responses.create(input="Hi", max_output_tokens=2, temperature=0)
            assert client.planning(response.id).experiment.pipeline.to_spec() == lean.to_spec(), "\n".join(
                row["reason"] for row in client.planning(response.id).to_spec()["rejections"])
            assert response.usage.output_tokens == 2
        assert all(row.active_sessions == 0 for row in discover(network, credentials=credentials).offers)
