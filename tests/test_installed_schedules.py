"""Advertised exact schedules select and execute through separate HTTP role children."""

from dataclasses import replace
import json
import time

import pytest

from pllm import Deployment, ExecutionBudget, Model, OpenAI
from pllm.deployment import ClientResources, LivePartyOffer, NetworkError, PartySpec, discover
from pllm.deployment.admission import ExecutionContract
from pllm.deployment.resources import _capacity
from pllm.deployment.selection import SelectedClientFactory
from pllm.model_loader import resolve_model
from pllm.modeling import lower_model
from pllm.profiles import MaskedLinearCpu
from pllm.runtime.benchmark_memory import estimate_plan_memory
from pllm.runtime.tiny_llama import create_tiny_llama_checkpoint
from pllm.schedulers import MlpPrefillTiling
from pllm.search import PlanningPolicy
from test_network_execution import http_hosts, setup  # noqa: F401

pytestmark = pytest.mark.rust


@pytest.fixture
def installed(setup, tmp_path):
    network, credentials, specs, _, _ = setup
    checkpoint = create_tiny_llama_checkpoint(tmp_path / "choice-checkpoint", hidden_size=64,
        intermediate_size=4096, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, head_dim=16)
    model = Model.path(str(checkpoint), model_id="schedule-choices")
    budget = ExecutionBudget(1, 64, 2)
    base = replace(specs[0].experiment, pipeline=MaskedLinearCpu(model), budget=budget)
    tiled = base.pipeline.with_params(scheduler=MlpPrefillTiling(tile_rows=16))
    source = resolve_model(model).source_lock_digest
    specs = tuple(replace(spec, experiment=base, source_lock_digest=source,
        role_ids=(role,), alternatives=(tiled,), max_sessions=2)
        for spec, role in zip(specs, ("inference", "preparation"), strict=True))
    graph = lower_model(json.loads((checkpoint / "config.json").read_bytes()),
                        batch=1, max_input_tokens=64, max_new_tokens=2)
    return network, credentials, specs, graph, source


def test_installed_records_reject_incompatible_or_duplicate_choices(installed):
    spec = installed[2][0]
    assert spec.to_spec()["schema"] == "pllm.party_spec.v2"
    assert PartySpec.from_spec(spec.to_spec()) == spec
    legacy = replace(spec, alternatives=())
    assert legacy.to_spec()["schema"] == "pllm.party_spec.v1"
    assert PartySpec.from_spec(legacy.to_spec()) == legacy
    with pytest.raises(NetworkError, match="duplicate"):
        replace(spec, alternatives=(spec.experiment.pipeline,))
    from pllm.quantization import SymmetricPerRow

    with pytest.raises(NetworkError, match="only supported client scheduling"):
        replace(spec, alternatives=(spec.alternatives[0].with_params(quantization=SymmetricPerRow(
            weight_bits=4, activation_bits=8)),))
    forged = spec.to_spec()
    forged["schema"] = "pllm.party_spec.v1"
    with pytest.raises(NetworkError, match="require party spec v2"):
        PartySpec.from_spec(forged)


@pytest.mark.integration
def test_automatic_memory_choice_replans_and_executes_without_private_admission(installed, tmp_path, monkeypatch):
    network, credentials, specs, graph, _ = installed
    full = specs[0].experiment
    tile = specs[0].alternatives[0]
    reserve = 512 << 20
    process = 128 << 20
    sizes = [estimate_plan_memory(graph, pipeline, max_input_tokens=64,
        max_output_tokens=2)["client_peak_bytes"] + (1 << 20) for pipeline in (full.pipeline, tile)]
    assert sizes[1] < sizes[0]
    available = [reserve + sizes[0] + (1 << 20)]
    monkeypatch.setattr(_capacity, "observe", lambda: ClientResources(time.time_ns() // 1_000_000,
        available[0], process, reserve, sum(_capacity._owners.values())))
    for name, secret in credentials.items():
        monkeypatch.setenv(name, secret)
    with http_hosts(installed, tmp_path) as (_, _, children):
        offered = discover(network, credentials=credentials)
        for offer in offered.offers:
            if isinstance(offer, LivePartyOffer):
                assert len(offer.configurations) == 2 and offer.workload == full.budget
                assert LivePartyOffer.from_spec(offer.to_spec()) == offer
        experiment = replace(full, deployment=Deployment.network(
            network_id=network.network_id, network_spec_digest=network.digest))
        with OpenAI(network=network, experiment=experiment) as client:
            outputs = []
            for free_bytes, expected in ((reserve + sizes[0] + (1 << 20), full.pipeline),
                                         (reserve + sizes[1], tile),
                                         (reserve + sizes[0] + (1 << 20), full.pipeline)):
                available[0] = free_bytes
                response = client.responses.create(input="abcde " * 4, max_output_tokens=2, temperature=0)
                assert response.usage.input_tokens > 16
                outputs.append(response.output_text)
                decision = client.planning(response.id)
                assert decision.experiment.pipeline == expected
                assert decision.request.client_resources.process_memory_bytes == process
                public = ExecutionContract.from_result(decision, decision.snapshot).canonical_bytes()
                for private in (b"client_resources", b"process_memory_bytes", b"reserved_memory_bytes",
                                b"planning_policy", b'"candidates"', b'"rejections"'):
                    assert private not in public
                assert not _capacity._owners
                assert all(offer.active_sessions == 0 for offer in discover(network, credentials=credentials).offers)
            assert len(set(outputs)) == 1
            audit = client.privacy_audit.to_dict()
            assert audit["plaintext_prompt_bytes_sent"] == audit["plaintext_token_ids_sent"] == 0
            available[0] = reserve + sizes[1] - 1
            with pytest.raises(NetworkError, match="NO_FEASIBLE"):
                client.responses.create(input="Hi", max_output_tokens=2)
            assert not _capacity._owners
        assert all(child.poll() is None for child in children)
        # Pinning removes alternatives even when they would fit; fresh discovery
        # cannot silently override an explicit composition requirement.
        available[0] = reserve + sizes[1]
        factory = SelectedClientFactory(network, experiment=experiment,
            policy=PlanningPolicy(candidate_selection="pinned"), credentials=credentials)
        with pytest.raises(NetworkError, match="NO_FEASIBLE"):
            factory.select()
        assert all(offer.active_sessions == 0 for offer in discover(network, credentials=credentials).offers)
