"""Decoder boundaries reject explicit forward intent before external work."""

from copy import copy
from dataclasses import replace
from types import MappingProxyType
from unittest.mock import Mock

import pytest

from pllm import Deployment, ExecutionBudget, Experiment, ForwardBudget, Model
from pllm.configuration import ConfigurationError
from pllm.compiler import plan, plan_on_load
from pllm.deployment import NetworkSnapshot, NetworkSpec, PartyOffer, open_execution
from pllm.deployment.selection import SelectedClientFactory
from pllm.modeling import lower_model
from pllm.numeric import SourceFloat32
from pllm.plan import PlanningResult
from pllm.profiles import ClientOnlyCpu
from pllm.runtime.benchmark_isolation import run_isolated_loopback_benchmark
from pllm.runtime.reference_benchmark import run_reference_benchmark
from pllm.search import GridSearch, PlanningPolicy, PlanningRequest, SearchSpace


pytestmark = pytest.mark.rust


@pytest.fixture
def decoder_inputs(tmp_path):
    model = Model.path(str(tmp_path / "unopened-checkpoint"))
    deployment = Deployment.local(root=str(tmp_path / "deployment"))
    decoder = Experiment("decoder", ClientOnlyCpu(model), deployment, ExecutionBudget(1, 3, 2))
    config = {
        "model_type": "qwen2", "hidden_size": 8, "intermediate_size": 16,
        "num_hidden_layers": 1, "num_attention_heads": 2, "num_key_value_heads": 1,
        "vocab_size": 32, "max_position_embeddings": 256, "hidden_act": "silu",
        "rms_norm_eps": 1e-6, "rope_theta": 10000.0, "tie_word_embeddings": True,
    }
    model_plan = lower_model(config, batch=1, max_input_tokens=3, max_new_tokens=2)
    policy = PlanningPolicy(evaluated_at_ms=100)
    request = PlanningRequest(model_plan, (decoder,), policy)
    client = PartyOffer("client", "client", "b" * 64, 1000, ("trusted_client",),
                        64 << 20, 64 << 20, 8, 0, ("*",), ("*",))
    snapshot = NetworkSnapshot("admission", (client,), (), "fixture", 0, 1000)
    network = NetworkSpec("admission", snapshot)
    result = plan(request, snapshot=snapshot)
    assert result.status == "feasible"
    return decoder, request, network, result


@pytest.fixture(params=["forward", "budget-bypass", "marker-other-slot"])
def forward(request, decoder_inputs):
    decoder = decoder_inputs[0]
    experiment = Experiment(
        "forward", ClientOnlyCpu(decoder.pipeline.model, numeric=SourceFloat32()),
        decoder.deployment, ForwardBudget(1, 3),
    )
    if request.param != "forward":
        # Preserve genuine public forward composition; simulate bypassed dataclass validation.
        experiment = copy(experiment)
        object.__setattr__(experiment, "budget", decoder.budget)
    if request.param == "marker-other-slot":
        pipeline = copy(experiment.pipeline)
        components = dict(pipeline.components)
        components["unfamiliar_slot"] = components.pop("numeric")
        object.__setattr__(pipeline, "components", MappingProxyType(components))
        object.__setattr__(experiment, "pipeline", pipeline)
    return experiment


def block_effects(monkeypatch):
    """Trap real boundary functions, not stand-in source/admission authorities."""
    paths = (
        "pllm.configuration.Experiment.resolve",
        "pllm.model_loader.resolve_model",
        "pllm.profiles.resolve_runtime_composition",
        "pllm.runtime.reference_benchmark.resolve_runtime_composition",
        "pllm.runtime.reference_benchmark.resolve_model",
        "pllm.runtime.transformer_engine.MaskedTransformerEngine.load",
        "pllm.runtime.transformer_engine.MaskedTransformerEngine._load_tokenizer_descriptor",
        "pllm.runtime.transformer_client.ClientBundle.tokenizer",
        "pllm.runtime.servers.build_roles",
        "pllm.deployment.selection.discover",
        "pllm.deployment.execution.discover",
        "pllm.deployment.execution.control_request",
        "pllm.search.performance.PlanningPrior.from_file",
        "pllm._native.resolve_experiment",
        "pllm._native.validate_network_placement",
        "pllm._native.lower_model",
        "multiprocessing.get_context",
        "subprocess.Popen",
        "httpx.Client.request",
    )
    traps = []
    for path in paths:
        trap = Mock(side_effect=AssertionError(f"unexpected effect: {path}"))
        monkeypatch.setattr(path, trap)
        traps.append(trap)
    return traps


def assert_no_effects(traps):
    for trap in traps:
        trap.assert_not_called()


def test_selected_factory_rejects_before_prior_resolution_or_discovery(
    monkeypatch, decoder_inputs, forward,
):
    network = decoder_inputs[2]
    traps = block_effects(monkeypatch)
    with pytest.raises(ConfigurationError, match="forward-only"):
        SelectedClientFactory(network, experiment=forward, performance_prior="unopened-prior.json")
    assert_no_effects(traps)


def test_selected_factory_prechecks_entire_request(monkeypatch, decoder_inputs, forward):
    decoder, request, network, _ = decoder_inputs
    request = copy(request)
    # Untrusted copied request remains the real public type; never grants execution authority.
    object.__setattr__(request, "candidates", (decoder, forward))
    traps = block_effects(monkeypatch)
    with pytest.raises(ConfigurationError, match="forward-only"):
        SelectedClientFactory(network, request=request)
    assert_no_effects(traps)


def test_open_execution_rejects_before_provisional_resolution_replay_and_native_validation(
    monkeypatch, decoder_inputs, forward,
):
    _, _, network, result = decoder_inputs
    # Copied public planning records retain the immutable original document, not
    # attestation of mutated fields. Deliberately untrusted, inconsistent config
    # must be refused before validation, replay, or side effects.
    original = result
    result = copy(original)
    object.__setattr__(result, "experiment", forward)
    assert type(result) is PlanningResult
    assert result is not original
    assert result.experiment is forward
    assert result.request is original.request
    assert result.snapshot is original.snapshot
    assert result.canonical_bytes() == original.canonical_bytes()
    assert original.experiment is decoder_inputs[0]
    traps = block_effects(monkeypatch)
    for name in ("from_spec", "validate"):
        trap = Mock(side_effect=AssertionError(f"unexpected planning {name}"))
        monkeypatch.setattr(PlanningResult, name, trap)
        traps.append(trap)
    with pytest.raises(ConfigurationError, match="forward-only"):
        open_execution(result, network=network)
    assert_no_effects(traps)


@pytest.mark.parametrize("entrypoint", ["factory", "execution"])
def test_selected_decoder_prechecks_embedded_request_candidates(
    monkeypatch, decoder_inputs, forward, entrypoint,
):
    decoder, request, network, result = decoder_inputs
    request = copy(request)
    object.__setattr__(request, "candidates", (decoder, forward))
    # Copied public planning records retain the immutable original document, not
    # attestation of mutated fields. Deliberately untrusted, inconsistent config
    # must be refused before validation, replay, or side effects.
    original = result
    result = copy(original)
    object.__setattr__(result, "request", request)
    assert type(result) is PlanningResult
    assert result is not original
    assert type(result.request) is PlanningRequest
    assert result.request is request
    assert result.request.candidates == (decoder, forward)
    assert result.experiment is decoder
    assert result.snapshot is original.snapshot
    assert result.canonical_bytes() == original.canonical_bytes()
    assert original.request is decoder_inputs[1]
    assert original.request.candidates == (decoder,)
    traps = block_effects(monkeypatch)
    for owner, name in (
        (PlanningResult, "from_spec"),
        (PlanningResult, "validate"),
        (PlanningRequest, "from_spec"),
    ):
        trap = Mock(side_effect=AssertionError(f"unexpected replay: {owner.__name__}.{name}"))
        monkeypatch.setattr(owner, name, trap)
        traps.append(trap)
    with pytest.raises(ConfigurationError, match="forward-only"):
        if entrypoint == "factory":
            SelectedClientFactory(network, result=result)
        else:
            open_execution(result, network=network)
    assert_no_effects(traps)


def test_plan_on_load_rejects_before_optimization_and_source(monkeypatch, decoder_inputs, forward):
    _, request, network, _ = decoder_inputs
    traps = block_effects(monkeypatch)
    optimization = Mock(side_effect=AssertionError("unexpected optimization"))
    monkeypatch.setattr("pllm.search.optimization_space", optimization)
    traps.append(optimization)
    with pytest.raises(ConfigurationError, match="forward-only"):
        plan_on_load(forward, snapshot=network.snapshot, policy=request.policy)
    assert_no_effects(traps)


def test_plan_on_load_prechecks_expanded_candidate_batch(monkeypatch, decoder_inputs, forward):
    decoder, request, network, _ = decoder_inputs
    space = SearchSpace(decoder, {})
    incumbent = GridSearch("load", space).candidates()[0]
    invalid = replace(incumbent, experiment=forward,
                      configuration_digest=forward.configuration_digest(), index=1)
    # Expanded rows own genuine Experiments, including an otherwise valid first candidate.
    monkeypatch.setattr(GridSearch, "candidates", lambda self: (incumbent, invalid))
    traps = block_effects(monkeypatch)
    with pytest.raises(ConfigurationError, match="forward-only"):
        plan_on_load(decoder, snapshot=network.snapshot, policy=request.policy, space=space)
    assert_no_effects(traps)


def test_isolation_rejects_before_serialization_or_spawn(monkeypatch, forward):
    traps = block_effects(monkeypatch)
    serialization = Mock(side_effect=AssertionError("unexpected serialization"))
    monkeypatch.setattr(Experiment, "to_spec", serialization)
    traps.append(serialization)
    with pytest.raises(ConfigurationError, match="forward-only"):
        run_isolated_loopback_benchmark(experiment=forward)
    assert_no_effects(traps)


@pytest.mark.parametrize("valid_first", [False, True])
def test_reference_prechecks_all_candidates_before_first_source(
    monkeypatch, decoder_inputs, forward, valid_first,
):
    decoder = decoder_inputs[0]
    candidates = [decoder, forward] if valid_first else [forward]
    traps = block_effects(monkeypatch)
    with pytest.raises(ConfigurationError, match="forward-only"):
        run_reference_benchmark(candidates, ["bounded prompt"])
    assert_no_effects(traps)
