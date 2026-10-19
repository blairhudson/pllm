"""Canonical public PUMA execution choices; no checkpoint/native execution."""
import json
from types import SimpleNamespace

import pytest

from pllm import Model
from pllm.configuration import ComponentRef, Deployment, ExecutionBudget, Experiment, Pipeline
from pllm.profiles import PumaThreePartyCpu, resolve_runtime_composition
from pllm.protocols import PumaThreePartyDecoder
from pllm.runtime import benchmark_memory, replicated_source
from pllm.runtime.replicated_protocol import (
    REQUEST_MEMORY, STREAM_MEMORY, execution_binding, execution_policy, worker_request_memory,
)


def experiment_for(**choices):
    return Experiment("policy-test", PumaThreePartyCpu(Model.path("/not-loaded/generated-source"),
        linear=PumaThreePartyDecoder(**choices)), Deployment.local(root="/not-started/roles"),
        ExecutionBudget(requests=2, max_input_tokens=4, max_new_tokens=2))


def test_legacy_defaults_keep_params_and_digests():
    for weights in ("private", "public"):
        original = experiment_for(weights=weights)
        explicit = experiment_for(weights=weights, range_policy="static", max_abs_activation=256,
                                  lookup="per_token", selection="expanded", peer_transport="http")
        assert original.canonical_bytes() == explicit.canonical_bytes()
        assert dict(original.pipeline.components["linear"].params) == ({"weights": "public"} if weights == "public" else {})
        assert execution_binding(original) is None
        assert worker_request_memory(original) == REQUEST_MEMORY


@pytest.mark.parametrize("choices", [
    {"range_policy": "checked"}, {"range_policy": "checked", "max_abs_activation": 1},
    {"range_policy": "checked", "max_abs_activation": 1024}, {"lookup": "tiled"},
    {"peer_transport": "streams"}, {"selection": "tiled"}, {"lookup": "trie"},
    {"range_policy": "checked", "max_abs_activation": 128, "lookup": "tiled", "selection": "tiled", "peer_transport": "streams"},
])
def test_public_choices_roundtrip_and_numeric_identity(choices):
    experiment = experiment_for(weights="public", **choices)
    restored = Experiment.from_spec(json.loads(experiment.canonical_bytes()))
    assert restored.canonical_bytes() == experiment.canonical_bytes()
    assert restored.configuration_digest() != experiment_for(weights="public").configuration_digest()
    policy = execution_policy(restored)
    assert policy == {**{"range_policy": "static", "max_abs_activation": 256,
                         "lookup": "per_token", "selection": "expanded", "peer_transport": "http"}, **choices}
    expected = (PumaThreePartyDecoder.CHECKED_NUMERIC_PROFILE if choices.get("range_policy") == "checked"
                else PumaThreePartyDecoder.PUBLIC_NUMERIC_PROFILE)
    assert resolve_runtime_composition(restored.pipeline).numeric_profile == expected
    assert worker_request_memory(restored) == REQUEST_MEMORY + (STREAM_MEMORY if choices.get("peer_transport") == "streams" else 0)
    assert PumaThreePartyDecoder(**restored.pipeline.components["linear"].get_params()).params == restored.pipeline.components["linear"].params


@pytest.mark.parametrize("choices", [
    {"range_policy": "unchecked"}, {"range_policy": True}, {"range_policy": None},
    {"max_abs_activation": 1}, {"range_policy": "checked", "max_abs_activation": 0},
    {"range_policy": "checked", "max_abs_activation": 1025},
    {"range_policy": "checked", "max_abs_activation": True},
    {"range_policy": "checked", "max_abs_activation": 256.0},
    {"lookup": "secret_index"}, {"lookup": []}, {"peer_transport": "peer-stream-v1"},
    {"peer_transport": "http-v1"}, {"peer_transport": False},
    {"selection": "pruned"}, {"selection": True}, {"selection": None},
])
def test_invalid_public_choices_rejected(choices):
    with pytest.raises(ValueError):
        PumaThreePartyDecoder(weights="public", **choices)


@pytest.mark.parametrize("choices", [{"range_policy": "checked"}, {"lookup": "tiled"}, {"lookup": "trie"}, {"selection": "tiled"}, {"peer_transport": "streams"}])
def test_private_weights_cannot_select_public_execution(choices):
    with pytest.raises(ValueError):
        PumaThreePartyDecoder(**choices)


@pytest.mark.parametrize("extra", [{"range_policy": "static"}, {"max_abs_activation": 256},
                                  {"lookup": "per_token"}, {"selection": "expanded"}, {"peer_transport": "http"}, {"unchecked": True}])
def test_noncanonical_component_substitution_has_no_runtime(extra):
    original = experiment_for(weights="public")
    # Generic serialized ComponentRefs cannot smuggle redundant/default selectors
    # past the typed component and native canonical-parameter agreement.
    component = ComponentRef(PumaThreePartyDecoder.descriptor.component, {"weights": "public", **extra})
    with pytest.raises(ValueError):
        execution_policy(SimpleNamespace(pipeline=SimpleNamespace(components={"linear": component})))
    if "unchecked" in extra:
        with pytest.raises(ValueError):
            Pipeline(model=original.pipeline.model, profile=original.pipeline.profile,
                     components={**original.pipeline.components, "linear": component})
    else:
        # The public Python constructor accepts explicit legacy defaults but
        # serialization removes them, so native sees only canonical parameters.
        pipeline = Pipeline(model=original.pipeline.model, profile=original.pipeline.profile,
                            components={**original.pipeline.components, "linear": component})
        assert pipeline.canonical_bytes() == original.pipeline.canonical_bytes()


@pytest.mark.parametrize("substitution", ["missing", "lookup", "selection", "missing_selection", "numeric"])
def test_native_contract_policy_substitutions_rejected(substitution):
    experiment = experiment_for(weights="public", range_policy="checked", lookup="tiled", peer_transport="streams")
    contract = {"execution": execution_binding(experiment),
                "numeric_profile": resolve_runtime_composition(experiment.pipeline).numeric_profile,
                "peer_stream_workspace_bytes_per_worker": STREAM_MEMORY}
    replicated_source.validate_execution(experiment, contract)
    if substitution == "missing":
        contract.pop("execution")
    elif substitution == "lookup":
        contract["execution"]["lookup"] = "per_token"
    elif substitution == "selection":
        contract["execution"]["selection"] = "tiled"
    elif substitution == "missing_selection":
        contract["execution"].pop("selection")
    else:
        contract["numeric_profile"] = PumaThreePartyDecoder.PUBLIC_NUMERIC_PROFILE
    with pytest.raises(ValueError, match="execution (contract|binding)"):
        replicated_source.validate_execution(experiment, contract)


@pytest.mark.parametrize("choices", [{}, {"weights": "public"}, {"weights": "public", "lookup": "trie"},
    {"weights": "public", "range_policy": "checked", "max_abs_activation": 128,
     "lookup": "tiled", "selection": "tiled", "peer_transport": "streams"}])
def test_publisher_passes_native_policy_without_changing_legacy_call(monkeypatch, choices):
    experiment = experiment_for(**choices)
    selected = execution_binding(experiment)
    calls, closed = [], []

    class Publisher:
        shapes = ()

        def append(self, *args):
            pass

        def __init__(self, *args, **kwargs):
            calls.append((args, kwargs))

        def seal(self, tokens):
            assert tokens == b"\x03\x00\x00\x00"
            return json.dumps({"execution": selected} if selected else {}).encode()

        def close(self):
            closed.append(True)

    monkeypatch.setattr(replicated_source._native, "ReplicatedPublisher", Publisher, raising=False)
    source = SimpleNamespace(config_bytes=b"{}", source_lock_digest="a" * 64,
        contract={"weights": choices.get("weights", "private")}, runtime_contract=lambda *args: None,
        _append_source=lambda *args: None)
    owner, _ = replicated_source.ReplicatedSource.publisher(source, experiment, (3,), 1, lambda: False)
    assert calls[0][1] == (selected or {})
    assert calls[0][0][-1] == choices.get("weights", "private")
    owner.close()
    assert closed == [True]


def test_stream_memory_priced_only_to_workers_with_ordinary_margin(monkeypatch):
    contract = {"execution_payload_estimate_bytes": REQUEST_MEMORY, "weights": "public",
                "plan": {"config_digest": "a" * 64}, "source_float32_bytes": 64}
    def native_contract(encoded, *args):
        experiment = Experiment.from_spec(json.loads(encoded))
        value = {**contract, "numeric_profile": resolve_runtime_composition(experiment.pipeline).numeric_profile}
        execution = execution_binding(experiment)
        if execution:
            value["execution"] = execution
        if execution and execution["peer_transport"] == "streams":
            value["peer_stream_workspace_bytes_per_worker"] = STREAM_MEMORY
        return json.dumps(value).encode()

    monkeypatch.setattr(replicated_source._native, "replicated_source_contract", native_contract, raising=False)
    estimates = [benchmark_memory.estimate_memory({}, experiment_for(weights="public", peer_transport=choice).pipeline,
        max_input_tokens=4, max_output_tokens=2) for choice in ("http", "streams")]
    baseline, stream = estimates
    assert baseline["client_peak_bytes"] == stream["client_peak_bytes"]
    delta = benchmark_memory._slack(STREAM_MEMORY, 25)
    for role in ("worker_a", "worker_b", "worker_c"):
        assert stream["provider_peak_bytes"][role] - baseline["provider_peak_bytes"][role] == delta
    assert stream["native_total_peak_bytes"] - baseline["native_total_peak_bytes"] == 3 * delta
    assert stream["components"]["peer_stream_workspace_bytes_per_worker"] == STREAM_MEMORY


def test_profile_facade_serializes_same_component_and_rejects_ambiguous_choices():
    model = Model.path("/not-loaded/generated-source")
    choices = {"weights": "public", "range_policy": "checked", "max_abs_activation": 128,
               "lookup": "tiled", "selection": "tiled", "peer_transport": "streams"}
    facade = PumaThreePartyCpu(model, **choices)
    explicit = PumaThreePartyCpu(model, linear=PumaThreePartyDecoder(**choices))
    assert facade.canonical_bytes() == explicit.canonical_bytes()
    assert facade.with_params(linear__max_abs_activation=64).linear.params["max_abs_activation"] == 64
    with pytest.raises(ValueError, match="not both"):
        PumaThreePartyCpu(model, linear=PumaThreePartyDecoder(weights="public"), peer_transport="streams", weights="public")


def test_discovery_and_capability_coverage_expose_explicit_choices():
    from pllm.components import list_components
    from pllm.metrics.coverage import benchmark_capabilities

    descriptor, = [item for item in list_components() if item.component == PumaThreePartyDecoder.descriptor.component]
    fields = descriptor.to_dict()["parameter_schema"]["properties"]
    assert fields["max_abs_activation"]["maximum"] == 1024
    assert fields["range_policy"]["enum"] == ["checked"]
    assert fields["lookup"]["enum"] == ["tiled", "trie"]
    assert fields["selection"]["enum"] == ["tiled"]
    assert fields["peer_transport"]["enum"] == ["streams"]
    requirements = [row.to_spec() for row in benchmark_capabilities(include_planned=False)
                    if row.to_spec()["component"] == descriptor.component]
    for name in ("range_policy", "max_abs_activation", "lookup", "selection", "peer_transport"):
        assert any(row["present"] == [name] for row in requirements)
    for name, value in (("range_policy", "checked"), ("lookup", "tiled"), ("selection", "tiled"), ("peer_transport", "streams")):
        assert any(row["parameters"] == {name: value} for row in requirements)


@pytest.mark.parametrize("choice", ["http", "streams"])
def test_supervisor_selector_confirms_but_cannot_override_experiment(choice):
    from pllm.runtime.replicated_protocol import validate_transport_selector

    experiment = experiment_for(weights="public", peer_transport=choice)
    assert validate_transport_selector(experiment, choice) == choice
    assert validate_transport_selector(experiment, None) == choice
    for invalid in ("streams" if choice == "http" else "http", "peer-stream-v1", "http-v1", "", True):
        with pytest.raises(ValueError, match="supervisor transport"):
            validate_transport_selector(experiment, invalid)


def test_supervisor_exports_experiment_choice_not_operator_override(monkeypatch):
    from pllm.runtime.servers import LocalTopology
    from pllm.runtime.replicated_protocol import ROLES

    monkeypatch.setenv("PLLM_REPLICATED_PEER_TRANSPORT", "http")
    experiment = experiment_for(weights="public", peer_transport="streams")
    topology = SimpleNamespace(_privacy_mode="replicated_public", _experiment=experiment,
        _role_ids=ROLES, _role_credentials={role: f"client-{role}-key" for role in ROLES},
        _role_urls={role: f"http://127.0.0.1:{9000 + index}" for index, role in enumerate(ROLES)},
        _peer_credentials={(a, b): f"{a}-to-{b}-key" for a in ROLES for b in ROLES if a != b},
        _hf_cache_dir=None)
    environment = LocalTopology._environment(topology, "worker_b")
    assert environment["PLLM_REPLICATED_PEER_TRANSPORT"] == "streams"
    assert environment["PLLM_ROLE_EXPERIMENT_JSON"].encode() == experiment.canonical_bytes()
    peers = json.loads(environment["PLLM_REPLICATED_PEERS_JSON"])
    assert set(peers) == {"worker_a", "worker_c"}
    assert peers["worker_a"][1:] == ["worker_b-to-worker_a-key", "worker_a-to-worker_b-key"]


@pytest.mark.parametrize("substitution", [True, 1.0, "1"])
def test_execution_binding_rejects_noninteger_json_bound(substitution):
    from pllm.runtime.replicated_protocol import validate_execution_binding

    experiment = experiment_for(weights="public", range_policy="checked", max_abs_activation=1)
    execution = execution_binding(experiment)
    execution["max_abs_activation"] = substitution
    with pytest.raises(ValueError, match="execution binding"):
        validate_execution_binding(experiment, {"execution": execution})


def test_source_rejects_unpriced_stream_workspace():
    experiment = experiment_for(weights="public", peer_transport="streams")
    contract = {"execution": execution_binding(experiment),
                "numeric_profile": resolve_runtime_composition(experiment.pipeline).numeric_profile}
    for workspace in (None, 0, STREAM_MEMORY - 1, STREAM_MEMORY + 1):
        if workspace is not None:
            contract["peer_stream_workspace_bytes_per_worker"] = workspace
        with pytest.raises(ValueError, match="execution contract"):
            replicated_source.validate_execution(experiment, contract)
