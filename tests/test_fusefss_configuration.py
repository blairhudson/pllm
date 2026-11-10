"""Pure immutable configuration tests, not native or SDK serving qualification."""
from __future__ import annotations

import json
from dataclasses import FrozenInstanceError

import pytest
from jsonschema import Draft202012Validator

import pllm
from pllm.components import create_component, get, list_components
from pllm.configuration import ConfigurationError, Experiment, Pipeline, loads_configuration
from pllm.kernels import AppleMetal, Cpu
from pllm.profiles import FuseFssSourceCpu, MaskedLinearCpu, PumaThreePartyCpu, resolve_runtime_composition
from pllm.protocols import FuseFssSourceDecoder, MaskedLinear, ProtocolMethod, PumaThreePartyDecoder
from pllm.roles import RoleTopology, ThreePartyReplicatedRoles, TrustedDealerTwoShareRoles, TwoOnlineDelegationRoles
from pllm.roles.topology import RoleGraph, graph_for_runtime, trusted_dealer_two_share_graph
from pllm.sources import TinyModel
from pllm.state import ClientLocalKv, FixedCountPrivatePrefix, ReplicatedPrivatePrefix, StateProtocol
from pllm.verification import FreivaldsVerify

PROFILES = (
    "fusefss.gpt2-complete-ring50-q12.v1",
    "fusefss.dense-rope-ring50-q12.v1",
    "fusefss.dense-rope-checked-ranges-ring50-q12.v1",
)


@pytest.fixture
def paper_examples_path(monkeypatch):
    from pathlib import Path

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))


def experiment(pipeline: Pipeline) -> Experiment:
    return Experiment(name="fss-contract-only", pipeline=pipeline,
        deployment=pllm.Deployment.local(root="local://fss-contract-only"),
        budget=pllm.ExecutionBudget(requests=1, max_input_tokens=3, max_new_tokens=2))


@pytest.mark.parametrize("numeric_profile", PROFILES)
def test_explicit_numeric_choice_compact_params_discovery_and_immutable_replay(numeric_profile):
    method = FuseFssSourceDecoder(numeric_profile=numeric_profile)
    assert isinstance(method, ProtocolMethod)
    assert method.to_spec() == {"component": "pllm/fusefss-source-decoder/v1",
                                "params": {"numeric_profile": numeric_profile}}
    assert method.get_params() == {"numeric_profile": numeric_profile, "weights": "public",
                                    "capacity": "capsule", "material_delivery": "single",
                                    "online_schedule": "serial"}
    assert method == FuseFssSourceDecoder(numeric_profile=numeric_profile, weights="public")
    for cls, instance, category in (
        (FuseFssSourceDecoder, method, "pllm/protocol-method"),
        (TrustedDealerTwoShareRoles, TrustedDealerTwoShareRoles(), "pllm/role-topology"),
        (FixedCountPrivatePrefix, FixedCountPrivatePrefix(), "pllm/state-protocol"),
    ):
        descriptor = cls.describe()
        assert get(descriptor.component) is cls
        assert descriptor in list_components()
        assert descriptor.category == category
        schema = descriptor.to_dict()["parameter_schema"]
        Draft202012Validator.check_schema(schema)
        Draft202012Validator(schema).validate(instance.to_spec()["params"])
        assert create_component(instance.component, instance.params) == instance
        with pytest.raises((FrozenInstanceError, AttributeError, TypeError)):
            instance.component = "other"
        with pytest.raises(TypeError):
            instance.params["unexpected"] = True
    assert isinstance(TrustedDealerTwoShareRoles(), RoleTopology)
    assert isinstance(FixedCountPrivatePrefix(), StateProtocol)
    pipeline = FuseFssSourceCpu(TinyModel(), numeric_profile=numeric_profile)
    assert pipeline.PROFILE == "research.fusefss_source_cpu"
    assert pipeline.SLOT_NAMES == ("linear", "topology", "state", "kernels")
    assert pipeline.linear == method
    assert isinstance(pipeline.topology, TrustedDealerTwoShareRoles)
    assert isinstance(pipeline.state, FixedCountPrivatePrefix)
    assert pipeline.kernels == Cpu()
    assert pipeline == FuseFssSourceCpu(TinyModel(), linear=method)
    replay = Pipeline.from_spec(json.loads(pipeline.canonical_bytes()))
    assert type(replay) is Pipeline
    assert replay.digest() == pipeline.digest()
    assert type(replay.components["linear"]) is FuseFssSourceDecoder
    original = experiment(pipeline)
    restored = loads_configuration(original.canonical_bytes().decode("utf-8"))
    assert restored == original
    assert restored.configuration_digest() == original.configuration_digest()
    assert resolve_runtime_composition(restored.pipeline) == resolve_runtime_composition(pipeline)


@pytest.mark.parametrize("numeric_profile", PROFILES)
def test_decoder_capacity_is_explicit_and_canonical(numeric_profile):
    original = FuseFssSourceCpu(TinyModel(), numeric_profile=numeric_profile)
    decoder = FuseFssSourceCpu(TinyModel(), numeric_profile=numeric_profile, capacity="decoder")
    method = decoder.components["linear"]
    assert original.digest() != decoder.digest()
    assert method.params == {"numeric_profile": numeric_profile, "capacity": "decoder"}
    assert FuseFssSourceDecoder(**method.get_params()) == method
    assert loads_configuration(experiment(decoder).canonical_bytes().decode()) == experiment(decoder)
    Draft202012Validator(method.describe().to_dict()["parameter_schema"]).validate(method.to_spec()["params"])
    assert "capacity" not in original.components["linear"].params
    for invalid in (True, None, [], "unbounded"):
        with pytest.raises(ConfigurationError):
            FuseFssSourceDecoder(numeric_profile=numeric_profile, capacity=invalid)


@pytest.mark.parametrize("numeric_profile", PROFILES)
def test_configuration_description_binds_fss_ring_roles_state_and_no_verification(numeric_profile):
    p = FuseFssSourceCpu(TinyModel(), numeric_profile=numeric_profile)
    options = resolve_runtime_composition(p)
    assert options is not None
    assert options.client_runtime == "compiled_fss_v1"
    assert options.runtime_kind == "fss"
    assert options.privacy_mode == "fss_public"
    assert options.privacy_protocol == "fss_ring50_q12"
    assert options.numeric_profile == numeric_profile
    assert (options.weight_bits, options.activation_bits) == (50, 50)
    assert (FuseFssSourceDecoder.RING_BITS, FuseFssSourceDecoder.FRACTIONAL_BITS) == (50, 12)
    assert options.setup_owner == "dealer"
    assert options.correlation_mode == "trusted_dealer"
    assert options.requires_preparation is True
    assert options.sharing == "additive_two_share"
    assert options.generation_policy == "fixed_length_greedy"
    assert options.verification_component is None
    assert options.verification_target_failure_bits == 0
    assert options.verification_window is None
    assert options.prefix_cache_bytes == 0 and not options.decode and not options.inflight
    assert options.inventory_policy == "none" and options.prepared_inventory_rows == 0
    assert not options.background_inventory_refill
    assert options.local_numeric_profile is None  # Never claim FP32/source parity.
    assert graph_for_runtime(options) == trusted_dealer_two_share_graph()


def test_independent_dealer_trust_graph_excludes_online_dealer_and_client_material():
    graph = trusted_dealer_two_share_graph()
    assert {r.id: r.capability for r in graph.roles} == {
        "client": "trusted_client", "dealer": "trusted_offline_dealer",
        "worker_a": "online_share_holder", "worker_b": "online_share_holder"}
    assert set(graph.separate_operators) == {("client", "dealer"), ("dealer", "worker_a"),
        ("dealer", "worker_b"), ("worker_a", "worker_b")}
    assert not any(c.phase == "online" and "dealer" in (c.source, c.target) for c in graph.channels)
    assert {(c.source, c.target) for c in graph.channels if c.phase == "offline"} == {
        ("dealer", "worker_a"), ("dealer", "worker_b"),
        ("worker_a", "dealer"), ("worker_b", "dealer"),
        ("client", "dealer"), ("dealer", "client"),
        ("client", "worker_a"), ("worker_a", "client"),
        ("client", "worker_b"), ("worker_b", "client")}
    assert {c.payload for c in graph.channels if c.target == "client" and c.phase == "online"} == {"final_token_id_shares"}
    assert {c.source for c in graph.channels if c.payload == "authenticated_fresh_preprocessing"} == {"dealer"}
    assert RoleGraph.from_spec(graph.to_spec()).digest() == graph.digest()
    owners = {r.id: r.id for r in graph.roles}
    assert graph.separation_violations(owners) == ()
    owners["dealer"] = owners["client"]
    assert graph.separation_violations(owners) == (("client", "dealer"),)
    descriptor = TrustedDealerTwoShareRoles.describe()
    assert "distinct-role-credentials" in descriptor.required_host_features
    assert "client-is-not-dealer" in descriptor.capabilities
    assert descriptor.artifacts[0]["role"] == "dealer"
    assert {"no-kv-cache", "no-eos-early-exit", "final-only-token-ids",
            "private-token-feedback", "protected-full-prefix-recomputation"} <= set(FixedCountPrivatePrefix.describe().capabilities)


@pytest.mark.parametrize("params", [
    {}, {"numeric_profile": None}, {"numeric_profile": True}, {"numeric_profile": []},
    {"numeric_profile": "fusefss.source-derived-ring50-q12.v1"},
    {"numeric_profile": "puma.source-derived-q18.v1"},
    {"numeric_profile": PROFILES[0], "weights": "private"},
    {"numeric_profile": PROFILES[0], "weights": False},
    {"numeric_profile": PROFILES[0], "weights": ["public"]},
    {"numeric_profile": PROFILES[0], "fractional_bits": 18},
    {"numeric_profile": PROFILES[0], "setup_owner": "client"},
    {"numeric_profile": PROFILES[2], "range_policy": "checked"},
    {"numeric_profile": PROFILES[2], "max_abs_activation": 256},
])
def test_malformed_method_parameters_reject_in_construction_discovery_and_replay(params):
    with pytest.raises((ConfigurationError, TypeError)):
        FuseFssSourceDecoder(**params)
    with pytest.raises((ConfigurationError, TypeError)):
        create_component(FuseFssSourceDecoder.descriptor.component, params)
    spec = FuseFssSourceCpu(TinyModel(), numeric_profile=PROFILES[0]).to_spec()
    spec["components"]["linear"]["params"] = params
    with pytest.raises((ConfigurationError, TypeError)):
        Pipeline.from_spec(spec)


@pytest.mark.parametrize("cls", [TrustedDealerTwoShareRoles, FixedCountPrivatePrefix])
@pytest.mark.parametrize("params", [{"client_is_dealer": True}, {"streaming": True}, {"kv_cache": True}, {"early_exit": True}, {"generation_count": 3}])
def test_generic_contracts_expose_no_boundary_weakening_knobs(cls, params):
    with pytest.raises(ConfigurationError):
        cls.from_params(params)
    with pytest.raises(ConfigurationError):
        create_component(cls.descriptor.component, params)


def test_profile_replacement_preserves_immutability_and_distinct_numeric_identity():
    p = FuseFssSourceCpu(TinyModel(), numeric_profile=PROFILES[0])
    adapted = p.with_params(linear__numeric_profile=PROFILES[1])
    assert type(adapted) is FuseFssSourceCpu
    assert adapted.digest() != p.digest()
    assert p.linear.params["numeric_profile"] == PROFILES[0]
    assert adapted.get_params()["linear__numeric_profile"] == PROFILES[1]
    assert adapted.with_params(linear__weights="public") == adapted
    assert resolve_runtime_composition(p.with_params(kernels__threads=2)) is not None
    with pytest.raises(ConfigurationError):
        p.with_params(linear__weights="private")
    with pytest.raises(ConfigurationError):
        p.with_params(linear=p.linear, linear__numeric_profile=PROFILES[1])
    with pytest.raises(ConfigurationError):
        FuseFssSourceCpu(TinyModel())
    with pytest.raises(ConfigurationError):
        FuseFssSourceCpu(TinyModel(), linear=p.linear, numeric_profile=PROFILES[0])


@pytest.mark.parametrize("slot,value", [
    ("linear", MaskedLinear()), ("linear", PumaThreePartyDecoder()),
    ("topology", ThreePartyReplicatedRoles()), ("topology", TwoOnlineDelegationRoles()),
    ("state", ClientLocalKv()), ("state", ReplicatedPrivatePrefix()), ("kernels", AppleMetal()),
])
def test_wrong_category_or_component_rejects_and_generic_composition_stays_incompatible(slot, value):
    p = FuseFssSourceCpu(TinyModel(), numeric_profile=PROFILES[0])
    kwargs = {"numeric_profile": PROFILES[0], slot: value}
    if slot == "linear":
        kwargs.pop("numeric_profile")
    with pytest.raises(ConfigurationError):
        FuseFssSourceCpu(TinyModel(), **kwargs)
    generic = Pipeline(model=p.model, components={**p.components, slot: value})
    assert resolve_runtime_composition(generic) is None


@pytest.mark.parametrize("slot,value", [
    ("verification", FreivaldsVerify()), ("cache", ClientLocalKv()),
    ("preparation", TrustedDealerTwoShareRoles()), ("nonlinear", MaskedLinear()),
])
def test_unapproved_extra_slots_do_not_resolve(slot, value):
    p = FuseFssSourceCpu(TinyModel(), numeric_profile=PROFILES[0])
    assert resolve_runtime_composition(Pipeline(model=p.model, components={**p.components, slot: value})) is None
    with pytest.raises(TypeError):
        FuseFssSourceCpu(TinyModel(), numeric_profile=PROFILES[0], **{slot: value})


@pytest.mark.parametrize("numeric_profile", PROFILES)
def test_production_resolution_calls_native_and_never_treats_description_as_admission(monkeypatch, numeric_profile):
    from pllm import _native

    p = FuseFssSourceCpu(TinyModel(), numeric_profile=numeric_profile)
    e = experiment(p)
    calls = []
    def unavailable(payload):
        calls.append(payload)
        raise ValueError("native FSS composition unavailable")
    monkeypatch.setattr(_native, "resolve_experiment", unavailable)
    with pytest.raises(ConfigurationError, match="requires native complete-decoder composition admission"):
        e.resolve()
    assert calls == [e.canonical_bytes()]


def test_checked_qwen_factory_is_distinct_explicit_and_bounded(monkeypatch):
    from pathlib import Path

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    from examples.papers.common import MODEL
    from examples.papers.fusefss import qwen_checked_smoke, qwen_smoke

    static, checked = qwen_smoke(), qwen_checked_smoke()
    assert static.pipeline.components["linear"].params == {"numeric_profile": PROFILES[1], "capacity": "decoder"}
    assert checked.pipeline.components["linear"].params == {"numeric_profile": PROFILES[2], "capacity": "decoder"}
    assert FuseFssSourceDecoder.DENSE_ROPE_CHECKED_NUMERIC_PROFILE == PROFILES[2]
    assert checked.pipeline.model == static.pipeline.model == MODEL
    assert checked.budget == static.budget == pllm.ExecutionBudget(requests=1, max_input_tokens=4, max_new_tokens=1)
    assert static.configuration_digest() != checked.configuration_digest()
    assert loads_configuration(checked.canonical_bytes().decode()) == checked
    assert "opt-in-checked-range-whole-request-outcome" in checked.pipeline.components["linear"].describe().capabilities
    assert checked.pipeline.with_params(components__linear__numeric_profile=PROFILES[1]) == static.pipeline


@pytest.mark.parametrize("factory_name,schedule", [
    ("qwen_checked_wavefront_smoke", "wavefront-v1"),
    ("qwen_checked_tensor_smoke", "tensor-v1"),
])
def test_checked_qwen_scheduled_factories_preserve_pinned_bounded_batch_contract(monkeypatch, factory_name, schedule):
    from pathlib import Path

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    from examples.papers import fusefss

    baseline = fusefss.qwen_checked_batch_smoke()
    scheduled = getattr(fusefss, factory_name)()
    assert scheduled.pipeline.model.to_spec() == {
        "source": "Qwen/Qwen2.5-0.5B-Instruct",
        "revision": "7ae557604adf67be50417f59c2c2f167def9a775",
        "local_files_only": True,
    }
    assert scheduled.budget == baseline.budget == pllm.ExecutionBudget(
        requests=1, max_input_tokens=4, max_new_tokens=1)
    assert scheduled.pipeline.components["linear"].params == {
        "numeric_profile": PROFILES[2], "capacity": "decoder",
        "material_delivery": "batch-v1", "online_schedule": schedule,
    }
    assert scheduled.name != baseline.name
    assert scheduled.deployment == baseline.deployment
    assert scheduled.configuration_digest() != baseline.configuration_digest()
    assert scheduled.pipeline.digest() != baseline.pipeline.digest()
    assert scheduled.with_params(
        name=baseline.name, pipeline__components__linear__online_schedule="serial") == baseline
    restored = loads_configuration(scheduled.canonical_bytes().decode())
    assert restored == scheduled
    assert restored.configuration_digest() == scheduled.configuration_digest()
    assert resolve_runtime_composition(restored.pipeline) == resolve_runtime_composition(scheduled.pipeline)


def test_qwen_empty_system_factory_preserves_pinned_tensor_composition(monkeypatch):
    from pathlib import Path

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    from examples.papers.fusefss import qwen_checked_tensor_smoke, qwen_empty_system

    baseline, candidate = qwen_checked_tensor_smoke(), qwen_empty_system()
    assert candidate.name == "fusefss-qwen-empty-system"
    assert candidate.pipeline.model.to_spec() == {
        "source": "Qwen/Qwen2.5-0.5B-Instruct",
        "revision": "7ae557604adf67be50417f59c2c2f167def9a775",
        "local_files_only": True,
    }
    assert candidate.pipeline.components["linear"].params == {
        "numeric_profile": PROFILES[2], "capacity": "decoder",
        "material_delivery": "batch-v1", "online_schedule": "tensor-v1",
    }
    assert candidate.budget == pllm.ExecutionBudget(requests=1, max_input_tokens=15, max_new_tokens=1)
    assert baseline.budget == pllm.ExecutionBudget(requests=1, max_input_tokens=4, max_new_tokens=1)
    assert candidate.pipeline == baseline.pipeline
    assert candidate.deployment == baseline.deployment
    assert candidate.with_params(name=baseline.name, budget=baseline.budget) == baseline
    assert loads_configuration(candidate.canonical_bytes().decode()) == candidate


def test_public_exports_and_old_defaults_remain_distinct():
    for name, cls in (("FuseFssSourceDecoder", FuseFssSourceDecoder),
                      ("FuseFssSourceCpu", FuseFssSourceCpu),
                      ("FixedCountPrivatePrefix", FixedCountPrivatePrefix),
                      ("TrustedDealerTwoShareRoles", TrustedDealerTwoShareRoles)):
        assert getattr(pllm, name) is cls
        assert name in pllm.__all__
        module = __import__(cls.__module__, fromlist=[name])
        assert name in module.__all__
    assert PumaThreePartyDecoder().to_spec() == {"component": "pllm/puma-three-party-decoder/v1", "params": {}}
    assert ReplicatedPrivatePrefix().component == "pllm/replicated-private-prefix/v1"
    assert ThreePartyReplicatedRoles().component == "pllm/three-party-replicated/v1"
    old = resolve_runtime_composition(PumaThreePartyCpu(TinyModel()))
    assert old.runtime_kind == "replicated" and old.setup_owner == "client"
    assert resolve_runtime_composition(MaskedLinearCpu(TinyModel())).client_runtime == "masked_transformer_v1"
    p = FuseFssSourceCpu(TinyModel(), numeric_profile=PROFILES[0])
    # Even an arbitrary profile label cannot disguise incompatible components.
    assert resolve_runtime_composition(Pipeline(model=p.model, components=p.components,
                                               profile="baseline.masked_linear_cpu")).runtime_kind == "fss"


@pytest.mark.parametrize("factory_name,name,numeric_profile,delivery,schedule,requests,inputs,outputs,root", [
    ("experiment", "fusefss", PROFILES[1], "single", "serial", 32, 64, 8, ".pllm/papers/fusefss"),
    ("qwen_smoke", "fusefss", PROFILES[1], "single", "serial", 1, 4, 1, ".pllm/papers/fusefss"),
    ("qwen_checked_smoke", "fusefss-checked-ranges", PROFILES[2], "single", "serial", 1, 4, 1, ".pllm/papers/fusefss-checked-ranges"),
    ("qwen_checked_batch_smoke", "fusefss-checked-ranges-batch", PROFILES[2], "batch-v1", "serial", 1, 4, 1, ".pllm/papers/fusefss-checked-ranges"),
    ("qwen_checked_wavefront_smoke", "fusefss-checked-ranges-wavefront", PROFILES[2], "batch-v1", "wavefront-v1", 1, 4, 1, ".pllm/papers/fusefss-checked-ranges"),
    ("qwen_checked_tensor_smoke", "fusefss-checked-ranges-tensor", PROFILES[2], "batch-v1", "tensor-v1", 1, 4, 1, ".pllm/papers/fusefss-checked-ranges"),
    ("qwen_checked_tensor_minimal_smoke", "fusefss-checked-ranges-tensor", PROFILES[2], "batch-v1", "tensor-v1", 1, 3, 1, ".pllm/papers/fusefss-checked-ranges"),
    ("qwen_empty_system", "fusefss-qwen-empty-system", PROFILES[2], "batch-v1", "tensor-v1", 1, 15, 1, ".pllm/papers/fusefss-checked-ranges"),
])
@pytest.mark.usefixtures('paper_examples_path')
def test_recipe_declaration_matches_independent_legacy_profile(
        factory_name, name, numeric_profile, delivery, schedule, requests, inputs, outputs, root):
    from examples.papers import fusefss
    from examples.papers.common import MODEL

    actual = getattr(fusefss, factory_name)()
    legacy = pllm.Experiment(name,
        FuseFssSourceCpu(MODEL, numeric_profile=numeric_profile, capacity="decoder",
                         material_delivery=delivery, online_schedule=schedule),
        pllm.Deployment.local(root=root),
        pllm.ExecutionBudget(requests=requests, max_input_tokens=inputs, max_new_tokens=outputs))
    assert type(actual.pipeline) is Pipeline
    assert actual.pipeline.model is MODEL
    assert actual.pipeline.profile == legacy.pipeline.profile == "research.fusefss_source_cpu"
    assert set(actual.pipeline.components) == {"linear", "topology", "state", "kernels"}
    assert actual.pipeline.components == legacy.pipeline.components
    assert actual.pipeline.canonical_bytes() == legacy.pipeline.canonical_bytes()
    assert actual.pipeline.digest() == legacy.pipeline.digest()
    assert actual.canonical_bytes() == legacy.canonical_bytes()
    assert actual.configuration_digest() == legacy.configuration_digest()
    assert resolve_runtime_composition(actual.pipeline) == resolve_runtime_composition(legacy.pipeline)
    restored = loads_configuration(actual.to_json())
    assert restored == actual
    assert restored.pipeline.profile is None  # Default JSON already omits this label.
    assert restored.configuration_digest() == actual.configuration_digest()


@pytest.mark.parametrize("model", [pllm.Model.tiny(), pllm.Model.path("/generated/config-only")])
@pytest.mark.usefixtures('paper_examples_path')
def test_recipe_model_identity_and_budget_override(model):
    from examples.papers.fusefss import experiment as recipe

    budget = pllm.ExecutionBudget(requests=2, max_input_tokens=3, max_new_tokens=1)
    actual = recipe(model=model, budget=budget)
    assert actual.pipeline.model is model
    assert actual.budget is budget
    legacy = pllm.Experiment("fusefss", FuseFssSourceCpu(model, numeric_profile=PROFILES[1], capacity="decoder"),
                             actual.deployment, budget)
    assert actual.canonical_bytes() == legacy.canonical_bytes()
    assert actual.configuration_digest() == legacy.configuration_digest()


@pytest.mark.usefixtures('paper_examples_path')
def test_recipe_structural_model_source_uses_to_spec():
    from examples.papers.fusefss import experiment as recipe
    from pllm.sources import ModelSource

    class Source:
        source, kind, model_id = "/generated/structural", "safetensors", "structural"

        def __init__(self):
            self.calls = 0

        def to_spec(self):
            self.calls += 1
            return {"source": self.source, "kind": self.kind, "model_id": self.model_id,
                    "local_files_only": True}

    source = Source()
    assert isinstance(source, ModelSource)
    actual = recipe(model=source)
    legacy = FuseFssSourceCpu(source, numeric_profile=PROFILES[1], capacity="decoder")
    assert source.calls == 2
    assert isinstance(actual.pipeline.model, pllm.Model)
    assert actual.pipeline.model is not source
    assert actual.pipeline.canonical_bytes() == legacy.canonical_bytes()
    assert actual.pipeline.digest() == legacy.digest()


@pytest.mark.parametrize("model", [None, "Qwen/Qwen2.5-0.5B-Instruct", {}, object()])
@pytest.mark.usefixtures('paper_examples_path')
def test_recipe_invalid_model_contract_is_not_defaulted(model):
    from examples.papers.fusefss import experiment as recipe

    for construct in (recipe, lambda model: FuseFssSourceCpu(model, numeric_profile=PROFILES[1], capacity="decoder")):
        with pytest.raises(ConfigurationError, match="^model must satisfy the ModelSource contract$"):
            construct(model=model)


@pytest.mark.usefixtures('paper_examples_path')
def test_recipe_generic_nested_overrides_and_reconstruction():
    from examples.papers.fusefss import qwen_checked_smoke

    original = qwen_checked_smoke()
    changed = original.with_params(pipeline__components__linear__material_delivery="batch-v1",
                                   pipeline__components__kernels__threads=2)
    assert original.pipeline.components["linear"].get_params()["material_delivery"] == "single"
    assert changed.get_params()["pipeline__components__linear__material_delivery"] == "batch-v1"
    assert changed.pipeline.components["kernels"] == Cpu(threads=2)
    rebuilt = Pipeline(**changed.pipeline.get_params(deep=False))
    assert rebuilt.canonical_bytes() == changed.pipeline.canonical_bytes()
    assert rebuilt.profile == changed.pipeline.profile
    assert resolve_runtime_composition(rebuilt) is not None
    assert changed.with_params(pipeline__components__linear__material_delivery="single",
                               pipeline__components__kernels__threads=1) == original


@pytest.mark.parametrize("slot,value", [("linear", MaskedLinear()), ("kernels", AppleMetal())])
@pytest.mark.usefixtures('paper_examples_path')
def test_recipe_unsupported_composition_resolve_refuses_before_facade_runner(monkeypatch, slot, value):
    from unittest.mock import Mock
    from examples.papers.fusefss import qwen_smoke
    from pllm import _native
    from pllm.runtime import client, servers

    changed = qwen_smoke().with_params(**{f"pipeline__components__{slot}": value})
    assert changed.pipeline.components[slot] == value  # No typed-setter timing claim.
    assert resolve_runtime_composition(changed.pipeline) is None
    failure = ValueError("unsupported composition (native refusal double)")
    admission = Mock(side_effect=failure)
    runner = Mock(side_effect=AssertionError("facade runner reached before refusal"))
    roles = Mock(side_effect=AssertionError("role builder reached before refusal"))
    monkeypatch.setattr(_native, "resolve_experiment", admission)
    monkeypatch.setattr(client, "_protected_client_type", runner)
    monkeypatch.setattr(servers, "build_roles", roles)
    # Real normal resolve path; only native refusal is doubled. This does not
    # establish native category enforcement or all source/trust refusal cases.
    with pytest.raises(ConfigurationError, match="^unsupported composition") as caught:
        client.OpenAI(experiment=changed, _automatic=False)
    assert caught.value.__cause__ is failure
    admission.assert_called_once_with(changed.canonical_bytes())
    runner.assert_not_called()
    roles.assert_not_called()


@pytest.mark.usefixtures('paper_examples_path')
def test_recipe_native_refusal_preserves_payload_and_cause(monkeypatch):
    from unittest.mock import Mock
    from examples.papers.fusefss import qwen_smoke
    from pllm import _native

    actual = qwen_smoke()
    failure = ValueError("native FSS composition unavailable")
    admission = Mock(side_effect=failure)
    monkeypatch.setattr(_native, "resolve_experiment", admission)
    with pytest.raises(ConfigurationError, match="requires native complete-decoder composition admission") as caught:
        actual.resolve()
    assert caught.value.__cause__ is failure
    admission.assert_called_once_with(actual.canonical_bytes())


@pytest.mark.parametrize("factory_name,name,numeric_profile,requests,delivery,schedule", [
    ("experiment", "generated-fss-f3-sdk", PROFILES[0], 8, "single", "serial"),
    ("dense_experiment", "generated-fss-dense-sdk", PROFILES[1], 8, "single", "serial"),
    ("dense_single_smoke", "generated-fss-dense-sdk", PROFILES[1], 1, "single", "serial"),
    ("dense_batch_smoke", "generated-fss-dense-batch", PROFILES[1], 1, "batch-v1", "serial"),
    ("dense_wavefront_smoke", "generated-fss-dense-wavefront", PROFILES[1], 1, "batch-v1", "wavefront-v1"),
    ("dense_tensor_smoke", "generated-fss-dense-tensor", PROFILES[1], 1, "batch-v1", "tensor-v1"),
])
@pytest.mark.usefixtures('paper_examples_path')
def test_generated_recipe_declaration_matches_independent_legacy_profile(
        tmp_path, factory_name, name, numeric_profile, requests, delivery, schedule):
    from examples.papers import fusefss_smoke

    actual = getattr(fusefss_smoke, factory_name)(tmp_path)
    legacy = pllm.Experiment(name,
        FuseFssSourceCpu(pllm.Model.path(str(tmp_path / "source")), numeric_profile=numeric_profile,
                         material_delivery=delivery, online_schedule=schedule),
        pllm.Deployment.local(root=str(tmp_path / "roles")),
        pllm.ExecutionBudget(requests=requests, max_input_tokens=1, max_new_tokens=2))
    assert type(actual.pipeline) is Pipeline
    assert actual.pipeline.profile == legacy.pipeline.profile == "research.fusefss_source_cpu"
    assert actual.pipeline.components == legacy.pipeline.components
    assert "capacity" not in actual.pipeline.components["linear"].params
    assert "weights" not in actual.pipeline.components["linear"].params
    assert actual.pipeline.canonical_bytes() == legacy.pipeline.canonical_bytes()
    assert actual.pipeline.digest() == legacy.pipeline.digest()
    assert actual.canonical_bytes() == legacy.canonical_bytes()
    assert actual.configuration_digest() == legacy.configuration_digest()
    restored = loads_configuration(actual.to_json())
    assert restored == actual and restored.pipeline.profile is None
    assert resolve_runtime_composition(restored.pipeline) == resolve_runtime_composition(legacy.pipeline)


@pytest.mark.parametrize("factory_name", ["experiment", "dense_experiment"])
@pytest.mark.usefixtures('paper_examples_path')
def test_generated_recipe_request_override_preserves_workload(tmp_path, factory_name):
    from examples.papers import fusefss_smoke

    actual = getattr(fusefss_smoke, factory_name)(tmp_path, requests=3)
    assert actual.budget == pllm.ExecutionBudget(requests=3, max_input_tokens=1, max_new_tokens=2)
