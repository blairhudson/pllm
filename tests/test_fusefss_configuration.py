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
)


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
    assert method.get_params() == {"numeric_profile": numeric_profile, "weights": "public"}
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


def test_production_resolution_calls_native_and_never_treats_description_as_admission(monkeypatch):
    from pllm import _native

    p = FuseFssSourceCpu(TinyModel(), numeric_profile=PROFILES[0])
    e = experiment(p)
    calls = []
    def unavailable(payload):
        calls.append(payload)
        raise ValueError("native FSS composition unavailable")
    monkeypatch.setattr(_native, "resolve_experiment", unavailable)
    with pytest.raises(ConfigurationError, match="requires native complete-decoder composition admission"):
        e.resolve()
    assert calls == [e.canonical_bytes()]


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
