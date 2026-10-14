"""Public decode-cache policy admission and additive resource envelope; no models loaded."""
from __future__ import annotations

import pytest
from jsonschema import Draft202012Validator, ValidationError

from pllm import Deployment, ExecutionBudget, Experiment, Model, Pipeline
from pllm.components import create_component
from pllm.configuration import ComponentRef, ConfigurationError, loads_configuration
from pllm.preparation import PreparedInventory
from pllm.profiles import MaskedLinearCpu, resolve_runtime_composition
from pllm.quantization import SymmetricPerRow
from pllm.state import ClientPrefixReuse


MiB = 1 << 20


def cache(**changes):
    return ClientPrefixReuse(**(dict(max_bytes=MiB, fixed_input_tokens=128,
        generated_prefixes=True, decode=True) | changes))


def pipeline(*, decode=True, decode_max_bytes=128 * MiB):
    return MaskedLinearCpu(Model.tiny(),
        quantization=SymmetricPerRow(causal_reduction="prefix_f32"),
        inventory=PreparedInventory(allocation="demand", refill="on-demand", decode_epoch_tokens=1),
        cache=cache(decode=decode, decode_max_bytes=decode_max_bytes))


def experiment(selected, deployment=None):
    return Experiment("decode-contract", selected, deployment or Deployment.local(root="local://decode"),
        ExecutionBudget(requests=8, max_input_tokens=128, max_new_tokens=64))


@pytest.mark.parametrize("bound", [8, 32, 128, 256])
def test_selected_policy_round_trip_and_resolution(bound):
    selected = pipeline(decode_max_bytes=bound * MiB)
    item = selected.cache
    Draft202012Validator(item.describe().to_dict()["parameter_schema"]).validate(item.to_spec()["params"])
    assert create_component(item.component, item.params) == item
    assert type(item).from_params(item.get_params()) == item
    replay = Pipeline.from_spec(selected.to_spec())
    assert replay.digest() == selected.digest() != pipeline(decode=False).digest()
    exp = experiment(selected)
    assert loads_configuration(exp.canonical_bytes().decode()).canonical_bytes() == exp.canonical_bytes()
    options = resolve_runtime_composition(replay)
    assert options.decode is True
    assert options.decode_max_bytes == bound * MiB
    assert options.preparation_decode_epoch_tokens == 1
    assert options.inflight is False


def test_omitted_false_fields_preserve_old_identity():
    old = ClientPrefixReuse(max_bytes=MiB, fixed_input_tokens=128, generated_prefixes=True)
    assert old == cache(decode=False, decode_max_bytes=128 * MiB)
    assert old.params == {"max_bytes": MiB, "fixed_input_tokens": 128, "generated_prefixes": True}
    assert "decode" not in old.params and "decode_max_bytes" not in old.params
    assert type(old).from_params(old.get_params()) == old
    assert resolve_runtime_composition(pipeline(decode=False)).decode is False


@pytest.mark.parametrize("changes", [dict(decode=1), dict(decode=None), dict(decode_max_bytes=True),
    dict(decode_max_bytes=0), dict(decode_max_bytes=7 * MiB), dict(decode_max_bytes=257 * MiB),
    dict(decode_max_bytes=8 * MiB + 1), dict(decode_max_bytes=32.0 * MiB),
    dict(decode_max_bytes="33554432"), dict(decode=False, decode_max_bytes=32 * MiB),
    dict(generated_prefixes=False), dict(inflight=True)])
def test_bad_constructor_contracts_reject(changes):
    with pytest.raises(ConfigurationError):
        cache(**changes)


@pytest.mark.parametrize("changes", [dict(decode=False), dict(decode_max_bytes=7 * MiB),
    dict(generated_prefixes=False), dict(inflight=True, inflight_max_requests=8, inflight_max_bytes=MiB)])
def test_schema_rejects_noncanonical_or_incompatible_fields(changes):
    item = cache()
    with pytest.raises(ValidationError):
        Draft202012Validator(item.describe().to_dict()["parameter_schema"]).validate(dict(item.params) | changes)


def test_schema_requires_decode_bound_and_flag_together():
    item = cache()
    validator = Draft202012Validator(item.describe().to_dict()["parameter_schema"])
    for field in ("decode", "decode_max_bytes"):
        value = dict(item.params)
        value.pop(field)
        with pytest.raises(ValidationError):
            validator.validate(value)


def forged_pipeline(selected, slot, component):
    # Exercise resolver defense independently of constructors' earlier rejection.
    raw = object.__new__(Pipeline)
    object.__setattr__(raw, "model", selected.model)
    object.__setattr__(raw, "components", {**selected.components, slot: component})
    object.__setattr__(raw, "profile", None)
    return raw


@pytest.mark.parametrize("slot,identity,params", [
    ("inventory", "pllm/prepared-inventory-policy/v1", {"policy": "request-sized", "rows": 1,
        "allocation": "demand", "refill": "on-demand", "decode_epoch_tokens": 2}),
    ("inventory", "pllm/prepared-inventory-policy/v1", {"policy": "request-sized", "rows": 1,
        "allocation": "demand", "refill": "on-demand", "decode_epoch_schedule": (1,)}),
    ("inventory", "pllm/prepared-inventory-policy/v1", {"policy": "request-sized", "rows": 1}),
    ("quantization", "pllm/symmetric-per-row-quantization/v1", {"weight_bits": 8, "activation_bits": 8}),
    ("kernels", "pllm/apple-metal-int8/v1", {}),
    ("linear", "pllm/two-online-offset-linear/v1", {}),
    ("linear", "pllm/cleartext-linear", {}),
    ("state", "pllm/client-local-kv", {}),
    ("scheduler", "pllm/mlp-prefill-tiling/v1", {}),
    ("verification", "pllm/freivalds-verify/v1", {"target_failure_bits": 40}),
    ("public_prefix", "pllm/public-prefix-capsule/v1", {}),
])
def test_unsupported_compositions_reject(slot, identity, params):
    selected = pipeline()
    assert resolve_runtime_composition(forged_pipeline(selected, slot, ComponentRef(identity, params))) is None


@pytest.mark.parametrize("changes", [dict(decode=False), dict(decode=None), dict(decode_max_bytes=None),
    dict(decode_max_bytes=7 * MiB), dict(generated_prefixes=False),
    dict(inflight=True, inflight_max_requests=8, inflight_max_bytes=MiB)])
def test_forged_cache_resolver_rejects(changes):
    selected = pipeline()
    forged = ComponentRef(selected.cache.component, dict(selected.cache.params) | changes)
    assert resolve_runtime_composition(forged_pipeline(selected, "cache", forged)) is None


def test_network_auto_planning_rejects_before_native_or_runtime_owner():
    network = Deployment.network(network_id="decode-network", network_spec_digest="1" * 64)
    with pytest.raises(ConfigurationError, match="caller-owned decode cache resources"):
        experiment(pipeline(), deployment=network)
    assert experiment(pipeline(decode=False), deployment=network).deployment.kind == "network"


@pytest.mark.parametrize("bound", [8, 32, 128, 256])
def test_decode_envelope_additive_not_substituted_for_ordinary_owners(monkeypatch, bound):
    from pllm.modeling import ModelPlan
    from pllm.runtime.benchmark_memory import estimate_memory
    from pllm.sources import _tiny_model_config

    runtime_schedule = ModelPlan.runtime_schedule
    # Legacy installed wheel lacks decode admission. Native source tests cover the
    # new schedule; use admitted baseline geometry to isolate arithmetic pricing.
    monkeypatch.setattr(ModelPlan, "runtime_schedule", lambda plan, composition:
        runtime_schedule(plan, pipeline(decode=False)))
    def estimate(selected):
        return estimate_memory(_tiny_model_config(), selected,
            max_input_tokens=128, max_output_tokens=64, inventory_rows=1)
    old = estimate(pipeline(decode=False))
    selected = estimate(pipeline(decode_max_bytes=bound * MiB))
    assert selected["components"]["client_decode_owner_bytes"] == bound * MiB
    for name in ("client_cache_bytes", "client_tensor_work_bytes", "client_state_work_bytes",
                 "client_live_tensor_bytes", "client_mask_bytes", "provider_correction_bytes"):
        assert selected["components"][name] == old["components"][name], name
    assert selected["inventory_rows"] == old["inventory_rows"]
    assert selected["compiled_input_bound"] == old["compiled_input_bound"]
    assert selected["provider_peak_bytes"] == old["provider_peak_bytes"]
    assert selected["client_peak_bytes"] - old["client_peak_bytes"] == bound * MiB * 5 // 4
    assert selected["scope"] == "conservative allocation estimate; not measured peak RSS"
