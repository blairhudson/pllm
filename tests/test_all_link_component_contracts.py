"""Opt-in identities, rejection boundaries and independent resource owners."""
from __future__ import annotations

import pytest
from jsonschema import Draft202012Validator

from pllm import Deployment, ExecutionBudget, Experiment, Model, Pipeline
from pllm.components import create_component
from pllm.configuration import ConfigurationError, load_configuration, loads_configuration
from pllm.preparation import ModelAwareCorrections, PreparedInventory
from pllm.profiles import MaskedLinearCpu, TwoOnlineOffsetCpu, VerifiedMaskedLinearCpu, resolve_runtime_composition
from pllm.protocols import ClientBundleTransport, MaskedLinear
from pllm.quantization import SymmetricPerRow
from pllm.state import ClientPrefixReuse


def inventory(**changes):
    return PreparedInventory(refill="on-demand", allocation="demand", **changes)


def pipeline(*, epoch=None, inflight=False):
    return MaskedLinearCpu(Model.tiny(), inventory=inventory(decode_epoch_tokens=epoch),
        quantization=SymmetricPerRow(causal_reduction="prefix_f32"),
        cache=ClientPrefixReuse(max_bytes=1 << 20, fixed_input_tokens=16, inflight=inflight))


def test_omitted_options_preserve_historical_specs_and_pipeline_digest():
    old_inventory = {"policy": "request-sized", "rows": 1, "refill": "on-demand", "allocation": "demand"}
    old_cache = {"max_bytes": 1 << 20, "fixed_input_tokens": 16}
    base = pipeline()
    assert base.inventory.params == old_inventory
    assert base.cache.params == old_cache
    assert inventory(decode_epoch_tokens=None) == inventory()
    assert base.digest() == Pipeline.from_spec(base.to_spec()).digest()
    assert "decode_epoch_tokens" not in base.to_spec()["components"]["inventory"]["params"]
    assert "inflight" not in base.to_spec()["components"]["cache"]["params"]


@pytest.mark.parametrize("epoch,inflight", [(4, False), (None, True), (32, True)])
def test_selected_configuration_serializes_and_loads(tmp_path, epoch, inflight):
    selected = Experiment("all-link", pipeline(epoch=epoch, inflight=inflight),
        Deployment.local(root="local://all-link"),
        ExecutionBudget(requests=8, max_input_tokens=16, max_new_tokens=64))
    text = selected.canonical_bytes().decode()
    path = tmp_path / "experiment.json"
    path.write_text(text)
    for replay in (loads_configuration(text), load_configuration(path)):
        assert replay.canonical_bytes() == selected.canonical_bytes()
        assert replay.pipeline.digest() == selected.pipeline.digest()
        resolved = resolve_runtime_composition(replay.pipeline)
        assert resolved.preparation_decode_epoch_tokens == epoch
        assert resolved.inflight is inflight


@pytest.mark.parametrize("epoch", [1, 4, 32, 64])
@pytest.mark.parametrize("inflight", [False, True])
def test_selected_options_round_trip_and_resolve(epoch, inflight):
    selected = pipeline(epoch=epoch, inflight=inflight)
    replay = Pipeline.from_spec(selected.to_spec())
    assert replay.digest() == selected.digest() != pipeline().digest()
    resolved = resolve_runtime_composition(replay)
    assert resolved.preparation_decode_epoch_tokens == epoch
    assert resolved.inflight is inflight
    for item in (selected.inventory, selected.cache):
        Draft202012Validator(item.describe().to_dict()["parameter_schema"]).validate(item.to_spec()["params"])
        assert create_component(item.component, item.params) == item
        assert type(item).from_params(item.get_params()) == item
    with pytest.raises((AttributeError, TypeError)):
        selected.cache.params["inflight"] = False


@pytest.mark.parametrize("epoch", [True, False, 0, 65, -1, 1.0, "4"])
def test_bad_epoch_counts_rejected(epoch):
    with pytest.raises(ConfigurationError):
        inventory(decode_epoch_tokens=epoch)


@pytest.mark.parametrize("changes", [dict(policy="prewarm"), dict(rows=2),
    dict(allocation="uniform"), dict(refill="idle")])
def test_epoch_requires_demand_policy(changes):
    params = dict(refill="on-demand", allocation="demand", decode_epoch_tokens=4) | changes
    with pytest.raises(ConfigurationError):
        PreparedInventory(**params)


@pytest.mark.parametrize("changes", [dict(inflight=1), dict(inflight_max_requests=1),
    dict(inflight_max_requests=33), dict(inflight_max_requests=True),
    dict(inflight_max_bytes=0), dict(inflight_max_bytes=(256 << 20) + 1),
    dict(inflight_max_bytes=(1 << 20) + 1), dict(inflight_max_bytes=True)])
def test_bad_inflight_limits_rejected(changes):
    with pytest.raises(ConfigurationError):
        ClientPrefixReuse(max_bytes=1 << 20, fixed_input_tokens=16, **(dict(inflight=True) | changes))


@pytest.mark.parametrize("changes", [dict(inflight_max_requests=2), dict(inflight_max_bytes=1 << 20)])
def test_nondefault_inflight_limits_require_opt_in(changes):
    with pytest.raises(ConfigurationError):
        ClientPrefixReuse(max_bytes=1 << 20, fixed_input_tokens=16, **changes)


@pytest.mark.parametrize("epoch,inflight", [(4, False), (None, True), (4, True)])
def test_unsupported_compositions_reject_and_safe_transport_composes(epoch, inflight):
    selected = pipeline(epoch=epoch, inflight=inflight)
    assert resolve_runtime_composition(selected.with_params(
        linear=MaskedLinear(output_encoding="row_residues", request_encoding="stage_packed"),
        preparation=ModelAwareCorrections(storage="paged", kernel="digit8"),
        delivery=ClientBundleTransport("artifacts", compression="zlib", storage="paged"))) is not None
    assert resolve_runtime_composition(selected.with_params(quantization=SymmetricPerRow())) is None
    if epoch:
        assert resolve_runtime_composition(selected.with_params(cache=None)) is None
    assert resolve_runtime_composition(VerifiedMaskedLinearCpu(Model.tiny(),
        inventory=selected.inventory, cache=selected.cache, quantization=selected.quantization)) is None
    if inflight:
        assert resolve_runtime_composition(TwoOnlineOffsetCpu(Model.tiny(),
            cache=selected.cache, quantization=selected.quantization)) is None


def test_epoch_memory_keeps_full_cap_and_inflight_adds_independent_owners(monkeypatch):
    from pllm.modeling import ModelPlan
    from pllm.runtime.benchmark_memory import estimate_memory
    from pllm.sources import _tiny_model_config

    # Resource unit test can use pre-promotion installed native geometry. Native
    # schedule equality and admission of new identities are tested in cargo.
    runtime_schedule = ModelPlan.runtime_schedule
    monkeypatch.setattr(ModelPlan, "runtime_schedule", lambda plan, composition:
        runtime_schedule(plan, pipeline()))

    def estimate(value):
        return estimate_memory(_tiny_model_config(), value,
            max_input_tokens=16, max_output_tokens=64, inventory_rows=1)

    base, epoch, shared = (estimate(value) for value in (pipeline(), pipeline(epoch=4), pipeline(inflight=True)))
    assert epoch["compiled_input_bound"] == base["compiled_input_bound"]
    assert epoch["inventory_rows"] == base["inventory_rows"] == 79
    assert epoch["components"]["client_tensor_work_bytes"] == base["components"]["client_tensor_work_bytes"]
    assert epoch["client_peak_bytes"] == base["client_peak_bytes"]
    assert shared["concurrent_response_bound"] == 8
    assert shared["components"]["client_cache_bytes"] == base["components"]["client_cache_bytes"]
    assert shared["components"]["client_inflight_owner_bytes"] == 64 << 20
    assert shared["components"]["client_inflight_copy_work_bytes"] == 128 << 20
    assert shared["components"]["client_tensor_work_bytes"] == 8 * base["components"]["client_tensor_work_bytes"]
    assert shared["components"]["provider_correction_bytes"] == 8 * base["components"]["provider_correction_bytes"]
    assert shared["client_peak_bytes"] > base["client_peak_bytes"] + (192 << 20)
