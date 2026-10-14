"""Public growing policy identity and full-cap admission controls."""
import pytest
from jsonschema import Draft202012Validator

from pllm import Deployment, ExecutionBudget, Experiment, Model, Pipeline
from pllm.components import create_component
from pllm.configuration import ConfigurationError, ExperimentProfile, loads_configuration
from pllm.preparation import PreparedInventory
from pllm.profiles import MaskedLinearCpu, resolve_runtime_composition
from pllm.quantization import SymmetricPerRow
from pllm.state import ClientPrefixReuse


def pipeline(schedule=None, fixed=None, inflight=False):
    return MaskedLinearCpu(Model.tiny(),
        inventory=PreparedInventory(refill="on-demand", allocation="demand",
            decode_epoch_tokens=fixed, decode_epoch_schedule=schedule),
        quantization=SymmetricPerRow(causal_reduction="prefix_f32"),
        cache=ClientPrefixReuse(max_bytes=1 << 20, fixed_input_tokens=128, inflight=inflight))


@pytest.mark.parametrize("schedule", [(4, 60), (1,), (2, 2, 64), (1,) * 64])
@pytest.mark.parametrize("inflight", [False, True])
def test_public_schedule_round_trip(schedule, inflight):
    selected = pipeline(schedule, inflight=inflight)
    inventory = selected.inventory
    Draft202012Validator(inventory.describe().to_dict()["parameter_schema"]).validate(inventory.to_spec()["params"])
    assert create_component(inventory.component, inventory.params) == inventory
    assert type(inventory).from_params(inventory.get_params()) == inventory
    exp = Experiment("growing", selected, Deployment.local(root="local://growing"),
        ExecutionBudget(requests=8, max_input_tokens=128, max_new_tokens=64))
    replay = loads_configuration(exp.canonical_bytes().decode())
    assert replay.canonical_bytes() == exp.canonical_bytes()
    assert Pipeline.from_spec(selected.to_spec()).digest() == selected.digest()
    assert resolve_runtime_composition(selected).preparation_decode_epoch_schedule == schedule


@pytest.mark.integration
def test_fresh_native_profile_carries_schedule():
    exp = Experiment("growing", pipeline((4, 60)), Deployment.local(root="local://growing"),
        ExecutionBudget(requests=8, max_input_tokens=128, max_new_tokens=64))
    assert ExperimentProfile(exp).preparation_decode_epoch_schedule == (4, 60)


@pytest.mark.parametrize("schedule", [(), [], (0,), (65,), (True,), (4.0,), "4", (4, 3), (1,) * 65])
def test_bad_schedule_rejected(schedule):
    with pytest.raises(ConfigurationError):
        pipeline(schedule)


def test_fixed_and_schedule_exclusive_and_omitted_identity():
    with pytest.raises(ConfigurationError, match="mutually exclusive"):
        pipeline((4, 60), fixed=4)
    assert pipeline().digest() == pipeline(None).digest()
    assert "decode_epoch_schedule" not in pipeline(fixed=4).inventory.params
    original = [4, 60]
    selected = pipeline(original)
    digest = selected.digest()
    original[0] = 1
    assert selected.digest() == digest
    with pytest.raises(TypeError):
        selected.inventory.params["decode_epoch_schedule"][0] = 1


@pytest.mark.parametrize("schedule", [None, (), (4, 3), (0,), (True,), (1,) * 65])
def test_forged_generic_component_rejected_by_runtime_resolver(schedule):
    from pllm.configuration import ComponentRef
    selected = pipeline()
    forged = ComponentRef(selected.inventory.component,
        {**selected.inventory.params, "decode_epoch_schedule": schedule})
    raw = object.__new__(Pipeline)
    object.__setattr__(raw, "model", selected.model)
    object.__setattr__(raw, "components", {**selected.components, "inventory": forged})
    object.__setattr__(raw, "profile", None)
    assert resolve_runtime_composition(raw) is None


def test_cumulative_allowances_repeat_and_clip():
    from pllm.runtime.client import RuntimeClient

    core = object.__new__(RuntimeClient)
    core.preparation_decode_epoch_tokens = None
    core.preparation_decode_epoch_schedule = (4, 60)
    assert core._decode_epochs_enabled()
    assert core._decode_epoch_allowance(0, 1) == 1
    assert [core._decode_epoch_allowance(i, remaining) for i, remaining in
            enumerate((130, 126, 66, 6))] == [4, 60, 60, 6]


def test_growing_memory_keeps_full_cap(monkeypatch):
    from pllm.modeling import ModelPlan
    from pllm.runtime.benchmark_memory import estimate_memory

    from pllm.sources import _tiny_model_config

    runtime_schedule = ModelPlan.runtime_schedule
    monkeypatch.setattr(ModelPlan, "runtime_schedule", lambda plan, composition:
        runtime_schedule(plan, pipeline()))
    def memory(selected):
        return estimate_memory(_tiny_model_config(), selected,
            max_input_tokens=128, max_output_tokens=64, inventory_rows=1)
    control, growing = memory(pipeline()), memory(pipeline((4, 60)))
    assert growing["compiled_input_bound"] == control["compiled_input_bound"]
    assert growing["inventory_rows"] == control["inventory_rows"] == 191
    assert growing["components"]["client_tensor_work_bytes"] == control["components"]["client_tensor_work_bytes"]
    assert growing["client_peak_bytes"] == control["client_peak_bytes"]
