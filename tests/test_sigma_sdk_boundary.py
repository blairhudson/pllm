"""D22 local source ownership must not silently qualify complete SIGMA SDK use."""

from __future__ import annotations

import importlib
import json

import pytest

from pllm import Deployment, ExecutionBudget, Experiment, Model, Pipeline
from pllm.components import (
    ComponentRef,
    NotYetImplementedError,
    create_component,
    get,
    list_components,
    planned_component,
)


@pytest.mark.parametrize("method", ["", "dealer-fss-keys", "shared-nonlinear"])
def test_local_d22_does_not_promote_planned_sigma_capabilities(method: str) -> None:
    plan = planned_component("sigma", method)
    assert plan.status == "pending"
    assert plan.identity not in {item.component for item in list_components()}
    implementation = getattr(importlib.import_module(plan.module), plan.name)
    for construct in (
        implementation,
        lambda: ComponentRef(plan.identity),
        lambda: create_component(plan.identity, {}),
        lambda: get(plan.identity),
    ):
        with pytest.raises(NotYetImplementedError) as captured:
            construct()
        assert captured.value.paper_id == "sigma"
        assert captured.value.method_id == method


@pytest.mark.parametrize("method", ["", "dealer-fss-keys", "shared-nonlinear"])
def test_sigma_pipeline_and_experiment_deserialization_fail_before_execution(method: str) -> None:
    plan = planned_component("sigma", method)
    # Start from ordinary configuration serialization, not a new SIGMA schema.
    control = Experiment(
        "sigma-local-is-not-a-decoder",
        Pipeline(model=Model.tiny(), components={}),
        Deployment.local(root="local://sigma-boundary-test"),
        ExecutionBudget(requests=1, max_input_tokens=1, max_new_tokens=1),
    )
    document = json.loads(control.canonical_bytes())
    document["pipeline"]["components"]["linear"] = {
        "component": plan.identity,
        "params": {},
    }
    # Deserialize actual JSON so no preconstructed object bypasses registry gates.
    restored = json.loads(json.dumps(document))
    with pytest.raises(NotYetImplementedError, match="sigma"):
        Pipeline.from_spec(restored["pipeline"])
    with pytest.raises(NotYetImplementedError, match="sigma"):
        Experiment.from_spec(restored)
