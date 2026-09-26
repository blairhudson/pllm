"""Model-neutral missing capabilities have named, fail-closed API contracts."""

from __future__ import annotations

import importlib
from dataclasses import replace

import pytest

from pllm import Experiment
from pllm.components import (
    ComponentRef,
    ModelCapabilityUnavailable,
    create_component,
    get,
    list_components,
    model_capabilities,
    model_capability,
)
from pllm.components import _model_capabilities
from pllm.configuration import ConfigurationError
from pllm.providers import discover_providers

from test_providers import provider_fixture


def test_missing_model_capabilities_have_one_importable_nonexecuting_class_each() -> None:
    stubs = model_capabilities()
    assert len(stubs) == 9
    assert len({stub.identity for stub in stubs}) == len(stubs)
    assert {item.component for item in list_components()}.isdisjoint({stub.identity for stub in stubs})
    for stub in stubs:
        assert model_capability(stub.slug) is stub
        assert stub.sdk_route == f"/sdk/models/capabilities/{stub.capability}/"
        module = importlib.import_module(stub.module)
        component = getattr(module, stub.name)
        assert stub.name in module.__all__
        assert component.planned_identity == stub.identity
        assert component.next_gate == stub.gate
        with pytest.raises(ModelCapabilityUnavailable) as captured:
            component()
        assert (captured.value.identity, captured.value.sdk_route) == (stub.identity, stub.sdk_route)
        assert stub.gate in str(captured.value)


def test_missing_model_capabilities_reject_direct_serialized_and_factory_selection() -> None:
    for stub in model_capabilities():
        for entry in (
            lambda: ComponentRef(stub.identity),
            lambda: create_component(stub.identity, {}),
            lambda: get(stub.identity),
        ):
            with pytest.raises(ModelCapabilityUnavailable, match=stub.name):
                entry()
        with pytest.raises(ModelCapabilityUnavailable, match=stub.name):
            Experiment.from_spec({
                "schema": "pllm.experiment.v2",
                "name": "unavailable-model-capability",
                "pipeline": {
                    "model": {"source": "org/model"},
                    "components": {"state": {"component": stub.identity, "params": {}}},
                },
                "deployment": {"kind": "local", "root": "local://unavailable-capability"},
                "budget": {"requests": 1, "max_input_tokens": 1, "max_new_tokens": 1},
            })
    for identity in (
        "pllm/planned-model-capability/unknown/v1",
        "pllm/planned-model-capability/gated-delta/v2",
    ):
        with pytest.raises(ConfigurationError, match="unknown reserved model capability"):
            ComponentRef(identity)
        with pytest.raises(ConfigurationError, match="unknown reserved model capability"):
            create_component(identity, {})


def test_provider_cannot_claim_missing_model_capability(tmp_path) -> None:
    stub = model_capability("gated-delta")
    entry_point, _manifest, _path, _package, _distribution = provider_fixture(
        tmp_path, component=stub.identity,
    )
    with pytest.raises(ModelCapabilityUnavailable, match=stub.name):
        discover_providers(entry_points=[entry_point])
    assert entry_point.loaded is False


def test_promoted_model_capability_retires_placeholder_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    stub = model_capability("gated-delta")
    monkeypatch.setattr(
        _model_capabilities,
        "_by_slug",
        lambda: {stub.slug: replace(stub, status="implemented")},
    )
    with pytest.raises(ConfigurationError, match="retired planned identity"):
        ComponentRef(stub.identity)
