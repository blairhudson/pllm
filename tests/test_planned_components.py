"""Pending research methods remain discoverable without becoming executable."""

from __future__ import annotations

import importlib
import ast
from dataclasses import replace
from pathlib import Path

import pytest

from pllm import Experiment
from pllm.components import (
    ComponentRef,
    NotYetImplementedError,
    create_component,
    get,
    list_components,
    planned_component,
    planned_components,
)
from pllm.configuration import ConfigurationError
from pllm.components import _planned
from pllm.providers import discover_providers

from test_providers import provider_fixture


def test_every_mapped_source_exposes_status_bound_component_methods() -> None:
    plans = planned_components()
    assert len({plan.paper_id for plan in plans}) == 85
    assert len({plan.paper_id for plan in plans if not plan.method_id}) == 85
    assert len(plans) > 85
    assert len({plan.identity for plan in plans}) == len(plans)
    assert len({(plan.module, plan.name) for plan in plans}) == len(plans)
    assert {descriptor.component for descriptor in list_components()}.isdisjoint(
        {plan.identity for plan in plans}
    )

    for plan in plans:
        module = importlib.import_module(plan.module)
        implementation = getattr(module, plan.name)
        assert module.__file__ is not None
        stub_path = Path(module.__file__).with_suffix(".pyi")
        typed = {
            node.name
            for node in ast.parse(stub_path.read_text(encoding="utf-8")).body
            if isinstance(node, ast.ClassDef)
        }
        assert plan.name in typed, f"missing typing declaration for {plan.module}.{plan.name}"
        assert plan.name in module.__all__
        assert implementation.__module__ == plan.module
        assert planned_component(plan.paper_id, plan.method_id) is plan
        assert plan.title and plan.summary
        assert plan.paper_url.startswith("https://pllm.run/research/papers/")
        assert plan.sdk_route.startswith("/sdk/reference/python/pllm/")
        if plan.status == "pending":
            assert implementation.paper_id == plan.paper_id
            assert implementation.method_id == plan.method_id
            assert implementation.paper_url == plan.paper_url
            assert implementation.planned_identity == plan.identity
            with pytest.raises(NotYetImplementedError) as captured:
                implementation()
            error = captured.value
            assert isinstance(error, ConfigurationError)
            assert isinstance(error, NotImplementedError)
            assert (error.paper_id, error.method_id, error.paper_url, error.identity, error.gate) == (
                plan.paper_id, plan.method_id, plan.paper_url, plan.identity, plan.gate
            )
            assert plan.paper_url in str(error)
            assert plan.gate in str(error)
        else:
            assert plan.status == "implemented"
            assert not issubclass(implementation, _planned.PendingComponent)
            if plan.kind == "candidate":
                assert issubclass(implementation, ComponentRef)


def test_planned_identity_fails_before_generic_configuration_or_search() -> None:
    for plan in (item for item in planned_components() if item.status == "pending"):
        for entry in (
            lambda: ComponentRef(plan.identity),
            lambda: create_component(plan.identity, {}),
            lambda: get(plan.identity),
        ):
            with pytest.raises(NotYetImplementedError, match=plan.paper_id):
                entry()

    for identity in ("pllm/planned/unknown/v1", "pllm/planned/ring-pcg/v2", "pllm/planned/ring-pcg/forged/v1"):
        with pytest.raises(ConfigurationError, match="reserved planned component"):
            ComponentRef(identity)
        with pytest.raises(ConfigurationError, match="reserved planned component"):
            create_component(identity, {})
        with pytest.raises(ConfigurationError, match="reserved planned component"):
            get(identity)

    with pytest.raises(ConfigurationError, match="ring-pcg"):
        Experiment.from_spec({
            "schema": "pllm.experiment.v2",
            "name": "cannot-search-pending",
            "pipeline": {
                "model": {"source": "org/model"},
                "components": {"correlation": {
                    "component": planned_component("ring-pcg").identity,
                    "params": {},
                }},
            },
            "deployment": {"kind": "local", "root": "local://cannot-search-pending"},
            "budget": {"requests": 1, "max_input_tokens": 1, "max_new_tokens": 1},
        })


@pytest.mark.parametrize("paper,method", [("ring-pcg", ""), ("breaking-euston", ""), ("maverick", "coded-verification")])
def test_provider_cannot_claim_a_planned_identity_without_importing_factory(tmp_path, paper: str, method: str) -> None:
    entry_point, _manifest, _path, _package, _distribution = provider_fixture(
        tmp_path, component=planned_component(paper, method).identity
    )
    with pytest.raises(ConfigurationError, match=paper):
        discover_providers(entry_points=[entry_point])
    assert entry_point.loaded is False


def test_promoted_method_cannot_reuse_the_planned_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    plan = planned_component("ring-pcg")
    monkeypatch.setattr(_planned, "_by_identity", lambda: {plan.identity: replace(plan, status="implemented")})
    with pytest.raises(ConfigurationError, match="retired planned identity"):
        ComponentRef(plan.identity)
