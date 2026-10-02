"""Provider-side admission for the native full-KV continuation extension."""

from __future__ import annotations

from typing import Any

from pllm.configuration import Pipeline
from pllm.modeling import ModelPlan
from pllm.profiles import resolve_runtime_composition

from .semantic_stages import client_owns_linear, scheduled_stage_specs


def admit_continuation(value: Any, plan: ModelPlan, engine: Any, model_id: str) -> dict:
    if type(value) is not dict or set(value) != {"composition", "contract"}:
        raise ValueError("invalid decoder continuation extension envelope")
    composition = Pipeline.from_spec(value["composition"])
    options = resolve_runtime_composition(composition)
    if options is None or options.privacy_mode != "public" or options.verification_component is not None:
        raise ValueError("continuation requires unverified prepared execution")
    expected = {
        "weight_bits": options.weight_bits,
        "activation_bits": options.activation_bits,
        "public_equalization_digest": options.public_equalization_digest,
        "remote_output_head": options.remote_output_head,
        "client_prefix_layers": options.client_prefix_layers,
        "client_linear_roles": tuple(options.client_linear_roles),
    }
    if any(getattr(engine, key, None) != item for key, item in expected.items()):
        raise ValueError("continuation numeric/placement contract differs from provider")
    if getattr(engine, "verification_component", "none") != "none":
        raise ValueError("verified provider cannot admit baseline continuation")
    specs = scheduled_stage_specs(plan, composition)
    remote = {
        spec.id: spec for spec in specs
        if spec.id != "token_lookup"
        and (spec.id != "lm_head" or options.remote_output_head)
        and not client_owns_linear(spec, client_prefix_layers=options.client_prefix_layers,
                                   client_linear_roles=tuple(options.client_linear_roles))
    }
    if set(remote) != set(engine.seeded_stage_ids(model_id)):
        raise ValueError("continuation stages differ from provider")
    actual = engine.models[model_id].stages
    for stage_id, spec in remote.items():
        installed = actual[stage_id].spec
        if (spec.in_features, spec.out_features, spec.weight_keys) != (
            installed.in_features, installed.out_features, installed.weight_keys
        ):
            raise ValueError("continuation stage shape/artifacts differ from provider")
    contract = plan.continuation_schedule(composition)
    expected_slot = contract.handshake_spec()
    if value["contract"] != expected_slot:
        raise ValueError("continuation native commitment differs from provider")
    # The maximum legal prefix/suffix split must fit before any inventory row is
    # reserved. Actual private prefix lengths remain entirely client-local.
    contract.admit(1, expected_slot["token_bound"] - 1)
    return expected_slot
