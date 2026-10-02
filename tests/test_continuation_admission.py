from __future__ import annotations

import copy
import dataclasses
import json

import httpx
import pytest

from pllm.runtime.continuation_admission import admit_continuation
from pllm.runtime.masked_runtime import ModelError
from pllm.runtime.servers import build_roles
from test_batched_prefix_reuse import prefix_experiment
from test_decoder_continuation import bound_checkpoint


@pytest.fixture
def bound_provider(tmp_path):
    from pllm.runtime.tiny_llama import create_tiny_llama_checkpoint

    root = create_tiny_llama_checkpoint(tmp_path / "model", seed=149, tie_word_embeddings=False)
    engine, compiled, contract, composition = bound_checkpoint(root)
    envelope = {"composition": composition.to_spec(), "contract": contract.handshake_spec()}
    return engine, compiled, envelope


@pytest.mark.parametrize("field", [
    "digest", "source_plan_digest", "model_config_digest", "composition_digest",
    "source_schedule_digest", "numeric_digest", "token_bound", "schema",
])
def test_provider_reconstructs_every_native_binding(bound_provider, field):
    engine, compiled, envelope = bound_provider
    assert admit_continuation(envelope, compiled._plan, engine, compiled._bundle.model_id) == envelope["contract"]
    forged = copy.deepcopy(envelope)
    forged["contract"][field] = 63 if field == "token_bound" else "f" * 64
    with pytest.raises(ValueError, match="native commitment"):
        admit_continuation(forged, compiled._plan, engine, compiled._bundle.model_id)
    assert all(stage.calls == 0 for stage in engine.models[compiled._bundle.model_id].stages.values())


@pytest.mark.parametrize("attribute,value", [
    ("weight_bits", 4), ("activation_bits", 4), ("remote_output_head", True),
    ("client_prefix_layers", 1), ("client_linear_roles", ("attention_out",)),
    ("public_equalization_digest", "a" * 64), ("verification_component", "pllm/freivalds-verify/v1"),
])
def test_provider_checks_actual_numeric_and_ownership(bound_provider, monkeypatch, attribute, value):
    engine, compiled, envelope = bound_provider
    monkeypatch.setattr(engine, attribute, value)
    with pytest.raises(ValueError, match="numeric/placement|verified provider"):
        admit_continuation(envelope, compiled._plan, engine, compiled._bundle.model_id)


@pytest.mark.parametrize("field", ["in_features", "out_features", "weight_keys"])
def test_provider_checks_installed_stage_geometry_and_weight_artifacts(bound_provider, monkeypatch, field):
    engine, compiled, envelope = bound_provider
    model_id = compiled._bundle.model_id
    stage = engine.models[model_id].stages[engine.seeded_stage_ids(model_id)[0]]
    value = ("other.weight",) if field == "weight_keys" else getattr(stage.spec, field) + 1
    monkeypatch.setattr(stage, "spec", dataclasses.replace(stage.spec, **{field: value}))
    with pytest.raises(ValueError, match="shape/artifacts"):
        admit_continuation(envelope, compiled._plan, engine, model_id)


def test_real_http_forged_contract_never_reserves_provider_rows_and_client_burns(tmp_path, monkeypatch):
    model_id, experiment = prefix_experiment(tmp_path)
    with build_roles(experiment, engine_threads=1) as topology:
        # Each independently issued horizon starts at row zero. Test actual
        # provider cursor as well as client ledger; no simulated stage counters.
        for field in (
            "digest", "source_plan_digest", "model_config_digest", "composition_digest",
            "source_schedule_digest", "numeric_digest", "token_bound", "schema", "extra_field",
            "body_fingerprint", "stage_commitment", "runtime_config_digest",
            "weight_bits", "activation_bits", "placement",
        ):
            with topology.client(background_inventory_refill=False) as client:
                client.preprocess(model_id, count=64)
                core = client._core
                inventory = core._transformer_states[model_id].prepared_inventory
                endpoint = f"/v1/runtime/inventories/{inventory.id}"
                before_provider = core.http.get(endpoint, headers=core.headers).json()
                before_client = inventory.status()
                original_post = core.http.post

                def forge(path, **kwargs):
                    if path == "/v1/runtime/sessions":
                        body = copy.deepcopy(kwargs["json"])
                        if field in {"body_fingerprint", "stage_commitment", "runtime_config_digest"}:
                            body["decoder_plan"][field] = "f" * 64
                        elif field in {"weight_bits", "activation_bits", "placement"}:
                            from pllm.configuration import Pipeline

                            document = body["decoder_continuation"]["composition"]
                            if field in {"weight_bits", "activation_bits"}:
                                document["components"]["quantization"]["params"][field] = 4
                            elif field == "placement":
                                document["components"]["placement"] = {
                                    "component": "pllm/client-owned-prefix-layers/v1", "params": {"layers": 1}
                                }
                            # Give the forged pipeline its own valid native digest;
                            # installed numeric/placement/source binding must still reject.
                            compiled = core._compiled_public_decoder(
                                core._transformer_states[model_id], max_input_tokens=248, max_new_tokens=1
                            )
                            body["decoder_continuation"]["contract"] = compiled._plan.continuation_schedule(
                                Pipeline.from_spec(json.loads(json.dumps(document)))
                            ).handshake_spec()
                        else:
                            body["decoder_continuation"]["contract"][field] = 247 if field == "token_bound" else "f" * 64
                        kwargs["json"] = body
                    return original_post(path, **kwargs)

                with monkeypatch.context() as patch:
                    patch.setattr(core.http, "post", forge)
                    with pytest.raises(Exception, match="continuation is invalid|body differs from provider"):
                        client.responses.create(model=model_id, input="admission forge", temperature=0, max_output_tokens=1)
                after_provider = core.http.get(endpoint, headers=core.headers).json()
                after_client = inventory.status()
                assert before_provider["next_row"] == after_provider["next_row"] == 0
                assert core.audit.inference_stage_calls == 0
                assert after_client["consumed"] == before_client["consumed"] == 0
                assert after_client["burned"] > before_client["burned"]
                assert after_client["reserved"] == 0


@pytest.mark.parametrize("missing", [True, False])
def test_real_http_missing_or_forged_ack_burns_reserved_rows_without_stage_calls(tmp_path, monkeypatch, missing):
    model_id, experiment = prefix_experiment(tmp_path)
    with build_roles(experiment, engine_threads=1) as topology, topology.client(background_inventory_refill=False) as client:
        base = "ordinary repeated prefix " * 3
        client.responses.create(model=model_id, input=base, temperature=0, max_output_tokens=1)
        branch = (base[:40] + "different suffix").ljust(len(base), "X")
        assert client.prepared_rows_for_response(branch, 1, model=model_id) < client.prepared_rows_for_response(branch, 1, model=model_id, store=False)
        client.preprocess(model_id, count=128)
        core = client._core
        inventory = core._transformer_states[model_id].prepared_inventory
        endpoint = f"/v1/runtime/inventories/{inventory.id}"
        before_provider = core.http.get(endpoint, headers=core.headers).json()
        before_client = inventory.status()
        calls = core.audit.inference_stage_calls
        original_post = core.http.post

        def tamper_ack(path, **kwargs):
            response = original_post(path, **kwargs)
            if path == "/v1/runtime/sessions" and response.is_success:
                value = response.json()
                if missing:
                    value.pop("decoder_continuation")
                else:
                    value["decoder_continuation"]["numeric_digest"] = "f" * 64
                return httpx.Response(response.status_code, json=value, request=response.request)
            return response

        with monkeypatch.context() as patch:
            patch.setattr(core.http, "post", tamper_ack)
            with pytest.raises(ModelError, match="did not admit"):
                client.responses.create(model=model_id, input=branch, temperature=0, max_output_tokens=1)
        after_provider = core.http.get(endpoint, headers=core.headers).json()
        after_client = inventory.status()
        reserved_rows = after_provider["next_row"] - before_provider["next_row"]
        assert reserved_rows > 0
        assert after_client["burned"] - before_client["burned"] == reserved_rows
        assert after_client["consumed"] == before_client["consumed"]
        assert after_client["reserved"] == 0 and core.audit.inference_stage_calls == calls
