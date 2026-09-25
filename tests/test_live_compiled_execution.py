import asyncio
import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from jsonschema import Draft202012Validator

from conftest import start_gateway, start_preparation
from pllm import Deployment, ExecutionBudget, Experiment, Model, Pipeline
from pllm.modeling import lower_model
from pllm.profiles import MaskedLinearCpu
from pllm.quantization import SymmetricPerRow
from pllm.roles import PreparedProviderRoles
from pllm.runtime import OpenAI
from pllm.runtime.loaders import load_hf_directory
from pllm.runtime.servers import build_roles
from pllm.runtime.tiny_llama import create_tiny_llama_checkpoint
from pllm.runtime.transformer_engine import MaskedTransformerEngine


@pytest.mark.parametrize(
    ("bits", "reject", "with_experiment", "model_type", "explicit_topology"),
    [
        (4, False, True, "qwen2", False),
        (8, False, True, "qwen2", False),
        (8, True, True, "qwen2", False),
        (8, False, False, "qwen2", False),
        (8, False, True, "llama", False),
        (8, False, True, "qwen2", True),
    ],
)
def test_live_prepared_decoder_binds_compiled_schedule_before_use(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bits: int,
    reject: bool,
    with_experiment: bool,
    model_type: str,
    explicit_topology: bool,
) -> None:
    root = create_tiny_llama_checkpoint(
        tmp_path / "model", num_hidden_layers=1, model_type=model_type,
        with_qkv_bias=model_type == "qwen2",
    )
    model_id = "live-compiled"
    engine = MaskedTransformerEngine(threads=1, weight_bits=bits, activation_bits=bits)
    gateway = start_gateway(engines={engine.capabilities.name: engine})
    preparation_engine = MaskedTransformerEngine(threads=1, weight_bits=bits, activation_bits=bits)
    asyncio.run(preparation_engine.load(load_hf_directory(root, model_id=model_id)))
    preparation = start_preparation(preparation_engine, gateway.base_url, gateway.push_api_key)
    from pllm.runtime import model_binding
    from pllm.runtime.model_binding import RuntimeBindingError

    original = model_binding.compile_runtime_model
    bound: list[tuple[str, str]] = []

    def record(plan, bundle, *, composition):
        if reject:
            raise RuntimeBindingError("incomplete compiler binding")
        compiled = original(plan, bundle, composition=composition)
        bound.append((compiled.runtime_schedule_digest, compiled.digest))
        return compiled

    monkeypatch.setattr(model_binding, "compile_runtime_model", record)
    try:
        with httpx.Client(base_url=gateway.base_url, timeout=30) as admin:
            response = admin.post(
                "/v1/runtime/models/load",
                headers={"Authorization": f"Bearer {gateway.api_key}"},
                json={
                    "engine": engine.capabilities.name,
                    "kind": "huggingface",
                    "path": str(root),
                    "model_id": model_id,
                },
            )
            assert response.status_code == 200, response.text
        selected = MaskedLinearCpu(
            Model.path(str(root), model_id=model_id),
            quantization=SymmetricPerRow(weight_bits=bits, activation_bits=bits),
        )
        pipeline = (
            Pipeline(
                model=selected.model,
                components={**selected.components, "topology": PreparedProviderRoles()},
            )
            if explicit_topology else selected
        )
        experiment = Experiment(
            name="live-compiled",
            pipeline=pipeline,
            deployment=Deployment.local(root="local://live-compiled"),
            budget=ExecutionBudget(requests=1, max_input_tokens=64, max_new_tokens=2),
        )
        with OpenAI(
            api_key=gateway.api_key,
            base_url=gateway.base_url,
            preparation_base_url=preparation.base_url,
            preparation_api_key=preparation.api_key,
            background_inventory_refill=False,
            experiment=experiment if with_experiment else None,
        ) as client:
            if reject:
                with pytest.raises(RuntimeBindingError, match="incomplete compiler binding"):
                    client.responses.create(
                        model=model_id, input="tiny parity", max_output_tokens=2, temperature=0
                    )
                assert client.privacy_audit.preparation_upload_bytes == 0
                assert client.privacy_audit.online_steps == 0
            else:
                result = client.responses.create(
                    model=model_id, input="tiny parity", max_output_tokens=2, temperature=0
                )
                assert result.usage is not None and result.usage.output_tokens <= 2
                assert client.privacy_audit.online_steps > 0
                assert bound and all(len(digest) == 64 for pair in bound for digest in pair)
                if not with_experiment:
                    continued = client.responses.create(
                        model=model_id,
                        previous_response_id=result.id,
                        input="follow up",
                        max_output_tokens=1,
                        temperature=0,
                    )
                    assert continued.usage is not None
                    assert client.privacy_audit.kv_continuation_hits == 1
            assert client.privacy_audit.plaintext_prompt_bytes_sent == 0
    finally:
        gateway.close()
        preparation.close()


def test_provider_checks_semantic_plan_before_reserving_material(tmp_path: Path) -> None:
    root = create_tiny_llama_checkpoint(tmp_path / "model", num_hidden_layers=1)
    model_id = "semantic-session-check"
    engine = MaskedTransformerEngine(threads=1, weight_bits=8, activation_bits=8)
    gateway = start_gateway(engines={engine.capabilities.name: engine})
    try:
        with httpx.Client(base_url=gateway.base_url, timeout=30) as admin:
            headers = {"Authorization": f"Bearer {gateway.api_key}"}
            loaded = admin.post(
                "/v1/runtime/models/load",
                headers=headers,
                json={
                    "engine": engine.capabilities.name,
                    "kind": "huggingface",
                    "path": str(root),
                    "model_id": model_id,
                },
            )
            assert loaded.status_code == 200, loaded.text
            valid = lower_model(
                engine.models[model_id].config,
                batch=1,
                max_input_tokens=2,
                max_new_tokens=2,
            ).digest
            engine.client_bundle(model_id)
            metadata = engine.models[model_id].manifest.metadata
            body = {
                "model": model_id,
                "execution": "seeded-preparation",
                "max_output_tokens": 2,
                "inventory_id": "not-created",
                "inventory_start": 0,
                "inventory_rows": 2,
                "decoder_plan": {
                    "schema": "pllm.decoder_session.v1",
                    "digest": "0" * 64,
                    "max_input_tokens": 2,
                    "body_fingerprint": metadata["body_fingerprint"],
                    "stage_commitment": metadata["seeded_stage_commitment"],
                    "runtime_config_digest": metadata["runtime_config_digest"],
                },
            }
            schema = json.loads(Path("schemas/decoder-session.schema.json").read_text())
            validator = Draft202012Validator(schema)
            assert validator.is_valid(body["decoder_plan"])
            assert not validator.is_valid({**body["decoder_plan"], "max_input_tokens": True})
            assert not validator.is_valid({**body["decoder_plan"], "token_ids": [1]})
            missing = admin.post(
                "/v1/runtime/sessions",
                headers=headers,
                json={key: value for key, value in body.items() if key != "decoder_plan"},
            )
            assert missing.status_code == 409
            assert "contract is required" in missing.text
            wrong = admin.post("/v1/runtime/sessions", headers=headers, json=body)
            assert wrong.status_code == 409
            assert "plan differs" in wrong.text
            body["decoder_plan"]["digest"] = valid
            body["decoder_plan"]["body_fingerprint"] = "0" * 64
            wrong_body = admin.post("/v1/runtime/sessions", headers=headers, json=body)
            assert wrong_body.status_code == 409
            assert "body differs" in wrong_body.text
            body["decoder_plan"]["body_fingerprint"] = metadata["body_fingerprint"]
            ready = admin.post("/v1/runtime/sessions", headers=headers, json=body)
            assert ready.status_code == 409
            assert "Prepared inventory is not ready" in ready.text
    finally:
        gateway.close()


@pytest.mark.parametrize(
    ("weight_bits", "model_type"),
    [(None, "qwen2"), (4, "qwen2"), (None, "llama")],
    ids=["implicit-w8a8", "experiment-w4a4", "dense-bias-free"],
)
def test_gateway_uses_compiled_decoder_with_real_local_role_children(
    tmp_path: Path, weight_bits: int | None, model_type: str
) -> None:
    root = create_tiny_llama_checkpoint(
        tmp_path / "gateway-model", num_hidden_layers=1,
        model_type=model_type, with_qkv_bias=model_type == "qwen2",
    )
    model_id = "tiny-compiled-gateway"
    source = Model.path(str(root), model_id=model_id)
    target = (
        source
        if weight_bits is None
        else Experiment(
            name="tiny compiled W4A4",
            pipeline=MaskedLinearCpu(
                source,
                quantization=SymmetricPerRow(weight_bits=weight_bits, activation_bits=weight_bits),
            ),
            budget=ExecutionBudget(requests=1, max_input_tokens=64, max_new_tokens=2),
            deployment=Deployment.local(root=str(tmp_path)),
        )
    )
    if isinstance(target, Experiment):
        target = Experiment.from_spec(target.to_spec())
        assert type(target.pipeline) is Pipeline
    topology = build_roles(
        target,
        engine_threads=1,
        log_dir=tmp_path / "role-logs",
    ).start()
    try:
        with TestClient(topology.gateway_app(local_api_key="loopback-key")) as gateway:
            response = gateway.post(
                "/v1/responses",
                headers={"Authorization": "Bearer loopback-key"},
                json={
                    "model": model_id,
                    "input": "hi",
                    "max_output_tokens": 1,
                    "temperature": 0,
                    "store": False,
                },
            )
            assert response.status_code == 200, response.text
            assert response.json()["usage"]["output_tokens"] == 1
    finally:
        topology.close()
