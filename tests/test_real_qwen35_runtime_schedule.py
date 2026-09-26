"""Opt-in pinned Qwen3.5 text-checkpoint binding and reference-quality gates."""

from __future__ import annotations

import asyncio
import gc
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from pllm import Model, load_model, lower_model
from pllm.metrics import measure_reference_agreement
from pllm.profiles import MaskedLinearCpu
from pllm.runtime.model_binding import compile_runtime_model
from pllm.runtime.quantization import dequantize_matmul, quantize_activation_per_row
from pllm.runtime.safetensors_store import SafeTensorStore
from pllm.runtime.semantic_tensors import required_client_tensors
from pllm.runtime.transformer_client import ClientBundle
from pllm.runtime.transformer_engine import MaskedTransformerEngine


REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
MODEL_ID = f"Qwen/Qwen3.5-4B@{REVISION}"
CONFIG_SHA256 = "ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c0621670"
SHARDS = {
    "model.safetensors-00001-of-00002.safetensors": "26a93f066e1916adb13453dae5a0c707c0fbc71299ed98779571a907b8e74c61",
    "model.safetensors-00002-of-00002.safetensors": "cb544bd9bfae93dc59b0f22b292f5933573854a7f9b97835c67060d7d910e188",
}


@pytest.mark.rust
@pytest.mark.slow
@pytest.mark.skipif(not os.environ.get("PLLM_REAL_QWEN35_PATH"), reason="set PLLM_REAL_QWEN35_PATH")
def test_pinned_text_checkpoint_binds_every_required_artifact() -> None:
    root = Path(os.environ["PLLM_REAL_QWEN35_PATH"]).resolve()
    config_bytes = (root / "config.json").read_bytes()
    assert hashlib.sha256(config_bytes).hexdigest() == CONFIG_SHA256
    assert root.name == REVISION
    manifest = load_model(Model.path(str(root), model_id=MODEL_ID))
    source_lock = manifest.metadata["source_lock"]
    assert source_lock["commit"] is None  # Local snapshot; file hashes bind its actual contents.
    files = {row["path"]: row["sha256"] for row in source_lock["files"]}
    assert {name: files[name] for name in SHARDS} == SHARDS
    assert len(manifest.stages) == 130

    plan = lower_model(config_bytes, batch=1, max_input_tokens=5, max_new_tokens=2)
    schedule = plan.runtime_schedule(MaskedLinearCpu(Model.path(str(root), model_id=MODEL_ID)))
    assert schedule.complete
    store = SafeTensorStore(root)
    operations = {row["id"]: row for row in plan.prefill["operations"]}
    expected: dict[str, tuple[int, ...]] = {}
    for step in schedule.to_dict()["prefill"]["steps"]:
        if step["executor"] != "remote_stage":
            continue
        for operation_id in step["operation_ids"]:
            operation = operations[operation_id]
            name = operation["attributes"].get("weight")
            assert name not in expected or expected[name] == store.tensor_shape(name)
            if operation["operator"] == "linear":
                source = operations[operation["inputs"][0]]
                expected[name] = (operation["output_shape"][-1], source["output_shape"][-1])
            elif operation["operator"] in {"token_lookup", "output_head"}:
                expected[name] = (manifest.vocab_size, manifest.hidden_size)
            else:
                raise AssertionError(f"unexpected remote operator {operation['operator']}")
    for name, shape in required_client_tensors(plan).items():
        if name in expected:
            assert expected[name] == shape
        expected[name] = shape
    assert len(expected) == 426
    mismatches = {
        name: (shape, store.tensor_shape(name), store.tensor_dtype(name))
        for name, shape in expected.items()
        if store.tensor_shape(name) != tuple(shape)
        or store.tensor_dtype(name) not in {"BF16", "F32"}
    }
    assert not mismatches, mismatches
    assert not any(name.startswith(("model.visual.", "mtp.")) for name in expected)


@pytest.mark.rust
@pytest.mark.slow
@pytest.mark.quality
@pytest.mark.skipif(not os.environ.get("PLLM_REAL_QWEN35_PATH"), reason="set PLLM_REAL_QWEN35_PATH")
def test_pinned_text_decoder_prefill_decode_against_bf16_reference(
    request: pytest.FixtureRequest,
) -> None:
    torch = pytest.importorskip("torch")
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration

    original_threads = torch.get_num_threads()
    torch.set_num_threads(4)
    request.addfinalizer(lambda: torch.set_num_threads(original_threads))
    root = Path(os.environ["PLLM_REAL_QWEN35_PATH"]).resolve()
    assert root.name == REVISION
    assert hashlib.sha256((root / "config.json").read_bytes()).hexdigest() == CONFIG_SHA256
    cohorts = ((9259,), (818, 5279, 529, 7001, 563))
    reference = Qwen3_5ForConditionalGeneration.from_pretrained(
        root, local_files_only=True, trust_remote_code=False,
        dtype=torch.bfloat16, low_cpu_mem_usage=True, attn_implementation="eager",
    ).eval()
    expected_rows: list[tuple[np.ndarray, int, np.ndarray]] = []
    with torch.inference_mode():
        for ids in cohorts:
            tensor = torch.tensor([ids], dtype=torch.long)
            expected = reference(
                input_ids=tensor, attention_mask=torch.ones_like(tensor), use_cache=False,
            ).logits[0, -1].float().numpy().copy()
            token = int(np.argmax(expected))
            continuation = torch.tensor([[*ids, token]], dtype=torch.long)
            decoded = reference(
                input_ids=continuation, attention_mask=torch.ones_like(continuation),
                use_cache=False,
            ).logits[0, -1].float().numpy().copy()
            expected_rows.append((expected, token, decoded))
    del reference
    gc.collect()

    manifest = load_model(Model.path(str(root), model_id=MODEL_ID))
    evidence = json.loads(
        (Path(__file__).resolve().parents[1] / "docs/evidence/qwen35-4b-text-reference-2026-09-26.json")
        .read_text(encoding="utf-8")
    )
    assert manifest.source_lock_digest == evidence["checkpoint"]["source_lock_digest"]
    engine = MaskedTransformerEngine(weight_bits=8, activation_bits=8, threads=4)
    asyncio.run(engine.load(manifest))
    bundle = ClientBundle.unpack(engine.client_bundle(MODEL_ID))
    plan = lower_model((root / "config.json").read_bytes(), batch=1, max_input_tokens=5, max_new_tokens=2)
    composition = MaskedLinearCpu(Model.path(str(root), model_id=MODEL_ID))
    compiled = compile_runtime_model(plan, bundle, composition=composition)
    assert compiled.complete and len(compiled.stage_bindings) == 130
    assert compiled.runtime_schedule_digest == plan.runtime_schedule(composition).digest

    stages = engine.models[MODEL_ID].stages

    def remote(stage_id: str, activation: np.ndarray) -> np.ndarray:
        stage = stages[stage_id]
        quantized = quantize_activation_per_row(activation, bits=stage.spec.activation_bits)
        integer = stage.compiled_weight.clear(quantized.values)
        output = dequantize_matmul(
            integer, quantized.scales, stage.weight.scales,
            output_shape=quantized.original_shape[:-1] + (stage.spec.out_features,),
        )
        if stage.bias is not None:
            output += stage.bias
        return np.ascontiguousarray(output, dtype=np.float32)

    measurements = []
    clear_store = SafeTensorStore(root)

    def clear_body(stage_id: str, activation: np.ndarray) -> np.ndarray:
        stage = stages[stage_id]
        keys = stage.spec.weight_keys
        matrices = [clear_store.get(key, dtype=np.float32) for key in keys]
        try:
            matrix = np.concatenate(matrices, axis=0) if len(matrices) > 1 else matrices[0]
            output = np.asarray(activation, dtype=np.float32) @ matrix.T
            if stage.bias is not None:
                output += stage.bias
            return np.ascontiguousarray(output, dtype=np.float32)
        finally:
            clear_store.drop(*keys)

    clear_measurements = []
    for ids, (expected, token, expected_decode) in zip(cohorts, expected_rows, strict=True):
        runtime = compiled.runtime(remote)
        _, actual, cache = runtime.prepare_ids(ids)
        decoded, _ = runtime.decode_step(token, cache)
        clear_runtime = compiled.runtime(clear_body)
        _, clear, clear_cache = clear_runtime.prepare_ids(ids)
        clear_decoded, _ = clear_runtime.decode_step(token, clear_cache)
        assert actual.shape == decoded.shape == expected.shape == (248320,)
        assert np.all(np.isfinite(actual)) and np.all(np.isfinite(decoded))
        measurements.append((
            measure_reference_agreement(actual, expected, top_k=5),
            measure_reference_agreement(decoded, expected_decode, top_k=5),
        ))
        clear_measurements.append((
            measure_reference_agreement(clear, expected, top_k=5),
            measure_reference_agreement(clear_decoded, expected_decode, top_k=5),
        ))
    assert len(measurements) == len(cohorts)
    for name, rows in (
        ("w8a8", measurements),
        ("diagnostic_float32_body_with_w8_token_head_boundary", clear_measurements),
    ):
        aggregate = [item for prefill_decode in rows for item in prefill_decode]
        actual_top1 = sum(item["top1_agreement"] for item in aggregate)
        assert f"{actual_top1:.0f}/{len(aggregate)}" == evidence[name]["top1_agreement"]
        assert max(item["max_abs_logit_error"] for item in aggregate) == pytest.approx(
            evidence[name]["worst_absolute_logit_error"], abs=0.0001,
        )
    print(f"Pinned Qwen3.5 text W8A8 local reference cohort: {measurements}")
    print(f"Pinned Qwen3.5 float32-body diagnostic (quantized token/head): {clear_measurements}")


@pytest.mark.rust
@pytest.mark.slow
@pytest.mark.skipif(not os.environ.get("PLLM_REAL_QWEN35_PATH"), reason="set PLLM_REAL_QWEN35_PATH")
def test_pinned_text_checkpoint_prepared_sdk_and_gateway(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    from pllm import Deployment, ExecutionBudget, Experiment
    from pllm.runtime import build_roles

    root = Path(os.environ["PLLM_REAL_QWEN35_PATH"]).resolve()
    assert root.name == REVISION
    assert hashlib.sha256((root / "config.json").read_bytes()).hexdigest() == CONFIG_SHA256
    experiment = Experiment(
        name="Pinned Qwen3.5 text prepared connectivity",
        pipeline=MaskedLinearCpu(Model.path(str(root), model_id=MODEL_ID)),
        deployment=Deployment.local(root=str(tmp_path / "deployment")),
        budget=ExecutionBudget(requests=2, max_input_tokens=64, max_new_tokens=1),
    )
    with build_roles(experiment) as topology:
        with topology.client() as client:
            result = client.responses.create(
                model=MODEL_ID, input="Hello", max_output_tokens=1,
            )
            assert result.usage.input_tokens > 0
            assert result.usage.output_tokens == 1
            assert client.privacy_audit.inference_stage_calls > 0
            assert client.privacy_audit.plaintext_prompt_bytes_sent == 0
            assert client.privacy_audit.plaintext_token_ids_sent == 0
        with TestClient(topology.gateway_app(local_api_key="hybrid-real-test")) as gateway:
            response = gateway.post(
                "/v1/responses", headers={"Authorization": "Bearer hybrid-real-test"},
                json={"model": MODEL_ID, "input": "World", "max_output_tokens": 1},
            )
            assert response.status_code == 200, response.text
            assert response.json()["usage"]["input_tokens"] > 0
            assert response.json()["usage"]["output_tokens"] == 1
