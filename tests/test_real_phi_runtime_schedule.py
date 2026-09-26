"""Opt-in pinned Phi checkpoint functionality and same-token quality gate."""

from __future__ import annotations

import asyncio
import gc
import hashlib
import os
from pathlib import Path

import numpy as np
import pytest

from pllm import Model, lower_model, load_model
from pllm.metrics import measure_reference_agreement
from pllm.profiles import MaskedLinearCpu
from pllm.quantization import PublicPerChannelEqualized, SymmetricPerRow
from pllm.runtime.model_binding import compile_runtime_model
from pllm.runtime.public_equalization import load_public_equalization_profile
from pllm.runtime.quantization import dequantize_matmul, quantize_activation_per_row
from pllm.runtime.safetensors_store import SafeTensorStore
from pllm.runtime.transformer_client import ClientBundle
from pllm.runtime.transformer_engine import MaskedTransformerEngine


REVISION = "cfbefacb99257ffa30c83adab238a50856ac3083"
MODEL_ID = f"microsoft/Phi-4-mini-instruct@{REVISION}"
CONFIG_SHA256 = "ac65d86061d3d0d704ee2511fd0eb8713ef19eb6eedba17c3080a4165d5b933b"
SHARDS = {
    "model-00001-of-00002.safetensors": "bc703090b63eda16f639fa4de7ac54635c23105ab1da2f6ec4d3403151d38ee6",
    "model-00002-of-00002.safetensors": "7ff79b9d2d31076bac2663393451f6530f4fc8ca49b09002116c92c373dba983",
}
EQUALIZATION_PROFILE_DIGEST = "6e2074d87d546ac98e2eb8518ff26495996a34b9f3202bfc2ee9ecc953eea683"


@pytest.mark.rust
@pytest.mark.slow
@pytest.mark.quality
@pytest.mark.skipif(not os.environ.get("PLLM_REAL_PHI_PATH"), reason="set PLLM_REAL_PHI_PATH")
def test_pinned_phi_full_decoder_functionality_and_quality(request: pytest.FixtureRequest) -> None:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    original_threads = torch.get_num_threads()
    torch.set_num_threads(4)
    request.addfinalizer(lambda: torch.set_num_threads(original_threads))
    root = Path(os.environ["PLLM_REAL_PHI_PATH"]).resolve()
    config_bytes = (root / "config.json").read_bytes()
    assert hashlib.sha256(config_bytes).hexdigest() == CONFIG_SHA256

    # Fixed, public, short-context same-token cohorts. No logits or IDs in reports.
    cohorts = ((9259,), (818, 5279, 529, 7001, 563))
    reference = transformers.AutoModelForCausalLM.from_pretrained(
        root, local_files_only=True, trust_remote_code=False,
        dtype=torch.float32, low_cpu_mem_usage=True, attn_implementation="eager",
    ).eval()
    expected_rows: list[tuple[np.ndarray, int, np.ndarray]] = []
    with torch.inference_mode():
        for ids in cohorts:
            expected = reference(input_ids=torch.tensor([ids]), use_cache=False).logits[0, -1].float().numpy().copy()
            selected = int(np.argmax(expected))
            decode = reference(
                input_ids=torch.tensor([[*ids, selected]]), use_cache=False,
            ).logits[0, -1].float().numpy().copy()
            expected_rows.append((expected, selected, decode))
    del reference
    gc.collect()

    manifest = load_model(Model.path(str(root), model_id=MODEL_ID))
    source_lock = manifest.metadata["source_lock"]
    assert source_lock["commit"] == REVISION
    files = {row["path"]: row["sha256"] for row in source_lock["files"]}
    assert {name: files[name] for name in SHARDS} == SHARDS
    engine = MaskedTransformerEngine(weight_bits=8, activation_bits=8, threads=4)
    asyncio.run(engine.load(manifest))
    bundle = ClientBundle.unpack(engine.client_bundle(MODEL_ID))
    plan = lower_model(config_bytes, batch=1, max_input_tokens=5, max_new_tokens=2)
    composition = MaskedLinearCpu(Model.path(str(root), model_id=MODEL_ID))
    compiled = compile_runtime_model(plan, bundle, composition=composition)
    assert compiled.complete and len(compiled.stage_bindings) == 130
    assert compiled.runtime_schedule_digest == plan.runtime_schedule(composition).digest

    stages = engine.models[MODEL_ID].stages
    store = SafeTensorStore(root)
    results = {}
    for mode in ("float32_body", "w8a8"):
        def remote(stage_id: str, activation: np.ndarray) -> np.ndarray:
            stage = stages[stage_id]
            if mode == "float32_body":
                keys = stage.spec.weight_keys
                source = [store.get(key) for key in keys]
                weight = source[0] if len(source) == 1 else np.concatenate(source)
                output = np.ascontiguousarray(activation @ weight.T, dtype=np.float32)
                del source, weight
                store.drop(*keys)
            else:
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
        for ids, (expected, reference_token, expected_decode) in zip(cohorts, expected_rows, strict=True):
            runtime = compiled.runtime(remote)
            _, actual, cache = runtime.prepare_ids(ids)
            decoded, _ = runtime.decode_step(reference_token, cache)
            assert actual.shape == decoded.shape == (200064,)
            assert np.all(np.isfinite(actual)) and np.all(np.isfinite(decoded))
            measurements.append((
                measure_reference_agreement(actual, expected, top_k=5),
                measure_reference_agreement(decoded, expected_decode, top_k=5),
            ))
        results[mode] = measurements

    # Float body is a semantic oracle, not a served precision option. W8A8
    # divergence is recorded without treating full execution as quality parity.
    for prefill, decode in results["float32_body"]:
        assert prefill["top1_agreement"] == decode["top1_agreement"] == 1.0
        assert prefill["max_abs_logit_error"] < 2.5
        assert decode["max_abs_logit_error"] < 2.5
    for prefill, decode in results["w8a8"]:
        assert prefill["max_abs_logit_error"] < 40.0
        assert decode["max_abs_logit_error"] < 40.0
    print(f"Phi original-context compiled numeric cohort: {results}")


@pytest.mark.rust
@pytest.mark.slow
@pytest.mark.skipif(not os.environ.get("PLLM_REAL_PHI_PATH"), reason="set PLLM_REAL_PHI_PATH")
def test_pinned_phi_equalized_prepared_sdk_through_two_children(tmp_path: Path) -> None:
    from pllm import Deployment, ExecutionBudget, Experiment
    from pllm.runtime import build_roles

    root = Path(os.environ["PLLM_REAL_PHI_PATH"]).resolve()
    source = Model.path(str(root), model_id=MODEL_ID)
    manifest = load_model(source)
    profile = load_public_equalization_profile(
        root, EQUALIZATION_PROFILE_DIGEST, manifest.source_lock_digest,
    )
    assert len(profile.stage_scales) == 128
    bodies = []
    for index, quantization in enumerate((
        SymmetricPerRow(weight_bits=8, activation_bits=8),
        PublicPerChannelEqualized(profile.digest),
    )):
        experiment = Experiment(
            name=f"Pinned Phi prepared numeric choice {index}",
            pipeline=MaskedLinearCpu(source, quantization=quantization),
            deployment=Deployment.local(root=str(tmp_path / f"deployment-{index}")),
            budget=ExecutionBudget(requests=1, max_input_tokens=32, max_new_tokens=1),
        )
        with build_roles(experiment, startup_timeout=600) as topology:
            assert {status.role for status in topology.statuses} == {"inference", "preparation"}
            with topology.client() as client:
                result = client.responses.create(
                    model=MODEL_ID, input="The capital of France is", max_output_tokens=1,
                )
                assert result.usage.input_tokens > 0
                audit = client.privacy_audit
                assert audit.inference_stage_calls == 128
                assert audit.plaintext_prompt_bytes_sent == 0
                assert audit.plaintext_token_ids_sent == 0
                assert audit.preparation_requests_during_online == 0
                assert audit.masked_online_upload_bytes > 0
                bodies.append((
                    audit.masked_online_upload_bytes, audit.masked_online_download_bytes,
                    audit.bundle_network_bytes,
                ))
    assert bodies[1][0] < bodies[0][0]
    assert bodies[1][1] < bodies[0][1]
    assert bodies[0][2] < bodies[1][2]
    print("Pinned Phi prepared body counts (baseline, calibrated):", bodies)
