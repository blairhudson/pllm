from __future__ import annotations

import asyncio
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

import pllm
from pllm.runtime.model_binding import compile_runtime_model
from pllm.profiles import MaskedLinearCpu, VerifiedMaskedLinearCpu
from pllm.runtime.quantization import dequantize_matmul, quantize_activation_per_row
from pllm.runtime.safetensors_store import SafeTensorStore
from pllm.runtime.transformer_client import ClientBundle
from pllm.runtime.transformer_engine import MaskedTransformerEngine

QWEN2_PATH = os.environ.get("PLLM_REAL_QWEN_PATH")
QWEN3_PATH = os.environ.get("PLLM_REAL_QWEN3_PATH")
pytestmark = [
    pytest.mark.rust,
    pytest.mark.slow,
]


def _quantized_remote(engine: MaskedTransformerEngine, model_id: str):
    model = engine.models[model_id]

    def remote(stage_id: str, activation: np.ndarray) -> np.ndarray:
        stage = model.stages[stage_id]
        quantized = quantize_activation_per_row(
            activation, bits=stage.spec.activation_bits
        )
        integer = stage.compiled_weight.clear(quantized.values)
        output = dequantize_matmul(
            integer,
            quantized.scales,
            stage.weight.scales,
            output_shape=quantized.original_shape[:-1] + (stage.spec.out_features,),
        )
        if stage.bias is not None:
            output = output + stage.bias
        return np.ascontiguousarray(output, dtype=np.float32)

    return remote


@pytest.mark.parametrize(
    ("model_path", "model_id", "expected_stages", "config_sha256"),
    [
        (QWEN2_PATH, "Qwen/Qwen2.5-0.5B-Instruct@runtime-schedule-test", 98, None),
        (
            QWEN3_PATH,
            "Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca",
            114,
            "660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd",
        ),
    ],
    ids=["qwen2.5-0.5b", "qwen3-0.6b"],
)
def test_real_qwen_checkpoint_binds_and_executes_complete_schedule(
    model_path: str | None, model_id: str, expected_stages: int, config_sha256: str | None
) -> None:
    if not model_path:
        pytest.skip("the pinned real checkpoint path is not configured")
    assert model_path is not None
    root = Path(model_path).resolve()
    config_bytes = (root / "config.json").read_bytes()
    if config_sha256 is not None:
        assert hashlib.sha256(config_bytes).hexdigest() == config_sha256
    config = json.loads(config_bytes)
    manifest = pllm.load_model(pllm.Model.path(str(root), model_id=model_id))
    if config_sha256 is not None:
        lock = manifest.metadata["source_lock"]
        assert lock["commit"] == "c1899de289a04d12100db370d81485cdf75e47ca"
        assert next(row for row in lock["files"] if row["path"] == "model.safetensors")[
            "sha256"
        ] == "f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b"
    engine = MaskedTransformerEngine(threads=4)
    asyncio.run(engine.load(manifest))
    bundle = ClientBundle.unpack(engine.client_bundle(model_id))
    plan = pllm.lower_model(config, batch=1, max_input_tokens=16, max_new_tokens=2)
    composition = MaskedLinearCpu(pllm.Model(model_id))
    schedule = plan.runtime_schedule(composition)
    compiled = compile_runtime_model(plan, bundle, composition=composition)
    remote = _quantized_remote(engine, model_id)
    runtime = compiled.runtime(remote)
    prompt_ids = runtime.encode_prompt("Hello")[-16:]
    session = compiled.session(remote)
    logits = session.prefill_ids(prompt_ids)
    first = session.select_next()
    next_logits = session.decode_selected()
    second = session.select_next()

    assert first == int(np.argmax(logits))
    assert second == int(np.argmax(next_logits))
    assert session.complete is True
    assert session.completeness_scope == "whole_decoder_runtime"
    assert session.status == "exhausted"
    assert session.position == len(prompt_ids) + 1
    assert len(manifest.checkpoint_digest or "") == 64
    assert len(manifest.source_lock_digest or "") == 64
    assert plan.coverage(composition).complete is True
    assert plan.coverage(VerifiedMaskedLinearCpu(pllm.Model(model_id))).complete is False
    assert schedule.complete is True
    assert schedule.protected_execution is False
    assert schedule.digest == compiled.runtime_schedule_digest
    assert len(compiled.stage_bindings) == expected_stages
    assert next_logits.shape == (config["vocab_size"],)
    assert np.all(np.isfinite(next_logits))
    assert 0 <= first < config["vocab_size"]
    assert 0 <= second < config["vocab_size"]
    assert runtime.bundle is bundle
    assert (
        bundle.stages["token_lookup"].client_weight
        is bundle.stages["lm_head"].client_weight
    )
    assert (
        bundle.stages["token_lookup"].client_weight_scales
        is bundle.stages["lm_head"].client_weight_scales
    )


@pytest.mark.skipif(not QWEN2_PATH, reason="PLLM_REAL_QWEN_PATH is not configured")
def test_real_qwen_offset_topology_executes_compiled_session(tmp_path: Path) -> None:
    from pllm import Deployment, ExecutionBudget, Experiment, Model
    from pllm.profiles import ClientOnlyCpu, TwoOnlineOffsetCpu
    from pllm.quantization import SymmetricPerRow
    from pllm.runtime.servers import build_roles

    assert QWEN2_PATH is not None
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    experiment = Experiment(
        name="real-offset-runtime",
        pipeline=TwoOnlineOffsetCpu(
            Model.path(QWEN2_PATH, model_id=model_id),
            quantization=SymmetricPerRow(weight_bits=8, activation_bits=8),
        ),
        deployment=Deployment.local(root=str(tmp_path / "offset")),
        budget=ExecutionBudget(requests=1, max_input_tokens=64, max_new_tokens=1),
    )
    with build_roles(experiment) as topology, topology.client() as client:
        response = client.responses.create(model=model_id, input="A", max_output_tokens=1)
        assert response.usage.output_tokens == 1
        assert client.privacy_audit.plaintext_prompt_bytes_sent == 0
        assert client.privacy_audit.plaintext_token_ids_sent == 0
    local = Experiment(
        name="real-offset-control",
        pipeline=ClientOnlyCpu(
            Model.path(QWEN2_PATH, model_id=model_id),
            quantization=SymmetricPerRow(weight_bits=8, activation_bits=8),
        ),
        deployment=Deployment.local(root=str(tmp_path / "control")),
        budget=experiment.budget,
    )
    with build_roles(local) as topology, topology.client() as client:
        control = client.responses.create(model=model_id, input="A", max_output_tokens=1)
    assert response.output_text == control.output_text


@pytest.mark.skipif(not QWEN2_PATH, reason="PLLM_REAL_QWEN_PATH is not configured")
def test_real_qwen_verified_prepared_topology_matches_client(tmp_path: Path) -> None:
    from pllm import Deployment, ExecutionBudget, Experiment, Model
    from pllm.profiles import ClientOnlyCpu, VerifiedMaskedLinearCpu
    from pllm.quantization import SymmetricPerRow
    from pllm.roles import PreparedProviderRoles
    from pllm.runtime.servers import build_roles
    from pllm.verification import FreivaldsVerify

    assert QWEN2_PATH is not None
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    model = Model.path(QWEN2_PATH, model_id=model_id)
    bits = SymmetricPerRow(weight_bits=8, activation_bits=8)
    budget = ExecutionBudget(requests=1, max_input_tokens=64, max_new_tokens=1)
    verified = Experiment(
        name="real-verified-runtime",
        pipeline=VerifiedMaskedLinearCpu(
            model,
            quantization=bits,
            verification=FreivaldsVerify(target_failure_bits=40),
            topology=PreparedProviderRoles(),
        ),
        deployment=Deployment.local(root=str(tmp_path / "verified")),
        budget=budget,
    )
    with build_roles(verified) as topology, topology.client() as client:
        result = client.responses.create(model=model_id, input="A", max_output_tokens=1)
        assert result.usage.output_tokens == 1
        assert client.privacy_audit.inference_stage_calls == 96
        assert client.privacy_audit.plaintext_prompt_bytes_sent == 0
        assert client.privacy_audit.plaintext_token_ids_sent == 0
        assert client.privacy_audit.preparation_requests_during_online == 0
    local = Experiment(
        name="real-verified-control",
        pipeline=ClientOnlyCpu(model, quantization=bits),
        deployment=Deployment.local(root=str(tmp_path / "control")),
        budget=budget,
    )
    with build_roles(local) as topology, topology.client() as client:
        control = client.responses.create(model=model_id, input="A", max_output_tokens=1)
    assert result.output_text == control.output_text


@pytest.mark.skipif(not QWEN3_PATH, reason="PLLM_REAL_QWEN3_PATH is not configured")
def test_real_qwen3_has_scoped_clear_reference_parity() -> None:
    assert QWEN3_PATH is not None
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    torch.set_num_threads(4)
    root = Path(QWEN3_PATH).resolve()
    model_id = "Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca"
    assert hashlib.sha256((root / "config.json").read_bytes()).hexdigest() == (
        "660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd"
    )
    manifest = pllm.load_model(pllm.Model.path(str(root), model_id=model_id))
    engine = MaskedTransformerEngine(weight_bits=8, activation_bits=8, threads=4)
    asyncio.run(engine.load(manifest))
    bundle = ClientBundle.unpack(engine.client_bundle(model_id))
    plan = pllm.lower_model(
        (root / "config.json").read_bytes(), batch=1, max_input_tokens=16, max_new_tokens=2
    )
    compiled = compile_runtime_model(
        plan, bundle, composition=MaskedLinearCpu(pllm.Model(model_id))
    )
    reference = transformers.AutoModelForCausalLM.from_pretrained(
        root,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.float32,
        attn_implementation="eager",
    ).eval()
    store = SafeTensorStore(root)

    # The float32-body comparison is a local semantic oracle, not a private
    # execution mode. W8A8 is a narrower numeric check on one prompt; separate
    # probes show W4A4 and short-prompt W8A8 do not meet these fidelity bounds.
    for mode, prompt, max_error in (
        ("float32_body", "Hello", 0.75),
        ("w8a8", "The capital of France is", 3.0),
    ):
        if mode == "float32_body":

            def remote(stage_id: str, activation: np.ndarray) -> np.ndarray:
                stage = engine.models[model_id].stages[stage_id]
                assert stage.bias is None
                weights = np.concatenate([store.get(key) for key in stage.spec.weight_keys])
                return np.ascontiguousarray(activation @ weights.T, dtype=np.float32)

        else:
            remote = _quantized_remote(engine, model_id)
        ids = compiled.runtime(remote).encode_prompt(prompt)[-16:]
        session = compiled.session(remote)
        actual = session.prefill_ids(ids)
        with torch.inference_mode():
            reference_prefill = reference(input_ids=torch.tensor([ids]), use_cache=True)
        expected = reference_prefill.logits[0, -1].float().numpy()
        assert int(np.argmax(actual)) == int(np.argmax(expected))
        assert len(set(np.argsort(actual)[-5:]) & set(np.argsort(expected)[-5:])) >= 4
        assert float(np.max(np.abs(actual - expected))) < max_error
        if mode == "w8a8":
            selected = session.select_next()
            actual_decode = session.decode_selected()
            with torch.inference_mode():
                reference_decode = reference(
                    input_ids=torch.tensor([[selected]]),
                    past_key_values=reference_prefill.past_key_values,
                    use_cache=True,
                )
            expected_decode = reference_decode.logits[0, -1].float().numpy()
            assert int(np.argmax(actual_decode)) == int(np.argmax(expected_decode))
            assert len(
                set(np.argsort(actual_decode)[-5:]) & set(np.argsort(expected_decode)[-5:])
            ) >= 4
            assert float(np.max(np.abs(actual_decode - expected_decode))) < 2.0
