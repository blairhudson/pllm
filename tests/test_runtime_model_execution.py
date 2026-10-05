from __future__ import annotations

import asyncio
import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest
import torch

import pllm
from pllm.nonlinear import (
    create_logrow_scaled_silu_q7_reference,
    fit_compact_silu_q7_reference,
)
from pllm.protocols import prepare_logrow_q7_session_reference
from pllm.runtime.loaders import load_hf_directory
from pllm.runtime.model_binding import compile_runtime_model
from pllm.runtime.model_execution import CompiledRuntimeSession, RuntimeExecutionError
from pllm.runtime.tiny_llama import create_tiny_llama_checkpoint
from pllm.runtime.transformer_client import (
    ClientBundle,
    MaskedTransformerClientRuntime,
    RemoteLinear,
    TransformerClientError,
)
from pllm.runtime.transformer_engine import MaskedTransformerEngine


def _tiny_phi_checkpoint(root: Path) -> tuple[Path, dict]:
    from safetensors.torch import save_file

    root.mkdir(parents=True)
    pinned = (
        Path(__file__).resolve().parents[1]
        / "crates/pllm-models/tests/fixtures/Phi-4-mini-instruct-cfbefac-config.json"
    )
    config = json.loads(pinned.read_text(encoding="utf-8"))
    config.update({
        "name_or_path": "tiny-phi-compiled", "hidden_size": 16, "intermediate_size": 32,
        "num_hidden_layers": 1, "num_attention_heads": 2, "num_key_value_heads": 1,
        "vocab_size": 258, "max_position_embeddings": 256,
        "original_max_position_embeddings": 64, "sliding_window": 256,
        "bos_token_id": 0, "eos_token_id": 1, "pad_token_id": 1,
        "pllm_test_tokenizer": "byte",
        "rope_scaling": {
            "type": "longrope", "short_factor": [1.0, 1.25, 2.0],
            "long_factor": [4.0, 5.0, 6.0],
        },
    })
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    weights = torch.Generator().manual_seed(67)

    def matrix(out_width: int, in_width: int) -> torch.Tensor:
        return torch.randn(out_width, in_width, generator=weights) * 0.08

    save_file({
        "model.embed_tokens.weight": matrix(258, 16),
        "model.norm.weight": torch.ones(16),
        "model.layers.0.input_layernorm.weight": torch.ones(16),
        "model.layers.0.post_attention_layernorm.weight": torch.ones(16),
        "model.layers.0.self_attn.qkv_proj.weight": matrix(32, 16),
        "model.layers.0.self_attn.o_proj.weight": matrix(16, 16),
        "model.layers.0.mlp.gate_up_proj.weight": matrix(64, 16),
        "model.layers.0.mlp.down_proj.weight": matrix(16, 32),
    }, root / "model.safetensors")
    return root, config


def _compiled(
    tmp_path: Path,
    *,
    max_input: int = 8,
    max_new: int = 3,
    model_type: str = "qwen2",
    gate_weight_scale: float = 0.08,
    tie_word_embeddings: bool = True,
):
    root = create_tiny_llama_checkpoint(
        tmp_path / "model",
        num_hidden_layers=2,
        model_type=model_type,
        with_qkv_bias=model_type == "qwen2",
        qk_norm=model_type == "qwen3",
        gate_weight_scale=gate_weight_scale,
        tie_word_embeddings=tie_word_embeddings,
    )
    model_id = "tiny-runtime-execution"
    manifest = load_hf_directory(root, model_id=model_id)
    engine = MaskedTransformerEngine(threads=1)
    asyncio.run(engine.load(manifest))
    bundle = ClientBundle.unpack(engine.client_bundle(model_id))
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    plan = pllm.lower_model(
        config,
        batch=1,
        max_input_tokens=max_input,
        max_new_tokens=max_new,
    )
    compiled = compile_runtime_model(plan, bundle)

    def remote(stage_id: str, activation: np.ndarray) -> np.ndarray:
        stage = bundle.stages[stage_id]
        runtime = engine.models[model_id].stages[stage_id]
        result = np.asarray(activation, dtype=np.float32) @ runtime.weight.dequantize().T
        if stage.bias is not None:
            result = result + stage.bias
        return np.ascontiguousarray(result, dtype=np.float32)

    return compiled, bundle, remote


@pytest.mark.parametrize("model_type", ["qwen2", "qwen3", "llama"])
@pytest.mark.parametrize("tokens", [[2], [2, 3, 5]])
def test_semantic_executor_matches_existing_numeric_decoder_across_phases(
    tmp_path: Path, model_type: str, tokens: list[int]
) -> None:
    compiled, bundle, remote = _compiled(tmp_path, model_type=model_type)
    semantic = compiled.runtime(remote)
    legacy = MaskedTransformerClientRuntime(bundle, remote)
    _, semantic_prefill, semantic_cache = semantic.prepare_ids(tokens)
    _, legacy_prefill, legacy_cache = legacy.prepare_ids(tokens)
    np.testing.assert_allclose(semantic_prefill, legacy_prefill, rtol=1e-5, atol=1e-5)
    chosen = int(np.argmax(legacy_prefill))
    np.testing.assert_allclose(
        semantic.decode_step(chosen, semantic_cache)[0],
        legacy.decode_step(chosen, legacy_cache)[0],
        rtol=1e-5,
        atol=1e-5,
    )


def test_dense_gated_decoder_binds_untied_output_head(tmp_path: Path) -> None:
    compiled, bundle, remote = _compiled(
        tmp_path, model_type="llama", tie_word_embeddings=False
    )
    assert compiled._plan.to_dict()["adapter"] == "pllm.dense_gated_decoder.v1"
    assert next(
        row["weight_keys"] for row in bundle.manifest["stages"] if row["id"] == "lm_head"
    ) == ["lm_head.weight"]
    semantic = compiled.runtime(remote)
    legacy = MaskedTransformerClientRuntime(bundle, remote)
    np.testing.assert_allclose(
        semantic.prepare_ids([2, 3, 5])[1],
        legacy.prepare_ids([2, 3, 5])[1],
        rtol=1e-5,
        atol=1e-5,
    )


@pytest.mark.parametrize("model_type", ["qwen2", "qwen3", "llama"])
def test_phase_rotary_reuse_preserves_every_logit_and_kv_bit(tmp_path, monkeypatch, model_type):
    from pllm.runtime import rotary_coefficients

    compiled, _, remote = _compiled(tmp_path, model_type=model_type)
    trajectories = []
    generated_tables = []
    original = rotary_coefficients.default_rotary_coefficients

    def counted(*args):
        generated_tables.append(1)
        return original(*args)

    monkeypatch.setattr(rotary_coefficients, "default_rotary_coefficients", counted)
    for limit in (0, 1 << 20):
        monkeypatch.setattr(rotary_coefficients, "MAX_ROTARY_COEFFICIENT_BYTES", limit)
        runtime = compiled.runtime(remote)
        values = []
        before = len(generated_tables)
        _, logits, cache = runtime.prepare_ids([2, 3, 5])
        for step in range(3):
            values.append(logits.copy())
            for retained in runtime.caches:
                values.extend((retained.key[:retained.length].copy(),
                               retained.value[:retained.length].copy()))
            assert runtime._phase_rotary is None
            if step < 2:
                logits, cache = runtime.decode_step(int(np.argmax(logits)), cache)
        trajectories.append(values)
        assert len(generated_tables) - before == (12 if limit == 0 else 3)
    for old, new in zip(*trajectories, strict=True):
        np.testing.assert_array_equal(old.view(np.uint32), new.view(np.uint32))


def test_phase_rotary_tables_retire_when_a_later_stage_fails(tmp_path):
    compiled, _, remote = _compiled(tmp_path)
    retained = []

    def failing(stage, value):
        if runtime._phase_rotary.retained_bytes:
            retained.append(runtime._phase_rotary)
            raise RuntimeError("remote stage failed")
        return remote(stage, value)

    runtime = compiled.runtime(failing)
    with pytest.raises(RuntimeError, match="remote stage failed"):
        runtime.prepare_ids([2, 3, 5])
    assert retained and runtime._phase_rotary is None
    assert all(item.retained_bytes == 0 and item.positions is None for item in retained)


@pytest.mark.quality
def test_scaled_rotary_checkpoint_binds_and_decodes_against_torch(tmp_path: Path) -> None:
    transformers = pytest.importorskip("transformers")
    from pllm.configuration import Model
    from pllm.profiles import MaskedLinearCpu
    from pllm.quantization import SymmetricPerRow
    from pllm.runtime.model_binding import RuntimeBindingError

    root = create_tiny_llama_checkpoint(
        tmp_path / "scaled", num_hidden_layers=2, model_type="llama",
        with_qkv_bias=False, tie_word_embeddings=False,
    )
    config_path = root / "config.json"
    source = json.loads(config_path.read_text(encoding="utf-8"))
    source["rope_scaling"] = {
        "rope_type": "llama3", "factor": 8.0,
        "original_max_position_embeddings": 32,
        "low_freq_factor": 1.0, "high_freq_factor": 4.0,
    }
    config_path.write_text(json.dumps(source), encoding="utf-8")
    model_id = "scaled-rotary-test"
    engine = MaskedTransformerEngine(threads=1, weight_bits=8, activation_bits=8)
    asyncio.run(engine.load(load_hf_directory(root, model_id=model_id)))
    bundle = ClientBundle.unpack(engine.client_bundle(model_id))
    plan = pllm.lower_model(source, batch=1, max_input_tokens=8, max_new_tokens=2)
    selected = MaskedLinearCpu(
        Model.path(str(root), model_id=model_id),
        quantization=SymmetricPerRow(weight_bits=8, activation_bits=8),
    )
    compiled = compile_runtime_model(plan, bundle, composition=selected)
    assert compiled.complete
    assert plan.runtime_schedule(selected).complete

    def remote(stage_id: str, activation: np.ndarray) -> np.ndarray:
        stage = engine.models[model_id].stages[stage_id]
        result = np.asarray(activation, np.float32) @ stage.weight.dequantize().T
        if stage.bias is not None:
            result += stage.bias
        return np.ascontiguousarray(result, dtype=np.float32)

    tokens = [2, 3, 5]
    runtime = compiled.runtime(remote)
    _, prefill, cache = runtime.prepare_ids(tokens)
    with torch.no_grad():
        reference = transformers.AutoModelForCausalLM.from_pretrained(
            root, local_files_only=True, trust_remote_code=False,
            dtype=torch.float32, attn_implementation="eager",
        ).eval()
        expected_prefill = reference(
            input_ids=torch.tensor([tokens]), use_cache=False,
        ).logits[0, -1].numpy()
        selected_token = int(np.argmax(expected_prefill))
        expected_decode = reference(
            input_ids=torch.tensor([[*tokens, selected_token]]), use_cache=False,
        ).logits[0, -1].numpy()
    decoded, _ = runtime.decode_step(selected_token, cache)
    assert float(np.max(np.abs(prefill - expected_prefill))) < 0.05
    assert float(np.max(np.abs(decoded - expected_decode))) < 0.05
    assert int(np.argmax(prefill)) == selected_token
    assert int(np.argmax(decoded)) == int(np.argmax(expected_decode))

    tampered = dataclasses.replace(bundle, cfg={
        **bundle.cfg, "rope_scaling": {
            **source["rope_scaling"], "low_freq_factor": 2.0,
        },
    })
    with pytest.raises(RuntimeBindingError):
        compile_runtime_model(plan, tampered, composition=selected)
    with pytest.raises(ValueError):
        pllm.lower_model(
            {**source, "rope_scaling": {"rope_type": "linear", "factor": 2.0}},
            batch=1, max_input_tokens=8, max_new_tokens=2,
        )


@pytest.mark.parametrize("client_owned", [False, True])
def test_scaled_rotary_uses_compiled_sdk_and_gateway(
    tmp_path: Path, client_owned: bool,
) -> None:
    from fastapi.testclient import TestClient

    from pllm.profiles import ClientOnlyCpu, MaskedLinearCpu
    from pllm.quantization import SymmetricPerRow
    from pllm.runtime import build_roles

    root = create_tiny_llama_checkpoint(
        tmp_path / "model", num_hidden_layers=1, model_type="llama",
        with_qkv_bias=False,
    )
    config_path = root / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["rope_scaling"] = {
        "rope_type": "llama3", "factor": 8.0,
        "original_max_position_embeddings": 32,
        "low_freq_factor": 1.0, "high_freq_factor": 4.0,
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")
    source = pllm.Model.path(str(root), model_id="scaled-runtime")
    quantization = SymmetricPerRow(weight_bits=8, activation_bits=8)
    experiment = pllm.Experiment(
        name="scaled-client" if client_owned else "scaled-prepared",
        pipeline=(ClientOnlyCpu if client_owned else MaskedLinearCpu)(
            source, quantization=quantization,
        ),
        deployment=pllm.Deployment.local(root=str(tmp_path / "deployment")),
        budget=pllm.ExecutionBudget(requests=2, max_input_tokens=64, max_new_tokens=2),
    )
    with build_roles(experiment) as topology:
        with topology.client() as client:
            result = client.responses.create(model="scaled-runtime", input="A", max_output_tokens=2)
            assert result.usage.input_tokens > 0
            assert (client.privacy_audit.inference_stage_calls == 0) == client_owned
            assert client.privacy_audit.plaintext_prompt_bytes_sent == 0
        with TestClient(topology.gateway_app(local_api_key="scaled-test")) as gateway:
            response = gateway.post(
                "/v1/responses", headers={"Authorization": "Bearer scaled-test"},
                json={"model": "scaled-runtime", "input": "B", "max_output_tokens": 2},
            )
            assert response.status_code == 200, response.text
            assert response.json()["usage"]["input_tokens"] > 0


@pytest.mark.quality
def test_bounded_fused_projection_checkpoint_matches_phi_torch_decoder(tmp_path: Path) -> None:
    transformers = pytest.importorskip("transformers")
    from pllm.configuration import Model
    from pllm.profiles import MaskedLinearCpu
    from pllm.quantization import SymmetricPerRow
    from pllm.runtime.model_binding import RuntimeBindingError

    root, config = _tiny_phi_checkpoint(tmp_path / "phi")
    model_id = "tiny-phi-compiled"
    engine = MaskedTransformerEngine(threads=1, weight_bits=8, activation_bits=8)
    asyncio.run(engine.load(load_hf_directory(root, model_id=model_id)))
    bundle = ClientBundle.unpack(engine.client_bundle(model_id))
    selected = MaskedLinearCpu(
        Model.path(str(root), model_id=model_id),
        quantization=SymmetricPerRow(weight_bits=8, activation_bits=8),
    )
    plan = pllm.lower_model(config, batch=1, max_input_tokens=8, max_new_tokens=2)
    compiled = compile_runtime_model(plan, bundle, composition=selected)
    assert compiled.complete
    assert any(
        stage["role"] == "semantic_linear"
        and stage["weight_keys"] == ["model.layers.0.self_attn.qkv_proj.weight"]
        for stage in bundle.manifest["stages"]
    )

    def remote(stage_id: str, activation: np.ndarray) -> np.ndarray:
        stage = engine.models[model_id].stages[stage_id]
        return np.ascontiguousarray(
            np.asarray(activation, dtype=np.float32) @ stage.weight.dequantize().T,
            dtype=np.float32,
        )

    tokens = [2, 3, 5]
    runtime = compiled.runtime(remote)
    _, prefill, cache = runtime.prepare_ids(tokens)
    with torch.no_grad():
        reference = transformers.AutoModelForCausalLM.from_pretrained(
            root, local_files_only=True, trust_remote_code=False,
            dtype=torch.float32, attn_implementation="eager",
        ).eval()
        expected_prefill = reference(
            input_ids=torch.tensor([tokens]), use_cache=False,
        ).logits[0, -1].numpy()
        selected_token = int(np.argmax(expected_prefill))
        expected_decode = reference(
            input_ids=torch.tensor([[*tokens, selected_token]]), use_cache=False,
        ).logits[0, -1].numpy()
    decoded, _ = runtime.decode_step(selected_token, cache)
    assert float(np.max(np.abs(prefill - expected_prefill))) < 0.05
    assert float(np.max(np.abs(decoded - expected_decode))) < 0.05
    assert int(np.argmax(prefill)) == selected_token
    assert int(np.argmax(decoded)) == int(np.argmax(expected_decode))

    session = compiled.session(remote)
    np.testing.assert_allclose(session.prefill_ids(tokens), prefill, atol=1e-6)
    assert session.select_next() == selected_token
    np.testing.assert_allclose(session.decode_selected(), decoded, atol=1e-6)
    session.finish()

    tampered = dataclasses.replace(bundle, cfg={
        **bundle.cfg, "rope_scaling": {
            **config["rope_scaling"], "short_factor": [2.0] * 3,
        },
    })
    with pytest.raises(RuntimeBindingError):
        compile_runtime_model(plan, tampered, composition=selected)
    extended = pllm.lower_model(config, batch=1, max_input_tokens=64, max_new_tokens=2)
    with pytest.raises(ValueError, match="cache re-rotation"):
        extended.runtime_schedule(selected)


@pytest.mark.parametrize("client_owned", [False, True])
def test_bounded_phi_uses_compiled_sdk_and_gateway(tmp_path: Path, client_owned: bool) -> None:
    from fastapi.testclient import TestClient

    from pllm.profiles import ClientOnlyCpu, MaskedLinearCpu
    from pllm.quantization import SymmetricPerRow
    from pllm.runtime import build_roles

    root, _ = _tiny_phi_checkpoint(tmp_path / "phi")
    source = pllm.Model.path(str(root), model_id="tiny-phi-compiled")
    selected = (ClientOnlyCpu if client_owned else MaskedLinearCpu)(
        source, quantization=SymmetricPerRow(weight_bits=8, activation_bits=8),
    )
    experiment = pllm.Experiment(
        name="phi-client" if client_owned else "phi-prepared",
        pipeline=selected,
        deployment=pllm.Deployment.local(root=str(tmp_path / "deployment")),
        budget=pllm.ExecutionBudget(requests=2, max_input_tokens=32, max_new_tokens=2),
    )
    with build_roles(experiment) as topology:
        with topology.client() as client:
            result = client.responses.create(
                model="tiny-phi-compiled", input="A", max_output_tokens=2,
            )
            assert result.usage.input_tokens > 0
            assert (client.privacy_audit.inference_stage_calls == 0) == client_owned
            assert client.privacy_audit.plaintext_prompt_bytes_sent == 0
        with TestClient(topology.gateway_app(local_api_key="phi-test")) as gateway:
            response = gateway.post(
                "/v1/responses", headers={"Authorization": "Bearer phi-test"},
                json={"model": "tiny-phi-compiled", "input": "B", "max_output_tokens": 2},
            )
            assert response.status_code == 200, response.text
            assert response.json()["usage"]["input_tokens"] > 0


def test_compiled_decode_rejects_nonfinite_restored_cache_before_remote_work(tmp_path: Path) -> None:
    compiled, _, remote = _compiled(tmp_path)
    calls: list[str] = []

    def counted(stage_id: str, values: np.ndarray) -> np.ndarray:
        calls.append(stage_id)
        return remote(stage_id, values)

    runtime = compiled.runtime(counted)
    _, _, cache = runtime.prepare_ids([2, 3])
    prior_calls = len(calls)
    cache[0].key[0, 0, 0] = np.nan
    with pytest.raises(TransformerClientError, match="cache state is unavailable"):
        runtime.decode_step(4, cache)
    assert len(calls) == prior_calls


@pytest.mark.parametrize("model_type", ["qwen2", "qwen3", "llama"])
def test_provider_materializes_stages_from_semantic_schedule(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model_type: str
) -> None:
    from pllm.configuration import Model
    from pllm.profiles import MaskedLinearCpu
    from pllm.runtime import transformer_engine
    from pllm.runtime.semantic_stages import scheduled_stage_specs

    root = create_tiny_llama_checkpoint(
        tmp_path / "model",
        model_type=model_type,
        qk_norm=model_type == "qwen3",
        with_qkv_bias=model_type == "qwen2",
    )
    config = json.loads((root / "config.json").read_text())
    plan = pllm.lower_model(config, batch=1, max_input_tokens=1, max_new_tokens=1)
    expected = scheduled_stage_specs(plan, MaskedLinearCpu(Model("semantic-provider")))

    def reject_legacy_stage_plan(**kwargs: object) -> None:
        raise AssertionError("provider used the handwritten stage plan")

    def reject_family_dispatch(*args: object) -> None:
        raise AssertionError("provider used family-name dispatch")

    monkeypatch.setattr(
        transformer_engine, "transformer_stage_plan", reject_legacy_stage_plan, raising=False
    )
    monkeypatch.setattr(transformer_engine, "classify_architecture", reject_family_dispatch)
    manifest = load_hf_directory(root, model_id="semantic-provider")
    engine = MaskedTransformerEngine(threads=1)
    asyncio.run(engine.load(manifest))
    bundle = ClientBundle.unpack(engine.client_bundle("semantic-provider"))
    assert (
        pllm.lower_model(
            engine.models["semantic-provider"].config,
            batch=1,
            max_input_tokens=2,
            max_new_tokens=2,
        ).digest
        == pllm.lower_model(
            bundle.cfg,
            batch=1,
            max_input_tokens=2,
            max_new_tokens=2,
        ).digest
    )
    assert [(row.id, row.role, row.weight_keys) for row in manifest.stages] == [
        (row.id, row.role, row.weight_keys) for row in expected
    ]


@pytest.mark.parametrize("model_type", ["qwen2", "qwen3"])
def test_bounded_logrow_research_executes_full_tiny_decoder_without_profile_promotion(
    tmp_path: Path,
    model_type: str,
) -> None:
    compiled, _, remote = _compiled(
        tmp_path,
        max_input=2,
        max_new=2,
        model_type=model_type,
        gate_weight_scale=0.008,
    )
    profile = fit_compact_silu_q7_reference(bytes(257 * 4))
    material = prepare_logrow_q7_session_reference(
        compiled._plan,
        profile,
        max_elements=128,
        max_evaluator_material_bytes=128 * 2144,
        max_decode_steps=1,
        max_session_evaluator_material_bytes=384 * 2144,
    )
    session = compiled.session(remote, research_logrow_material=material)
    assert not session.complete
    assert session.completeness_scope == "bounded_local_research_reference"
    assert session.nonlinear_method == "pllm/logrow-q7-local-reference/v1"
    assert session.binding_digest != compiled.digest
    assert session.runtime_schedule_digest == compiled.runtime_schedule_digest
    logits = session.prefill_ids([0, 2])
    assert np.all(np.isfinite(logits))
    assert material.remaining_tensors == 2
    session.select_next()
    decoded = session.decode_selected()
    assert np.all(np.isfinite(decoded))
    assert material.remaining_tensors == 0
    session.select_next()
    assert session.status == "exhausted"


def test_bounded_logrow_research_rejects_short_prefill_and_burns_every_tensor(
    tmp_path: Path,
) -> None:
    compiled, _, remote = _compiled(tmp_path, max_input=2, max_new=2)
    profile = fit_compact_silu_q7_reference(bytes(257 * 4))
    material = prepare_logrow_q7_session_reference(
        compiled._plan,
        profile,
        max_elements=128,
        max_evaluator_material_bytes=128 * 2144,
        max_decode_steps=1,
        max_session_evaluator_material_bytes=384 * 2144,
    )
    session = compiled.session(remote, research_logrow_material=material)
    with pytest.raises(RuntimeExecutionError, match="exact planned token count"):
        session.prefill_ids([0])
    assert material.remaining_tensors == 0


def test_bounded_logrow_research_refuses_out_of_domain_checkpoint_gates(tmp_path: Path) -> None:
    compiled, _, remote = _compiled(tmp_path, max_input=2, max_new=2)
    profile = fit_compact_silu_q7_reference(bytes(257 * 4))
    material = prepare_logrow_q7_session_reference(
        compiled._plan,
        profile,
        max_elements=128,
        max_evaluator_material_bytes=128 * 2144,
        max_decode_steps=1,
        max_session_evaluator_material_bytes=384 * 2144,
    )
    session = compiled.session(remote, research_logrow_material=material)
    with pytest.raises(RuntimeExecutionError, match="research LogRow nonlinear execution failed"):
        session.prefill_ids([0, 2])
    assert session.status == "poisoned"
    assert material.remaining_tensors == 0


@pytest.mark.parametrize("model_type", ["qwen2", "qwen3"])
def test_public_range_logrow_reference_runs_ordinary_tiny_checkpoint(
    tmp_path: Path, model_type: str
) -> None:
    compiled, _, remote = _compiled(tmp_path, max_input=2, max_new=2, model_type=model_type)
    profile = create_logrow_scaled_silu_q7_reference(4)
    material = prepare_logrow_q7_session_reference(
        compiled._plan,
        profile,
        max_elements=128,
        max_evaluator_material_bytes=128 * 2144,
        max_decode_steps=1,
        max_session_evaluator_material_bytes=384 * 2144,
    )
    session = compiled.session(remote, research_logrow_material=material)
    assert session.nonlinear_method == "pllm/logrow-q7-local-reference/v1"
    assert not session.complete
    logits = session.prefill_ids([0, 2])
    assert np.all(np.isfinite(logits))
    assert material.remaining_tensors == 2
    session.select_next()
    decoded = session.decode_selected()
    assert np.all(np.isfinite(decoded))
    assert material.remaining_tensors == 0
    session.select_next()
    assert session.status == "exhausted"


def test_bounded_logrow_research_rejects_unbound_and_replayed_material(tmp_path: Path) -> None:
    compiled, _, remote = _compiled(tmp_path, max_input=2, max_new=2, gate_weight_scale=0.008)
    profile = fit_compact_silu_q7_reference(bytes(257 * 4))
    with pytest.raises(TypeError, match="native session handle"):
        compiled.session(remote, research_logrow_material=object())
    material = prepare_logrow_q7_session_reference(
        compiled._plan,
        profile,
        max_elements=128,
        max_evaluator_material_bytes=128 * 2144,
        max_decode_steps=1,
        max_session_evaluator_material_bytes=384 * 2144,
    )
    session = compiled.session(remote, research_logrow_material=material)
    with pytest.raises(ValueError, match="already bound"):
        compiled.session(remote, research_logrow_material=material)
    assert material.remaining_tensors == 4
    material.abort()
    with pytest.raises(RuntimeExecutionError, match="research LogRow nonlinear execution failed"):
        session.prefill_ids([0, 2])
    assert session.status == "poisoned"
    assert material.remaining_tensors == 0


def test_session_enforces_greedy_prefill_decode_and_bounds(tmp_path: Path) -> None:
    compiled, bundle, remote = _compiled(tmp_path, max_new=2)
    session = compiled.session(remote)
    assert isinstance(session, CompiledRuntimeSession)
    assert session.complete is True
    assert session.completeness_scope == "whole_decoder_runtime"
    assert session.binding_digest == compiled.digest
    assert session.runtime_schedule_digest == compiled.runtime_schedule_digest
    assert session.status == "new"
    assert session.position == 0
    assert session.generated_tokens == 0
    assert session.remaining_tokens == 2

    logits = session.prefill_ids([0, 2])
    assert logits.flags.writeable is False
    assert session.status == "ready"
    assert session.position == 2
    first = session.select_next()
    assert first == int(np.argmax(logits))
    assert session.status == "selected"
    assert session.generated_tokens == 1
    assert session.remaining_tokens == 1

    next_logits = session.decode_selected()
    assert next_logits.flags.writeable is False
    assert session.status == "ready"
    assert session.position == 3
    second = session.select_next()
    assert second == int(np.argmax(next_logits))
    assert session.status == "exhausted"
    assert session.generated_tokens == 2
    assert session.remaining_tokens == 0
    assert session._runtime.position == 0
    assert all(cache.key is None and cache.value is None for cache in session._runtime.caches)
    with pytest.raises(RuntimeExecutionError, match="no available logits"):
        _ = session.logits
    with pytest.raises(RuntimeExecutionError, match="no selected token"):
        session.decode_selected()
    with pytest.raises(RuntimeExecutionError, match="only once"):
        session.prefill_ids([0])

    direct = compiled.runtime(remote)
    _, direct_logits, caches = direct.prepare_ids([0, 2])
    direct_first = int(np.argmax(direct_logits))
    direct_next, _ = direct.decode_step(direct_first, caches)
    assert (first, second) == (direct_first, int(np.argmax(direct_next)))
    assert bundle.stages["token_lookup"].client_weight is bundle.stages["lm_head"].client_weight


def test_generate_ids_executes_locked_feedback_chain(tmp_path: Path) -> None:
    compiled, _, remote = _compiled(tmp_path, max_new=3)
    generated = compiled.session(remote).generate_ids([0, 2], max_new_tokens=2)
    assert len(generated) == 2
    assert all(isinstance(token, int) for token in generated)

    full = compiled.session(remote)
    assert len(full.generate_ids([0, 2])) == 3
    assert full.status == "exhausted"
    assert full.position == 4
    assert full.generated_tokens == 3


def test_session_executes_masked_stage_protocol(tmp_path: Path) -> None:
    root = create_tiny_llama_checkpoint(tmp_path / "masked" / "model", num_hidden_layers=1)
    model_id = "tiny-runtime-masked-execution"
    manifest = load_hf_directory(root, model_id=model_id)
    engine = MaskedTransformerEngine(threads=1)
    asyncio.run(engine.load(manifest))
    bundle = ClientBundle.unpack(engine.client_bundle(model_id))
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    plan = pllm.lower_model(config, batch=1, max_input_tokens=4, max_new_tokens=2)
    compiled = compile_runtime_model(plan, bundle)

    class Provider:
        model_id = "tiny-runtime-masked-execution"

        def take_many(self, stage, count):
            return engine.create_local_correlations(model_id, stage.id, count)

    def exchange(stage_id: str, payloads: list[bytes]) -> list[bytes]:
        stage = engine.models[model_id].stages[stage_id].spec
        return asyncio.run(engine.execute_stage(model_id, stage, payloads))

    masked = RemoteLinear(bundle.stages, Provider(), exchange)
    masked_tokens = compiled.session(masked).generate_ids([0, 2], max_new_tokens=2)

    def clear(stage_id: str, activation: np.ndarray) -> np.ndarray:
        stage = engine.models[model_id].stages[stage_id]
        result = np.asarray(activation, np.float32) @ stage.weight.dequantize().T
        if stage.bias is not None:
            result = result + stage.bias
        return np.ascontiguousarray(result, np.float32)

    assert masked_tokens == compiled.session(clear).generate_ids([0, 2], max_new_tokens=2)
    assert masked.stats.calls == 8
    assert masked.stats.correlations > 0
    assert masked.stats.upload_bytes > 0
    assert masked.stats.download_bytes > 0


def test_session_rejects_invalid_inputs_without_advancing(tmp_path: Path) -> None:
    compiled, bundle, remote = _compiled(tmp_path, max_input=3, max_new=2)
    session = compiled.session(remote)
    for token_ids in ([], [0, 1, 2, 3], [0.5], [-1], [int(bundle.cfg["vocab_size"])]):
        with pytest.raises(RuntimeExecutionError):
            session.prefill_ids(token_ids)
        assert session.status == "new"
        assert session.position == 0
    with pytest.raises(RuntimeExecutionError, match="not ready"):
        session.select_next()
    for limit in (0, 3, True, 1.5):
        with pytest.raises(RuntimeExecutionError):
            session.generate_ids([0], max_new_tokens=limit)
        assert session.status == "new"


def test_session_poisoning_blocks_partial_state_reuse(tmp_path: Path) -> None:
    compiled, bundle, remote = _compiled(tmp_path, max_new=2)
    calls = 0

    def failing(stage_id: str, activation: np.ndarray) -> np.ndarray:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("remote failure")
        return remote(stage_id, activation)

    failed = compiled.session(failing)
    with pytest.raises(RuntimeExecutionError, match="prefill failed"):
        failed.prefill_ids([0, 2])
    assert failed.status == "poisoned"
    assert failed.position == 0
    with pytest.raises(RuntimeExecutionError):
        failed.prefill_ids([0])

    advanced = compiled.session(remote)
    advanced.prefill_ids([0, 2])
    advanced.select_next()
    stage_id = "layers.0.self_attn.qkv_proj"
    original = bundle.stages[stage_id]
    bundle.stages[stage_id] = dataclasses.replace(
        original,
        weight_scales=original.weight_scales * 2,
    )
    with pytest.raises(RuntimeExecutionError, match="decode failed"):
        advanced.decode_selected()
    assert advanced.status == "poisoned"
    bundle.stages[stage_id] = original


def test_session_rejects_forged_runtime_output_and_direct_construction(tmp_path: Path) -> None:
    compiled, bundle, remote = _compiled(tmp_path)
    with pytest.raises(
        RuntimeExecutionError,
        match="CompiledRuntimeSession must be created by CompiledRuntimeModel.session",
    ):
        CompiledRuntimeSession()

    for kind in ("shape", "dtype", "finite"):

        def malformed(stage_id: str, activation: np.ndarray) -> np.ndarray:
            if not stage_id.endswith("o_proj"):
                return remote(stage_id, activation)
            stage = bundle.stages[stage_id]
            if kind == "shape":
                return np.zeros((activation.shape[0], 1), np.float32)
            if kind == "dtype":
                return np.zeros((activation.shape[0], stage.out_features), np.float64)
            return np.full((activation.shape[0], stage.out_features), np.nan, np.float32)

        session = compiled.session(malformed)
        with pytest.raises(RuntimeExecutionError, match="output is malformed"):
            session.prefill_ids([0, 2])
        assert session.status == "poisoned"
        session.close()
        assert session.status == "closed"


def test_session_can_finish_before_plan_bound(tmp_path: Path) -> None:
    compiled, _, remote = _compiled(tmp_path, max_new=3)
    session = compiled.session(remote)
    session.prefill_ids([0])
    session.select_next()
    assert session.status == "selected"
    session.finish()
    assert session.status == "exhausted"
    assert session.generated_tokens == 1
    with pytest.raises(RuntimeExecutionError, match="no available logits"):
        _ = session.logits
    session.close()
    assert session.status == "closed"
