"""Generated hybrid checkpoint: one shared plan, native stages, and independent Torch oracle."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import numpy as np
import pytest

import pllm
from pllm.profiles import ClientOnlyCpu
from pllm.runtime.loaders import load_hf_directory
from pllm.runtime.model_binding import compile_runtime_model
from pllm.runtime.model_execution import RuntimeExecutionError
from pllm.runtime.semantic_executor import SemanticDecoderRuntime
from pllm.runtime.semantic_tensors import required_client_tensors
from pllm.runtime.transformer_client import ClientBundle
from pllm.runtime.transformer_engine import MaskedTransformerEngine


class _TraceHybrid(SemanticDecoderRuntime):
    def _local(self, operation, values, state_kinds, pending_keys):
        result = super()._local(operation, values, state_kinds, pending_keys)
        if operation["id"] == "layer.3.self_attention.q_norm":
            self.query_norm = np.asarray(result).copy()
        return result


def _generated_hybrid_checkpoint(root: Path):
    torch = pytest.importorskip("torch")
    from safetensors.torch import save_file
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    pinned = Path(__file__).resolve().parents[1] / "crates/pllm-models/tests/fixtures/Qwen3.5-4B-851bf6e-config.json"
    config = json.loads(pinned.read_text(encoding="utf-8"))
    text = config["text_config"]
    text.update({
        "hidden_size": 64, "intermediate_size": 128,
        "num_hidden_layers": 4, "num_attention_heads": 2,
        "num_key_value_heads": 1, "head_dim": 32, "vocab_size": 512,
        "max_position_embeddings": 128, "linear_num_key_heads": 2,
        "linear_num_value_heads": 4, "linear_key_head_dim": 8,
        "linear_value_head_dim": 8,
        "layer_types": ["linear_attention"] * 3 + ["full_attention"],
        "bos_token_id": 0, "eos_token_id": 1, "pad_token_id": 1,
        "pllm_test_tokenizer": "byte",
    })
    text["rope_parameters"]["mrope_section"] = [2, 1, 1]
    config.update({
        "name_or_path": "tiny-qwen35-hybrid", "image_token_id": 500,
        "video_token_id": 501, "vision_start_token_id": 502,
        "vision_end_token_id": 503, "pllm_test_tokenizer": "byte",
    })
    torch.manual_seed(355)
    upstream = Qwen3_5ForCausalLM(Qwen3_5TextConfig(**text)).eval()
    plan = pllm.lower_model(config, batch=1, max_input_tokens=5, max_new_tokens=2)
    source = pllm.Model("tiny-qwen35-hybrid")
    composition = ClientOnlyCpu(source)
    schedule = plan.runtime_schedule(composition)
    assert schedule.complete
    needed = {
        name for step in schedule.to_dict()["prefill"]["steps"]
        for name in step.get("weight_ids", [])
    }
    needed.update(required_client_tensors(plan))
    # Text-only Torch model stores tensors under `model.*`. Official outer
    # checkpoint nests the same decoder under `model.language_model.*`.
    upstream_weights = {
        "model.language_model." + name.removeprefix("model."): value
        for name, value in upstream.state_dict().items()
    }
    absent = sorted(needed - set(upstream_weights))
    assert not absent, f"semantic artifacts missing from upstream text decoder: {absent}"
    root.mkdir(parents=True)
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    save_file({name: upstream_weights[name].contiguous().clone() for name in sorted(needed)}, root / "model.safetensors")
    return root, config, upstream, plan, composition


def test_generated_hybrid_checkpoint_binds_and_matches_torch(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    root, config, upstream, plan, composition = _generated_hybrid_checkpoint(tmp_path / "hybrid")
    model_id = "tiny-qwen35-hybrid"
    manifest = load_hf_directory(root, model_id=model_id)
    assert manifest.stages and manifest.stages[0].role == "token_lookup"
    engine = MaskedTransformerEngine(weight_bits=8, activation_bits=8, threads=2)
    asyncio.run(engine.load(manifest))
    bundle = ClientBundle.unpack(engine.client_bundle(model_id, placement="client"))
    prefill_ops = {op["id"]: op for op in plan.prefill["operations"]}
    decode_ops = {op["id"]: op for op in plan.decode["operations"]}
    prefill_only = set(prefill_ops) - set(decode_ops)
    assert prefill_only == {
        name for name, op in prefill_ops.items()
        if op["operator"] in {"state_initialize", "kv_cache_initialize", "last_token"}
    }
    assert set(decode_ops) <= set(prefill_ops)
    compiled = compile_runtime_model(plan, bundle, composition=composition)
    with pytest.raises(RuntimeExecutionError, match="text-only|multimodal"):
        compiled.session(compiled.client_linear_executor(engine)).prefill_ids(
            [config["image_token_id"]]
        )
    session = compiled.session(compiled.client_linear_executor(engine))
    tokens = [3, 9, 15]
    logits = session.prefill_ids(tokens)
    with torch.no_grad():
        reference = upstream(torch.tensor(tokens, dtype=torch.long)[None], use_cache=False).logits[0, -1]
    weights = {
        "model.language_model." + name.removeprefix("model."): value
        for name, value in upstream.state_dict().items()
    }
    def clear_body(stage_id: str, activation: np.ndarray) -> np.ndarray:
        stage = engine.models[model_id].stages[stage_id]
        ordered = np.concatenate([weights[key].numpy() for key in stage.spec.weight_keys], axis=0)
        out = np.asarray(activation, dtype=np.float32) @ ordered.T
        if stage.bias is not None:
            out = out + stage.bias
        return np.ascontiguousarray(out, dtype=np.float32)
    # Isolated numeric oracle bypasses client-owned stage admission deliberately;
    # the serving path above uses only its binding-checked client kernel.
    clear = _TraceHybrid(
        bundle, clear_body, plan=plan, schedule=plan.runtime_schedule(composition).to_dict(),
        stages={operation: stage.stage_id for stage in compiled._stages for operation in stage.semantic_operations},
        tensors={row["weight_id"]: row["key"] for row in compiled.to_spec()["local_tensors"]},
    )
    clear.query_norm = None
    _, clear_logits, _ = clear.prepare_ids(tokens)
    captured = []
    hook = upstream.model.layers[3].self_attn.q_norm.register_forward_hook(
        lambda _module, _input, output: captured.append(output)
    )
    with torch.no_grad():
        upstream(torch.tensor(tokens, dtype=torch.long)[None], use_cache=False)
    hook.remove()
    assert clear.query_norm is not None and captured
    np.testing.assert_allclose(
        clear.query_norm[0, :, -1, :], captured[0][0, -1].numpy(), atol=0.03, rtol=0.03,
    )
    np.testing.assert_allclose(clear_logits, reference.numpy(), atol=0.05, rtol=0.05)
    np.testing.assert_allclose(logits, reference.numpy(), atol=0.05, rtol=0.05)
    assert np.all(np.isfinite(logits))
    assert session.position == len(tokens)
    selected = session.select_next()
    assert selected == int(np.argmax(logits))
    decoded = session.decode_selected()
    with torch.no_grad():
        reference_decode = upstream(torch.tensor(tokens + [selected], dtype=torch.long)[None], use_cache=False).logits[0, -1]
    np.testing.assert_allclose(decoded, reference_decode.numpy(), atol=0.05, rtol=0.05)
    assert session.position == len(tokens) + 1
    assert config["text_config"]["layer_types"].count("linear_attention") == 3


def test_generated_hybrid_runs_prepared_sdk_and_gateway(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    from pllm import Deployment, ExecutionBudget, Experiment, Model
    from pllm.profiles import MaskedLinearCpu
    from pllm.runtime import build_roles

    root, _, _, _, _ = _generated_hybrid_checkpoint(tmp_path / "hybrid")
    experiment = Experiment(
        name="Generated bounded hybrid text decoder",
        pipeline=MaskedLinearCpu(Model.path(str(root), model_id="tiny-qwen35-hybrid")),
        deployment=Deployment.local(root=str(tmp_path / "deployment")),
        budget=ExecutionBudget(requests=2, max_input_tokens=64, max_new_tokens=2),
    )
    with build_roles(experiment) as topology:
        with topology.client() as client:
            result = client.responses.create(
                model="tiny-qwen35-hybrid", input="A", max_output_tokens=2,
            )
            assert result.usage.input_tokens > 0
            assert result.usage.output_tokens == 2
            assert client.privacy_audit.inference_stage_calls > 0
            assert client.privacy_audit.plaintext_prompt_bytes_sent == 0
            assert client.privacy_audit.plaintext_token_ids_sent == 0
        with TestClient(topology.gateway_app(local_api_key="hybrid-test")) as gateway:
            response = gateway.post(
                "/v1/responses", headers={"Authorization": "Bearer hybrid-test"},
                json={"model": "tiny-qwen35-hybrid", "input": "B", "max_output_tokens": 2},
            )
            assert response.status_code == 200, response.text
            assert response.json()["usage"]["input_tokens"] > 0
            assert response.json()["usage"]["output_tokens"] == 2
