from __future__ import annotations

import asyncio
import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

import pllm
from pllm.runtime.loaders import load_hf_directory
from pllm.runtime.model_binding import compile_runtime_model
from pllm.runtime.model_execution import CompiledRuntimeSession, RuntimeExecutionError
from pllm.runtime.tiny_llama import create_tiny_llama_checkpoint
from pllm.runtime.transformer_client import ClientBundle, RemoteLinear
from pllm.runtime.transformer_engine import MaskedTransformerEngine


def _compiled(tmp_path: Path, *, max_input: int = 8, max_new: int = 3):
    root = create_tiny_llama_checkpoint(tmp_path / "model", num_hidden_layers=2)
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
    assert all(
        cache.key is None and cache.value is None
        for cache in session._runtime.caches
    )
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

    assert masked_tokens == compiled.session(clear).generate_ids(
        [0, 2], max_new_tokens=2
    )
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
