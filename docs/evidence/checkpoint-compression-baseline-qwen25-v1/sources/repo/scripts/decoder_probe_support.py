"""Pinned compiled decoder fixture for non-selectable research diagnostics."""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

import numpy as np

from pllm import Model, lower_model
from pllm.model_loader import resolve_model
from pllm.profiles import MaskedLinearCpu
from pllm.quantization import SymmetricPerRow
from pllm.runtime.model_binding import compile_runtime_model
from pllm.runtime.quantization import dequantize_matmul, quantize_activation_per_row
from pllm.runtime.transformer_client import ClientBundle
from pllm.runtime.transformer_engine import MaskedTransformerEngine

ROOT = Path(__file__).resolve().parents[1]
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
BODY = "5d631be30158b3ea2a72cae355ce09a3b304bbfddde48b5758c339c346a34974"
PROMPT = "Explain why neither server can see the prompt."


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


class DecoderFixture:
    def __init__(self, *, inputs: int = 39, outputs: int = 32):
        self.source = resolve_model(Model.hf(MODEL, revision=REVISION))
        if self.source.path is None or self.source.checkpoint_digest is None:
            raise ValueError("pinned checkpoint must resolve")
        self.engine = MaskedTransformerEngine(weight_bits=8, activation_bits=8, threads=4)
        asyncio.run(self.engine.load(self.source.manifest))
        self.model = self.engine.models[self.source.manifest.id]
        if self.model.manifest.metadata.get("body_fingerprint") != BODY:
            raise ValueError("W8A8 fingerprint differs from locked controls")
        self.bundle = ClientBundle.unpack(self.engine.client_bundle(self.source.manifest.id))
        self.plan = lower_model(
            (self.source.path / "config.json").read_bytes(),
            batch=1,
            max_input_tokens=inputs,
            max_new_tokens=outputs,
        )
        self.composition = MaskedLinearCpu(
            Model.hf(MODEL, revision=REVISION),
            quantization=SymmetricPerRow(weight_bits=8, activation_bits=8),
        )
        self.compiled = compile_runtime_model(self.plan, self.bundle, composition=self.composition)
        self.body = {
            key: stage
            for key, stage in self.model.stages.items()
            if stage.spec.role not in {"token_lookup", "lm_head"}
        }

    def tokens(self, text: str) -> list[int]:
        runtime = self.compiled.runtime(self.remote)
        rendered = self.bundle.render_prompt([{"role": "user", "content": text}])
        return runtime.encode_prompt(rendered)[-self.plan.prefill["query_sequence"] :]

    def remote(self, stage_id: str, activation: np.ndarray) -> np.ndarray:
        stage = self.model.stages[stage_id]
        return stage_output(stage, activation, stage.spec.activation_bits)

    def lock(self) -> dict:
        return {
            "model": MODEL,
            "revision": REVISION,
            "checkpoint_digest": self.source.checkpoint_digest,
            "source_lock_digest": self.source.source_lock_digest,
            "body_fingerprint": BODY,
            "plan_digest": self.plan.digest,
            "schedule_digest": self.compiled.runtime_schedule_digest,
            "composition_digest": self.composition.digest(),
            "binding_digest": self.compiled.digest,
        }


def stage_output(stage, activation: np.ndarray, bits: int) -> np.ndarray:
    quantized = quantize_activation_per_row(activation, bits=bits)
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


def trajectory(fixture: DecoderFixture, ids: list[int], remote, *, forcing=None, count=2):
    runtime = fixture.compiled.runtime(remote)
    _, logits, _ = runtime.prepare_ids(ids)
    scores, selected = [logits.copy()], [int(np.argmax(logits))]
    for index in range(count - 1):
        token = selected[-1] if forcing is None else forcing[index]
        logits = runtime.forward_ids([token])[-1]
        scores.append(logits.copy())
        selected.append(int(np.argmax(logits)))
    return scores, selected
