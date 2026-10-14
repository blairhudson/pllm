"""The measured audit detects exact-state differences and retires instrumentation."""
import asyncio
import gc
import json
import weakref

import numpy as np
import pytest

from pllm import Deployment, ExecutionBudget, Experiment, Model, lower_model
from pllm.metrics import BenchmarkTarget, Scenario
from pllm.profiles import ClientOnlyCpu
from pllm.quantization import SymmetricPerRow
from pllm.runtime.benchmark_response_audit import response_state_audit
from pllm.runtime.loaders import load_hf_directory
from pllm.runtime.model_binding import compile_runtime_model
from pllm.runtime.tiny_llama import create_tiny_llama_checkpoint
from pllm.runtime.transformer_client import ClientBundle, MaskedTransformerClientRuntime
from pllm.runtime.transformer_engine import MaskedTransformerEngine


def test_target_audit_contract():
    exp = Experiment("audit", ClientOnlyCpu(Model.tiny()), Deployment.local(root="local://audit"), ExecutionBudget(1, 128, 8))
    ordinary = BenchmarkTarget.response(exp)
    assert "state_audit" not in ordinary.to_spec()["parameters"]
    audited = BenchmarkTarget.response(exp, state_audit=True)
    assert audited.digest != ordinary.digest
    assert BenchmarkTarget.from_spec(audited.to_spec()).digest == audited.digest
    for kw in ({"state_audit": 1}, {"state_audit": True, "backend": "docker"},
               {"state_audit": True, "scenario": Scenario.repeat()},
               {"state_audit": True, "scenario": Scenario.cold(max_output_tokens=65)},
               {"state_audit": True, "client_metadata": True},
               {"state_audit": True, "concurrent_requests": 2}):
        with pytest.raises(ValueError):
            BenchmarkTarget.response(exp, **kw)


def compiled_fixture(tmp_path, family):
    path = create_tiny_llama_checkpoint(tmp_path / "model", model_type=family,
        with_qkv_bias=family == "qwen2", qk_norm=family == "qwen3")
    engine = MaskedTransformerEngine(threads=1, weight_bits=8, activation_bits=8)
    asyncio.run(engine.load(load_hf_directory(path, model_id="audit")))
    bundle = ClientBundle.unpack(engine.client_bundle("audit", placement="client"))
    plan = lower_model(json.loads((path / "config.json").read_text()), batch=1,
                       max_input_tokens=16, max_new_tokens=3)
    compiled = compile_runtime_model(plan, bundle, composition=ClientOnlyCpu(Model("audit"),
        quantization=SymmetricPerRow(causal_reduction="prefix_f32")))
    return compiled, compiled.client_linear_executor(engine)


@pytest.mark.parametrize("family", ["qwen2", "qwen3", "llama"])
def test_audit_preserves_decoding_detects_bit_change_and_keeps_no_runtime(tmp_path, family):
    compiled, remote = compiled_fixture(tmp_path, family)
    all_rows, all_tokens = [], []
    original = MaskedTransformerClientRuntime.sample
    for changed in (False, False, True):
        runtime = compiled.runtime(remote)
        rng = runtime.rng
        logits = runtime.forward_ids([0, 2, 3])[-1]
        if changed:
            runtime.caches[0].key.view(np.uint32).flat[0] ^= 1
        tokens = []
        with response_state_audit(b"x" * 32) as records:
            for _ in range(3):
                token = runtime.sample(logits, temperature=0)
                tokens.append(token)
                if len(tokens) < 3:
                    logits = runtime.forward_ids([token])[-1]
        assert runtime.rng is rng
        assert MaskedTransformerClientRuntime.sample is original
        assert all(row["rng_before"] == row["rng_after"] for row in records[0]["selections"])
        all_rows.append(records)
        all_tokens.append(tokens)
        reference = weakref.ref(runtime)
        runtime.close()
        del runtime
        gc.collect()
        assert reference() is None
    assert all_rows[0] == all_rows[1]
    assert all_tokens[0] == all_tokens[1]
    assert all_rows[0][0]["selections"][0]["logits_and_active_kv"] != all_rows[2][0]["selections"][0]["logits_and_active_kv"]


def test_failure_restores_sampler_rng_and_lock(tmp_path, monkeypatch):
    compiled, remote = compiled_fixture(tmp_path, "llama")
    runtime = compiled.runtime(remote)
    rng = runtime.rng
    logits = runtime.forward_ids([0, 2])[-1]
    def failure(*args, **kwargs):
        raise RuntimeError("injected sampler failure")
    monkeypatch.setattr(MaskedTransformerClientRuntime, "sample", failure)
    with pytest.raises(RuntimeError, match="injected"):
        with response_state_audit(b"x" * 32) as records:
            runtime.sample(logits)
    assert records[0]["failed"]
    assert runtime.rng is rng
    assert MaskedTransformerClientRuntime.sample is failure
    with response_state_audit(b"x" * 32):
        pass
    runtime.close()
