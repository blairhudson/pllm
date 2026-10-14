"""Research-only lookup substitution preserves the complete generated decoder."""
import asyncio
import json

import numpy as np
import pytest

from pllm import Model, _native, lower_model
from pllm.profiles import ClientOnlyCpu
from pllm.quantization import SymmetricPerRow
from pllm.runtime.loaders import load_hf_directory
from pllm.runtime.model_binding import compile_runtime_model
from pllm.runtime.tiny_llama import create_tiny_llama_checkpoint
from pllm.runtime.transformer_client import ClientBundle
from pllm.runtime.transformer_engine import MaskedTransformerEngine


@pytest.mark.parametrize("family", ["qwen2", "qwen3", "llama"])
@pytest.mark.parametrize("reduction", [None, "prefix_f32"])
def test_full_vocabulary_logits_kv_sampling_and_token_feedback(tmp_path, monkeypatch, family, reduction):
    path = create_tiny_llama_checkpoint(tmp_path / "model", model_type=family,
        num_hidden_layers=2, with_qkv_bias=family == "qwen2", qk_norm=family == "qwen3")
    config = json.loads((path / "config.json").read_text())
    engine = MaskedTransformerEngine(threads=1, weight_bits=8, activation_bits=8)
    asyncio.run(engine.load(load_hf_directory(path, model_id="lookup-parity")))
    bundle = ClientBundle.unpack(engine.client_bundle("lookup-parity", placement="client"))
    stage = bundle.stages["token_lookup"]
    assert stage.client_weight_layout == "embedding"
    weights = stage.client_weight
    n, d = weights.shape
    records = np.empty((n, d + 4), np.uint8)
    records[:, :d] = weights.view(np.uint8)
    records[:, d:] = stage.client_weight_scales.astype("<f4").view(np.uint8).reshape(n, 4)
    servers = [_native.PrivatePageServer(records.tobytes(), d + 4, p) for p in (0, 1)]
    plan = lower_model(config, batch=1, max_input_tokens=16, max_new_tokens=4)
    pipeline = ClientOnlyCpu(Model("lookup-parity"), quantization=SymmetricPerRow(causal_reduction=reduction))
    compiled = compile_runtime_model(plan, bundle, composition=pipeline)
    executor = compiled.client_linear_executor(engine)
    original = ClientBundle.local_token_lookup
    calls = []

    def lookup(owner, ids):
        if owner is not bundle:
            return original(owner, ids)
        indices = np.asarray(ids, np.int64).reshape(-1)
        queries = [_native.private_page_issue(*servers[0].descriptor(), int(i), leaf_bits=128) for i in indices]
        try:
            replies = [server.evaluate_batch([q[p] for q in queries]) for p, server in enumerate(servers)]
            raw = b"".join(q[2].decode(replies[0][i], replies[1][i]) for i, q in enumerate(queries))
            records = np.frombuffer(raw, np.uint8).reshape(len(indices), d + 4)
            result = records[:, :d].view(np.int8).astype(np.float32) * records[:, d:].copy().view("<f4")
            expected = original(owner, ids)
            np.testing.assert_array_equal(result.view("u4"), expected.view("u4"))
            calls.append(len(indices))
            return result
        finally:
            for q in queries:
                q[2].cancel()

    results = []
    for private in (False, True):
        if private:
            monkeypatch.setattr(ClientBundle, "local_token_lookup", lookup)
        runtime = compiled.runtime(executor)
        runtime.rng = np.random.default_rng(417)
        _, logits, _ = runtime.prepare_ids([0, 2, 3, 5, 7])
        phases = []
        for step in range(4):
            assert logits.shape[-1] == config["vocab_size"]  # No finite-language slicing.
            token = runtime.sample(logits, temperature=0.7, top_p=0.9)
            phases.append((logits.tobytes(), token, json.dumps(runtime.rng.bit_generator.state, sort_keys=True),
                [(c.key[:c.length].tobytes(), c.value[:c.length].tobytes()) for c in runtime.caches]))
            if step < 3:
                logits = runtime.forward_ids([token])[-1]
        results.append(phases)
    assert results[0] == results[1]
    assert calls == [5, 1, 1, 1]  # Original generated tokens return through lookup.
