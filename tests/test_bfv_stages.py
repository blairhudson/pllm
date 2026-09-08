import asyncio
import os
from pathlib import Path

import msgpack
import numpy as np
import pytest

from pllm.runtime.client import _BFVStageClient
from pllm.runtime.he_runtime import BFVCorrelationServer
from pllm.runtime.loaders import load_hf_directory
from pllm.runtime.tiled_bfv import TiledBFVClient
from pllm.runtime.tiny_gemma import create_tiny_gemma4_checkpoint
from pllm.runtime.transformer_engine import MaskedTransformerEngine, TransformerEngineError

PYDEPS = os.environ.get("HE_OPENAI_PYDEPS", "")


def run(value):
    return asyncio.run(value)


@pytest.mark.he
def test_real_bfv_stage_correlation_is_exact(tmp_path: Path):
    root = create_tiny_gemma4_checkpoint(
        tmp_path / "tiny",
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
    )
    manifest = load_hf_directory(root, model_id="tiny-bfv")
    engine = MaskedTransformerEngine(threads=1, tenseal_path=PYDEPS)
    run(engine.load(manifest))
    runtime = engine.models["tiny-bfv"].stages["token_lookup"]
    client = _BFVStageClient(plain_modulus=runtime.modulus, pydeps_path=PYDEPS)
    engine.register_bfv_context("tiny-bfv", "ctx", client.public_context)
    rng = np.random.default_rng(7)
    mask = rng.integers(0, runtime.modulus, size=runtime.spec.in_features, dtype=np.uint32)
    encrypted = client.encrypt(mask)
    transformed = engine.evaluate_bfv_correlation("tiny-bfv", "token_lookup", "ctx", encrypted)
    actual = client.decrypt(transformed, stage_id="token_lookup")
    expected = engine.kernel(runtime.weight.values, mask[None, :], runtime.modulus)[0]
    assert np.array_equal(actual, expected)
    assert runtime.modulus == 65_537


@pytest.mark.he
def test_tiled_bfv_stage_batch_is_exact(tmp_path: Path):
    root = create_tiny_gemma4_checkpoint(
        tmp_path / "tiny-tiled",
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
    )
    manifest = load_hf_directory(root, model_id="tiny-tiled")
    engine = MaskedTransformerEngine(threads=1, tenseal_path=PYDEPS)
    run(engine.load(manifest))
    runtime = engine.models["tiny-tiled"].stages["token_lookup"]
    client = TiledBFVClient(
        runtime.spec.in_features,
        runtime.spec.out_features,
        plain_modulus=runtime.modulus,
        tenseal_path=PYDEPS,
    )
    engine.register_bfv_context("tiny-tiled", "ctx", client.public_context)
    rng = np.random.default_rng(19)
    masks = rng.integers(
        0,
        runtime.modulus,
        size=(5, runtime.spec.in_features),
        dtype=np.uint32,
    )
    encrypted = client.encrypt_many(masks)
    responses = engine.evaluate_bfv_correlations("tiny-tiled", "token_lookup", "ctx", encrypted)

    actual = client.decrypt_many(responses, client.group_sizes(len(masks)))
    expected = runtime.compiled_weight.modular(masks, runtime.modulus)
    assert np.array_equal(actual, expected)

    malformed = msgpack.unpackb(client.public_context, raw=False)
    malformed["galois_keys"] = b"invalid"
    with pytest.raises(TransformerEngineError, match="invalid tiled BFV public context"):
        engine.register_bfv_context(
            "tiny-tiled", "bad", msgpack.packb(malformed, use_bin_type=True)
        )

    replacement = TiledBFVClient(
        runtime.spec.in_features,
        runtime.spec.out_features,
        plain_modulus=runtime.modulus,
        tenseal_path=PYDEPS,
    )
    engine.register_bfv_context("tiny-tiled", "ctx", replacement.public_context)
    replacement_request = replacement.encrypt_many(masks[:1])
    replacement_response = engine.evaluate_bfv_correlations(
        "tiny-tiled", "token_lookup", "ctx", replacement_request
    )
    replacement_actual = replacement.decrypt_many(replacement_response, [1])
    assert np.array_equal(replacement_actual, expected[:1])


@pytest.mark.he
def test_bfv_reference_chunks_wide_inputs_exactly():
    modulus = 786_433
    width = 4_097
    rng = np.random.default_rng(11)
    weight = rng.integers(-7, 8, size=(1, width), dtype=np.int64)
    mask = rng.integers(0, modulus, size=width, dtype=np.uint32)
    client = _BFVStageClient(plain_modulus=modulus, pydeps_path=PYDEPS)
    server = BFVCorrelationServer(weight, pydeps_path=PYDEPS)
    server.register_context("ctx", client.public_context)
    encrypted = client.encrypt(mask)
    envelope = msgpack.unpackb(encrypted, raw=False)
    assert len(envelope["chunks"]) == 3
    raw = server.evaluate("ctx", encrypted)
    result = msgpack.unpackb(raw, raw=False)
    actual = np.asarray(
        [
            int(client.ts.bfv_vector_from(client.context, item).decrypt()[0]) % modulus
            for item in result["ciphertexts"]
        ]
    )
    expected = (weight @ mask.astype(np.int64)) % modulus
    assert np.array_equal(actual, expected)
