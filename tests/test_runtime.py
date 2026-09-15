from __future__ import annotations

import asyncio
import os

import numpy as np
import pytest

from pllm.runtime.bfv_correlations import BFVCorrelationClient, BFVCorrelationServer
from pllm.runtime.masked_runtime import (
    BigramStageExecutor,
    CorrelationPool,
    ModelError,
    LocalCorrelationFactory,
    MaskCorrelation,
    MaskedBigramClientSession,
    MaskedBigramModel,
    StageBatchScheduler,
    centered_mod,
)

PYDEPS = os.environ.get("PLLM_TENSEAL_PATH", "")


def test_local_correlation_exact_bigram_generation():
    model = MaskedBigramModel(alphabet="private\n", phrase="private\n")
    pool = CorrelationPool()
    pool.extend(LocalCorrelationFactory(model.weight, model.modulus, seed=1).create(16))
    client = MaskedBigramClientSession(model, pool)
    token = model.tokenizer.bos_token_id
    output = []
    for _ in range(16):
        request, correlation = client.request_for_token(token)
        token = client.complete_token(model.evaluate_masked(request.pack()), correlation)
        if token == model.tokenizer.eos_token_id:
            break
        output.append(token)
    assert model.tokenizer.decode(output) == "private\n"


def test_correlation_pool_is_single_use():
    pool = CorrelationPool()
    row = MaskCorrelation("x", np.array([1]), np.array([2]))
    pool.put(row)
    assert pool.take().id == "x"
    with pytest.raises(ModelError):
        pool.put(row)


def test_masked_input_is_uniformly_shifted_and_exact():
    model = MaskedBigramModel(alphabet="abc\n", phrase="a\n")
    correlation = LocalCorrelationFactory(model.weight, model.modulus, seed=7).create(1)[0]
    one_hot = np.zeros(model.tokenizer.vocab_size, dtype=np.int64)
    one_hot[model.tokenizer.bos_token_id] = 1
    masked = centered_mod(one_hot + correlation.mask, model.modulus)
    assert not np.array_equal(masked, one_hot)
    assert np.array_equal(
        centered_mod(model.weight @ masked - correlation.transformed_mask, model.modulus),
        centered_mod(model.weight @ one_hot, model.modulus),
    )


@pytest.mark.asyncio
async def test_stage_scheduler_batches_concurrent_same_weight_requests():
    model = MaskedBigramModel(alphabet="private\n", phrase="private\n")
    scheduler = StageBatchScheduler(BigramStageExecutor(model), max_batch_size=32, max_wait_ms=20)
    factory = LocalCorrelationFactory(model.weight, model.modulus, seed=9)
    payloads = []
    for row in factory.create(32):
        pool = CorrelationPool(); pool.put(row)
        client = MaskedBigramClientSession(model, pool)
        request, _ = client.request_for_token(model.tokenizer.bos_token_id)
        payloads.append(request.pack())
    try:
        results = await asyncio.gather(*(scheduler.submit(payload) for payload in payloads))
        assert len(results) == 32
        stats = scheduler.stats()
        assert stats["items"] == 32
        assert stats["max_batch_size"] >= 16
        assert stats["mean_batch_size"] >= 8
    finally:
        await scheduler.close()


@pytest.mark.he
def test_real_bfv_correlation_roundtrip_and_server_has_no_secret():
    model = MaskedBigramModel(alphabet="private\n", phrase="private\n")
    client = BFVCorrelationClient(dimension=model.tokenizer.vocab_size, plain_modulus=model.modulus, pydeps_path=PYDEPS)
    server = BFVCorrelationServer(model.weight, pydeps_path=PYDEPS)
    server.register_context("ctx", client.public_context)
    assert not server.contexts["ctx"].has_secret_key()
    mask = np.arange(model.tokenizer.vocab_size, dtype=np.int64)
    transformed = client.decrypt_transformed(server.evaluate("ctx", client.encrypt_mask(mask)))
    expected = centered_mod(model.weight @ mask, model.modulus)
    assert np.array_equal(transformed, expected)


@pytest.mark.he
def test_server_rejects_bfv_context_with_secret_key():
    model = MaskedBigramModel(alphabet="private\n", phrase="private\n")
    client = BFVCorrelationClient(dimension=model.tokenizer.vocab_size, plain_modulus=model.modulus, pydeps_path=PYDEPS)
    server = BFVCorrelationServer(model.weight, pydeps_path=PYDEPS)
    private = client.context.serialize(save_public_key=True, save_secret_key=True, save_galois_keys=True, save_relin_keys=False)
    with pytest.raises(ModelError, match="secret key"):
        server.register_context("bad", private)
