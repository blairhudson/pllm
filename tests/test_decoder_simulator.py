from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pytest

from pllm.runtime.authenticated_mpc import (
    AuthenticatedMPC,
    AuthenticationError,
    TrustedPreprocessor,
)
from pllm.runtime.he_authenticated_preprocessing import HEAuthenticatedPreprocessor
from pllm.runtime.preprocessing_inventory import PreparedInventory, RecordingPreprocessor
from pllm.runtime.linear_integrity import (
    LinearIntegrityError,
    check_linear_result,
    create_linear_check_key,
)
from pllm.runtime.secure_selection import secure_argmax, verify_one_hot
from pllm.runtime.secure_transformer import (
    ClearKVCache,
    SecureDecoder,
    SecureDecoderConfig,
    SecureDecoderWeights,
    SecureKVCache,
)


def _decoder(seed: int = 7) -> SecureDecoder:
    config = SecureDecoderConfig(
        vocabulary_size=8,
        hidden_size=8,
        head_count=2,
        head_size=4,
        intermediate_size=12,
        maximum_context=4,
    )
    weights = SecureDecoderWeights.random(config, seed=seed)
    return SecureDecoder(config, weights, TrustedPreprocessor(seed=seed + 1))


@pytest.mark.parametrize("choices", [2, 3, 4, 8, 16])
def test_secure_argmax_matches_clear_result(choices: int) -> None:
    preprocessor = TrustedPreprocessor(seed=100 + choices)
    runtime = AuthenticatedMPC(preprocessor)
    rng = np.random.default_rng(200 + choices)
    clear = rng.integers(0, 4096, size=(3, choices), dtype=np.int64)
    shared = preprocessor.share(clear)
    result = secure_argmax(
        runtime,
        shared,
        bit_mask=preprocessor.bit_mask(clear.shape),
    )
    np.testing.assert_array_equal(result.indices, np.argmax(clear, axis=-1))
    assert result.opened_values > 0
    assert runtime.stats.online_rounds > 0


def test_private_token_validation_rejects_invalid_input() -> None:
    preprocessor = TrustedPreprocessor(seed=23)
    runtime = AuthenticatedMPC(preprocessor)
    invalid = np.asarray([[1, 1, 0, 0]], dtype=np.int64)
    shared = runtime.input(invalid, preprocessor.input_mask(invalid.shape))
    with pytest.raises(AuthenticationError):
        verify_one_hot(runtime, shared)


def test_parallel_multiplication_uses_one_round() -> None:
    preprocessor = TrustedPreprocessor(seed=31)
    runtime = AuthenticatedMPC(preprocessor)
    left = [preprocessor.share(np.arange(8, dtype=np.int64) + index) for index in range(4)]
    right = [preprocessor.share(np.arange(8, dtype=np.int64) + index + 2) for index in range(4)]
    triples = [preprocessor.multiplication_triple((8,)) for _ in range(4)]
    outputs = runtime.multiply_many(left, right, triples)
    assert runtime.stats.multiplication_rounds == 1
    clear = runtime.open_many(outputs)
    for index, item in enumerate(clear):
        expected = (np.arange(8, dtype=np.int64) + index) * (
            np.arange(8, dtype=np.int64) + index + 2
        )
        np.testing.assert_array_equal(item, expected % preprocessor.modulus)


def test_secure_decoder_matches_clear_decoder_and_reuses_cache() -> None:
    decoder = _decoder()
    secure_cache = SecureKVCache()
    clear_cache = ClearKVCache()
    tokens = [2, 5, 1]
    for token in tokens:
        secure = decoder.forward_token([token], cache=secure_cache)
        clear_token, clear_logits = decoder.clear_forward_token([token], cache=clear_cache)
        np.testing.assert_array_equal(secure.token_ids, clear_token)
        opened_logits = AuthenticatedMPC(decoder.preprocessor).open(secure.logits)
        np.testing.assert_array_equal(opened_logits, clear_logits)
        assert secure.online_rounds > 0
        assert secure.uploaded_bytes > 0
        assert secure.downloaded_bytes > 0
    assert secure_cache.length == clear_cache.length == len(tokens)


def test_secure_decoder_can_reveal_only_final_logits_for_lower_latency() -> None:
    decoder = _decoder(seed=29)
    result = decoder.forward_token([4], output_disclosure="logits")
    clear_token, clear_logits = decoder.clear_forward_token([4])
    np.testing.assert_array_equal(result.token_ids, clear_token)
    np.testing.assert_array_equal(result.disclosed_logits, clear_logits)
    assert result.selection is None
    assert result.online_rounds < 20


def test_secure_decoder_detects_tampered_cache_share() -> None:
    decoder = _decoder(seed=19)
    with pytest.raises(AuthenticationError):
        decoder.forward_token([3], tamper_cache=True)


def test_hidden_linear_check_detects_corruption() -> None:
    rng = np.random.default_rng(41)
    modulus = 65537
    weight = rng.integers(-7, 8, size=(48, 32), dtype=np.int64)
    values = rng.integers(0, modulus, size=(6, 32), dtype=np.int64)
    output = values @ (weight % modulus).T % modulus
    key = create_linear_check_key(weight, modulus=modulus, checks=2, seed=11)
    assert check_linear_result(values, output, key).valid
    corrupted = output.copy()
    corrupted[0, 7] += 1
    with pytest.raises(LinearIntegrityError):
        check_linear_result(values, corrupted, key)


def test_recorded_preprocessing_plan_replays_once() -> None:
    config = SecureDecoderConfig()
    weights = SecureDecoderWeights.random(config, seed=71)
    recorder = RecordingPreprocessor(TrustedPreprocessor(seed=72))
    planned_decoder = SecureDecoder(config, weights, recorder)  # type: ignore[arg-type]
    planned_decoder.forward_token([3])
    assert recorder.plan.operation_count > 10
    assert recorder.plan.counts()["multiplication_triple"] > 0

    inventory = PreparedInventory.generate(recorder.plan, TrustedPreprocessor(seed=73))
    decoder = SecureDecoder(config, weights, inventory)  # type: ignore[arg-type]
    result = decoder.forward_token([3])
    clear, _ = decoder.clear_forward_token([3])
    np.testing.assert_array_equal(result.token_ids, clear)
    assert inventory.remaining == 0
    with pytest.raises(Exception):
        decoder.forward_token([3])


@pytest.mark.he
def test_he_generated_packed_triples_are_exact() -> None:
    path = os.environ.get("PLLM_PYDEPS")
    preprocessor = HEAuthenticatedPreprocessor(
        seed=53,
        threads=1,
        tenseal_path=path,
        enable_linear_correlations=False,
    )
    triple = preprocessor.multiplication_triple((128,))
    runtime = AuthenticatedMPC(preprocessor)  # type: ignore[arg-type]
    a, b, c = runtime.open_many([triple.a, triple.b, triple.c])
    np.testing.assert_array_equal(c, a * b % preprocessor.modulus)
    assert preprocessor.stats.multiplication_triples == 128
    assert preprocessor.stats.public_context_bytes < 1_000_000


@pytest.mark.he
def test_he_generated_small_linear_correlation_is_exact() -> None:
    path = os.environ.get("PLLM_PYDEPS")
    preprocessor = HEAuthenticatedPreprocessor(
        seed=61,
        threads=1,
        tenseal_path=path,
        enable_linear_correlations=True,
    )
    weight = np.asarray(
        [[1, 2, -1, 0], [0, -3, 2, 1], [2, 0, 1, -2]],
        dtype=np.int64,
    )
    correlation = preprocessor.linear_correlation(weight, (1, 4))
    runtime = AuthenticatedMPC(preprocessor)  # type: ignore[arg-type]
    random_input, transformed = runtime.open_many(
        [correlation.random_input, correlation.transformed_input]
    )
    np.testing.assert_array_equal(
        transformed,
        random_input @ (weight % preprocessor.modulus).T % preprocessor.modulus,
    )
    assert preprocessor.stats.public_context_bytes > 1_000_000


@pytest.mark.he
def test_complete_secure_decoder_uses_he_generated_material() -> None:
    path = os.environ.get("PLLM_PYDEPS")
    config = SecureDecoderConfig()
    weights = SecureDecoderWeights.random(config, seed=81)
    recorder = RecordingPreprocessor(TrustedPreprocessor(seed=82))
    SecureDecoder(config, weights, recorder).forward_token([2])  # type: ignore[arg-type]
    generator = HEAuthenticatedPreprocessor(
        seed=83,
        threads=1,
        tenseal_path=path,
        enable_linear_correlations=True,
    )
    inventory = PreparedInventory.generate(recorder.plan, generator)
    result = SecureDecoder(config, weights, inventory).forward_token([2])  # type: ignore[arg-type]
    clear, _ = SecureDecoder(
        config, weights, TrustedPreprocessor(seed=84)
    ).clear_forward_token([2])
    np.testing.assert_array_equal(result.token_ids, clear)
    assert inventory.remaining == 0
    assert generator.stats.multiplication_triples > 0
    assert generator.stats.linear_correlations == 6


def test_user_facing_source_has_no_retired_brand_or_proof_wording() -> None:
    root = Path(__file__).parents[1]
    files = [
        root / "README.md",
        root / "docs" / "content" / "docs" / "assurance" / "index.mdx",
        root / "SECURITY.md",
    ]
    forbidden = ("zero" + " knowledge", "zk" + "ai", "pllm" + "-inference")
    for path in files:
        value = path.read_text(encoding="utf-8").lower()
        for phrase in forbidden:
            assert phrase not in value, f"{phrase!r} remains in {path}"


def test_public_package_is_named_pllm() -> None:
    import pllm

    assert pllm.__version__ == __import__("pllm._version", fromlist=["__version__"]).__version__
    pyproject = Path(__file__).parents[1] / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")
    assert 'name = "pllm"' in text
    assert 'dynamic = ["version"]' in text
