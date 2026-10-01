"""Two distinct additive-share roles around an actual BFV nonlinear island."""

from __future__ import annotations

import msgpack
import numpy as np
import pytest

from pllm.metrics import EncryptedQuadraticShareCostProbe
from pllm.runtime.he_quadratic_share import (
    HEQuadraticShareClient, HEQuadraticShareError, HEQuadraticShareEvaluator,
)
from pllm.runtime.he_authenticated_preprocessing import HEAuthenticatedPreprocessor


def test_quadratic_share_probe_rejects_unbounded_work_without_he() -> None:
    for args in ({"width": 65}, {"rows": 0}, {"islands": 33}, {"width": True}):
        with pytest.raises(ValueError, match="bounded BFV islands"):
            EncryptedQuadraticShareCostProbe(**args)


@pytest.mark.he
@pytest.mark.parametrize("rows", (1, 8, 39))
def test_encrypted_polynomial_returns_only_shares_with_exact_parity(rows: int) -> None:
    pytest.importorskip("tenseal")
    shape = rows, 32
    client = HEQuadraticShareClient(*shape)
    evaluator = HEQuadraticShareEvaluator(client.public_context, *shape)
    assert not evaluator.context.has_secret_key()
    clear = np.arange(rows * 32, dtype=np.int64).reshape(shape) % 41 - 20
    worker_a = np.random.default_rng(31).integers(0, 65_537, size=shape, dtype=np.int64)
    worker_b = (clear - worker_a) % 65_537
    request = client.issue(worker_a)
    evaluated = evaluator.evaluate(request, worker_b)
    assert len(request) < 1 << 20 and len(evaluated.response) < 1 << 20
    first = client.finish(evaluated.response)
    expected = (clear * clear + 3 * clear + 7) % 65_537
    np.testing.assert_array_equal((first + evaluated.worker_b_share) % 65_537, expected)
    assert not np.array_equal(first, expected)
    assert not np.array_equal(evaluated.worker_b_share, expected)
    with pytest.raises(HEQuadraticShareError, match="replay"):
        evaluator.evaluate(request, worker_b)
    with pytest.raises(HEQuadraticShareError, match="no outstanding"):
        client.finish(evaluated.response)


@pytest.mark.he
def test_invalid_result_burns_client_request_but_new_session_recovers() -> None:
    pytest.importorskip("tenseal")
    shape = 1, 32
    client = HEQuadraticShareClient(*shape)
    evaluator = HEQuadraticShareEvaluator(client.public_context, *shape)
    a = np.arange(32, dtype=np.int64).reshape(shape)
    b = np.full(shape, 31, dtype=np.int64)
    request = client.issue(a)
    with pytest.raises(HEQuadraticShareError, match="already pending"):
        client.issue(a)
    evaluated = evaluator.evaluate(request, b)
    corrupted = msgpack.unpackb(evaluated.response, raw=False)
    corrupted["session_id"] = b"wrong-session-id"
    with pytest.raises(HEQuadraticShareError, match="session mismatch"):
        client.finish(msgpack.packb(corrupted, use_bin_type=True))
    with pytest.raises(HEQuadraticShareError, match="no outstanding"):
        client.finish(evaluated.response)
    again = evaluator.evaluate(client.issue(a), b)
    result = client.finish(again.response)
    np.testing.assert_array_equal(
        (result + again.worker_b_share) % 65_537,
        ((a + b) ** 2 + 3 * (a + b) + 7) % 65_537,
    )


@pytest.mark.he
def test_wrong_context_invalid_shares_and_cancel_fail_closed() -> None:
    pytest.importorskip("tenseal")
    client = HEQuadraticShareClient(1, 32)
    evaluator = HEQuadraticShareEvaluator(client.public_context, 1, 32)
    a = np.zeros((1, 32), dtype=np.int64)
    b = np.ones((1, 32), dtype=np.int64)
    with pytest.raises(HEQuadraticShareError, match="residues"):
        client.issue(np.full((1, 32), -1, dtype=np.int64))
    with pytest.raises(HEQuadraticShareError, match="shape"):
        evaluator.evaluate(client.issue(a), np.ones((2, 32), dtype=np.int64))
    client.cancel()
    with pytest.raises(HEQuadraticShareError, match="no outstanding"):
        client.finish(b"bad")
    request = client.issue(a)
    altered = msgpack.unpackb(request, raw=False)
    altered["context_digest"] = "0" * 64
    with pytest.raises(HEQuadraticShareError, match="binding mismatch"):
        evaluator.evaluate(msgpack.packb(altered, use_bin_type=True), b)
    evaluated = evaluator.evaluate(request, b)
    client.finish(evaluated.response)


@pytest.mark.he
@pytest.mark.parametrize("islands", (1, 8, 32))
def test_quadratic_share_sdk_probe_counts_actual_ciphertext_and_context(islands: int) -> None:
    pytest.importorskip("tenseal")
    result = EncryptedQuadraticShareCostProbe(islands=islands).run()
    assert result["schema"] == "pllm.encrypted_quadratic_share_cost_probe.v1"
    assert result["exact_modular_parity"] and not result["public_evaluator_has_secret_key"]
    assert not result["whole_decoder_executable"] and not result["full_wire_measured"]
    assert result["online_all_link_body_bytes"] == (
        result["worker_a_to_b_body_bytes"] + result["worker_b_to_a_body_bytes"]
    )
    assert result["covered_cold_body_bytes"] == (
        result["online_all_link_body_bytes"] + result["public_context_body_bytes"]
    )
    if result["process_peak_rss_after_bytes"] is not None:
        assert result["process_peak_rss_after_bytes"] >= result["process_peak_rss_before_bytes"]
    if islands in (8, 32):
        # Matched pinned Qwen2.5 39+N warm all-link controls: this *one*
        # island per output token already loses the hundredfold byte gate.
        measured_prepared = {8: 116_843_966, 32: 178_970_558}[islands]
        assert result["online_all_link_body_bytes"] > measured_prepared // 100
    assert not any("secret" in key and key != "public_evaluator_has_secret_key" for key in result)


@pytest.mark.he
def test_existing_he_triple_factory_does_not_generate_compact_session_material() -> None:
    pytest.importorskip("tenseal")
    preprocessor = HEAuthenticatedPreprocessor(
        seed=7, threads=1, enable_linear_correlations=False,
    )
    triple = preprocessor.multiplication_triple((4096,))
    a = (triple.a.client.value + triple.a.server.value) % 65_537
    b = (triple.b.client.value + triple.b.server.value) % 65_537
    c = (triple.c.client.value + triple.c.server.value) % 65_537
    np.testing.assert_array_equal(c, (a * b) % 65_537)
    assert preprocessor.stats.multiplication_triples == 4096
    assert preprocessor.stats.uploaded_bytes + preprocessor.stats.downloaded_bytes > 4096 * 100
