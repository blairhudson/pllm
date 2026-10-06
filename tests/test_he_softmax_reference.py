"""HE-method numeric screens keep public mask provenance and rejection explicit."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module", autouse=True)
def source_imports():
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(ROOT))
        yield


@pytest.fixture(scope="module")
def oracle():
    from benchmarks.research.he_softmax_reference import Oracle
    worker = Oracle()
    try:
        yield worker
    finally:
        worker.close()


def test_native_exponential_matches_independent_equations_and_public_causal_rows(oracle):
    scores = np.array([[[[-3, -np.inf, -np.inf], [-1, 2, -np.inf], [-8, 0, 8]]]], dtype=np.float32)
    coefficients = np.array([.99999, .99999, .50002, .16667, .041653, .0083316,
        .0013920, .00019871, .000024471, .0000027286, .00000029466, .000000026438,
        .0000000014896, .00000000012104, .000000000020777, .0000000000013376])
    for method in ("nexus_exp", "thor_exp"):
        actual = oracle.evaluate(method, scores)
        for row, values in enumerate(scores[0, 0]):
            valid = values[:row + 1].astype(np.float64)
            if method == "nexus_exp":
                expected = (1 + (valid - 16) / 256) ** 256
            else:
                expected = np.polynomial.polynomial.polyval(valid / 32, coefficients)
                expected = expected ** 32  # independent of the iterative normalizer
            expected /= expected.sum()
            np.testing.assert_allclose(actual[0, 0, row, :row + 1], expected, rtol=1e-6, atol=1e-7)
            assert np.all(actual[0, 0, row, row + 1:] == 0)
        np.testing.assert_allclose(oracle.evaluate(method, scores[:, :, -1:], 2), actual[:, :, -1:], rtol=0, atol=0)


def test_reject_private_nonfinite_values_and_outside_domain_without_clipping(oracle):
    from benchmarks.research.he_softmax_reference import DomainRejected
    for value in (np.nan, np.inf, -np.inf):
        with pytest.raises(ValueError, match="publicly masked"):
            oracle.evaluate("nexus_exp", np.array([[[[value]]]], dtype=np.float32))
    with pytest.raises(ValueError, match="attention scores"):
        oracle.evaluate("nexus_exp", np.zeros((1, 1, 1, 3), dtype=np.float32))
    for value in (-16.1, 16.1):
        with pytest.raises(DomainRejected):
            oracle.evaluate("nexus_exp", np.array([[[[value]]]], dtype=np.float32))


def test_checkpoint_rejections_never_acquire_quality_or_ciphertext_scores():
    from benchmarks.research.he_softmax_reference import CONFIGURATION
    report = json.loads((ROOT / "docs/evidence/research-he-softmax-reference-qwen25.json").read_text())
    assert report["configuration"] == CONFIGURATION
    assert not report["executable_sdk"] and not report["ciphertext_execution"]
    assert not report["whole_generation_quality"]
    assert report["whole_response_network_bytes"] is None
    for source, digest in report["source_sha256"].items():
        assert hashlib.sha256((ROOT / source).read_bytes()).hexdigest() == digest
    for method, summary in report["summary"].items():
        rows = [row for row in report["observations"] if row["method"] == method]
        assert summary["attempted_prompts"] == len(rows)
        assert summary["rejected_prompts"] == sum(row["rejection"] is not None for row in rows)
        assert summary["completed_positions"] == sum(row["completed_positions"] for row in rows)
        if not summary["completed_positions"]:
            assert summary["worst_abs_logit_error"] is None
