from __future__ import annotations

import numpy as np
import pytest

from pllm.components import create_component
from pllm.metrics import ReferenceAgreement, measure_reference_agreement


def test_reference_agreement_is_a_bound_metric_and_scores_same_token_logits() -> None:
    metric = ReferenceAgreement(
        dataset_digest="a" * 64, reference_checkpoint_digest="b" * 64, top_k=3
    )
    assert create_component(metric.component, metric.params) == metric
    assert metric.describe().capabilities == (
        "ratio",
        "higher-is-better",
        "reference-executed",
        "same-token",
    )
    assert measure_reference_agreement([2.0, 1.0, 1.0, 0.0], [2.0, 1.0, 1.0, 0.0], top_k=3) == {
        "top1_agreement": 1.0,
        "top_k_recall": 1.0,
        "max_abs_logit_error": 0.0,
    }
    # Ties resolve by token ID; top-1 and top-k can diverge.
    assert measure_reference_agreement([0.0, 1.0, 1.0, 2.0], [2.0, 1.0, 1.0, 0.0], top_k=3) == {
        "top1_agreement": 0.0,
        "top_k_recall": 2 / 3,
        "max_abs_logit_error": 2.0,
    }


@pytest.mark.parametrize(
    ("candidate", "reference", "top_k"),
    [
        ([0.0, float("nan")], [0.0, 1.0], 1),
        ([0.0, 1.0], [0.0], 1),
        ([0.0, 1.0], [[0.0, 1.0]], 1),
        ([0.0, 1.0], [0.0, 1.0], 3),
        (np.zeros(1_000_001), np.zeros(1_000_001), 1),
        ([1e308], [-1e308], 1),
    ],
)
def test_reference_agreement_rejects_invalid_vocab_cohorts(candidate, reference, top_k) -> None:
    with pytest.raises(ValueError, match="reference logits"):
        measure_reference_agreement(candidate, reference, top_k=top_k)


def test_reference_agreement_rejects_unbound_or_unavailable_reference() -> None:
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        ReferenceAgreement(dataset_digest="public", reference_checkpoint_digest="b" * 64)
    with pytest.raises(ValueError, match="top_k"):
        measure_reference_agreement([1.0], [1.0], top_k=True)
