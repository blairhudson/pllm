from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pllm.runtime.share_linear_feedback_reference import (
    ClearFeedbackReference,
    FeedbackReferenceError,
    FeedbackWeights,
    SharedFeedbackReference,
    _FEEDBACK,
    _INPUT,
    _MODULUS,
    _pack,
    project_feedback_cost,
)


def _weights(*, layers: int = 3, hidden: int = 8, rank: int = 4, vocab: int = 8) -> FeedbackWeights:
    rng = np.random.default_rng(421)

    def matrix(shape: tuple[int, ...]) -> np.ndarray:
        return rng.integers(-2, 3, size=shape, dtype=np.int16)

    return FeedbackWeights.bind(
        recurrent=tuple(matrix((hidden, hidden)) for _ in range(layers)),
        incoming=tuple(matrix((hidden, rank if layer == 0 else hidden)) for layer in range(layers)),
        project=tuple(matrix((rank, hidden)) for _ in range(layers)),
        expand=tuple(matrix((hidden, rank)) for _ in range(layers)),
        head=matrix((vocab, hidden)),
        token_codes=rng.integers(0, _MODULUS, size=(vocab, rank), dtype=np.uint16),
    )


def test_two_independent_state_shares_match_clear_prefill_and_decode() -> None:
    weights = _weights()
    shared = SharedFeedbackReference(weights, max_rows=5)
    clear = ClearFeedbackReference(weights, max_rows=5)
    np.testing.assert_array_equal(shared.prefill([2, 3, 7]), clear.prefill([2, 3, 7]))
    for _ in range(2):
        actual_token, actual_scores = shared.decode_selected()
        expected_token, expected_scores = clear.decode_selected()
        assert actual_token == expected_token
        np.testing.assert_array_equal(actual_scores, expected_scores)
    assert shared.workers[0].position == shared.workers[1].position == clear.position == 5
    assert shared.workers[0].party != shared.workers[1].party
    assert any(not np.array_equal(first, original) for first, original in zip(
        shared.workers[0]._states, clear.states, strict=True,
    ))
    ledger = shared.ledger()
    assert ledger == {
        "schema": "pllm.share_linear_feedback_reference.v1",
        **{
            key: project_feedback_cost(
                layers=3, rank=4, rows=5, scored_rows=3,
                hidden=8, vocabulary=8, word_bytes=2,
            )[key]
            for key in (
                "client_to_workers_body_bytes", "workers_to_client_body_bytes",
                "all_link_body_bytes", "worker_to_worker_body_bytes",
                "offline_one_use_body_bytes", "remote_integer_macs",
                "client_nonlinear_elements",
            )
        },
        "dependent_client_cuts": 15,
        "positions": 5,
    }
    assert ledger["all_link_body_bytes"] == 4_456
    assert ledger["remote_integer_macs"] > 80 * ledger["client_nonlinear_elements"]
    with pytest.raises(FeedbackReferenceError):
        shared.decode_selected()
    assert all(worker.closed for worker in shared.workers)
    assert all(not state.any() for worker in shared.workers for state in worker._states)


def test_replay_or_cross_session_input_fails_closed_and_erases_worker_state() -> None:
    weights = _weights()
    first = SharedFeedbackReference(weights, max_rows=3)
    second = SharedFeedbackReference(weights, max_rows=3)
    assert first.binding != second.binding
    first.prefill([0])
    stale = _pack(first.binding, _INPUT, 0, 0, 0, np.zeros(weights.rank, dtype=np.uint16))
    with pytest.raises(FeedbackReferenceError):
        first.workers[0].start_token(stale)
    assert first.workers[0].closed
    assert all(not state.any() for state in first.workers[0]._states)
    with pytest.raises(FeedbackReferenceError):
        second.workers[0].start_token(stale)
    assert second.workers[0].closed


def test_malformed_feedback_burns_pending_cut_without_replay() -> None:
    weights = _weights()
    session = SharedFeedbackReference(weights, max_rows=1)
    worker = session.workers[0]
    token = _pack(session.binding, _INPUT, 0, 0, 0, np.zeros(weights.rank, dtype=np.uint16))
    worker.start_token(token)
    worker.begin_layer()
    good = _pack(session.binding, _FEEDBACK, 0, 0, 0, np.zeros(weights.rank, dtype=np.uint16))
    bad = good[:-2] + _MODULUS.to_bytes(2, "little")
    with pytest.raises(FeedbackReferenceError):
        worker.finish_layer(bad)
    assert worker.closed
    with pytest.raises(FeedbackReferenceError):
        worker.finish_layer(good)


def test_public_weights_and_cost_bound_fail_closed() -> None:
    weights = _weights()
    with pytest.raises(ValueError):
        weights.recurrent[0][0, 0] = 7
    for altered in (
        np.zeros((8, 9), dtype=np.int16),
        np.zeros((8, 8), dtype=np.float32),
        np.full((8, 8), 3, dtype=np.int16),
    ):
        with pytest.raises(FeedbackReferenceError):
            FeedbackWeights.bind(
                recurrent=(altered,), incoming=(np.zeros((8, 4), dtype=np.int16),),
                project=(np.zeros((4, 8), dtype=np.int16),),
                expand=(np.zeros((8, 4), dtype=np.int16),),
                head=np.zeros((8, 8), dtype=np.int16),
                token_codes=np.zeros((8, 4), dtype=np.uint16),
            )
    with pytest.raises(FeedbackReferenceError):
        project_feedback_cost(
            layers=24, rank=32, rows=70, scored_rows=32,
            hidden=896, vocabulary=151_936, word_bytes=4,
        )


def test_qwen_shape_is_a_projection_only_and_reports_round_bound() -> None:
    rank32 = project_feedback_cost(
        layers=24, rank=32, rows=70, scored_rows=32,
        hidden=896, vocabulary=64, word_bytes=4,
    )
    rank16 = project_feedback_cost(
        layers=24, rank=16, rows=70, scored_rows=32,
        hidden=896, vocabulary=64, word_bytes=4,
    )
    assert rank16["all_link_body_bytes"] < rank32["all_link_body_bytes"] < 1_789_706
    assert rank32["remote_integer_macs"] > 1_000 * rank32["client_nonlinear_elements"]
    assert rank32["implemented_dependent_client_cuts"] == 1_680
    assert rank32["wavefront_cut_round_lower_bound_if_batched"] == 806


def test_cost_gate_keeps_the_same_control_and_bounded_word_contract() -> None:
    root = Path(__file__).parents[1]
    gate = json.loads((root / "docs/evidence/share-linear-feedback-cost-gate-2026-09-29.json").read_text())
    control = json.loads((root / gate["measured_control"]["source"]).read_text())
    assert gate["measured_control"]["body_fingerprint"] == control["source"]["body_fingerprint"]
    for outputs, label in ((8, "eight"), (32, "thirty_two")):
        cohort = next(row for row in control["cohorts"] if row["output_tokens"] == outputs)
        measured = cohort["prepared_control"]["covered_all_link_body_bytes"]
        assert gate["measured_control"][f"{label}_output_covered_all_link_body_bytes"] == measured
        for rank in (16, 32):
            for word, descriptor in ((4, "four_byte_word"), (8, "eight_byte_word")):
                if word == 8 and outputs == 8:
                    continue
                projected = project_feedback_cost(
                    layers=24, rank=rank, rows=39 + outputs - 1,
                    scored_rows=outputs, hidden=896, vocabulary=64, word_bytes=word,
                )
                recorded = gate["hypothetical_24_layer_896_wide_costs"][descriptor]
                assert recorded[f"{label}_outputs_rank{rank}_all_link_body_bytes"] == projected["all_link_body_bytes"]
                if word == 4:
                    assert projected["all_link_body_bytes"] < measured // 100
