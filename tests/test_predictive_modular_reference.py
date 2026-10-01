from __future__ import annotations

import json
import os
import subprocess
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pytest

from pllm.runtime.predictive_modular_reference import (
    ModularRequest,
    ModularResponse,
    PredictiveModularError,
    certified_error_bound,
    issue_certified_modular_stage,
    minimum_certified_ring_bits,
    sparse_predictor_options,
)


_PLAN = "f" * 64
_ROOT = Path(__file__).resolve().parents[1]
_EVIDENCE = _ROOT / "docs/evidence/predictive-modular-qwen25-2026-09-30.json"


def _issue(weight: np.ndarray, predictor: np.ndarray, *, q: int = 4, domain: int = 127):
    return issue_certified_modular_stage(
        weight,
        predictor,
        modulus=q,
        rows=1,
        stage_id="test.semantic.projection",
        plan_digest=_PLAN,
        max_abs_input=domain,
    )


@pytest.mark.parametrize("q", [4, 256, 65536, 1 << 32])
def test_certified_lift_reconstructs_full_signed_integer_not_just_residue(q: int) -> None:
    weight = np.array([[127, -63], [-7, 38]], dtype=np.int8)
    predictor = weight.copy()
    if q == 256:
        predictor[0, 0] = 126  # B=127 < q/2; even worst-case input is certified.
    client, provider = _issue(weight, predictor, q=q)
    input_row = np.array([[-127, 126]], dtype=np.int8)
    request = client.mask_input(input_row)
    response = provider.evaluate(request)
    actual = client.recover(response)
    expected = input_row.astype(np.int64) @ weight.astype(np.int64).T
    np.testing.assert_array_equal(actual, expected)
    if q < 65536:
        assert abs(int(expected[0, 0])) > q
    with pytest.raises(PredictiveModularError, match="consumed"):
        provider.evaluate(request)
    with pytest.raises(PredictiveModularError, match="awaiting"):
        client.recover(response)


def test_public_certificate_covers_every_input_in_a_small_integer_domain() -> None:
    weight = np.array([[3, -2], [-1, 5]], dtype=np.int8)
    predictor = weight.copy()
    predictor[1, 1] -= 1
    np.testing.assert_array_equal(certified_error_bound(weight, predictor, max_abs_input=3), [0, 3])
    for x, y in product(range(-3, 4), repeat=2):
        client, provider = _issue(weight, predictor, q=8, domain=3)
        row = np.array([[x, y]], dtype=np.int8)
        actual = client.recover(provider.evaluate(client.mask_input(row)))
        np.testing.assert_array_equal(actual, row.astype(np.int64) @ weight.astype(np.int64).T)


def test_two_bit_sparse_predictor_is_not_an_exact_w8a8_contract() -> None:
    weight = np.array([[3, -2]], dtype=np.int8)
    predictor = np.array([[3, -1]], dtype=np.int8)
    np.testing.assert_array_equal(certified_error_bound(weight, predictor), [127])
    with pytest.raises(PredictiveModularError, match="whole-input exact lift certificate"):
        _issue(weight, predictor)
    with pytest.raises(PredictiveModularError, match="whole-input exact lift certificate"):
        _issue(weight, predictor, q=128)
    client, provider = _issue(weight, predictor, q=256)
    x = np.array([[0, -127]], dtype=np.int8)
    np.testing.assert_array_equal(
        client.recover(provider.evaluate(client.mask_input(x))),
        x.astype(np.int64) @ weight.astype(np.int64).T,
    )


def test_public_sparse_column_certificate_matches_full_matrix_contract() -> None:
    weight = np.array(
        [[3, -2, 1, 0, 4, 2, -1, 3, -2, 7], [-2, 1, 4, 0, -3, 5, 2, -2, 3, 1]],
        dtype=np.int8,
    )
    options = sparse_predictor_options(weight)
    for fraction, option in options.items():
        selected = option["selected"]
        assert len(selected) == weight.shape[1] * fraction // 100
        predictor = np.zeros_like(weight)
        predictor[:, selected] = weight[:, selected]
        exact = certified_error_bound(weight, predictor)
        assert option["public_worst_error"] == int(np.max(exact))


def test_strict_ring_width_and_locked_projection_never_authorize_two_bit_w8a8() -> None:
    assert [minimum_certified_ring_bits(value) for value in (0, 1, 2, 3, 127)] == [
        2,
        2,
        3,
        3,
        8,
    ]
    locked = json.loads(_EVIDENCE.read_text(encoding="utf-8"))
    short, long = locked["cohorts"].values()
    assert (
        short["covered_prepared_all_link_body_bytes"] < long["covered_prepared_all_link_body_bytes"]
    )
    assert short["remote_integer_macs"] < long["remote_integer_macs"]
    for row in (short, long):
        assert (
            row["unattained_uniform_two_bit_arithmetic_floor_bytes"]
            < row["tenfold_all_link_budget_bytes"]
        )
        for candidate in row["predictors"].values():
            assert candidate["certified_ring_bits_min_max"][0] >= 19
            assert (
                candidate["optimistic_certified_all_link_arithmetic_bytes"]
                > row["tenfold_all_link_budget_bytes"]
            )
            assert (
                candidate["optimistic_certified_online_arithmetic_bytes"]
                > row["tenfold_online_budget_bytes"]
            )
    assert long["predictors"]["90"]["client_integer_macs"] > (
        9 * long["remote_integer_macs"] // 10 - 100_000_000
    )


@pytest.mark.quality
@pytest.mark.skipif(
    os.environ.get("PLLM_RUN_REAL_QWEN25") != "1",
    reason="opt in to the pinned cached real Qwen2.5 checkpoint",
)
@pytest.mark.parametrize("output_tokens", [8, 32])
def test_real_compiled_sparse_predictor_cost_gate_matches_locked_evidence(
    output_tokens: int,
) -> None:
    hub = os.environ.get("HF_HUB_CACHE") or str(
        Path(os.environ["HF_HOME"]) / "hub"
        if "HF_HOME" in os.environ
        else Path.home() / ".cache/huggingface/hub"
    )
    process = subprocess.run(
        [
            sys.executable,
            str(_ROOT / "scripts/probe_predictive_modular.py"),
            "--max-output-tokens",
            str(output_tokens),
            "--summary",
        ],
        cwd=_ROOT,
        env={**os.environ, "HF_HUB_CACHE": hub},
        capture_output=True,
        text=True,
        check=False,
        timeout=900,
    )
    assert process.returncode == 0, process.stderr
    actual = json.loads(process.stdout)
    evidence = json.loads(_EVIDENCE.read_text(encoding="utf-8"))
    cohort = evidence["cohorts"][f"39_plus_{output_tokens}"]
    assert actual["source_lock_digest"] == evidence["source_lock_digest"]
    assert actual["checkpoint_digest"] == evidence["checkpoint_digest"]
    assert actual["model_body_fingerprint"] == evidence["body_fingerprint"]
    assert actual["remote_stage_count"] == 96
    for key in (
        "token_cohort_digest",
        "semantic_plan_digest",
        "runtime_schedule_digest",
        "stage_summaries_sha256",
        "covered_prepared_all_link_body_bytes",
        "covered_prepared_online_body_bytes",
        "tenfold_all_link_budget_bytes",
        "tenfold_online_budget_bytes",
        "unattained_uniform_two_bit_arithmetic_floor_bytes",
    ):
        assert actual[key] == cohort[key]
    for fraction, expected in cohort["predictors"].items():
        found = actual["predictors"][fraction]
        for key in (
            "client_integer_macs",
            "public_i8_predictor_bytes_including_u16_indices",
            "optimistic_certified_all_link_arithmetic_bytes",
            "optimistic_certified_online_arithmetic_bytes",
        ):
            assert found[key] == expected[key]
        assert (
            found["certified_minimum_ring_bits"],
            found["certified_maximum_ring_bits"],
        ) == tuple(expected["certified_ring_bits_min_max"])
        assert (
            found["observed_maximum_ring_bits"]
            == expected["observed_ring_bits_max_not_an_admission"]
        )
        assert found["remote_integer_macs"] == cohort["remote_integer_macs"]
        assert found["stages_certified_at_two_bits"] == 0
        assert found["stages_certified_at_eight_bits_or_less"] == 0


def test_ambiguous_half_modulus_and_invalid_inputs_are_never_issued() -> None:
    weight = np.array([[2]], dtype=np.int8)
    predictor = np.array([[0]], dtype=np.int8)
    with pytest.raises(PredictiveModularError, match="certificate"):
        _issue(weight, predictor, q=4, domain=1)  # B=q/2, two nearest lifts.
    for malformed in (0, 3, 12, 1 << 33, True):
        with pytest.raises(PredictiveModularError, match="contract"):
            _issue(weight, weight, q=malformed)
    with pytest.raises(PredictiveModularError, match="matrix"):
        _issue(weight, predictor.astype(np.float32))
    with pytest.raises(PredictiveModularError, match="input domain"):
        certified_error_bound(weight, weight, max_abs_input=0)


def test_client_and_provider_material_are_distinct_one_use_views() -> None:
    weight = np.array([[2, 1]], dtype=np.int8)
    client, provider = _issue(weight, weight)
    assert not hasattr(provider, "input_mask")
    assert not hasattr(provider, "output_mask")
    assert not hasattr(client, "correction")
    forged = ModularRequest("0" * 32, client.binding, np.zeros((1, 2), dtype=np.uint32))
    with pytest.raises(PredictiveModularError, match="matching ticket"):
        provider.evaluate(forged)
    request = client.mask_input(np.array([[-127, 127]], dtype=np.int8))
    with pytest.raises(PredictiveModularError, match="consumed"):
        client.mask_input(np.array([[0, 0]], dtype=np.int8))
    malformed = ModularRequest(request.ticket, "0" * 64, request.masked_input)
    with pytest.raises(PredictiveModularError, match="committed stage"):
        provider.evaluate(malformed)
    with pytest.raises(PredictiveModularError, match="consumed"):
        provider.evaluate(request)
    np.testing.assert_array_equal(provider.correction, 0)
    client.abort()
    with pytest.raises(PredictiveModularError, match="awaiting"):
        client.recover(
            ModularResponse(request.ticket, request.binding, np.zeros((1, 1), dtype=np.uint32))
        )
    np.testing.assert_array_equal(client.input_mask, 0)
    np.testing.assert_array_equal(client.output_mask, 0)


def test_bad_response_burns_client_ticket_and_out_of_domain_attempt_burns_masks() -> None:
    weight = np.array([[1]], dtype=np.int8)
    client, provider = _issue(weight, weight)
    request = client.mask_input(np.array([[12]], dtype=np.int8))
    bad = ModularResponse(request.ticket, request.binding, np.array([[4]], dtype=np.uint32))
    with pytest.raises(PredictiveModularError, match="bound ticket"):
        client.recover(bad)
    with pytest.raises(PredictiveModularError, match="awaiting"):
        client.recover(provider.evaluate(request))
    other, _ = _issue(weight, weight, domain=3)
    with pytest.raises(PredictiveModularError, match="fixed public domain"):
        other.mask_input(np.array([[4]], dtype=np.int8))
    np.testing.assert_array_equal(other.input_mask, 0)
    np.testing.assert_array_equal(other.output_mask, 0)
