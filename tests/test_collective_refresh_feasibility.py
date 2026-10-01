"""Correctness/privacy-bound and directed-link regression oracles, not HE tests."""

from __future__ import annotations

import importlib.util
from fractions import Fraction
from pathlib import Path

import pytest


_PATH = Path(__file__).resolve().parents[1] / "scripts/probe_collective_refresh_feasibility.py"
_SPEC = importlib.util.spec_from_file_location("collective_refresh_probe", _PATH)
assert _SPEC and _SPEC.loader
probe = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(probe)


@pytest.mark.parametrize("q", [64, 128, 256])
def test_distributed_rounding_failure_bound_exhaustive(q: int) -> None:
    for message in range(-8, 9):
        failures = sum(
            probe.distributed_round_coefficient(first, q)
            + probe.distributed_round_coefficient(message - first, q)
            != message
            for first in range(q)
        )
        assert failures <= 2 * (abs(message) + 1)
        if message == 0:
            assert failures == 2  # Exact interval-endpoint convention, tiny even q.


def test_odd_RNS_modulus_has_coefficient_roundoff() -> None:
    for q in (65, 257):
        result = probe.rounding_oracle(q)
        assert result["small_coefficient_errors"] == [-1, 0]
        assert result["is_secure_refresh"] is False


def test_individual_success_probability_is_not_response_bound() -> None:
    # Per-call q meets 2^-50, yet 1,116 calls need additional modulus bits.
    individual = probe.required_rounding_modulus_bits(16384, 1 << 40, 1, 50)
    response = probe.required_rounding_modulus_bits(16384, 1 << 40, 1116, 50)
    assert individual == 105
    assert response == 116
    assert 2 * 16384 * (1 << 40) * 1116 * (1 << 50) <= 1 << response


def test_scale_and_mask_reserve_veto_naive_sampled_depth() -> None:
    assert probe.minimum_mask_limbs((60, 40, 40), 40) is None
    assert probe.minimum_mask_limbs((60, 40, 40, 40, 40), 40) == 4
    # A direct-product/session TV guard can exhaust all usable levels.
    assert probe.minimum_mask_limbs((60, 40, 40, 40, 40), 40, 144) == 5
    assert probe.refresh_epochs(96, 1) == 95
    assert probe.refresh_epochs(96, 4) == 23


def test_both_worker_directions_and_both_masked_shares_count() -> None:
    masked = probe.worker_refresh_payload(16384, 4, 5, "masked_pair")
    rounding = probe.worker_refresh_payload(16384, 2, 5, "distributed_rounding")
    assert masked["A_to_B"] == 524288
    assert masked["B_to_A"] == 1179648
    assert masked["both_links"] == 1703936
    assert rounding["both_links"] == 1572864
    assert masked["messages_per_ciphertext_without_batching"] == 2


def test_uniform_interval_shift_privacy_needs_range_guard() -> None:
    assert probe.interval_shift_tv(1 << 168, 1 << 40) == Fraction(1, 1 << 128)
    assert probe.interval_shift_tv(1 << 168, 128 * (1 << 40)) == Fraction(1, 1 << 121)
    assert probe.interval_shift_tv(256, 512) == 1


def test_locked_evidence_and_optimistic_known_floors() -> None:
    result = probe.run()
    for cohort in result["cohorts"]:
        assert cohort["executed_rows"] == (46 if cohort["outputs"] == 8 else 70)
        for protocol in cohort["protocols"]:
            floor = protocol.get("one_live_ciphertext_raw_link_floor_bytes")
            if floor is not None:
                assert floor > cohort["budgets"]["all_link"]
    assert result["secure_collective_refresh_executed"] is False
    assert result["real_Qwen_numeric_fidelity_validated"] is False


@pytest.mark.parametrize(
    "action",
    [
        lambda: probe.refresh_epochs(96, 0),
        lambda: probe.minimum_mask_limbs((), 40),
        lambda: probe.worker_refresh_payload(16384, 2, 5, "unknown"),
        lambda: probe.rounding_oracle(65536),
    ],
)
def test_invalid_or_unbounded_oracle_fails_closed(action) -> None:
    with pytest.raises(ValueError):
        action()
