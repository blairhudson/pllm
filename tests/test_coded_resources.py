"""Public preprocessing storage lower bounds from complete semantic schedules."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pllm import Model, lower_model
from pllm.profiles import ClientOnlyCpu, MaskedLinearCpu
from pllm.quantization import SymmetricPerRow
from pllm.runtime.coded_resources import (
    CodedResourceError,
    coded_delegation_storage_lower_bound,
)

_QWEN25_CONFIG = {
    "model_type": "qwen2",
    "hidden_size": 896,
    "intermediate_size": 4864,
    "num_hidden_layers": 24,
    "num_attention_heads": 14,
    "num_key_value_heads": 2,
    "vocab_size": 151936,
    "max_position_embeddings": 32768,
    "hidden_act": "silu",
    "rms_norm_eps": 1e-6,
    "rope_theta": 1000000.0,
    "tie_word_embeddings": True,
}


def test_coded_cost_is_bound_to_compiled_qwen25_remote_stages() -> None:
    plan = lower_model(_QWEN25_CONFIG, batch=1, max_input_tokens=30, max_new_tokens=1)
    composition = MaskedLinearCpu(Model("Qwen/Qwen2.5-0.5B-Instruct"))
    report = coded_delegation_storage_lower_bound(
        plan, composition, resident_budget_bytes=2 << 30, cached_budget_bytes=6 << 30,
    )
    assert report["remote_stages"] == 96
    assert report["plan_digest"] == plan.digest
    assert report["composition_digest"] == composition.digest()
    assert report["preprocessing_bytes_lower_bound"] == 64 * report["public_w8_weight_bytes"]
    assert report["preprocessing_bytes_lower_bound"] > 6 << 30
    assert not report["fully_resident_within_budget"]
    assert not report["fully_cached_within_budget"]
    assert report["executable"] is False
    assert report["full_response_compute_cap_checked"] is False
    assert report["total_wire_bytes_measured"] is False
    assert all(row["input_width"] <= 4864 for row in report["stage_dimensions"])
    evidence = json.loads(
        (Path(__file__).parents[1] / "docs/evidence/maverick-coded-storage-lower-bound-2026-09-27.json")
        .read_text()
    )
    for name in (
        "remote_stages", "public_w8_weight_bytes", "privacy_p_bytes_lower_bound",
        "verification_q_bytes_lower_bound", "preprocessing_bytes_lower_bound", "executable",
    ):
        assert report[name] == evidence[name]
    for name in ("plan_digest", "schedule_digest", "composition_digest"):
        assert report[name] == evidence["source"][name]
    assert evidence["largest_stage_auxiliary_bytes_lower_bound"] == max(
        row["privacy_p_bytes_lower_bound"] + row["verification_q_bytes_lower_bound"]
        for row in report["stage_dimensions"]
    )


def test_coded_cost_rejects_unsupported_numeric_and_local_placement() -> None:
    plan = lower_model(_QWEN25_CONFIG, batch=1, max_input_tokens=8, max_new_tokens=1)
    source = Model("Qwen/Qwen2.5-0.5B-Instruct")
    with pytest.raises(CodedResourceError, match="W8A8"):
        coded_delegation_storage_lower_bound(
            plan,
            MaskedLinearCpu(source, quantization=SymmetricPerRow(weight_bits=4, activation_bits=4)),
            resident_budget_bytes=2 << 30,
            cached_budget_bytes=6 << 30,
        )
    with pytest.raises(CodedResourceError, match="no remote"):
        coded_delegation_storage_lower_bound(
            plan,
            ClientOnlyCpu(source),
            resident_budget_bytes=2 << 30,
            cached_budget_bytes=6 << 30,
        )
    with pytest.raises(CodedResourceError, match="budget"):
        coded_delegation_storage_lower_bound(
            plan, MaskedLinearCpu(source), resident_budget_bytes=0, cached_budget_bytes=1,
        )
