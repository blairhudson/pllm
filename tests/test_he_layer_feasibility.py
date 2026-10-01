"""Complete semantic dependency gate, not a runnable encrypted decoder."""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from pllm import Model, lower_model
from pllm.modeling import ModelPlan
from pllm.profiles import MaskedLinearCpu
from pllm.quantization import SymmetricPerRow
from pllm.runtime.he_layer_feasibility import EncryptedLayerGateError, token_boundary_he_layer_gate

from test_shared_resources import CONFIG


_ROOT = Path(__file__).parents[1]
_EVIDENCE = _ROOT / "docs/evidence/he-token-boundary-feasibility-qwen25-2026-09-30.json"
_HUB_CACHE = os.environ.get("HF_HUB_CACHE") or str(
    Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"
)


def _composition() -> MaskedLinearCpu:
    return MaskedLinearCpu(
        Model.hf(
            "Qwen/Qwen2.5-0.5B-Instruct",
            revision="7ae557604adf67be50417f59c2c2f167def9a775",
        ),
        quantization=SymmetricPerRow(weight_bits=8, activation_bits=8),
    )


def _gate(output_tokens: int) -> dict:
    return token_boundary_he_layer_gate(
        lower_model(CONFIG, batch=1, max_input_tokens=39, max_new_tokens=output_tokens),
        _composition(),
        response_new_tokens=output_tokens,
        covered_all_link_body_budget_bytes={8: 11_684_396, 32: 17_897_055}[output_tokens],
        covered_online_body_budget_bytes={8: 7_383_174, 32: 11_354_502}[output_tokens],
    )


@pytest.mark.parametrize(("output_tokens", "ideal_ciphertexts"), [(8, 24), (32, 72)])
def test_compiler_bound_layer_requires_serial_attention_and_mlp_products(
    output_tokens: int, ideal_ciphertexts: int
) -> None:
    report = _gate(output_tokens)
    assert report["layer_count"] == 24
    assert report["hidden_width"] == 896
    assert report["idealized_both_direction_ciphertexts"] == ideal_ciphertexts
    assert report["idealized_client_to_provider_ciphertexts"] == 9 + output_tokens - 1
    assert report["idealized_provider_to_client_ciphertexts"] == output_tokens
    assert report["client_final_norm_and_output_head_required_but_not_he_placed"]
    for phase in ("prefill", "decode"):
        contract = report["semantic_phase_contracts"][phase]
        assert contract["layer_count"] == len(contract["layers"]) == 24
        assert contract["optimistic_serial_ciphertext_product_depth"] == 96
        assert [row["forced_chain_depth_at_gate"] for row in contract["layers"]] == [
            4 * (index + 1) for index in range(24)
        ]
        for layer in contract["layers"]:
            assert layer["operator_counts"]["rms_norm"] == 2
            assert layer["operator_counts"]["softmax"] == 1
            assert layer["operator_counts"]["kv_cache_append"] == 2
            assert layer["operator_counts"]["linear"] == 7
    assert report["semantic_phase_contracts"]["prefill"]["layers"][0]["score_shape"] == (
        1,
        14,
        39,
        39,
    )
    assert not report["numeric_fidelity_validated"]
    assert not report["whole_decoder_executable"]
    assert any("RMSNorm" in missing for missing in report["omitted_cryptographic_work"])


def test_he_gate_rejects_incomplete_feedback_domain_and_adapters() -> None:
    plan = lower_model(CONFIG, batch=1, max_input_tokens=3, max_new_tokens=2)
    kwargs = {
        "response_new_tokens": 2,
        "covered_all_link_body_budget_bytes": 2_000_000,
        "covered_online_body_budget_bytes": 1_000_000,
    }
    for change in (
        {"response_new_tokens": 3},
        {"slots_per_ciphertext": 0},
        {"covered_all_link_body_budget_bytes": True},
        {"covered_online_body_budget_bytes": 3_000_000},
    ):
        with pytest.raises(EncryptedLayerGateError):
            token_boundary_he_layer_gate(plan, _composition(), **{**kwargs, **change})

    changed = copy.deepcopy(plan.to_dict())
    changed["token_feedback"] = False
    with pytest.raises(EncryptedLayerGateError, match="feedback"):
        token_boundary_he_layer_gate(
            ModelPlan(json.dumps(changed, sort_keys=True).encode()), _composition(), **kwargs
        )

    # Q/K normalization changes this path; another adapter must earn its own
    # complete operator/numeric schedule rather than inherit Qwen2's estimate.
    qwen3 = {
        **CONFIG,
        "model_type": "qwen3",
        "hidden_size": 128,
        "intermediate_size": 256,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 32,
        "vocab_size": 256,
        "max_window_layers": 2,
        "layer_types": ["full_attention"] * 2,
        "attention_bias": False,
        "attention_dropout": 0.0,
        "use_cache": True,
        "use_sliding_window": False,
        "sliding_window": None,
        "rope_scaling": None,
    }
    with pytest.raises(EncryptedLayerGateError, match="complete attention/MLP semantics"):
        token_boundary_he_layer_gate(
            lower_model(qwen3, batch=1, max_input_tokens=3, max_new_tokens=2),
            _composition(),
            **kwargs,
        )


def test_secret_free_tenseal_context_reaches_only_measured_depth() -> None:
    pytest.importorskip("tenseal")
    completed = subprocess.run(
        [sys.executable, str(_ROOT / "scripts/probe_he_token_boundary.py"), "--backend-only"],
        cwd=_ROOT,
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["tenseal_version"] == "0.3.17"
    assert result["python_bootstrap_methods"] == []
    assert [row["confirmed_serial_ciphertext_product_depth"] for row in result["contexts"]] == [
        2,
        4,
    ]
    for row in result["contexts"]:
        assert not row["provider_has_secret_key"]
        assert row["first_failed_depth"]["attempted_depth"] == (
            row["confirmed_serial_ciphertext_product_depth"] + 1
        )
        assert row["first_failed_depth"]["message"] == "scale out of bounds"
        assert row["public_context_with_rotation_and_relin_keys_body_bytes"] > 17_897_055
        assert row["one_ciphertext_request_body_bytes"] > 0
        assert all(
            sample["sampled_max_absolute_error"] < 1e-3 for sample in row["bounded_depth_samples"]
        )


def test_locked_he_evidence_keeps_incomplete_numeric_and_context_costs_explicit() -> None:
    evidence = json.loads(_EVIDENCE.read_text(encoding="utf-8"))
    assert evidence["schema"] == "pllm.token_boundary_he_feasibility_evidence.v1"
    assert not evidence["whole_layer_executable"] and not evidence["numeric_fidelity_validated"]
    assert not evidence["full_wire_measured"]
    for output_tokens in (8, 32):
        report = _gate(output_tokens)
        locked = evidence["cohorts"][str(output_tokens)]
        assert (
            report["idealized_both_direction_ciphertexts"]
            == locked["idealized_both_direction_ciphertexts"]
        )
        assert (
            report["maximum_average_ciphertext_bytes_for_online_tenfold"]
            == locked["maximum_average_ciphertext_bytes_for_online_tenfold"]
        )
        for phase in ("prefill", "decode"):
            assert (
                report["semantic_phase_contracts"][phase][
                    "optimistic_serial_ciphertext_product_depth"
                ]
                == locked["optimistic_minimum_serial_product_depth_per_forward"]
            )
        assert [
            sample["sampled_boundary_exceeds_online_tenfold_budget"]
            for sample in locked["sampled_contexts"]
        ] == ([False, True] if output_tokens == 8 else [True, True])
        assert all(
            sample["one_client_key_public_eval_context_exceeds_all_link_tenfold_budget"]
            for sample in locked["sampled_contexts"]
        )
        assert all(
            sample["depth_shortfall_even_ignoring_norms_and_softmax"] >= 92
            for sample in locked["sampled_contexts"]
        )


@pytest.mark.skipif(not os.getenv("PLLM_RUN_REAL_QWEN25"), reason="requires cached pinned config")
def test_official_config_reproduces_compiler_he_depth_and_token_boundaries() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(_ROOT / "scripts/probe_he_token_boundary.py"),
            "--no-backend",
            "--summary",
        ],
        cwd=_ROOT,
        env={**os.environ, "HF_HUB_CACHE": _HUB_CACHE},
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    evidence = json.loads(_EVIDENCE.read_text(encoding="utf-8"))
    assert report["source"] == evidence["source"]
    for output_tokens in (8, 32):
        row = report["cohorts"][str(output_tokens)]
        locked = evidence["cohorts"][str(output_tokens)]
        assert row["plan_digest"] == locked["plan_digest"]
        assert row["schedule_digest"] == locked["schedule_digest"]
        assert row["composition_digest"] == locked["composition_digest"]
        for phase in ("prefill", "decode"):
            assert (
                row["semantic_phase_contracts"][phase]["layer_provenance_sha256"]
                == locked["layer_provenance_sha256"][phase]
            )
