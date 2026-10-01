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
from pllm.runtime.region_contract_cost import compiler_region_contract_cost
from pllm.runtime.shared_resources import SharedResourceError

from test_shared_resources import CONFIG

_ROOT = Path(__file__).parents[1]
_EVIDENCE = _ROOT / "docs/evidence/compiler-region-contract-qwen25-2026-09-30.json"
_SHARED_HUB_CACHE = os.environ.get("HF_HUB_CACHE") or str(
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


def _report(output_tokens: int) -> dict:
    return compiler_region_contract_cost(
        lower_model(CONFIG, batch=1, max_input_tokens=39, max_new_tokens=output_tokens),
        _composition(),
        response_new_tokens=output_tokens,
        maximum_online_all_link_body_bytes={8: 7_383_174, 32: 11_354_502}[output_tokens],
        maximum_total_all_link_body_bytes={8: 11_684_396, 32: 17_897_055}[output_tokens],
    )


@pytest.mark.parametrize(
    ("outputs", "boundary_floor", "resident_24", "resident_12"),
    [
        (8, 13_848_576, 12_174_848, 6_239_744),
        (32, 21_073_920, 18_669_056, 9_637_376),
    ],
)
def test_compiler_bound_two_worker_interfaces_veto_incomplete_placements(
    outputs: int,
    boundary_floor: int,
    resident_24: int,
    resident_12: int,
) -> None:
    report = _report(outputs)
    assert report["input_tokens"] == 39
    assert report["executed_rows"] == 39 + outputs - 1
    assert report["layer_count"] == len(report["layer_provenance"]) == 24
    assert report["protected_composition_selected"] is False
    cut = report["client_attention_remote_mlp"]
    assert cut["known_online_body_floor_bytes"] == boundary_floor
    assert cut["known_floor_exceeds_online_budget"]
    assert cut["known_floor_exceeds_all_link_budget"]
    assert cut["remote_fraction_of_tracked_body_projection_macs"] > 0.8
    assert cut["client_attention_projection_integer_macs"] > 0
    assert cut["client_output_head_integer_macs"] > 0
    assert cut["additional_client_attention_i8_weight_and_f32_scale_bytes_once_per_model"] > 0
    assert (
        cut["remote_mlp_projection_integer_macs_per_worker"]
        > cut["client_attention_projection_integer_macs"]
    )
    links = cut["directed_known_online_links"]
    assert {(row["source"], row["destination"]) for row in links} == {
        ("client", "worker_a"),
        ("client", "worker_b"),
        ("worker_a", "client"),
        ("worker_b", "client"),
    }
    assert sum(row["optimistic_body_bytes"] for row in links) == boundary_floor
    for bits, floor, veto in (("24", resident_24, True), ("12", resident_12, False)):
        resident = report["two_worker_resident"][bits]
        assert resident["known_online_body_floor_bytes"] == floor
        assert resident["known_floor_exceeds_all_link_budget"] is veto
        assert resident["known_floor_exceeds_online_budget"] is veto
        assert resident["all_link_cost_known"] is False
        assert resident["byte_admitted"] is False
        edges = resident["directed_known_online_links"]
        assert {(edge["source"], edge["destination"]) for edge in edges[:2]} == {
            ("worker_a", "worker_b"),
            ("worker_b", "worker_a"),
        }
        assert sum(edge["optimistic_body_bytes"] for edge in edges) == floor
        assert any(
            row["link"] == "dealer→worker_a" for row in resident["unknown_required_links_and_work"]
        )
        assert any(
            row["link"] == "dealer→worker_b" for row in resident["unknown_required_links_and_work"]
        )
        assert all(row["body_bytes"] is None for row in resident["unknown_required_links_and_work"])


def test_region_contract_rejects_forged_semantic_sources_and_invalid_budgets() -> None:
    plan = lower_model(CONFIG, batch=1, max_input_tokens=2, max_new_tokens=2)
    good = {
        "response_new_tokens": 2,
        "maximum_online_all_link_body_bytes": 1_000_000,
        "maximum_total_all_link_body_bytes": 2_000_000,
    }
    for changes in (
        {"response_new_tokens": 3},
        {"maximum_online_all_link_body_bytes": 3_000_000},
        {"resident_source_bits": (12, 12)},
        {"mlp_input_share_bits": 12},
        {"maximum_total_all_link_body_bytes": True},
    ):
        with pytest.raises(SharedResourceError):
            compiler_region_contract_cost(plan, _composition(), **{**good, **changes})
    altered = copy.deepcopy(plan.to_dict())
    attention = next(
        row for row in altered["prefill"]["operations"] if row.get("operator") == "attention_scores"
    )
    attention["inputs"] = ["missing.query", *attention["inputs"][1:]]
    forged = ModelPlan(json.dumps(altered, sort_keys=True).encode())
    with pytest.raises((SharedResourceError, ValueError)):
        compiler_region_contract_cost(forged, _composition(), **good)


def test_alternate_semantic_adapter_uses_same_role_and_source_contract() -> None:
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
    plan = lower_model(qwen3, batch=1, max_input_tokens=3, max_new_tokens=2)
    report = compiler_region_contract_cost(
        plan,
        MaskedLinearCpu(
            Model("org/alternate-dense-decoder"),
            quantization=SymmetricPerRow(weight_bits=8, activation_bits=8),
        ),
        response_new_tokens=2,
        maximum_online_all_link_body_bytes=1_000_000,
        maximum_total_all_link_body_bytes=2_000_000,
    )
    assert plan.to_dict()["adapter"] == "pllm.qwen3.v1"
    assert report["layer_count"] == 2
    assert report["executed_rows"] == 4
    assert report["client_attention_remote_mlp"]["known_online_body_floor_bytes"] > 0
    assert report["two_worker_resident"]["12"]["known_online_body_floor_bytes"] > 0
    assert not report["client_attention_remote_mlp"]["byte_admitted"]


def test_locked_region_cost_cohorts_keep_unpriced_work_explicit() -> None:
    evidence = json.loads(_EVIDENCE.read_text(encoding="utf-8"))
    assert evidence["schema"] == "pllm.compiler_region_contract_cost_evidence.v1"
    assert not evidence["whole_response_byte_admitted"]
    for cohort in evidence["cohorts"]:
        output_tokens = cohort["output_tokens"]
        report = _report(output_tokens)
        prepared = cohort["matched_prepared"]
        assert report["layer_count"] == cohort["semantic_layers"]
        assert (
            report["maximum_online_all_link_body_bytes"] == prepared["tenfold_online_budget_bytes"]
        )
        assert (
            report["maximum_total_all_link_body_bytes"] == prepared["tenfold_all_link_budget_bytes"]
        )
        cut = report["client_attention_remote_mlp"]
        locked_cut = cohort["client_attention_remote_mlp"]
        assert (
            cut["known_online_body_floor_bytes"]
            == locked_cut["known_online_body_floor_bytes"]
            == sum(
                locked_cut[f"{direction}_body_bytes"]
                for direction in (
                    "client_to_worker_a",
                    "client_to_worker_b",
                    "worker_a_to_client",
                    "worker_b_to_client",
                )
            )
        )
        assert (
            cut["additional_client_attention_i8_weight_and_f32_scale_bytes_once_per_model"]
            == locked_cut["additional_client_i8_weight_and_f32_scale_bytes_once_per_model"]
        )
        assert (
            cut["provider_mlp_i8_weight_and_f32_scale_bytes_per_worker_once_per_model"]
            == locked_cut["provider_mlp_i8_weight_and_f32_scale_bytes_per_worker_once_per_model"]
        )
        for metric in (
            "client_attention_projection_integer_macs",
            "remote_mlp_projection_integer_macs_per_worker",
            "client_output_head_integer_macs",
            "remote_fraction_of_tracked_body_projection_macs",
        ):
            assert cut[metric] == locked_cut[metric]
        assert (
            cut["current_quadratic_numerator_dealer_bodies_both_parties_comparator_bytes"]
            == locked_cut["current_quadratic_numerator_two_party_dealer_comparator_bytes"]
        )
        assert not cut["byte_admitted"] and cut["unknown_required_links_and_work"]
        for bits in ("24", "12"):
            resident = report["two_worker_resident"][bits]
            locked_resident = cohort["two_worker_resident"]
            assert (
                resident["known_online_body_floor_bytes"]
                == locked_resident[f"{bits}_bit_known_online_body_floor_bytes"]
            )
            assert resident["byte_admitted"] is False
            assert all(
                item["body_bytes"] is None for item in resident["unknown_required_links_and_work"]
            )
        expected = cohort["two_worker_resident"]
        ring_12 = report["two_worker_resident"]["12"]
        assert (
            ring_12["known_online_body_floor_bytes"] - expected["12_bit_peer_opening_body_bytes"]
            == expected["token_boundary_body_bytes"]
        )
        assert (
            prepared["tenfold_online_budget_bytes"] - ring_12["known_online_body_floor_bytes"]
            == expected["12_bit_remaining_online_budget_bytes"]
        )
        assert (
            prepared["tenfold_all_link_budget_bytes"] - ring_12["known_all_link_body_floor_bytes"]
            == expected["12_bit_remaining_all_link_budget_bytes"]
        )


def test_12_bit_source_shares_need_private_carry_to_lift_into_wide_linear_ring() -> None:
    # Plain reinterpretation of separately held Z_4096 shares as wide-ring
    # shares changes even one public i8 weight product. The trusted-client
    # oracle below derives the missing carry; providers must NOT learn it.
    modulus = 1 << 12
    wide_modulus = 1 << 32
    possible_carries: dict[int, set[int]] = {}
    for value in (-127, -1, 0, 1, 127):
        carries: set[int] = set()
        for left in range(modulus):
            right = (value - left) % modulus
            carry, remainder = divmod(left + right - value, modulus)
            assert remainder == 0
            assert 0 <= carry <= 2
            carries.add(carry)
            assert (left + right - carry * modulus) % wide_modulus == value % wide_modulus
            if carry:
                assert 73 * (left + right) % wide_modulus != 73 * value % wide_modulus
        possible_carries[value] = carries
    # Publishing the carry is not a privacy-compatible repair: a zero carry
    # excludes negative values, while a two-carry excludes positive values.
    assert possible_carries[127] == {0, 1}
    assert possible_carries[-127] == {1, 2}


@pytest.mark.skipif(not os.getenv("PLLM_RUN_REAL_QWEN25"), reason="requires cached pinned config")
@pytest.mark.parametrize("output_tokens", [8, 32])
def test_real_pinned_cost_probe_binds_config_and_schedule(output_tokens: int) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(_ROOT / "scripts/probe_region_contract_cost.py"),
            "--max-output-tokens",
            str(output_tokens),
            "--summary",
        ],
        cwd=_ROOT,
        env={**os.environ, "HF_HUB_CACHE": _SHARED_HUB_CACHE},
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    evidence = json.loads(_EVIDENCE.read_text(encoding="utf-8"))
    cohort = next(item for item in evidence["cohorts"] if item["output_tokens"] == output_tokens)
    assert report["plan_digest"] == cohort["plan_digest"]
    assert report["schedule_digest"] == cohort["schedule_digest"]
    assert report["layer_provenance_sha256"] == cohort["layer_provenance_sha256"]
    assert report["body_fingerprint_from_control"] == evidence["source"]["body_fingerprint"]
    assert report["composition_digest"] == evidence["source"]["composition_digest"]
    assert (
        report["client_attention_remote_mlp"]["known_online_body_floor_bytes"]
        == cohort["client_attention_remote_mlp"]["known_online_body_floor_bytes"]
    )
    for bits in ("12", "24"):
        assert (
            report["two_worker_resident"][bits]["known_online_body_floor_bytes"]
            == cohort["two_worker_resident"][f"{bits}_bit_known_online_body_floor_bytes"]
        )
    assert not report["whole_decoder_executable"]
