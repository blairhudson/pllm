"""No-training structural and held-out cut gates for existing text decoders.

Structure uses complete native semantic schedules and the *optimistic* u16
prepared-stage body size at 39+32; it cannot bound a redesigned MPC protocol.
Optional held-out evaluation modifies only a frozen, public SmolLM checkpoint
after the original float32 MLP runs. It cannot establish protected execution.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from pllm import Model, lower_model
from pllm.profiles import MaskedLinearCpu
from pllm.quantization import SymmetricPerRow
from pllm.runtime.rank_interface_reference import OutputRankBasis, fit_output_rank_basis
from pllm.runtime.semantic_stages import scheduled_stage_specs

from probe_rank_interface import (
    _collect_outputs,
    _logits_and_cache,
    _observe,
    _project_outputs,
    _sha256,
    _token_ids,
)

_ROOT = Path(__file__).resolve().parents[1]
_FIXTURES = _ROOT / "crates/pllm-models/tests/fixtures"
_SMOL = "HuggingFaceTB/SmolLM2-135M-Instruct"
_SMOL_REVISION = "12fd25f77366fa6b3b4b768ec3050bf629380bac"
_QWEN = "Qwen/Qwen2.5-0.5B-Instruct"
_QWEN_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
_ROWS = 70  # 39 prefill rows plus 31 decode rows; 32 generated tokens.
_CONTROL = 178_970_558
_TENFOLD = _CONTROL // 10
_HUNDREDFOLD = _CONTROL // 100


def _cached_config(model: str, revision: str) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            model,
            "config.json",
            revision=revision,
            local_files_only=True,
        )
    )


def structural_screen() -> dict[str, Any]:
    configs = (
        (_QWEN, _QWEN_REVISION, _cached_config(_QWEN, _QWEN_REVISION)),
        (
            "Qwen/Qwen3-0.6B",
            "c1899de289a04d12100db370d81485cdf75e47ca",
            _FIXTURES / "Qwen3-0.6B-c1899de-config.json",
        ),
        ("Qwen/Qwen3.5-4B", "851bf6e", _FIXTURES / "Qwen3.5-4B-851bf6e-config.json"),
        (
            "microsoft/Phi-4-mini-instruct",
            "cfbefac",
            _FIXTURES / "Phi-4-mini-instruct-cfbefac-config.json",
        ),
        ("google/gemma-4-E2B-it", "3e22461f", _FIXTURES / "gemma-4-E2B-it-3e22461f-config.json"),
        (_SMOL, _SMOL_REVISION, _FIXTURES / "SmolLM2-135M-Instruct-12fd25f-config.json"),
    )
    candidates: dict[str, Any] = {}
    for model_id, revision, config_path in configs:
        source = json.loads(config_path.read_text(encoding="utf-8"))
        plan = lower_model(source, batch=1, max_input_tokens=39, max_new_tokens=32)
        composition = MaskedLinearCpu(
            Model.hf(model_id, revision=revision),
            quantization=SymmetricPerRow(weight_bits=8, activation_bits=8),
        )
        schedule = plan.runtime_schedule(composition)
        if not schedule.complete or schedule.protected_execution:
            raise ValueError(f"{model_id} has no complete public text schedule")
        stages = scheduled_stage_specs(plan, composition)
        remote = [stage for stage in stages if stage.role not in {"token_lookup", "lm_head"}]
        if not remote or any(stage.in_features <= 0 or stage.out_features <= 0 for stage in remote):
            raise ValueError("unbound remote stage dimensions")
        remote_linear_macs = _ROWS * sum(s.in_features * s.out_features for s in remote)
        local_head_macs = 32 * stages[-1].in_features * stages[-1].out_features
        # One u16 input and one u16 output online, one u16 correction offline;
        # no tickets, frame headers, bundle, or control. This favors the plan.
        optimistic_u16_bodies = 2 * _ROWS * sum(s.in_features + 2 * s.out_features for s in remote)
        hidden = int(source.get("text_config", source).get("hidden_size", 0))
        layers = int(source.get("text_config", source).get("num_hidden_layers", 0))
        if not 0 < hidden <= 8192 or not 0 < layers <= 128:
            raise ValueError("source has no bounded text hidden/layer dimensions")
        candidates[model_id] = {
            "config_sha256": _sha256(config_path),
            "source_revision_or_fixture": revision,
            "semantic_plan_digest": plan.digest,
            "schedule_digest": schedule.digest,
            "hidden_width": hidden,
            "layers": layers,
            "remote_stages": len(remote),
            "remote_body_linear_macs": remote_linear_macs,
            "local_head_linear_macs": local_head_macs,
            "remote_fraction_of_tracked_linear_macs": (
                remote_linear_macs / (remote_linear_macs + local_head_macs)
            ),
            "optimistic_u16_prepared_stage_bodies_39_plus_32_bytes": optimistic_u16_bodies,
            "optimistic_two_hidden_source_online_openings_bytes": (2 * layers * _ROWS * hidden * 4),
            "public_i8_remote_weight_snapshot_bytes_excluding_scales": sum(
                s.in_features * s.out_features for s in remote
            ),
        }
    return {
        "schema": "pllm.pretrained_bottleneck_screen.v1",
        "candidate_sources": candidates,
        "shape": {"input_tokens": 39, "output_tokens": 32, "executed_rows": _ROWS},
        "qwen_prepared_control_covered_body_bytes": _CONTROL,
        "qwen_tenfold_budget_bytes": _TENFOLD,
        "qwen_hundredfold_budget_bytes": _HUNDREDFOLD,
        "stage_floor_scope": (
            "optimistic existing prepared stage protocol at u16 for all stages; "
            "not a lower bound on new cryptographic protocols or a measured candidate response"
        ),
        "two_source_scope": (
            "optimistic two independent four-byte hidden-source openings per row and layer; "
            "excludes attention arithmetic, nonlinear keys, rescaling, setup and feedback"
        ),
    }


def heldout_screen() -> dict[str, Any]:
    import torch
    import transformers

    torch.set_num_threads(4)
    path = _cached_config(_SMOL, _SMOL_REVISION).parent
    if _sha256(path / "config.json") != _sha256(
        _FIXTURES / "SmolLM2-135M-Instruct-12fd25f-config.json"
    ):
        raise ValueError("pinned public model configuration differs from the source mirror")
    prompt_path = _ROOT / "examples/benchmarks/selective_subspace_prompts.json"
    confirm_path = _ROOT / "examples/benchmarks/selective_subspace_confirmation.json"
    prompts = json.loads(prompt_path.read_text(encoding="utf-8"))
    prompts["confirmation"] = json.loads(confirm_path.read_text(encoding="utf-8"))
    if (
        set(prompts) != {"calibration", "tuning", "heldout", "confirmation"}
        or [len(prompts[k]) for k in ("calibration", "tuning", "heldout", "confirmation")]
        != [20, 6, 12, 12]
        or any(type(text) is not str or not text for group in prompts.values() for text in group)
        or len({text for group in prompts.values() for text in group}) != 50
    ):
        raise ValueError("held-out public cohorts are not disjoint or bounded")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        path,
        local_files_only=True,
        trust_remote_code=False,
    )
    cohorts = {
        group: [_token_ids(tokenizer, text, 39) for text in prompts[group]] for group in prompts
    }
    if len({tuple(ids) for group in cohorts.values() for ids in group}) != 50:
        raise ValueError("different public prompts collapse to one token sequence")
    model = transformers.AutoModelForCausalLM.from_pretrained(
        path,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.float32,
        attn_implementation="eager",
    ).eval()
    if (
        len(model.model.layers) != 30
        or model.config.hidden_size != 576
        or model.config.intermediate_size != 1536
    ):
        raise ValueError("frozen candidate checkpoint differs from the bound architecture")
    with _collect_outputs(model) as (_, outputs):
        for ids in cohorts["calibration"]:
            _logits_and_cache(model, ids)
    calibration = [np.concatenate(rows) for rows in outputs]
    if any(not 128 < rows.shape[0] <= 780 for rows in calibration):
        raise ValueError("candidate calibration rows cannot fit the checked rank cap")
    full_basis = [fit_output_rank_basis(rows, 128) for rows in calibration]
    profiles = {
        (rank, stride): [
            OutputRankBasis(basis.mean, basis.directions[:, :rank], basis.calibration_rows)
            for basis in full_basis
        ]
        for rank, stride in ((16, 1), (32, 1), (64, 1), (128, 1), (64, 10), (128, 10))
    }
    results: dict[str, Any] = {}
    for cohort in ("heldout", "confirmation"):
        reference: list[tuple[np.ndarray, np.ndarray, int]] = []
        for ids in cohorts[cohort]:
            logits, cache = _logits_and_cache(model, ids)
            selected = int(np.argmax(logits))
            decode, _ = _logits_and_cache(model, [selected], cache=cache)
            reference.append((logits, decode, selected))
        scores = {}
        for (rank, stride), bases in profiles.items():
            prefill_matches = decode_matches = free_matches = 0
            max_error = 0.0
            for ids, (expected_prefill, expected_decode, selected) in zip(
                cohorts[cohort],
                reference,
                strict=True,
            ):
                with _project_outputs(model, bases, stride):
                    actual_prefill, cache = _logits_and_cache(model, ids)
                    actual_selected = int(np.argmax(actual_prefill))
                    actual_decode, _ = _logits_and_cache(model, [selected], cache=cache)
                    if actual_selected == selected:
                        actual_free = actual_decode
                    else:
                        _, fresh_cache = _logits_and_cache(model, ids)
                        actual_free, _ = _logits_and_cache(
                            model,
                            [actual_selected],
                            cache=fresh_cache,
                        )
                prefill_matches += actual_selected == selected
                decode_matches += _observe(actual_decode, expected_decode)["top1_match"]
                free_matches += _observe(actual_free, expected_decode)["top1_match"]
                max_error = max(max_error, float(np.max(np.abs(actual_prefill - expected_prefill))))
            cut_layers = 30 // stride
            scores[f"rank{rank}_every{stride}"] = {
                "retained_rank": rank,
                "cut_layers": cut_layers,
                "prefill_matches": int(prefill_matches),
                "same_token_decode_matches": int(decode_matches),
                "free_second_token_matches": int(free_matches),
                "worst_prefill_logit_error": max_error,
                "optimistic_two_worker_cut_body_projection_bytes": (
                    4 * _ROWS * (cut_layers * 50 + cut_layers * rank * 8)
                ),
            }
        results[cohort] = scores
    return {
        "source_config_sha256": _sha256(path / "config.json"),
        "source_weight_sha256": _sha256(path / "model.safetensors"),
        "public_calibration_sha256": _sha256(prompt_path),
        "disjoint_confirmation_sha256": _sha256(confirm_path),
        "cohort_lengths": {key: len(items) for key, items in cohorts.items()},
        "token_lengths": {
            key: {"minimum": min(map(len, items)), "maximum": max(map(len, items))}
            for key, items in cohorts.items()
        },
        "float32_optimistic_output_rank_cut": results,
        "qualifier": (
            "Original MLP executes before every projection; same-token decode uses the "
            "reference first token and free decode uses the candidate token. One decode "
            "step, not a complete 32-token response. Projection omits all remaining "
            "remote stages, offline keys, full wire, CPU, and protected numeric arithmetic."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structure-only", action="store_true")
    args = parser.parse_args()
    result = structural_screen()
    if not args.structure_only:
        result["heldout_numeric_oracle"] = heldout_screen()
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
