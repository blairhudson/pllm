"""Offline output-sensitivity and shared-subspace gates for a frozen Qwen decoder.

The original MLP is evaluated before projection. Projections cannot implement
protected nonlinear execution or prove any whole-response network savings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from pllm.runtime.rank_interface_reference import OutputRankBasis, fit_output_rank_basis
from probe_rank_interface import (
    _MODEL_ID,
    _REVISION,
    _collect_outputs,
    _logits_and_cache,
    _observe,
    _sha256,
    _token_ids,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

_PROMPTS = (
    Path(__file__).resolve().parents[1] / "examples/benchmarks/selective_subspace_prompts.json"
)
_LAYERS = 24
_HIDDEN = 896
_LOW, _BASE, _HIGH = 32, 64, 96
_ROWS = 70  # Pinned 39-input, 32-output control: 39 prefill + 31 decode.
_WORD_BYTES = 8  # Hypothetical two-worker numeric representation, not executable.
_FRAME_BYTES = 50  # Same projected typed-envelope size as other client-cut gates.


@contextmanager
def _project(model: object, bases: dict[int, OutputRankBasis]) -> Iterator[None]:
    import torch

    handles = []
    try:
        for index, basis in bases.items():
            mean = torch.tensor(basis.mean, dtype=torch.float32)
            directions = torch.tensor(basis.directions, dtype=torch.float32)

            def project(
                _module: object,
                _inputs: object,
                output: object,
                *,
                mean=mean,
                directions=directions,
            ):
                return mean + ((output - mean) @ directions) @ directions.T

            handles.append(model.model.layers[index].mlp.register_forward_hook(project))
        yield
    finally:
        for handle in handles:
            handle.remove()


def _js(left: np.ndarray, right: np.ndarray) -> float:
    """Full-vocabulary Jensen-Shannon divergence (natural-log units)."""
    a = left.astype(np.float64)
    b = right.astype(np.float64)
    p = np.exp(a - np.max(a))
    q = np.exp(b - np.max(b))
    p /= p.sum()
    q /= q.sum()
    midpoint = (p + q) / 2
    with np.errstate(divide="ignore", invalid="ignore"):
        result = (
            np.sum(np.where(p > 0, p * np.log(p / midpoint), 0))
            + np.sum(np.where(q > 0, q * np.log(q / midpoint), 0))
        ) / 2
    return float(result)


def _summarize(observations: list[dict]) -> dict:
    return {
        "top1_matches": sum(row["top1_match"] for row in observations),
        "mean_top5_recall": sum(row["top5_recall"] for row in observations) / len(observations),
        "worst_logit_error": max(row["worst_logit_error"] for row in observations),
        "mean_js_divergence": sum(row["js_divergence"] for row in observations) / len(observations),
        "retrospective_margin_witnesses": sum(
            row["retrospective_margin_witness"] for row in observations
        ),
    }


def _cost(ranks: tuple[int, ...]) -> dict:
    projected_cut_bytes = 4 * _ROWS * (_LAYERS * _FRAME_BYTES + sum(ranks) * _WORD_BYTES)
    return {
        "total_rank": sum(ranks),
        "projected_two_worker_cut_body_bytes": projected_cut_bytes,
        "round_trips_if_sequential": _ROWS * _LAYERS,
        "covered_prepared_body_control_bytes": 178_970_558,
        "tenfold_covered_body_limit_bytes": 17_897_055,
        "tenfold_online_body_limit_bytes": 11_354_502,
    }


def probe(prompts_path: Path = _PROMPTS, *, confirmation_path: Path | None = None) -> dict:
    import torch
    import transformers
    from huggingface_hub import hf_hub_download

    torch.set_num_threads(4)
    cache = os.environ.get("HF_HUB_CACHE") or str(
        Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"
    )
    config_path = Path(
        hf_hub_download(
            _MODEL_ID,
            "config.json",
            revision=_REVISION,
            local_files_only=True,
            cache_dir=cache,
        )
    )
    checkpoint = config_path.parent
    prompts = json.loads(prompts_path.read_text(encoding="utf-8"))
    if confirmation_path is not None:
        prompts["heldout"] = json.loads(confirmation_path.read_text(encoding="utf-8"))
    if (
        set(prompts) != {"calibration", "tuning", "heldout"}
        or len(prompts["calibration"]) != 20
        or len(prompts["tuning"]) != 6
        or len(prompts["heldout"]) != 12
        or any(not isinstance(text, str) or not text for rows in prompts.values() for text in rows)
        or len(set(sum(prompts.values(), []))) != 38
    ):
        raise ValueError("subspace probe requires disjoint, bounded public cohorts")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        checkpoint,
        local_files_only=True,
        trust_remote_code=False,
    )
    cohorts = {
        group: [_token_ids(tokenizer, prompt, 39) for prompt in rows]
        for group, rows in prompts.items()
    }
    if len({tuple(ids) for rows in cohorts.values() for ids in rows}) != 38:
        raise ValueError("calibration, tuning and held-out token sequences must be distinct")
    model = transformers.AutoModelForCausalLM.from_pretrained(
        checkpoint,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.float32,
        attn_implementation="eager",
    ).eval()
    if len(model.model.layers) != _LAYERS or model.config.hidden_size != _HIDDEN:
        raise ValueError("frozen checkpoint no longer matches the declared decoder")

    with _collect_outputs(model) as (_, outputs):
        for ids in cohorts["calibration"]:
            _logits_and_cache(model, ids)
    calibration = [np.concatenate(rows) for rows in outputs]
    if any(not 128 < rows.shape[0] <= 512 for rows in calibration):
        raise ValueError("public calibration rows exceed the fitted-subspace bound")
    fitted = [fit_output_rank_basis(rows, 128) for rows in calibration]

    def local(rank: int, index: int) -> OutputRankBasis:
        original = fitted[index]
        return OutputRankBasis(
            original.mean, original.directions[:, :rank], original.calibration_rows
        )

    # Pluralis-style shared representation: single set of directions for every
    # layer, but each layer retains its own public mean. No checkpoint training.
    covariance = np.zeros((_HIDDEN, _HIDDEN), dtype=np.float64)
    normalized_covariance = np.zeros_like(covariance)
    for rows in calibration:
        centered = rows.astype(np.float64) - rows.mean(axis=0, dtype=np.float64)
        gram = centered.T @ centered
        covariance += gram
        normalized_covariance += gram / np.trace(gram)
    eigvals, eigenvectors = np.linalg.eigh(covariance)
    directions = np.ascontiguousarray(eigenvectors[:, -128:], dtype=np.float32)
    directions.setflags(write=False)
    normalized_eigvals, normalized_vectors = np.linalg.eigh(normalized_covariance)
    balanced_directions = np.ascontiguousarray(normalized_vectors[:, -128:], dtype=np.float32)
    balanced_directions.setflags(write=False)
    shared = {
        index: OutputRankBasis(fitted[index].mean, directions, fitted[index].calibration_rows)
        for index in range(_LAYERS)
    }
    balanced_shared = {
        index: OutputRankBasis(
            fitted[index].mean, balanced_directions, fitted[index].calibration_rows
        )
        for index in range(_LAYERS)
    }
    pooled_retained_energy = float(eigvals[-128:].sum() / eigvals.clip(min=0).sum())
    balanced_retained_energy = float(
        normalized_eigvals[-128:].sum() / normalized_eigvals.clip(min=0).sum()
    )

    def layer_energies(outputs: list[np.ndarray], bases: dict[int, OutputRankBasis]) -> list[float]:
        values = []
        for index, rows in enumerate(outputs):
            centered = rows.astype(np.float64) - fitted[index].mean
            projected = centered @ bases[index].directions.astype(np.float64)
            values.append(float(np.sum(projected**2) / np.sum(centered**2)))
        return values

    uniform128_bases = {index: local(128, index) for index in range(_LAYERS)}
    calibration_energies = {
        "per_layer128": layer_energies(calibration, uniform128_bases),
        "shared128": layer_energies(calibration, shared),
        "balanced_shared128": layer_energies(calibration, balanced_shared),
    }

    # References and rankings may inspect tuning cohort; held-out cohort remains
    # unread until all rank allocations and shared bases have been fixed.
    tuning_reference = []
    for ids in cohorts["tuning"]:
        logits, cache_state = _logits_and_cache(model, ids)
        first = int(np.argmax(logits))
        next_logits, _ = _logits_and_cache(model, [first], cache=cache_state)
        tuning_reference.append((ids, first, logits, next_logits))
    sensitivities = []
    for index in range(_LAYERS):
        js = []
        with _project(model, {index: local(_LOW, index)}):
            for ids, first, reference_prefill, reference_decode in tuning_reference:
                actual_prefill, cache_state = _logits_and_cache(model, ids)
                actual_decode, _ = _logits_and_cache(model, [first], cache=cache_state)
                js.extend(
                    (_js(actual_prefill, reference_prefill), _js(actual_decode, reference_decode))
                )
        sensitivities.append(sum(js) / len(js))
    ascending = sorted(range(_LAYERS), key=lambda index: (sensitivities[index], index))

    def allocated(order: list[int]) -> tuple[int, ...]:
        ranks = [_BASE] * _LAYERS
        for index in order[:8]:
            ranks[index] = _LOW
        for index in order[-8:]:
            ranks[index] = _HIGH
        return tuple(ranks)

    schedules = {
        "uniform64": (_BASE,) * _LAYERS,
        "sensitivity32_64_96": allocated(ascending),
        "reverse32_64_96": allocated(list(reversed(ascending))),
        "uniform128": (128,) * _LAYERS,
        "shared128": (128,) * _LAYERS,
        "balanced_shared128": (128,) * _LAYERS,
    }
    for count in (2, 4):
        for label, selection in (("top", ascending[-count:]), ("bottom", ascending[:count])):
            ranks = [_BASE] * _LAYERS
            for index in selection:
                ranks[index] = _HIDDEN
            schedules[f"full{count}_{label}64"] = tuple(ranks)
    if len(set(schedules["sensitivity32_64_96"])) != 3 or (
        sum(schedules["uniform64"]) != sum(schedules["sensitivity32_64_96"])
    ):
        raise ValueError("selective and uniform profiles must have equal total rank")

    results = {
        key: {"prefill": [], "decode_same_token": [], "free_second_token_matches": 0}
        for key in schedules
    }
    heldout_outputs: list[list[np.ndarray]] = [[] for _ in range(_LAYERS)]
    for ids in cohorts["heldout"]:
        with _collect_outputs(model) as (_, reference_outputs):
            ref_prefill, ref_cache = _logits_and_cache(model, ids)
            selected = int(np.argmax(ref_prefill))
            ref_decode, _ = _logits_and_cache(model, [selected], cache=ref_cache)
        for index, group in enumerate(reference_outputs):
            heldout_outputs[index].extend(group)
        for name, ranks in schedules.items():
            basis = (
                shared
                if name == "shared128"
                else balanced_shared
                if name == "balanced_shared128"
                else {
                    index: local(rank, index) for index, rank in enumerate(ranks) if rank != _HIDDEN
                }
            )
            with _project(model, basis):
                candidate_prefill, candidate_cache = _logits_and_cache(model, ids)
                candidate_selected = int(np.argmax(candidate_prefill))
                candidate_decode, _ = _logits_and_cache(model, [selected], cache=candidate_cache)
                if candidate_selected == selected:
                    free_logits = candidate_decode
                else:
                    _, free_cache = _logits_and_cache(model, ids)
                    free_logits, _ = _logits_and_cache(
                        model, [candidate_selected], cache=free_cache
                    )
            for phase, actual, reference in (
                ("prefill", candidate_prefill, ref_prefill),
                ("decode_same_token", candidate_decode, ref_decode),
            ):
                observed = _observe(actual, reference)
                observed["js_divergence"] = _js(actual, reference)
                results[name][phase].append(observed)
            results[name]["free_second_token_matches"] += int(
                np.argmax(free_logits) == np.argmax(ref_decode)
            )

    heldout_rows = [np.concatenate(group) for group in heldout_outputs]
    heldout_energies = {
        "per_layer128": layer_energies(heldout_rows, uniform128_bases),
        "shared128": layer_energies(heldout_rows, shared),
        "balanced_shared128": layer_energies(heldout_rows, balanced_shared),
    }
    summaries = {}
    for name, values in results.items():
        ranks = schedules[name]
        summaries[name] = {
            "ranks_by_layer": list(ranks),
            "projected_cost": _cost(ranks),
            "prefill": _summarize(values["prefill"]),
            "decode_same_token": _summarize(values["decode_same_token"]),
            "free_second_token_matches": values["free_second_token_matches"],
            "total_heldout_prompts": len(cohorts["heldout"]),
        }
    return {
        "schema": "pllm.selective_subspace_quality_gate.v1",
        "source": {
            "model": _MODEL_ID,
            "revision": _REVISION,
            "config_sha256": _sha256(config_path),
            "weights_sha256": _sha256(checkpoint / "model.safetensors"),
            "prompts_sha256": _sha256(prompts_path),
            "confirmation_prompts_sha256": (
                _sha256(confirmation_path) if confirmation_path is not None else None
            ),
            "token_cohort_digest": hashlib.sha256(
                json.dumps(cohorts, separators=(",", ":"), sort_keys=True).encode(),
            ).hexdigest(),
        },
        "calibration_rows_per_layer": calibration[0].shape[0],
        "tuning_input_token_counts": list(map(len, cohorts["tuning"])),
        "heldout_input_token_counts": list(map(len, cohorts["heldout"])),
        "sensitivity_rank": _LOW,
        "layer_sensitivities_js": sensitivities,
        "ascending_sensitivity_layers": ascending,
        "shared128_calibration_centered_output_energy_fraction": pooled_retained_energy,
        "balanced_shared128_calibration_centered_output_energy_fraction": balanced_retained_energy,
        "calibration_output_energy_by_layer": calibration_energies,
        "heldout_reference_output_energy_by_layer": heldout_energies,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "threads": torch.get_num_threads(),
        },
        "results": summaries,
        "limits": [
            "Sensitivity order is fitted on public tuning only; no held-out token IDs or logits determine any rank or basis.",
            "Every original Qwen MLP computes its full nonlinear output before the rank hook. No lower-cost protected Qwen execution, client feedback, field rescaling or independent worker is implemented.",
            "Projected cut bytes assume four rank-sized typed bodies per layer and row in the separate 39+32-token control shape; quality prompts have shorter prefills and one decode, so no matched cost/quality cohort exists. The sketch omits original MLP computation traffic, attention, embedding/head, material, full wire, cold distribution and dependent round-trip latency.",
            "One-step same-token and free-feedback checks on twelve new public prompts are narrow numeric diagnostics, not generation-quality or private-inference evidence.",
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirmation", type=Path, help="Independent public held-out prompt file")
    arguments = parser.parse_args()
    print(json.dumps(probe(confirmation_path=arguments.confirmation), indent=2, sort_keys=True))
