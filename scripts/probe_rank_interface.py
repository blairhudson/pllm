"""Research-only optimistic low-rank output gate against a locked local checkpoint.

Full MLPs are still evaluated before their outputs are projected. Passing this
gate cannot establish a protected low-rank implementation or a trained model.
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

from pllm.runtime.rank_interface_reference import (
    AffineResidualBasis,
    fit_affine_residual_basis,
    fit_output_rank_basis,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
_DEFAULT_PROMPTS = Path(__file__).resolve().parents[1] / "examples/benchmarks/rank_interface_prompts.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@contextmanager
def _collect_outputs(model: object) -> Iterator[tuple[list[list[np.ndarray]], list[list[np.ndarray]]]]:
    inputs: list[list[np.ndarray]] = [[] for _ in model.model.layers]
    outputs: list[list[np.ndarray]] = [[] for _ in model.model.layers]
    handles = []
    try:
        for index, layer in enumerate(model.model.layers):
            def capture(_module: object, values: object, output: object, *, index: int = index) -> None:
                inputs[index].append(values[0].detach().reshape(-1, output.shape[-1]).float().cpu().numpy().copy())
                outputs[index].append(output.detach().reshape(-1, output.shape[-1]).float().cpu().numpy().copy())

            handles.append(layer.mlp.register_forward_hook(capture))
        yield inputs, outputs
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def _project_outputs(model: object, bases: list[object], cut_stride: int) -> Iterator[None]:
    import torch

    handles = []
    try:
        for index, (layer, basis) in enumerate(zip(model.model.layers, bases, strict=True)):
            if (index + 1) % cut_stride:
                continue
            if isinstance(basis, AffineResidualBasis):
                input_mean = torch.tensor(basis.input_mean, dtype=torch.float32)
                output_mean = torch.tensor(basis.output_mean, dtype=torch.float32)
                public_map = torch.tensor(basis.public_map, dtype=torch.float32)
                mean = torch.tensor(basis.residual.mean, dtype=torch.float32)
                directions = torch.tensor(basis.residual.directions, dtype=torch.float32)

                def project(
                    _module: object, values: object, output: object,
                    *, input_mean=input_mean, output_mean=output_mean,
                    public_map=public_map, mean=mean, directions=directions,
                ):
                    bypass = output_mean + (values[0] - input_mean) @ public_map
                    remainder = output - bypass
                    return bypass + mean + ((remainder - mean) @ directions) @ directions.T

            else:
                mean = torch.tensor(basis.mean, dtype=torch.float32)
                directions = torch.tensor(basis.directions, dtype=torch.float32)

                def project(_module: object, _inputs: object, output: object, *, mean=mean, directions=directions):
                    return mean + ((output - mean) @ directions) @ directions.T

            handles.append(layer.mlp.register_forward_hook(project))
        yield
    finally:
        for handle in handles:
            handle.remove()


def _token_ids(tokenizer: object, prompt: str, limit: int) -> list[int]:
    values = tokenizer.encode(prompt, add_special_tokens=False)
    if not 0 < len(values) <= limit:
        raise ValueError("public prompt length exceeds the locked rank-probe bound")
    return values


def _logits_and_cache(model: object, ids: list[int], *, cache: object = None) -> tuple[np.ndarray, object]:
    import torch

    with torch.inference_mode():
        input_ids = torch.tensor([ids], dtype=torch.long)
        if cache is None:
            result = model(input_ids=input_ids, use_cache=True)
        else:
            result = model(input_ids=input_ids, past_key_values=cache, use_cache=True)
        return result.logits[0, -1].float().cpu().numpy().copy(), result.past_key_values


def _observe(actual: np.ndarray, reference: np.ndarray) -> dict[str, float | bool]:
    reference_top = int(np.argmax(reference))
    actual_top = int(np.argmax(actual))
    error = float(np.max(np.abs(actual - reference)))
    runner_up = float(np.partition(actual, -2)[-2])
    return {
        "top1_match": actual_top == reference_top,
        "top5_recall": len(set(np.argpartition(actual, -5)[-5:]) & set(np.argpartition(reference, -5)[-5:])) / 5,
        "worst_logit_error": error,
        # This is retrospective: measuring error with the full reference does
        # not supply an admissible error bound for an unobserved prompt.
        "retrospective_margin_witness": float(actual[actual_top]) - runner_up > 2 * error,
    }


def probe(
    *, prompts_path: Path, ranks: tuple[int, ...],
    cut_strides: tuple[int, ...] = (1, 2, 4), max_input_tokens: int = 39,
    mode: str = "output",
) -> dict:
    import torch
    import transformers
    from huggingface_hub import hf_hub_download

    torch.set_num_threads(4)
    cache = os.environ.get("HF_HUB_CACHE") or str(
        Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"
    )
    config_path = Path(hf_hub_download(
        _MODEL_ID, "config.json", revision=_REVISION, local_files_only=True,
        cache_dir=cache,
    ))
    checkpoint = config_path.parent
    source = json.loads(prompts_path.read_text(encoding="utf-8"))
    if (
        set(source) != {"calibration", "evaluation"}
        or not all(isinstance(items, list) and items for items in source.values())
        or not all(type(prompt) is str and prompt for items in source.values() for prompt in items)
        or len(source["calibration"]) > 32 or len(source["evaluation"]) > 16
    ):
        raise ValueError("public rank-probe cohort is invalid or unbounded")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        checkpoint, local_files_only=True, trust_remote_code=False,
    )
    cohorts = {
        group: [_token_ids(tokenizer, prompt, max_input_tokens) for prompt in source[group]]
        for group in ("calibration", "evaluation")
    }
    if type(ranks) is not tuple or not ranks or len(set(ranks)) != len(ranks):
        raise ValueError("rank probe needs distinct explicit positive ranks")
    if mode not in ("output", "affine_residual"):
        raise ValueError("rank probe requires an explicit supported operator decomposition")
    if (
        type(cut_strides) is not tuple or not cut_strides
        or len(set(cut_strides)) != len(cut_strides)
        or any(type(stride) is not int or stride < 1 or 24 % stride for stride in cut_strides)
    ):
        raise ValueError("public cut stride must divide the pinned 24-layer plan")
    model = transformers.AutoModelForCausalLM.from_pretrained(
        checkpoint, local_files_only=True, trust_remote_code=False,
        dtype=torch.float32, attn_implementation="eager",
    ).eval()
    if len(model.model.layers) != 24 or model.config.hidden_size != 896:
        raise ValueError("pinned checkpoint no longer has the checked 24 × 896 architecture")

    with _collect_outputs(model) as (inputs, outputs):
        for ids in cohorts["calibration"]:
            _logits_and_cache(model, ids)
    calibration = [np.concatenate(rows) for rows in outputs]
    calibration_inputs = [np.concatenate(rows) for rows in inputs]
    if any(rows.shape[0] > 512 for rows in calibration):
        raise ValueError("rank calibration exceeded the 512-row-per-layer bound")
    bases = {
        rank: [
            fit_output_rank_basis(y, rank)
            if mode == "output" else fit_affine_residual_basis(x, y, rank)
            for x, y in zip(calibration_inputs, calibration, strict=True)
        ] for rank in ranks
    }
    totals = {
        (rank, stride): {"prefill": [], "decode_same_token": [], "decode_free_token": [], "residual_energy": []}
        for rank in ranks for stride in cut_strides
    }
    heldout_inputs: list[list[np.ndarray]] = [[] for _ in model.model.layers]
    heldout_outputs: list[list[np.ndarray]] = [[] for _ in model.model.layers]
    for ids in cohorts["evaluation"]:
        with _collect_outputs(model) as (reference_inputs, reference_outputs):
            reference_prefill, cache = _logits_and_cache(model, ids)
            selected = int(np.argmax(reference_prefill))
            reference_decode, _ = _logits_and_cache(model, [selected], cache=cache)
        for layer in range(len(heldout_outputs)):
            heldout_inputs[layer].extend(reference_inputs[layer])
            heldout_outputs[layer].extend(reference_outputs[layer])
        for rank, stride in totals:
            with _project_outputs(model, bases[rank], stride):
                actual_prefill, candidate_cache = _logits_and_cache(model, ids)
                actual_selected = int(np.argmax(actual_prefill))
                # Same-token parity separates numeric drift from feedback.
                actual_decode, _ = _logits_and_cache(model, [selected], cache=candidate_cache)
                if actual_selected == selected:
                    free_decode = actual_decode
                else:
                    _, free_cache = _logits_and_cache(model, ids)
                    free_decode, _ = _logits_and_cache(model, [actual_selected], cache=free_cache)
            totals[(rank, stride)]["prefill"].append(_observe(actual_prefill, reference_prefill))
            totals[(rank, stride)]["decode_same_token"].append(_observe(actual_decode, reference_decode))
            totals[(rank, stride)]["decode_free_token"].append(
                int(np.argmax(free_decode)) == int(np.argmax(reference_decode))
            )
    for rank, stride in totals:
        for layer, outputs in enumerate(heldout_outputs):
            if (layer + 1) % stride:
                continue
            full = np.concatenate(outputs)
            basis = bases[rank][layer]
            reduced = (
                basis.project(np.concatenate(heldout_inputs[layer]), full)
                if isinstance(basis, AffineResidualBasis) else basis.project(full)
            )
            totals[(rank, stride)]["residual_energy"].append(
                float(np.linalg.norm(full - reduced) / max(1e-12, np.linalg.norm(full)))
            )

    summary = {}
    for (rank, stride), result in totals.items():
        name = f"rank{rank}_cuts{24 // stride}"
        summary[name] = {
            phase: {
                "top1_matches": sum(row["top1_match"] for row in result[phase]),
                "total": len(result[phase]),
                "mean_top5_recall": sum(row["top5_recall"] for row in result[phase]) / len(result[phase]),
                "worst_logit_error": max(row["worst_logit_error"] for row in result[phase]),
                "retrospective_margin_witnesses": sum(row["retrospective_margin_witness"] for row in result[phase]),
            }
            for phase in ("prefill", "decode_same_token")
        }
        summary[name]["free_second_token_matches"] = sum(result["decode_free_token"])
        summary[name]["maximum_layer_eval_trajectory_relative_residual"] = max(result["residual_energy"])

    return {
        "schema": f"pllm.rank_interface_{mode}_oracle.v1",
        "scope": (
            "full-reference-MLP-output-projected-after-nonlinearity; no protected execution or actual low-rank MLP"
            if mode == "output" else
            "ridge-fit public affine bypass plus full-reference-computed nonlinear residual projected after evaluation; no protected execution"
        ),
        "model": _MODEL_ID, "revision": _REVISION,
        "config_sha256": _sha256(config_path),
        "weights_sha256": _sha256(checkpoint / "model.safetensors"),
        "prompts_sha256": _sha256(prompts_path),
        "calibration_rows_per_layer": calibration[0].shape[0],
        "calibration_input_token_counts": [len(ids) for ids in cohorts["calibration"]],
        "evaluation_input_token_counts": [len(ids) for ids in cohorts["evaluation"]],
        "token_cohort_digest": hashlib.sha256(
            json.dumps(cohorts, separators=(",", ":"), sort_keys=True).encode()
        ).hexdigest(),
        "environment": {
            "python": platform.python_version(), "torch": torch.__version__,
            "transformers": transformers.__version__, "reference_dtype": "float32",
            "threads": torch.get_num_threads(),
        },
        "results": summary,
        "limitations": [
            "Full proprietary nonlinear outputs are calculated before every rank projection; this is an optimistic oracle, not a rank-width client cut or a trainable factorization.",
            "Sparse-cut variants retain all unprojected full-precision MLPs; they do not model a decoder trained with only six or twelve nonlinear cuts.",
            "Residual measurements use held-out reference trajectories, not calibration rows.",
            "Affine bypass weights are fitted only from public calibration inputs/outputs; their cold distribution and execution bytes are not measured.",
            "Observed errors on reference queries are not a priori error bounds or live token-decision certificates.",
            "No two-worker execution, wire accounting, cold model cost, or representative generation quality is established.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts-file", type=Path, default=_DEFAULT_PROMPTS)
    parser.add_argument("--ranks", type=int, nargs="+", default=[16, 32])
    parser.add_argument("--cut-strides", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--max-input-tokens", type=int, default=39)
    parser.add_argument("--mode", choices=("output", "affine_residual"), default="output")
    args = parser.parse_args()
    report = probe(
        prompts_path=args.prompts_file, ranks=tuple(args.ranks),
        cut_strides=tuple(args.cut_strides),
        max_input_tokens=args.max_input_tokens,
        mode=args.mode,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
