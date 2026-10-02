"""Bounded pinned-checkpoint float oracle for polynomial correlation hypotheses.

Eight public calibration/evaluation prompts, two token decisions, four fixed
candidates. No weight training, downloads or protected decoder execution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from pllm.runtime.polynomial_numeric_reference import (
    evaluate_channel_polynomial,
    fit_channel_polynomial,
)
from probe_rank_interface import (
    _MODEL_ID,
    _REVISION,
    _logits_and_cache,
    _observe,
    _sha256,
    _token_ids,
)

_ROOT = Path(__file__).resolve().parents[1]
_PROMPTS = _ROOT / "examples/benchmarks/projected_polynomial_prompts.json"


@contextmanager
def _replace_activations(model, profiles):
    import torch

    class Polynomial(torch.nn.Module):
        def __init__(self, profile):
            super().__init__()
            self.profile = profile

        def forward(self, gate):
            return evaluate_channel_polynomial(gate, self.profile)

    originals = [(layer.mlp, layer.mlp.act_fn) for layer in model.model.layers]
    try:
        for index, (mlp, _) in enumerate(originals):
            mlp.act_fn = Polynomial(None if profiles is None else profiles[index])
        yield
    finally:
        for mlp, original in originals:
            mlp.act_fn = original


def probe() -> dict:
    import torch
    import transformers
    from huggingface_hub import hf_hub_download

    torch.set_num_threads(4)
    cache = os.environ.get("HF_HUB_CACHE") or str(
        Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"
    )
    config_path = Path(
        hf_hub_download(
            _MODEL_ID, "config.json", revision=_REVISION, cache_dir=cache, local_files_only=True
        )
    )
    checkpoint = config_path.parent
    dataset = json.loads(_PROMPTS.read_text())
    if set(dataset) != {"calibration", "evaluation"} or any(len(v) != 8 for v in dataset.values()):
        raise ValueError("numeric gate requires the locked eight/eight public cohort")
    if set(dataset["calibration"]) & set(dataset["evaluation"]):
        raise ValueError("calibration and held-out prompts must be disjoint")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        checkpoint, local_files_only=True, trust_remote_code=False
    )
    cohorts = {
        key: [_token_ids(tokenizer, text, 39) for text in texts] for key, texts in dataset.items()
    }
    model = transformers.AutoModelForCausalLM.from_pretrained(
        checkpoint,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.float32,
        attn_implementation="eager",
    ).eval()
    if (
        len(model.model.layers) != 24
        or model.config.hidden_size != 896
        or model.config.intermediate_size != 4864
        or model.config.hidden_act != "silu"
    ):
        raise ValueError("pinned source differs from the polynomial gate")
    if any(
        getattr(layer.mlp, name).bias is not None
        for layer in model.model.layers
        for name in ("gate_proj", "up_proj", "down_proj")
    ):
        raise ValueError("polynomial gate requires the original bias-free weights")

    collected = {name: [[] for _ in model.model.layers] for name in ("gate", "up")}
    counts = {name: [0] * len(model.model.layers) for name in collected}
    handles = []
    try:
        for index, layer in enumerate(model.model.layers):
            for name in collected:

                def capture(_module, _values, output, *, index=index, name=name):
                    rows = (
                        output.detach()
                        .reshape(-1, 4864)[: 256 - counts[name][index]]
                        .float()
                        .clone()
                    )
                    collected[name][index].append(rows)
                    counts[name][index] += len(rows)

                handles.append(getattr(layer.mlp, name + "_proj").register_forward_hook(capture))
        for ids in cohorts["calibration"]:
            _logits_and_cache(model, ids)
    finally:
        for handle in handles:
            handle.remove()
    profiles = {f"public_degree_{degree}": [] for degree in (1, 2, 4)}
    for index in range(24):
        g = torch.cat(collected["gate"][index])
        u = torch.cat(collected["up"][index])
        for degree in (1, 2, 4):
            profiles[f"public_degree_{degree}"].append(fit_channel_polynomial(g, u, degree=degree))
    collected.clear()
    candidates = {"fixed_taylor_degree_2": None, **profiles}

    # Optimistic sparse-correction gate on unmodified reference trajectories.
    # The true per-row dynamic scale is supplied for free; it is not a protocol.
    residuals = {
        name: {
            "elements": 0,
            "nonzero_code_corrections": 0,
            "rows": 0,
            "maximum_active_fraction": 0.0,
            "maximum_active_count": 0,
            "rows_at_most_one_percent_active": 0,
        }
        for name in candidates
    }
    pending = {}
    handles = []
    try:
        for index, layer in enumerate(model.model.layers):

            def capture_g(_module, _args, output, *, index=index):
                pending[index] = output.detach()

            def inspect_u(_module, _args, output, *, index=index):
                g = pending.pop(index)
                u = output.detach()
                target = torch.nn.functional.silu(g) * u
                scale = target.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / 127
                encoded = torch.round(target / scale).clamp(-127, 127)
                for name, values in candidates.items():
                    fitted = (
                        evaluate_channel_polynomial(g, None if values is None else values[index])
                        * u
                    )
                    other = torch.round(fitted / scale).clamp(-127, 127)
                    active = encoded != other
                    fractions = active.float().mean(dim=-1)
                    row = residuals[name]
                    row["elements"] += active.numel()
                    row["nonzero_code_corrections"] += int(active.sum())
                    row["rows"] += fractions.numel()
                    row["maximum_active_fraction"] = max(
                        row["maximum_active_fraction"], float(fractions.max())
                    )
                    row["maximum_active_count"] = max(
                        row["maximum_active_count"], int(active.sum(dim=-1).max())
                    )
                    row["rows_at_most_one_percent_active"] += int((fractions <= 0.01).sum())

            handles.extend(
                (
                    layer.mlp.gate_proj.register_forward_hook(capture_g),
                    layer.mlp.up_proj.register_forward_hook(inspect_u),
                )
            )
        references = []
        for ids in cohorts["evaluation"]:
            prefill, cache_state = _logits_and_cache(model, ids)
            selected = int(np.argmax(prefill))
            decode, _ = _logits_and_cache(model, [selected], cache=cache_state)
            references.append((ids, selected, prefill, decode))
    finally:
        for handle in handles:
            handle.remove()
    results = {}
    for name, values in candidates.items():
        observations = {"prefill": [], "decode_same_token": []}
        with _replace_activations(model, values):
            for ids, selected, reference_prefill, reference_decode in references:
                prefill, state = _logits_and_cache(model, ids)
                decode, _ = _logits_and_cache(model, [selected], cache=state)
                for phase, actual, expected in (
                    ("prefill", prefill, reference_prefill),
                    ("decode_same_token", decode, reference_decode),
                ):
                    if not np.isfinite(actual).all():
                        raise ValueError(f"nonfinite polynomial oracle output: {name}/{phase}")
                    observations[phase].append(_observe(actual, expected))
        row = residuals[name]
        row["mean_active_fraction"] = row["nonzero_code_corrections"] / row["elements"]
        results[name] = {
            **{
                phase: {
                    "top1_matches": sum(v["top1_match"] for v in data),
                    "total_prompts": len(data),
                    "mean_top5_recall": sum(v["top5_recall"] for v in data) / len(data),
                    "worst_logit_error": max(v["worst_logit_error"] for v in data),
                }
                for phase, data in observations.items()
            },
            "optimistic_quantized_residual": row,
            "profile_digests": None if values is None else [p["digest"] for p in values],
        }
        print(
            f"{name}: {results[name]['prefill']['top1_matches']}/8 prefill, "
            f"{results[name]['decode_same_token']['top1_matches']}/8 decode; "
            f"{row['mean_active_fraction']:.1%} codes need correction",
            flush=True,
        )
    return {
        "schema": "pllm.projected_polynomial_quality_gate.v1",
        "source": {
            "model": _MODEL_ID,
            "revision": _REVISION,
            "config_sha256": _sha256(config_path),
            "weights_sha256": _sha256(checkpoint / "model.safetensors"),
            "dataset_sha256": _sha256(_PROMPTS),
            "token_cohort_digest": hashlib.sha256(
                json.dumps(cohorts, sort_keys=True).encode()
            ).hexdigest(),
        },
        "code_sha256": {
            path: _sha256(_ROOT / path)
            for path in (
                "scripts/probe_projected_polynomial_quality.py",
                "python/pllm/runtime/polynomial_numeric_reference.py",
            )
        },
        "calibration_rows_per_layer": counts["gate"],
        "all_24_layers_replaced": True,
        "evaluation_token_counts": list(map(len, cohorts["evaluation"])),
        "results": results,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "threads": 4,
        },
        "protected_execution": False,
        "whole_generation_quality": False,
        "limitations": [
            "Optimistic float32 oracle; no ring rounding, numeric bridge or protected transport",
            "Per-channel public fits are more flexible than the fixed native numerator",
            "Eight independent public prompts, one teacher-forced decode; no broad quality claim",
            "Residual sparsity receives true activation scales for free and still hides no index positions",
            "Weights are unchanged; public offline least squares is not gradient training",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = probe()
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
