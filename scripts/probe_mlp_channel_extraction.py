"""No-training extraction gate for pinned public gated-MLP weights.

The client-side test replaces selected original MLP operators with a public
affine plus exact pretrained channels. All other operators execute normally.
This is a bounded *numeric* diagnostic, not protected multi-party execution.
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

from pllm.runtime.mlp_channel_extraction import (
    calibrate_gated_mlp_channels,
    projected_channel_cut_bodies,
)
from probe_rank_interface import (
    _DEFAULT_PROMPTS,
    _MODEL_ID,
    _REVISION,
    _logits_and_cache,
    _observe,
    _sha256,
    _token_ids,
)

if TYPE_CHECKING:
    from collections.abc import Iterator


@contextmanager
def _extract_forward(model: object, extracted: dict[int, object]) -> Iterator[None]:
    import torch

    original = {}
    try:
        for index, artifact in extracted.items():
            module = model.model.layers[index].mlp
            original[index] = module.forward
            affine = torch.tensor(artifact.affine, dtype=torch.float32)
            offset = torch.tensor(artifact.offset, dtype=torch.float32)
            gate = torch.tensor(artifact.gate, dtype=torch.float32)
            up = torch.tensor(artifact.up, dtype=torch.float32)
            down = torch.tensor(artifact.down, dtype=torch.float32)
            gate_slope = torch.tensor(artifact.gate_slope, dtype=torch.float32)
            up_slope = torch.tensor(artifact.up_slope, dtype=torch.float32)
            intercept = torch.tensor(artifact.intercept, dtype=torch.float32)

            def forward(
                x: object, *, affine=affine, offset=offset, gate=gate,
                up=up, down=down, gate_slope=gate_slope,
                up_slope=up_slope, intercept=intercept,
            ):
                linear = x @ affine.T + offset
                if not gate.shape[0]:
                    return linear
                gate_values = x @ gate.T
                up_values = x @ up.T
                residual = torch.nn.functional.silu(gate_values) * up_values - (
                    gate_values * gate_slope + up_values * up_slope + intercept
                )
                return linear + residual @ down.T

            module.forward = forward
        yield
    finally:
        for index, forward in original.items():
            model.model.layers[index].mlp.forward = forward


def probe(
    *, prompt_path: Path, layers: tuple[int, ...], widths: tuple[int, ...],
    affine_method: str = "taylor", calibration_phase: str = "all_prefill",
) -> dict:
    import torch
    import transformers
    from huggingface_hub import hf_hub_download

    if (
        type(layers) is not tuple or not 1 <= len(layers) <= 4
        or any(type(index) is not int or not 0 <= index < 24 for index in layers)
        or len(layers) != len(set(layers)) or type(widths) is not tuple
        or not 1 <= len(widths) <= 8 or len(widths) != len(set(widths))
        or any(type(width) is not int or not 0 <= width <= 8192 for width in widths)
        or affine_method not in ("taylor", "least_squares")
        or calibration_phase not in ("all_prefill", "all_prefill_and_decode", "decision_and_decode")
    ):
        raise ValueError("public channel extraction needs at most four unique semantic layer slots and eight widths")
    torch.set_num_threads(4)
    cache = os.environ.get("HF_HUB_CACHE") or str(
        Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"
    )
    config_path = Path(hf_hub_download(
        _MODEL_ID, "config.json", revision=_REVISION, local_files_only=True, cache_dir=cache,
    ))
    checkpoint = config_path.parent
    prompts = json.loads(prompt_path.read_text(encoding="utf-8"))
    if (
        set(prompts) != {"calibration", "evaluation"}
        or not all(isinstance(items, list) and items for items in prompts.values())
        or len(prompts["calibration"]) > 32 or len(prompts["evaluation"]) > 16
        or not all(type(p) is str and p for items in prompts.values() for p in items)
    ):
        raise ValueError("public calibration or evaluation text exceeds the bounded cohort")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        checkpoint, local_files_only=True, trust_remote_code=False,
    )
    cohorts = {
        group: [_token_ids(tokenizer, prompt, 39) for prompt in prompts[group]]
        for group in ("calibration", "evaluation")
    }
    model = transformers.AutoModelForCausalLM.from_pretrained(
        checkpoint, local_files_only=True, trust_remote_code=False,
        dtype=torch.float32, attn_implementation="eager",
    ).eval()
    if (
        len(model.model.layers) != 24 or model.config.hidden_size != 896
        or model.config.intermediate_size != 4864 or model.config.hidden_act != "silu"
        or any(
            layer.mlp.gate_proj.bias is not None or layer.mlp.up_proj.bias is not None
            or layer.mlp.down_proj.bias is not None for layer in model.model.layers
        )
    ):
        raise ValueError("pinned source no longer matches the bounded bias-free SiLU graph")
    if any(width > 4864 for width in widths):
        raise ValueError("requested extraction exceeds pretrained MLP channel width")

    collected: dict[int, list[np.ndarray]] = {index: [] for index in layers}
    phase = ["prefill"]
    handles = []
    try:
        for index in layers:
            def collect(_module: object, values: object, _output: object, *, index: int = index) -> None:
                rows = values[0].detach().reshape(-1, 896).float().cpu().numpy()
                if calibration_phase == "decision_and_decode" and phase[0] == "prefill":
                    rows = rows[-1:]
                collected[index].append(rows.copy())

            handles.append(model.model.layers[index].mlp.register_forward_hook(collect))
        for ids in cohorts["calibration"]:
            logits, cache_state = _logits_and_cache(model, ids)
            if calibration_phase != "all_prefill":
                phase[0] = "decode"
                _logits_and_cache(model, [int(np.argmax(logits))], cache=cache_state)
                phase[0] = "prefill"
    finally:
        for handle in handles:
            handle.remove()
    calibrated = {}
    for index in layers:
        if not collected[index]:
            raise ValueError("public calibration did not execute a declared MLP")
        inputs = np.concatenate(collected[index])
        mlp = model.model.layers[index].mlp
        calibrated[index] = calibrate_gated_mlp_channels(
            gate_weight=mlp.gate_proj.weight.detach().cpu().numpy(),
            up_weight=mlp.up_proj.weight.detach().cpu().numpy(),
            down_weight=mlp.down_proj.weight.detach().cpu().numpy(),
            calibration_inputs=inputs,
            affine_method=affine_method,
        )

    # Use independent held-out hidden rows to check algebraic parity at r=M.
    checked: dict[int, list[np.ndarray]] = {index: [] for index in layers}
    hooks = []
    try:
        for index in layers:
            def collect(_module: object, values: object, _output: object, *, index: int = index) -> None:
                checked[index].append(values[0].detach().reshape(-1, 896).float().cpu().numpy().copy())

            hooks.append(model.model.layers[index].mlp.register_forward_hook(collect))
        _logits_and_cache(model, cohorts["evaluation"][0])
    finally:
        for hook in hooks:
            hook.remove()
    independent_max_error = {}
    for index in layers:
        inputs = torch.from_numpy(np.concatenate(checked[index]))
        original = model.model.layers[index].mlp(inputs).detach()
        with _extract_forward(model, {index: calibrated[index].select(4864)}):
            candidate = model.model.layers[index].mlp(inputs).detach()
        independent_max_error[str(index)] = float(torch.max(torch.abs(original - candidate)))
        if independent_max_error[str(index)] >= 0.05:
            raise ValueError("all-channel extraction changes the pinned float32 operator")

    references = []
    for ids in cohorts["evaluation"]:
        reference_prefill, cache = _logits_and_cache(model, ids)
        selected = int(np.argmax(reference_prefill))
        reference_decode, _ = _logits_and_cache(model, [selected], cache=cache)
        references.append((ids, selected, reference_prefill, reference_decode))

    results = {}
    for width in widths:
        extracted = {index: calibrated[index].select(width) for index in layers}
        prefill = []
        decode = []
        free = []
        with _extract_forward(model, extracted):
            for ids, selected, reference_prefill, reference_decode in references:
                candidate_prefill, candidate_cache = _logits_and_cache(model, ids)
                candidate_token = int(np.argmax(candidate_prefill))
                candidate_decode, _ = _logits_and_cache(model, [selected], cache=candidate_cache)
                if candidate_token == selected:
                    free_logits = candidate_decode
                else:
                    _, free_cache = _logits_and_cache(model, ids)
                    free_logits, _ = _logits_and_cache(model, [candidate_token], cache=free_cache)
                prefill.append(_observe(candidate_prefill, reference_prefill))
                decode.append(_observe(candidate_decode, reference_decode))
                free.append(int(np.argmax(free_logits)) == int(np.argmax(reference_decode)))
        def summarize(values: list[dict]) -> dict:
            return {"top1_matches": sum(entry["top1_match"] for entry in values),
                    "mean_top5_recall": sum(entry["top5_recall"] for entry in values) / len(values),
                    "worst_logit_error": max(entry["worst_logit_error"] for entry in values),
                    "retrospective_margin_witnesses": sum(entry["retrospective_margin_witness"] for entry in values)}

        results[str(width)] = {
            "prefill": summarize(prefill), "decode_same_token": summarize(decode),
            "free_second_token_matches": sum(free), "total_prompts": len(prefill),
            "public_extracted_profile_digests": {str(index): value.digest for index, value in extracted.items()},
            "cost_if_every_layer_used_this_width": {
                "rows": 70, "word_bytes": 8,
                **projected_channel_cut_bodies(layers=24, rows=70, channels_per_layer=width, word_bytes=8),
            },
        }

    return {
        "schema": "pllm.extracted_gated_mlp_probe.v1",
        "affine_method": affine_method,
        "calibration_phase": calibration_phase,
        "source": {"model": _MODEL_ID, "revision": _REVISION,
                   "config_sha256": _sha256(config_path),
                   "weights_sha256": _sha256(checkpoint / "model.safetensors"),
                   "prompts_sha256": _sha256(prompt_path),
                   "token_cohort_digest": hashlib.sha256(
                       json.dumps(cohorts, separators=(",", ":"), sort_keys=True).encode(),
                   ).hexdigest()},
        "calibration_rows": sum(row.shape[0] for row in collected[layers[0]]),
        "evaluated_layers": list(layers), "evaluation_token_counts": list(map(len, cohorts["evaluation"])),
        "full_channel_reconstruction_max_error": independent_max_error,
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "transformers": transformers.__version__, "threads": torch.get_num_threads()},
        "results": results,
        "limits": [
            "Only selected pretrained channels are evaluated exactly; omitted channels use public offline affine coefficients without changing model weights or using gradient training.",
            "Evaluated layers are fewer than the full decoder and every other original Qwen operator remains in its original placement.",
            "The extracted MLP executes locally in float32. Protected worker shares, ring rescale, logits and token feedback are not implemented.",
            "Per-layer-cut bytes are a two-worker typed-body projection, not a full response, full-wire or CPU measurement.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts-file", type=Path, default=_DEFAULT_PROMPTS)
    parser.add_argument("--layers", type=int, nargs="+", default=[23])
    parser.add_argument("--widths", type=int, nargs="+", default=[0, 64, 128, 256, 512, 1024, 2048, 4864])
    parser.add_argument("--affine-method", choices=("taylor", "least_squares"), default="taylor")
    parser.add_argument("--calibration-phase", choices=(
        "all_prefill", "all_prefill_and_decode", "decision_and_decode",
    ),
                        default="all_prefill")
    args = parser.parse_args()
    print(json.dumps(probe(
        prompt_path=args.prompts_file, layers=tuple(args.layers), widths=tuple(args.widths),
        affine_method=args.affine_method, calibration_phase=args.calibration_phase,
    ), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
