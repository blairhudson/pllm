"""No-training, real-checkpoint KV-selection quality and network-placement gate.

Apply a fixed, public attention-key subset to decode only. Keep the original
full KV tensors allocated so Torch can supply correct absolute positions; this
is an optimistic numeric oracle, not executable KV eviction or private gather.
"""

from __future__ import annotations

import argparse
import copy
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator
from unittest.mock import patch

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from transformers.models.qwen2 import modeling_qwen2

from pllm import Model, lower_model
from pllm.passes import KvCacheEviction
from pllm.profiles import MaskedLinearCpu
from pllm.quantization import SymmetricPerRow


_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
_PREFIX = "The same setup describes masked linear inference and immutable client state. " * 12
_SUFFIXES = ("Beta.", "Alpha.", "Gamma.")
_CONFIRMATION = (
    "Public maps allocate two independent online workers and audit token-boundary ownership. " * 10,
    "A compiler binds semantic attention state before protected inference and rejects missing coverage. "
    * 10,
    "The city library catalog indexes astronomy history with dates and careful citations. " * 11,
)
_OUTPUT_TOKENS = 8


def _static_indices(length: int, recent: int, older: int) -> list[int]:
    """Keep recent tokens and evenly spaced earlier tokens, independent of data."""
    if length <= recent + older:
        return list(range(length))
    earlier = length - recent
    historical = [position * (earlier - 1) // (older - 1) for position in range(older)]
    return sorted(set(historical) | set(range(earlier, length)))


@contextmanager
def _filtered_attention(recent: int, older: int, counts: dict[str, int]) -> Iterator[None]:
    original = modeling_qwen2.eager_attention_forward

    def filtered(
        module: Any, query: Any, key: Any, value: Any, mask: Any, *args: Any, **kwargs: Any
    ) -> Any:
        length = key.shape[-2]
        if query.shape[-2] != 1:
            raise ValueError("KV oracle must run only after full, unmodified prefill")
        positions = _static_indices(length, recent, older)
        indices = torch.tensor(positions, dtype=torch.long, device=key.device)
        counts["calls"] += 1
        counts["original_key_positions"] += length
        counts["retained_key_positions"] += len(positions)
        if len(positions) < length:
            counts["changed_calls"] += 1
        return original(
            module,
            query,
            key.index_select(-2, indices),
            value.index_select(-2, indices),
            mask.index_select(-1, indices) if mask is not None else None,
            *args,
            **kwargs,
        )

    with patch.object(modeling_qwen2, "eager_attention_forward", filtered):
        yield


def _decode(
    model: Any,
    cache: Any,
    initial: int,
    *,
    forcing: list[int] | None,
    recent: int | None = None,
    older: int | None = None,
) -> tuple[list[int], list[torch.Tensor], dict[str, int]]:
    selected = [initial]
    logits: list[torch.Tensor] = []
    counts = {
        "calls": 0,
        "changed_calls": 0,
        "original_key_positions": 0,
        "retained_key_positions": 0,
    }

    def steps() -> None:
        for step in range(_OUTPUT_TOKENS - 1):
            token = selected[-1] if forcing is None else forcing[step]
            forward = model(
                input_ids=torch.tensor([[token]], dtype=torch.long),
                past_key_values=cache,
                use_cache=True,
            )
            scores = forward.logits[0, -1].detach().float()
            logits.append(scores)
            selected.append(int(scores.argmax()))

    if recent is None or older is None:
        steps()
    else:
        with _filtered_attention(recent, older, counts):
            steps()
    return selected, logits, counts


def _remote_shapes(phase: dict[str, Any]) -> list[tuple[Any, ...]]:
    return [
        (row["layer"], tuple(row["output_shape"]), tuple(row["inputs"]))
        for row in phase["operations"]
        if row["operator"] == "linear"
    ]


def run(*, confirmation: bool = False) -> dict[str, Any]:
    from huggingface_hub import hf_hub_download

    config = json.loads(
        Path(
            hf_hub_download(
                _MODEL,
                "config.json",
                revision=_REVISION,
                local_files_only=True,
            )
        ).read_text(encoding="utf-8")
    )
    model = AutoModelForCausalLM.from_pretrained(
        _MODEL,
        revision=_REVISION,
        local_files_only=True,
        attn_implementation="eager",
        dtype=torch.float32,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        _MODEL,
        revision=_REVISION,
        local_files_only=True,
    )
    prompts = (
        [text + "Explain the conclusion." for text in _CONFIRMATION]
        if confirmation
        else [_PREFIX + suffix for suffix in _SUFFIXES]
    )
    prompt_ids = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=True,
            add_generation_prompt=True,
        )["input_ids"]
        for text in prompts
    ]
    lengths = [len(ids) for ids in prompt_ids]
    if (not confirmation and lengths != [175] * 3) or (confirmation and lengths != [173, 173, 176]):
        raise ValueError("pinned public Qwen input cohort changed")

    composition = MaskedLinearCpu(
        Model.hf(_MODEL, revision=_REVISION),
        quantization=SymmetricPerRow(weight_bits=8, activation_bits=8),
    )
    plan = lower_model(
        config, batch=1, max_input_tokens=max(lengths), max_new_tokens=_OUTPUT_TOKENS
    )
    schedule = plan.runtime_schedule(composition)
    changed = plan.apply(KvCacheEviction())
    original, transformed = plan.to_dict(), changed.to_dict()
    shapes_equal = all(
        _remote_shapes(original[phase]) == _remote_shapes(transformed[phase])
        for phase in ("prefill", "decode")
    )
    if not shapes_equal or changed.coverage(composition).complete:
        raise RuntimeError("cache transform stage-placement or admission gate changed")
    try:
        changed.runtime_schedule(composition)
    except ValueError:
        pass
    else:
        raise RuntimeError("transformed KV plan unexpectedly has a runnable schedule")

    candidates = {
        "recent8_spaced8": (8, 8),
        "recent32_spaced32": (32, 32),
        "recent64_spaced64": (64, 64),
    }
    scores = {
        name: {
            "same_token_top1_matches": 0,
            "free_token_matches": 0,
            "worst_same_token_logit_error": 0.0,
            "attention_calls": 0,
            "retained_key_positions": 0,
            "original_key_positions": 0,
        }
        for name in candidates
    }
    with torch.inference_mode():
        for ids in prompt_ids:
            prefill = model(
                input_ids=torch.tensor([ids], dtype=torch.long),
                past_key_values=DynamicCache(),
                use_cache=True,
            )
            saved_cache = prefill.past_key_values
            initial = int(prefill.logits[0, -1].argmax())
            reference, reference_logits, _ = _decode(
                model,
                copy.deepcopy(saved_cache),
                initial,
                forcing=None,
            )
            for name, (recent, older) in candidates.items():
                selected, logits, counts = _decode(
                    model,
                    copy.deepcopy(saved_cache),
                    initial,
                    forcing=reference,
                    recent=recent,
                    older=older,
                )
                free, _, _ = _decode(
                    model,
                    copy.deepcopy(saved_cache),
                    initial,
                    forcing=None,
                    recent=recent,
                    older=older,
                )
                row = scores[name]
                row["same_token_top1_matches"] += sum(
                    actual == expected for actual, expected in zip(selected, reference, strict=True)
                )
                row["free_token_matches"] += sum(
                    actual == expected for actual, expected in zip(free, reference, strict=True)
                )
                row["worst_same_token_logit_error"] = max(
                    row["worst_same_token_logit_error"],
                    *(
                        float(torch.max(torch.abs(actual - expected)))
                        for actual, expected in zip(logits, reference_logits, strict=True)
                    ),
                )
                row["attention_calls"] += counts["calls"]
                row["retained_key_positions"] += counts["retained_key_positions"]
                row["original_key_positions"] += counts["original_key_positions"]
                if counts["changed_calls"] != 24 * (_OUTPUT_TOKENS - 1):
                    raise RuntimeError("filtered attention did not cover every decode layer")

    total_selected = len(prompts) * _OUTPUT_TOKENS
    for row in scores.values():
        row["selections_checked"] = total_selected
        row["attention_position_fraction"] = (
            row["retained_key_positions"] / row["original_key_positions"]
        )
    baseline_decode = original["decode"]
    transformed_decode = transformed["decode"]
    return {
        "schema": "pllm.kv_selection_quality_gate.v1",
        "source": {
            "model": _MODEL,
            "revision": _REVISION,
            "input_token_counts": lengths,
            "cohort": "confirmation" if confirmation else "primary",
            "output_tokens": _OUTPUT_TOKENS,
            "public_prompt_count": len(prompts),
            "reference": "upstream Qwen2 float32 eager attention",
        },
        "plan_digest": plan.digest,
        "schedule_digest": schedule.digest,
        "transformed_plan_digest": changed.digest,
        "stage_linear_shapes_unchanged": shapes_equal,
        "transformed_schedule_executable": False,
        "decode_attention_scores_width_before": next(
            row["output_shape"][-1]
            for row in baseline_decode["operations"]
            if row["operator"] == "attention_scores"
        ),
        "decode_attention_scores_width_transformed": next(
            row["output_shape"][-1]
            for row in transformed_decode["operations"]
            if row["operator"] == "attention_scores"
        ),
        "candidates": scores,
        "prepared_stage_body_reduction_estimate_bytes": 0,
        "reason": "cache and attention execute locally; grouped remote linear tensor shapes unchanged",
        "runtime_kv_evicted": False,
        "provider_protocol_measured": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirmation", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(confirmation=args.confirmation), indent=2))
