"""Bounded public-weight head-index screen; checkpoints after each rank."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from pllm.metrics import PrivateHeadRetrievalProbe


PROMPTS = (
    "Name the instrument used to measure atmospheric pressure.",
    "What is twelve multiplied by seven?",
    "Complete this Python expression for a list of squares: [x*x for x in",
    "Give one antonym for the word ancient.",
    "Which planet is known for its prominent rings?",
    "Return a JSON object with a boolean field named enabled.",
    "Briefly explain why wet clothes dry more slowly in humid air.",
    "Write the first three prime numbers separated by commas.",
)

CONFIRMATION = (
    "Which chemical element has the symbol Fe?",
    "Convert fifteen hundred metres to kilometres.",
    "What does the Python len function return?",
    "Complete the sentence: Pure water normally freezes at",
    "Write a valid empty JSON array.",
    "Name the largest organ of the human body.",
    "What colour results from mixing blue and yellow paint?",
    "Complete this SQL fragment: SELECT name FROM",
)


def pinned_inputs(query_calibrated=False):
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from pllm.runtime.quantization import quantize_activation_per_row
    from probe_private_pages import pinned_boundary
    from probe_rank_interface import _MODEL_ID, _REVISION, _token_ids

    torch.set_num_threads(4)
    snapshot = Path(
        hf_hub_download(_MODEL_ID, "config.json", revision=_REVISION, local_files_only=True)
    ).parent
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        snapshot, dtype=torch.float32, attn_implementation="eager", local_files_only=True
    ).eval()
    hidden = []

    def capture(_module, args):
        hidden.append(args[0][0, -1].detach().cpu().numpy().copy())

    hook = model.lm_head.register_forward_pre_hook(capture)
    calibration_prompts = (
        json.loads(
            (
                Path(__file__).resolve().parents[1]
                / "examples/benchmarks/rank_interface_prompts.json"
            ).read_text()
        )["calibration"]
        if query_calibrated
        else []
    )
    prompts = CONFIRMATION if query_calibrated else PROMPTS
    with torch.no_grad():
        for prompt in (*calibration_prompts, *prompts):
            ids = torch.tensor([_token_ids(tokenizer, prompt, 64)])
            cache = None
            for _ in range(4):
                result = model(ids, past_key_values=cache, use_cache=True)
                cache = result.past_key_values
                ids = result.logits[:, -1].argmax(dim=-1, keepdim=True)
    hook.remove()
    del model, result, cache
    all_queries = quantize_activation_per_row(np.stack(hidden), bits=8)
    split = len(calibration_prompts) * 4
    calibration = (
        (all_queries.values[:split].astype(np.float64) * all_queries.scales[:split, None])
        if split
        else None
    )
    owner, weights, scales, source = pinned_boundary()
    source.update(
        {
            "hidden_source": "upstream float32, original W8 head, A8 final hidden",
            "prefill_positions": len(prompts),
            "teacher_forced_decode_positions": 3 * len(prompts),
            "public_calibration_positions": split,
            "cohort_digest": hashlib.sha256(json.dumps(prompts).encode()).hexdigest(),
        }
    )
    return (
        dict(
            weights=weights,
            scales=scales,
            inputs=all_queries.values[split:],
            input_scales=all_queries.scales[split:],
            public_calibration=calibration,
        ),
        source,
        owner,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pinned", action="store_true")
    parser.add_argument("--query-calibrated", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.query_calibrated and not args.pinned:
        parser.error("--query-calibrated requires --pinned public calibration")
    data, source, owner = (
        pinned_inputs(args.query_calibrated)
        if args.pinned
        else ({}, {"fixture": "bounded deterministic public weights and queries"}, None)
    )
    ranks = (32, 64) if args.pinned else (4, 8)
    counts = (1, 8, 32, 128) if args.pinned else (1, 8, 32)
    report = {
        "schema": "pllm.private_head_retrieval_benchmark.v1",
        "source": source,
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "cases": [],
    }
    for rank in ranks:
        result = PrivateHeadRetrievalProbe(
            rank=rank, candidate_counts=counts, page_records=32 if args.pinned else 8
        ).run(**data)
        report["cases"].append(result)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
        print(
            f"rank={rank}: index={result['client_index_payload_bytes']} B; "
            f"agreement={[v['winner_agreement'] for v in result['candidates']]}; "
            f"certificates={[v['certified_winners'] for v in result['candidates']]}",
            flush=True,
        )
    del owner


if __name__ == "__main__":
    main()
