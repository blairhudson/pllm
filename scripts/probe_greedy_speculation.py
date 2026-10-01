"""Pinned W8A8 target and public pretrained client draft, with charged rejected rows.

Current compiled decode admits one row: this is an exact local verification
controller/clear-kernel trace, not parallel protected verification or WAN timing.
"""

from __future__ import annotations

import json
import time
from collections import Counter

from pllm import Model
from pllm.model_loader import resolve_model
from pllm.runtime.greedy_verification_reference import verify_greedy

from decoder_probe_support import DecoderFixture, PROMPT, digest, stage_output, trajectory


def run() -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    fixture = DecoderFixture(inputs=39, outputs=32)
    draft_source = resolve_model(
        Model.hf(
            "HuggingFaceTB/SmolLM2-135M-Instruct",
            revision="12fd25f77366fa6b3b4b768ec3050bf629380bac",
        )
    )
    if draft_source.path is None:
        raise ValueError("public pretrained client draft must be cached")
    torch.set_num_threads(4)
    draft = AutoModelForCausalLM.from_pretrained(
        draft_source.path,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.float32,
        attn_implementation="eager",
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        draft_source.path, local_files_only=True, trust_remote_code=False
    )
    codec = fixture.compiled.runtime(fixture.remote)
    requests = {}
    for count in (8, 32):
        tokens = fixture.tokens(PROMPT)
        if len(tokens) != 39:
            raise ValueError("control token count changed")
        _, expected = trajectory(fixture, tokens, fixture.remote, count=count)
        for width in (2, 4):
            counters = Counter()

            def remote(key, activation):
                stage = fixture.body[key]
                rows = activation.size // stage.spec.in_features
                counters["stage_calls"] += 1
                counters["stage_rows"] += rows
                counters["body_integer_macs"] += (
                    rows * stage.spec.in_features * stage.spec.out_features
                )
                counters["online_arithmetic_bytes"] += (
                    rows
                    * (stage.spec.in_features + stage.spec.out_features)
                    * stage.seeded_profile.wire_bits
                    // 8
                )
                counters["correction_arithmetic_bytes"] += (
                    rows * stage.spec.out_features * stage.seeded_profile.wire_bits // 8
                )
                return stage_output(stage, activation, 8)

            def propose(history, cap):
                start = time.process_time()
                inputs = tokenizer(
                    codec.tokenizer.decode(history), return_tensors="pt", add_special_tokens=False
                )["input_ids"]
                if inputs.shape[1] > 256:
                    raise ValueError("draft context exceeds public research bound")
                with torch.inference_mode():
                    result = draft.generate(
                        inputs,
                        max_new_tokens=cap + 2,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                suffix = tokenizer.decode(result[0, inputs.shape[1] :], skip_special_tokens=True)
                proposals = codec.tokenizer.encode(suffix, add_bos=False)[:cap]
                counters["draft_cpu_seconds"] += time.process_time() - start
                counters["draft_generated_tokens"] += int(result.shape[1] - inputs.shape[1])
                if not proposals:
                    raise ValueError("empty cross-tokenizer draft; no invented zero-cost fallback")
                return [int(value) for value in proposals]

            start = time.process_time()
            result = verify_greedy(
                fixture.compiled.runtime(remote), tokens, propose, outputs=count, width=width
            )
            counters["aggregate_local_cpu_seconds"] = time.process_time() - start
            if result["tokens"] != expected:
                raise AssertionError("draft verification changed greedy target output")
            result["output_digest"] = digest(result.pop("tokens"))
            result["target_greedy_parity"] = True
            result["counts"] = dict(counters)
            result["baseline_target_decode_rows"] = count - 1
            result["target_row_multiplier"] = (39 + result["target_decode_rows"]) / (39 + count - 1)
            requests[f"39+{count}_draft{width}"] = result
    return {
        "schema": "pllm.greedy_speculation_gate.v1",
        "source": fixture.lock(),
        "draft_checkpoint_digest": draft_source.checkpoint_digest,
        "draft_source_lock_digest": draft_source.source_lock_digest,
        "draft_tokenizer_differs": True,
        "sampling": "greedy; fixed output cap, no EOS early stop",
        "requests": requests,
        "batched_compiled_decode": False,
        "scope": "single-process clear-kernel controller; arithmetic traffic projected from actual consumed stage rows; no HTTP/privacy/deployment claim",
        "decision": "one-row compiled decode provides no stage-round batching; rejected rows increase bodies",
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
