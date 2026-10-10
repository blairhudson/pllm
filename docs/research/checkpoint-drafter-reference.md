# Existing-checkpoint drafting with exact target verification

`CheckpointDrafter` privately bridges an existing smaller checkpoint into the
bounded batched greedy verifier. The target remains the original pinned Qwen2.5
W8A8 decoder under canonical `prefix_f32`. No retraining, replacement architecture
or stochastic sampling equivalence is assumed.

Different vocabularies bridge through client-local text. A proposal is admitted
only when target decoding/re-encoding preserves the complete executed token
prefix. Changed BPE boundaries, lossy Unicode and incompatible special tokens
cannot silently replace that prefix. Boundary failure offers an explicit fallback
proposal that still requires target verification.

The drafter reuses only its matching executed prefix. Rejected rows are erased;
the final kept token is recomputed so rollback cannot retain stale logits.
Every emitted token comes from exact target greedy verification. Draft generation,
teacher forcing, rollback, rejected target rows, both model imports and both
weight/state owners remain charged. Failure closes the drafter. This local-clear
reference neither issues provider material nor adds an SDK speculative protocol.

## Pinned checkpoint gate

[`research-checkpoint-drafter-r3.json`](../evidence/research-checkpoint-drafter-r3.json)
records nine fresh processes: three per choice, each executing the same twelve
held-out prompts and sixteen selected outputs per prompt. All target selections,
observed logit vectors and final active KV agree exactly: 299,649,024 checked
values, with zero observed swap growth. The auxiliary checkpoint is pinned
SmolLM2-135M-Instruct; the target remains pinned Qwen2.5-0.5B-Instruct W8A8.

| Per client, twelve responses | Serial target | Two-token draft | Four-token draft |
| --- | ---: | ---: | ---: |
| Accepted / offered draft tokens | — | 96 / 152 | 116 / 216 |
| Target linear calls | 18,432 | 9,216 | 7,296 |
| Target linear stage rows | 64,416 | 69,792 | 74,016 |
| Estimated target ring bodies, decimal MB | 1,069.82 | 1,159.11 | 1,229.26 |
| Median execution CPU, seconds | 62.151 | 112.917 | 119.884 |
| Median execution elapsed, seconds | 30.239 | 66.815 | 69.672 |
| Median process lifetime peak RSS, decimal MB | 1,210.24 | 1,418.22 | 1,422.70 |
| Median process lifetime CPU, seconds | 89.250 | 145.209 | 152.041 |

Draft acceptance is 63.16%/53.70%. Fewer dependent target calls come with
8.35%/14.90% more target rows and estimated bodies. Execution CPU rises
81.68%/92.89%, elapsed time 120.96%/130.41%, and peak RSS 17.18%/17.56%.
Both model owners remain resident. Auxiliary import costs another 5.48/5.37 CPU
seconds, already included in process-lifetime CPU. Execution windows include
draft generation, rejected work, teacher forcing, rollback and vocabulary
conversion; model import is outside those windows. This local-clear gate does
not justify serving promotion. Reduced call counts need an implemented protected
transport and matched WAN measurements before they can establish latency savings.

The earlier [`research-checkpoint-drafter.json`](../evidence/research-checkpoint-drafter.json)
was produced while two invocations overlapped. It retains its own parity and
cost records, but its timings are excluded from the table and are not pooled
with the corrected cohort. The shared runner now takes a nonblocking per-user
cohort lock before admission and releases it on failure or cancellation.
This excludes other participating frontier runners, not arbitrary host activity.

## Replay

```sh
uv run --no-sync pytest -q tests/test_checkpoint_drafter.py tests/test_batched_greedy_verification.py
uv run --no-sync python scripts/probe_checkpoint_drafter.py \
  --output target/checkpoint-drafter.json --processes 3
```

The probe freezes twelve public held-out prompts and compares serial target
execution with two/four-token blocks, using pinned SmolLM2-135M-Instruct as the
auxiliary model. Three fresh processes per choice check all selected outputs,
every observed target logit vector and final active KV. Reports retain acceptance,
all target rows/calls, estimated ring bodies, drafter work, wall/CPU time and
process-lifetime peak RSS. Ring bodies are geometry estimates, not measured
transport; useful local batching does not imply WAN benefit.
