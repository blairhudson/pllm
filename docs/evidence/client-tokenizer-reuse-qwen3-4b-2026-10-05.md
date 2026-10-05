# Qwen3-4B client tokenizer-lifetime screen

A matched fresh-process pair reduced client/dashboard lifetime peak RSS from
**390.32 to 350.45 MB (10.21%, 1.11×)** by constructing one tokenizer instead of
three. End-of-run RSS increased from **337.54 to 350.19 MB**: the probe retains
its tokenizer through process exit. This is a bounded allocation experiment,
not a shipped SDK cache or a tenfold whole-client result.

## Cohort and mechanism

Both candidates used pinned `Qwen/Qwen3-4B` revision
`1cfa9a7208912126459214e8b04321603b3df60c`, W8A8, 16 rendered input tokens,
eight greedy outputs, no warmups and one cold client-bundle request. Each ran
the ordinary benchmark driver in a fresh client process with separate native
Inference and Preparation children and a fresh artifact cache. The checkpoint
was already cached. Both shared one private cohort salt, the same source lock
and Pipeline digest; their Experiment names differ.

The existing paged/lazy client constructs tokenizers for the benchmark's full
input admission check, preparation-row sizing and actual response. The ordinary
response already shares its tokenizer with the decoder. The probe replaces only
the factory inside its isolated client worker, retaining the first tokenizer for
that one bundle instance. It rejects another bundle instance. This does not
change tokenization, model execution, one-use material or the live Pipeline.

| Measurement | Existing paged client | Probe reuse |
| --- | ---: | ---: |
| Process-lifetime peak RSS, decimal MB | 390.32 | 350.45 |
| End-of-run RSS, decimal MB | 337.54 | 350.19 |
| Tokenizer constructions | 3 | 1 |
| Client CPU from benchmark startup through first response, s | 30.04 | 29.60 |
| Aggregate cold-first-response CPU, s | 183.02 | 182.94 |
| Full request time, s | 71.06 | 70.75 |
| Covered application bodies, MB | 571.58 | 571.58 |
| Online application bodies, MB | 141.71 | 141.71 |
| Observed new host swap, bytes | 0 | 0 |

Source lock, body fingerprint, Pipeline, prompt digest, authoritative input/output
counts and output-text digest match. All runtime/audit checks pass. The tiny
transport control also matches and confirms three versus one constructions.
The real pair compares selected output text, not every logit or KV value.

Phase instrumentation records nested, non-additive process samples. In the
control, the largest tokenizer call adds 113.87 MB to the previous high-water
mark; prefill adds another 24.35 MB at its own boundary. These increments are
not standalone object sizes. A separate unpaired diagnostic recorded 366.61 MB
peak and a 104.19 MB tokenizer increment; it is not substituted for the paired
control.

## Decision

Keep the optimization probe-only. The measured reduction warrants investigating
explicit request-lifetime reuse across setup and execution, with cleanup on
success, failure and cancellation. Persistent tokenizer retention is a different
tradeoff. Reducing tokenizer representation and setup/import overhead remains a
candidate for a larger client-memory reduction.

These are single samples with cached checkpoints, co-located roles and process
RSS accounting. They exclude filesystem cache and total device memory. CPU and
latency establish neither a general speedup nor a full-response compute cap.
No matched resident 4B whole-client control was admitted, so the earlier 4B
import-only reduction and this result must not be multiplied into a tenfold
claim. The earlier 421.31 MB CLI sample remains a separate cohort.

## Reproduction

With the checkpoint cached and enough headroom, choose an unused output path:

```bash
HF_HUB_OFFLINE=1 uv run --no-sync python scripts/probe_client_runtime_memory.py \
  --four-b --tokenizer-reuse-ablation \
  --output docs/build/qwen3-tokenizer-ablation.json
```

Use `--tiny` instead of `--four-b` for the transport control. Ordinary admission
and host-pressure guards remain active. The reuse mode is test-local and does
not add an executable Experiment component.

Data:

- [Matched pair](client-runtime-memory-qwen3-4b-tokenizer-reuse-2026-10-05.json)
- [Unpaired phase diagnostic](client-runtime-memory-qwen3-4b-profile-2026-10-05.json)
- [Earlier complete-response CLI sample](qwen3-4b-client-paged-2026-10-05.md)
