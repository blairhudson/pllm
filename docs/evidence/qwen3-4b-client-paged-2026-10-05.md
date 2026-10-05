# Qwen3-4B paged-client complete-response diagnostic

## Result

The pinned `Qwen/Qwen3-4B` checkpoint completed one local native prepared-role
benchmark with 16 authoritative input tokens and eight generated tokens. The
response reached its configured output cap (`response_status: incomplete`),
and all benchmark completion, runtime, privacy-audit and memory-guard checks
passed. The source was already cached. This is one functionality/resource
sample, not a numeric-reference or generation-quality comparison.

Exact report: [qwen3-4b-client-paged-2026-10-05.json](qwen3-4b-client-paged-2026-10-05.json).

| Measurement | Result |
| --- | ---: |
| Client/dashboard process-lifetime peak RSS | 421,314,560 bytes (421.31 MB) |
| Inference process-lifetime peak RSS | 4,497,391,616 bytes (4.50 GB) |
| Preparation process-lifetime peak RSS | 4,026,957,824 bytes (4.03 GB) |
| Time to first token | 12.9172 s |
| Recorded post-first-output throughput | 1.0581 output tokens/s |
| Full request duration | 38.1335 s |
| Aggregate startup-through-first-response CPU | 94.2995 CPU s |
| Online covered application bodies | 141,705,107 bytes |
| Setup-inclusive covered application bodies | 571,576,415 bytes |
| Observed new host swap | 0 bytes |

The client sample includes the in-process benchmark dashboard and excludes both
provider processes and the OS filesystem cache. Process lifetime high-water
marks occur at potentially different times and are not a simultaneous topology
peak. The full-response CPU-cap field remains unchecked. No matched resident
4B control ran; the **10× whole-client memory target remains unestablished**.

## Configuration and provenance

- Source revision: `1cfa9a7208912126459214e8b04321603b3df60c`.
- Source lock: `6b2609e5ce487f01691cf3c991da6d7b9279f7adf0b018eca4e789d2941215a0`.
- Body fingerprint: `8dd9e09f7e920487870d67d30ad38162d9372ac4cad852092ff87fbdc936fefc`.
- Example: `examples/benchmarks/client_memory.py:qwen3_paged`.
- Native CPU, one thread, W8A8, `prefix_f32`, row-residue corrections,
  request-sized on-demand inventory with a four-stage window, compressed batched
  artifacts and paged client storage. The exact Experiment is embedded in JSON.
- Configured workload bound: 64 input and eight output tokens. Actual workload
  was 16+8, not 64+8.
- Implementation baseline: `c4f6a7c`, with the user-requested host guard change:
  preflight reserve `max(1 GiB, physical RAM / 8)` and runtime floor
  `max(1 GiB, reserve / 2)`. On this 32 GiB host those are 4 and 2 GiB.
  Allocation slack remains 25%; the watchdog still aborts above 256 MiB of new
  swap growth. Existing swap is not counted as RAM.

## Executed command

From the repository root:

```bash
HF_HUB_OFFLINE=1 .venv/bin/python -m pllm benchmark run \
  --experiment examples/benchmarks/client_memory.py:qwen3_paged \
  --trust-python --backend native \
  --prompt "Explain private inference in one sentence." \
  --max-output-tokens 8 --warmups 0 --repetitions 1 \
  --temperature 0 --capture-output-digest --output qwen3-4b-paged.json
```

The resulting JSON was moved unchanged to the linked evidence location. Runtime
checks reported zero plaintext prompt/token-ID bytes sent and no online
Preparation requests. Separate local child roles remain co-located; this does
not establish operator independence or a cryptographic security proof. Covered
application bodies do not include complete physical-wire or checkpoint-download
costs. Historical client-memory cohorts use different workloads and salts; their
reduction factors cannot be multiplied into this result.

The later [tokenizer-lifetime screen](client-tokenizer-reuse-qwen3-4b-2026-10-05.md)
uses its own matched fresh-process control and records a 10.21% peak-RSS reduction.
It does not replace this sample or establish the tenfold target.
