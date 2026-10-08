# Prepared CPU kernels and bounded bundle decoding

## Contract

This implementation ablation keeps the prepared Client–Preparation–Inference
topology, numeric body, one-use inventory and serialized protocol unchanged.
Preparation computes fresh mask-dependent corrections offline; only Inference
participates in online stages. All body projections remain provider-owned.

On NEON, `Matrix::modular` uses the wrapping 32-bit dot product when its validated
modulus is a power of two. For every admitted such modulus `p`, `p` divides
`2^32`; reducing a wrapping sum modulo `p` therefore equals reducing the original
signed integer sum. Four independent vector accumulators remove the former
single-accumulator dependency chain. Non-power-of-two moduli retain widened
arithmetic. The existing shape, residue and overflow admission still applies.
This also accelerates NEON callers of the separate `wrap32` API.

Compressed public-bundle decoding passes a bounded input-buffer view directly to
zlib. It releases that view before resizing the buffer, including on failure.
Declared lengths, decoded-frame limits, trailing-data checks and bundle/object
digest verification remain mandatory. This removes temporary encoded-frame
copies, not the decoded block or the caller's output owner.

## Isolated screen

`scripts/probe_prepared_path.py --probe kernels` compares two installed builds
in six fresh processes, alternating their order. Each process checks one- and
23-row products at three Qwen3-4B-derived shapes against an independent signed
int64 NumPy oracle. Weights are synthetic public values, not checkpoint weights.
The ring is `2^24`; the Rust suite separately covers power-of-two and odd moduli,
signed weights, SIMD tails, scalar execution and multiple thread counts.

The first screen checks **2,138,112 output positions** across both builds.
Each position is checked through both modular and wrap32 entry points. Median
process CPU for modular execution falls **24–44%** across the six shapes.
The complete fresh-process samples and installed-build hashes are in
`docs/evidence/research-prepared-kernel-screen.json`; the control is commit
`ec7d88d6f17a5f566d99d17d3283112f3d3f4cd6`.

| Rows × input × output | Control CPU ms | Candidate CPU ms | Reduction |
| --- | ---: | ---: | ---: |
| 1 × 2,560 × 2,560 | 1.247 | 0.853 | 31.60% |
| 23 × 2,560 × 2,560 | 24.428 | 13.821 | 43.42% |
| 1 × 2,560 × 9,728 | 4.719 | 2.656 | 43.72% |
| 23 × 2,560 × 9,728 | 109.639 | 81.044 | 26.08% |
| 1 × 9,728 × 2,560 | 4.509 | 2.651 | 41.21% |
| 23 × 9,728 × 2,560 | 109.573 | 82.924 | 24.32% |

Values are medians of the three fresh processes' nine-call medians per build.
The four-MiB incompressible frame control preserves bytes and wire counters while
reducing median Python-traced decoder peak allocation from **5,417,340** to
**4,368,309 bytes**. This is a decoder allocation measurement, not process RSS
or whole-response memory evidence.

Kernel timings omit checkpoint import, paging, mask expansion, transport,
reconstruction and nonlinear work. Their percentage reduction is not a complete
Preparation or request-speed claim.

## Response measurement

The same probe's `responses` mode runs one cold and one exact-repeat request per
fresh installed-build client through the ordinary SDK benchmark. Each client
gets a separate disposable public-artifact cache, shares each party's native
100 Mbps download / 40 Mbps upload cap, and uses one parent-owned private cohort
salt. The selected Experiment uses on-demand correction issuance and has no KV
prefix reuse: the repeat still performs fresh preparation and model execution.

Implementation hashes are collected after response measurements. No allocation
tracing, monkeypatching or forced collection runs inside measured responses.
Both installed Python trees must match except for bundle-frame decoding, and
their dependency versions must agree. The native library has a separate digest.
Canonical comparison checks require matching numeric, workload, output and
transport identities. Client peaks span both requests; cold-first CPU covers
startup through the first response, not both requests combined.

The initial Qwen3-4B attempt was rejected by whole-topology allocation admission,
including with a response-sized public input bound. A Qwen2.5-0.5B
150+8-token pair uses the same prepared stack under sufficient
headroom. Neither kernel geometry nor a smaller checkpoint establishes a
Qwen3-4B whole-response result.

## Matched Qwen2.5 result

`docs/evidence/research-prepared-path-qwen25.json` retains both complete reports,
their canonical comparison and installed-build identities. All four responses
and comparison checks pass. The pinned W8A8 body, source lock, ordered token
cohort and output digests match; both clients observe zero new swap. This is one
cold/repeat sequence per build, not repeated samples of each lifecycle state.

| Metric | Control cold / repeat | Candidate cold / repeat |
| --- | ---: | ---: |
| Complete request s | 141.383 / 108.648 | 139.009 / 105.803 |
| Request tokens/s, eight outputs | 0.05658 / 0.07363 | 0.05755 / 0.07561 |
| After-first-output tokens/s, seven outputs | 2.554 / 2.493 | 2.666 / 2.789 |
| Preparation elapsed s | 42.608 / 42.715 | 41.561 / 41.289 |
| Preparation run-window CPU s | 14.139 / 13.979 | 8.132 / 7.846 |
| Inference run-window CPU s | 25.821 / 20.562 | 20.897 / 15.749 |
| Client run-window CPU s | 22.912 / 17.724 | 22.900 / 17.579 |
| Covered all-link application bodies, MB | 476.953 / 352.048 | 476.953 / 352.048 |

Preparation's run-window CPU falls **42.48%/43.87%** and Inference's falls
**19.07%/23.41%**. Including startup, Preparation's cold-first CPU falls from
**25.14 to 18.87 seconds**; aggregate cold-first CPU falls from **87.30 to
76.38 seconds (12.52%)**. Client cold-first CPU rises from **26.48 to 26.74
seconds**, while its run-window CPU is nearly unchanged.

Complete request latency improves only **1.68%/2.62%**, despite the larger
kernel and provider-CPU reductions. Correction pushes still account for
approximately **36–37 seconds** per request under the shared access caps.
Covered bodies are identical; two-request TCP-stream totals are **830.670375
and 830.670512 MB**, differing by 137 framing/control bytes. Neither this
implementation change nor the capped measurements establish a network-volume
reduction. After-first-output throughput improves **4.38%/11.88%** in these
single samples; it excludes preparation and time to first output.

Client lifetime peak RSS **rises from 248.87 to 261.82 MB (5.20%)** across the
two responses. Sampled simultaneous topology peak RSS falls from **1,608.09 to
1,482.10 MB (7.83%)**; sampling can miss short peaks and count shared pages
twice. The isolated frame-allocation saving therefore does not establish a
whole-client memory improvement. The measured outcome is lower provider CPU
with modest capped latency improvement, while the client-memory target remains
open.

## Reproduction

Build and install the control and candidate wheels into separate environments
with matching dependencies. The control is the parent implementation of this
slice. Keep the shared Hub checkpoint cache populated and public tokenizer
artifacts pre-positioned; the artifact builder is
`scripts/build_incremental_sdk_experiments.py`. Then run from the checkout:

```sh
.venv/bin/python -m scripts.probe_prepared_path --probe kernels \
  --baseline-python target/venvs/prepared-selection-frontier/bin/python \
  --candidate-python target/venvs/prepared-kernels/bin/python \
  --output target/evidence/prepared-kernel-reproduction.json

PLLM_RESEARCH_ARTIFACTS=target/prepared-frontier-artifacts \
.venv/bin/python -m scripts.probe_prepared_path --probe responses \
  --baseline-python target/venvs/prepared-selection-frontier/bin/python \
  --candidate-python target/venvs/prepared-kernels/bin/python \
  --experiment benchmarks.research.prepared_frontier:serial \
  --prompt-file target/prepared-frontier-artifacts/prompt.txt \
  --output target/evidence/prepared-path-reproduction.json
```

Output paths must be new. The public tokenizer artifact's distribution and
compiler cost, checkpoint distribution, physical wire, Internet RTT and
independent operators remain outside the response measurement. These are
implementation-performance checks under an existing numeric contract, not a
new checkpoint-quality or privacy result.
