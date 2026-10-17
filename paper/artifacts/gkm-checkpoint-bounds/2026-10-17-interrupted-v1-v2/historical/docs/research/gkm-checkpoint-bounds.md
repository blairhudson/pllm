# GKM checkpoint bounds: pinned common-model source gate

Date: **17 October 2026**. Result: **blocked by observed host swapout; no source
numeric admission**. Header/semantic geometry is available. Weight maxima,
out-of-range counts and L1 bounds are **unmeasured**, not zero.

## Scope and receipts

This gate follows [D1/D3](../../paper/methods/DECISIONS.md) and
[GKM §§2.2–2.3](../../paper/methods/delegation-contracts.md#22-normative-computation-and-numeric-boundary):
original nearest/2048 signed-i16 weights, ring64 linear arithmetic and separate
float32 embedding bytes. It does not execute a model or construct a Pipeline.

Owned implementation:

- `scripts/probe_gkm_checkpoint_bounds.py`: bounded, read-only cached-source scan;
- `scripts/gkm_checkpoint_oracle.py`: independent scalar integer/IEEE-bit oracle;
- `tests/test_gkm_checkpoint_bounds.py`: focused independent controls.

Receipts were written under `target/` before this document:

1. `target/research/gkm-checkpoint-bounds-2026-10-17-v1/report.json`:
   rejected during startup, before any source reads. Host swapout grew
   **1,933,312 bytes**. Scanner lifetime peak RSS: **47,251,456 bytes**;
   wall time: **0.174704 seconds**.
2. A bounded 5.124-second host observation then showed no swap usage or swapout
   growth, with at least 12,062,097,408 available bytes. This justified one fresh
   attempt with identical limits; the earlier receipt was retained.
3. `target/research/gkm-checkpoint-bounds-2026-10-17-v2/admission.json` and
   `report.json`: metadata preflight passed; stopped while hashing the source
   because host swapout grew **4,276,224 bytes**. No tensor was decoded.
   Hashing read **369,722,916 source bytes** before rejection. Scanner user/system
   CPU: **0.756150 / 0.165318 seconds**; wall: **1.054402 seconds**; observed and
   lifetime peak RSS: **66,977,792 bytes**. There were no child processes.

These costs belong to **publication/source inspection**. They are neither model
throughput nor serving RSS. Swap used stayed constant at 3,115,057,152 bytes;
increased cumulative swapout independently triggered rejection. The receipts do
not attribute host-wide swapout to this process. No guard was weakened and no
further source scan was attempted after the second failure.

## Source identity

- Repository: `Qwen/Qwen2.5-0.5B-Instruct`.
- Revision: `7ae557604adf67be50417f59c2c2f167def9a775`, read without importing the
  W8A8 baseline from `benchmarks/research/common.py`.
- Cached official tree SHA-256:
  `239a6e2f056c95d690248c1f7df3d926989441c1cdbd893765adb12791e7e31a`.
- `config.json` SHA-256, verified against its cached official Git blob identity:
  `18e18afcaccafade98daf13a54092927904649e1dd4eba8299ab717d5d94ff45`.
- Safetensors header SHA-256 (32,280 header bytes; length prefix excluded):
  `bd808652be103f0e4e29786b9aa8829e0f38607880486926c7246ee2de577a76`.
- **Expected**, not completely reverified, `model.safetensors` LFS SHA-256:
  `fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe`.
  Official size: **988,097,824 bytes**. Full hashing was interrupted.
- Semantic plan digest:
  `f7cb9f16061e34582333fba0c26c9e864c7dbc35668f1ed1484bbf4511d3d030`.

The second attempt completed official hash checks for `config.json`,
`generation_config.json` and `merges.txt`. It did **not** complete the full
checkpoint/tokenizer source lock or before/after content/identity verification.
Header geometry below therefore remains provisional until complete source-lock
verification succeeds. No download or source-cache modification occurred.

The configuration declares 24 layers, hidden width 896, MLP width 4,864,
14 attention heads, two KV heads, vocabulary 151,936, tied embeddings,
`rms_norm_eps=1e-6`, and BF16 checkpoint precision. These are configuration
observations, not approval of the lowering's local numerical graph.

## Model-neutral ownership and geometry

`lower_model(config, batch=1, max_input_tokens=1, max_new_tokens=1)` supplies
semantic operations and source weight/bias identities. Existing
`required_checkpoint_tensors` and `required_client_tensors` supply shape and
local-owner requirements. The scanner requires exact declared keys, validates
all shapes, rejects unowned tensors, and checks agreement between prefill and
decode. It does not recognize tensor path names as architectural roles.

The inspected header contains **290 tensors**. The lowered inventory contains
**168 body linear operations**, **one full head**, **72 bias tensors**, and
**49 other client-local coefficient tensors**. The head and token table refer
to the same source tensor but require different encoded owners.

| Owner / representation | Payload bytes |
| --- | ---: |
| Retained seven official source files, disk | 999,586,347 |
| Source tensor payload, excluding header | 988,065,536 |
| Body i16 matrices, per worker | 715,653,120 |
| Full head i16 matrix, per worker | 272,269,312 |
| Raw little-endian float32 token table, per worker | 544,538,624 |
| Linear matrices plus table, per worker | 1,532,461,056 |
| Two workers, matrices plus table | 3,064,922,112 |
| Client float32 small coefficients, including biases | 286,208 |
| Bias subset, client float32 | 110,592 |
| Published embedding row SHA-256 vector | 4,861,952 |

The token/head source shape is **151,936 × 896**. A shared source identity does
not deduplicate the 544,538,624-byte float32 table against the 272,269,312-byte
i16 head. The scanner itself retains neither representation.

### Group and publication-layout blocker

Native `classify_remote_groups` is private to
`crates/pllm-compiler/src/decoder_runtime_schedule.rs`. The existing Python
group/schedule API requires a runtime composition; no source-numeric GKM
composition is admitted by this probe. It does not construct a W8A8 composition
to stand in for GKM or invent a fused-stage schedule. **Fused group roles and
final publication layout remain blocked.** Exact per-operation source bounds
can still be computed by a successful scan.

For the explicitly **conditional unfused** layout of one matrix per semantic
linear/head operation, geometry alone gives:

- 169 matrices; global `R` length 151,936; total `S` length 246,656;
- 13,153,536 bytes for 33-byte compressed point payloads;
- 4,861,952 bytes for the publisher's 32-byte secret-scalar payload;
- 22,487,040 bytes for per-matrix client input/output scalar accumulators.

No points, secret scalars, accumulators or publication were generated. These are
payload formulas, not complete owner memory estimates. Native decoded points,
MSM scratch, queues, copies, delivery, KV and local execution remain additional.
The final grouped layout can have different publication geometry.

## Numerical and integrity procedure

The probe supports BF16, F16 and F32 explicitly in little-endian safetensors
encoding. Reads use `os.pread`, at most **1 MiB**, with row-aligned chunks.
There is no full-file mapping or whole-tensor import. Header admission rejects
duplicate keys, inconsistent shapes, unsupported types, overlap, gaps,
truncation and oversized rows/headers before value reads.

Finite source values decode exactly into binary64. Multiplication by `2^11`
is exact for all three admitted source formats; `rint` implements nearest-even.
Only after checking `-32768 <= Q(w) <= 32767` may a linear weight be cast.
Non-finite values reject. Local coefficients/biases retain float32 ownership;
their i16-range diagnostics cannot reject or admit a delegated matrix. Bias
placement and addition order still require an explicit numerical graph.

For each linear matrix, the successful-scan procedure computes every row's
integer `L1 = sum(abs(Q(w)))`, hashes the streamed row-L1 vector, records its
maximum and row index, and reports:

```text
safe_uniform_encoded_input_max = floor((2^63 - 1) / max_row_L1)
conditional_integer_output_abs_max(B) = B * max_row_L1
```

For zero L1, weights impose no bound; the symmetric signed-i64 input cap remains.
Fixed candidate encoded bounds are 2,048, 32,768 and 2,097,152. Their reports
are conditional arithmetic checks only. The actual activation range is never
inferred from a weight scan.

The independent scalar oracle interprets sign/exponent/fraction bits using
Python integers, forms an exact rational value, performs quotient/remainder
ties-to-even rounding, and derives float32 bits independently. Every chunk
checks endpoints, midpoint and finite extrema against it. Source extrema,
rounded extrema, failures and sample-transcript hashes are recorded. Float32
table hashing preserves signed zero; embedding row hashes and i16 head hashes
are separate.

Full source hashes are checked against cached official Git/LFS identities before
numeric scanning and recomputed afterwards. Open descriptor/path identities
include device, inode, size, modification/change timestamps, symlink and resolved
target. Code/oracle hashes are also checked before/after successful scans.
Interrupted runs retain partial evidence and cannot mark source admission passed.
JSON output is atomically and exclusively published; prior failure receipts
cannot be overwritten.

## Resource admission

Bootstrap ceiling is **512 MiB**. Complete scan estimate is **487,587,840 bytes
(465 MiB)**: baseline/imports 256 MiB, raw buffer 1 MiB, bounded overlapping
arrays 32 MiB, metadata/oracle/report/serialization 96 MiB, allocator slack
64 MiB, and telemetry/helper allowance 16 MiB. The estimate is frozen before
source hashing or value decoding. There is one persistent process and one
numerical thread, with 300-second wall and 240-second CPU caps.

Available host memory must preserve `max(4 GiB, physical_memory / 8)` plus the
bootstrap ceiling at initial admission. Runtime monitoring preserves the full
host reserve and rejects **any** new swap usage or cumulative swapout. Existing
swap is not counted as capacity. Checks occur between bounded reads, normally
at least every 0.25 seconds; maximum observed sampling gap is recorded. This is
cooperative monitoring, not an OS hard limit or proof against missed peaks.

An eventual retry needs a host that remains within these unchanged limits.
Run into a **new** receipt directory:

```sh
.venv/bin/python scripts/probe_gkm_checkpoint_bounds.py \
  --output target/research/gkm-checkpoint-bounds-next
```

## Tests and frozen code

Executed:

```sh
.venv/bin/python -m pytest tests/test_gkm_checkpoint_bounds.py -q
# 22 passed in 0.59s
.venv/bin/ruff check scripts/probe_gkm_checkpoint_bounds.py \
  scripts/gkm_checkpoint_oracle.py tests/test_gkm_checkpoint_bounds.py
# All checks passed!
```

Controls cover all 65,536 patterns of each 16-bit format, F32 boundary/random-bit
checks against exact rationals, signed zeros, halfway ties, asymmetric i16
endpoints, out-of-range rejection without clipping, non-finite rejection,
independent exact L1/hash calculations, tied owners, separate biases, malformed
headers, bounded/short reads, mutation and symlink retargeting, immutable receipts,
resource failures before scanning, and opaque model-neutral tensor identities.

SHA-256 identities recorded in both attempt receipts:

| File | SHA-256 |
| --- | --- |
| Probe | `eafe9cb90b51e01b6670fc6cc3864aacaa587df81235b3fed3d241c4336d0349` |
| Scalar oracle | `ad8683a96813e6b428627594831010126c8d21fc54a50b79bce8ecc2aa701a17` |
| Tests | `0d2429ca5066f907edfdf93b1b5c36eacdc28929b686514d347e22e0567473fe` |
| Installed metadata-lowering native module, second attempt | `6b9370fba12ddd3df25979631d2ae9d37097d60bf0ae9bba1cd1bc2e8389a41e` |

## Admission result and remaining gates

| Question | Result |
| --- | --- |
| Source numeric scan completed? | **No**, host swapout guard rejected both attempts. |
| Finite / i16 out-of-range counts? | **Unknown**; no tensor decoded. |
| Actual weight or rounded maxima? | **Unknown**. |
| Exact matrix L1 / encoded-input maxima? | **Unknown**. |
| Complete checkpoint/tokenizer identity verified? | **No**, source hashing interrupted. |
| Header/semantic owner geometry? | Available, provisional pending complete source lock. |
| Source-faithful fused GKM groups/publication layout? | Blocked on metadata/API integration. |
| Actual activation bounds, quality, privacy, serving eligibility? | Not established by this gate. |

After resource admission and a complete successful scan, remaining work still
includes actual encoded-input bounds at every stage, the original local numeric
graph and bias order, publication/verification bindings, verified acceptance and
lifecycle, original-model validation, and frozen common-model quality criteria.
