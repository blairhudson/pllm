# Complete Curl/Haar gated reference

The installed `pllm.nonlinear.CurlWaveletGatedMultiplyQ9` independently composes
Curl's private Haar-table lookup with exact protected index reduction, exact
up/output rescaling and fresh Beaver multiplication. It evaluates complete
`SiLU(gate) * up` blocks under a fixed Q18-to-Q9 numeric contract. The operator
and separate clear checkpoint diagnostic are available for user-driven research;
there is no compiler tensor schedule or protected decoder transport binding.

## Source contract and adaptation

- Curl, <https://eprint.iacr.org/2024/1127>, Sections 4.1.1 and 4.2.1,
  Figures 4 and 6. Acquired PDF SHA-256:
  `dc930fdb3219c989106f96eb9ae1f00f28aae16ec5d17eef0c5421eb4f241c1e`.
- Function Secret Sharing for Mixed-Mode and Fixed-Point Secure Computation,
  <https://eprint.iacr.org/2020/1392>, Figure 14 universal interval keys.
- Rust owns public fitting, all party-local execution and material; Python
  independently checks integer results. Upstream code is not a dependency.

Curl Figure 6 truncates before opening a masked compressed index and performing
the rotated one-hot inner product. This implementation preserves that order and
uses local ring reduction only **after** protected truncation. Exact floor index
reduction and exact ties-even up/output rescaling replace the paper's
probabilistic truncation. Generic 32-bit universal-interval helper keys also
compute an unused sign share; that full cost is charged. These are costs of the
PLLM adaptation, not original Curl system measurements or an impossibility bound.

The public table has 4,096 mathematical SiLU samples on `[-64,64)` at spacing
1/32, rounded to Q9. Haar levels 0–8 replace each contiguous power-of-two block
with its integer mean, rounded once. The immutable profile binds coefficients,
ranges and rounding. This differs from the historical publicly fitted Q7
activation-only profile in [the earlier frontier](nonlinear-frontier.md).

Inputs are Q18 integers with public gate/up bounds ±48/±128. The gate gets a
public offset of 64 on one share, then protected floor division by
`2**(13 + levels)`. Up is divided by 512 with exact ties-even; the Q9 lookup
result and Q9 up share enter a fresh 32-bit Beaver product, followed by exact
ties-even division by 512. The whole table and public up bound prove that the
signed product fits 32 bits. Output Q9 converts exactly to float32 in the
clear diagnostic. Input float32-to-Q18 conversion uses ties-even and rejects
out-of-range/non-finite values. Protected float conversion and range proofs
remain absent; the trusted client owns those boundaries.

## One-use execution and accounting

The bounded operator admits 1–256 lanes under a conservative 64 MiB combined
allocation estimate. Preflight precedes any material issuance. Each party owns
only its local rescale, one-hot and Beaver shares; trusted issuance remains
in-process. Context, session, profile, first index, lane order and fresh issuance
bind the fixed four-round execution:

1. Index and up rescaling execute in parallel.
2. Fresh masked table indices open; each party evaluates its own one-hot share.
3. Fresh Beaver multiplication opens masked differences.
4. Protected output rescaling returns opaque output shares.

The client/test oracle reconstructs only at the boundary. No reusable secret
handles escape the Python API. Completion, malformed frames, wrong context,
replay and cancellation burn all remaining material. Framing is not authenticated
transport or malicious-output verification. Integer parity is not independent
cryptographic review.

## Bounded operator measurements

[Raw report](../evidence/research-wavelet-gated-cost.json): nine fresh processes,
levels 0/2/4, lane counts 1/64/256, three fresh samples each. All **2,889** outputs
match the independent Python integer oracle. The 256-lane cases measure:

| Haar levels | Key payload per party | Combined allocation estimate | Issuance, median | Local online, median | Peer frames, both directions |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 5,079,040 B | 39,395,776 B | 134.81 ms | 201.38 ms | 46,304 B |
| 2 | 1,944,576 B | 20,588,992 B | 115.07 ms | 198.06 ms | 46,304 B |
| 4 | 1,169,408 B | 15,937,984 B | 111.34 ms | 198.39 ms | 46,304 B |

Public fitting, clear numeric execution and native-call process CPU are retained
separately. Local online timing includes input sharing and final reconstruction.
It is not distributed latency. Allocation is not measured process peak memory.

### Hypothetical complete-MLP geometry

The historical Qwen2.5 39-input/eight-output geometry executes
`24 * 4,864 * (39 + 8 - 1) = 5,369,856` gated elements, or 20,976 exact 256-lane
chunks. No protected checkpoint request executes in this screen.

| Haar levels | Both-party keys | Lookup keys | Three exact rescalings | Peer frames |
| --- | ---: | ---: | ---: | ---: |
| 0 | 213.08 GB | 175.98 GB | 36.97 GB | 971.27 MB |
| 2 | 81.58 GB | 44.01 GB | 37.44 GB | 971.27 MB |
| 4 | 49.06 GB | 11.02 GB | 37.91 GB | 971.27 MB |

Fresh Beaver payload is another 128.88 MB, already included in the keys column.
Each layout additionally estimates 128.88 MB for client share-entry/output-exit
payloads and 714.44 MB of public per-issuance metadata. Public tables for both
parties cost 32,768/8,192/2,048 bytes once and may be reused. Secret one-hot,
rescale and Beaver material is fresh for every element and invocation.
Offline key framing, physical wire, other decoder operators and private float
conversion are unmeasured/additional. Compression reduces table shares but does
not eliminate the complete exact-rescaling cost or peer rounds.

## Fresh held-out checkpoint screen

[Raw report](../evidence/research-wavelet-gated-quality-qwen25.json): twelve new
public prompts, frozen levels 0/2/4, pinned Qwen2.5-0.5B-Instruct W8A8 compiled
checkpoint. Each configuration starts independent state. Same-token decode uses
three reference-selected tokens; free-running generation instead feeds each
candidate's own token selections for eight outputs. A separate unchanged
fresh-runtime control matches all 48 logits/KV states and 12 trajectories.

| Haar levels | Prefill top-1 | Same-token decode top-1 | Exact eight-token trajectories | Worst absolute logit error |
| --- | ---: | ---: | ---: | ---: |
| 0 | 10/12 | 35/36 | 10/12 | 2.9478 |
| 2 | 7/12 | 33/36 | 5/12 | 3.3277 |
| 4 | 8/12 | 33/36 | 4/12 | 5.4929 |

No candidate has an exact logit vector or full KV state. All declared input
ranges pass without clipping; maximum checked teacher-forced gate/up magnitudes
remain below 36.60/102.03. Each candidate's native clear block matches
**130,043,904** independent integer outputs across the teacher-forced and
generation paths. Observed swap growth is zero. These are clear-checkpoint
diagnostics, not protected checkpoint execution, FP32/upstream-reference quality
or a representative task ranking. The fresh prompt cohort cannot inherit the
earlier SIGMA or Q7 screen's scores.

Every candidate fails this fidelity gate; complete material also remains too
large for the target decoder. The bounded operator stays available to benchmark
negative results. No best-case table-only estimate is substituted for complete
cost. Next admission work needs both a better declared numeric profile and
cheaper protected conversion/rescaling before tensor scheduling or transport.

## Reproduction

Run from the checkout with the release native extension installed. Evidence paths
must be new; the driver never overwrites existing observations.

```sh
uv run --no-sync maturin develop --release --locked
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run --no-sync python \
  -m benchmarks.research.wavelet_gated --mode cost --output wavelet-cost-new.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  uv run --no-sync python -m benchmarks.research.wavelet_gated --mode quality \
  --output wavelet-quality-new.json
```

The quality run needs the pinned Qwen2.5 checkpoint already in the shared Hub
cache and 4 GiB available beyond the host reserve. It retains exact source/body,
public prompt, profile, native binary and diagnostic-source hashes.

Installed SDK-only controls:

The [separate installed-wheel report](../evidence/research-wavelet-gated-operators-sdk.json)
comes from the following public example at 64 lanes and three fresh samples per
profile. It verifies the installed API and replay contract; its local timings
are separate observations from the nine-process cost cohort above.

```sh
python examples/benchmarks/wavelet_gated.py --lanes 64 --samples 3 \
  --output wavelet-operators-new.json
```

Reports use `benchmark_operator` and `replay_operator_benchmark`, preserve
configuration factories and separate numeric identities, and omit private
input/output values. Caller-owned inputs must be supplied again for replay.
