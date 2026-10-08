# Fused prepared-client boundaries

## Contract

The ordinary prepared client fuses float32 row quantization, one-use mask
subtraction and bit packing in Rust. For row-residue replies, it fuses unpacking,
mask reconstruction, centered decoding and float32 dequantization. The native
binding releases the interpreter lock and enforces the existing 4,096-row,
four-million-value bound. Encoded inputs, scales and outputs preserve the
original arithmetic order and exact bits, including ties-to-even rounding,
zero rows, mixed output widths and optional biases.

This removes intermediate quantized, masked and int64-output tensors. Python
still owns bounded input/output byte conversions, tickets and inventory leases.
One-use reservations, stage bindings, cancellation and output admission retain
their existing contracts. Freivalds execution retains integer outputs until
verification completes. Complete body computation remains provider-owned;
Preparation stays offline and only Inference receives online stage requests.

## Isolated boundary screen

`scripts/probe_prepared_boundaries.py` compares the fused and separate paths in
18 fresh processes with alternating order. Synthetic Qwen3-shaped input/output
geometry checks **22,192,128 values** across those processes; these are reproducible
private-value stand-ins, not checkpoint activations. Every packed input, scale and
float32 output digest matches. Timing and allocation tracing are separate.

| Rows × input × output | Separate CPU ms | Fused CPU ms | Separate / fused traced peak MB |
| --- | ---: | ---: | ---: |
| 1 × 2,560 × 9,728 | 0.093 | 0.081 | 0.216 / 0.163 |
| 150 × 2,560 × 9,728 | 9.831 | 8.660 | 32.211 / 12.856 |
| 150 × 9,728 × 2,560 | 13.201 | 8.085 | 19.205 / 15.870 |

Values are medians of three fresh-process medians. Boundary CPU falls 12–39%;
the largest traced-allocation reduction is 60.09%. These figures exclude matrix
execution, masks issued before the call, transport, complete client RSS and
whole-response work. Evidence: `docs/evidence/research-prepared-boundaries.json`.

## Matched installed-SDK response pair

`docs/evidence/research-prepared-client-fusion-qwen25.json` retains a separate
Qwen2.5-0.5B W8A8 **150+8-token cold/repeat** comparison. Both fresh clients use
the ordinary serial prepared Experiment, paged client/Preparation weights,
indexed tokenization, demand issuance, compressed artifacts and packed/residue
codecs. Each has an empty private artifact cache. Native relays enforce shared
100 Mbps download / 40 Mbps upload per party. A single parent-owned cohort salt
binds matching outputs and workload; there is no KV reuse on the repeat.

The control is the installed `94b738c` implementation. The candidate Python tree
differs only in stage framing and transformer-client boundaries; the native
library is separately hashed. Dependencies agree. All four responses and the
canonical comparison pass their checks, with zero observed new swap.

| Metric | Control cold / repeat | Fused cold / repeat |
| --- | ---: | ---: |
| Complete request s | 137.446 / 105.020 | 138.218 / 104.606 |
| After-first-output tokens/s, seven outputs | 2.831 / 2.668 | 2.901 / 2.855 |
| Preparation elapsed s | 40.589 / 40.636 | 40.627 / 40.607 |
| Client run-window CPU s | 20.442 / 16.439 | 20.889 / 15.898 |
| Preparation run-window CPU s | 7.546 / 7.537 | 7.605 / 7.379 |
| Covered all-link application bodies, MB | 476.953 / 352.048 | 476.953 / 352.048 |

Client process-lifetime peak RSS falls **246.79 to 240.60 MB (2.51%)** across
the two requests. Cold-first aggregate CPU rises **68.37 to 69.14 seconds
(1.12%)**. Cold request latency rises **0.56%**, while repeat latency falls
**0.39%**. Thus the isolated boundary improvement produces a small measured
client-memory reduction, with inconclusive complete-request speed benefit.
These single sequences establish neither a 10× client-memory reduction nor
a Qwen3-4B whole-response result.

Checkpoint distribution, the pre-positioned public tokenizer artifact, physical
wire, Internet RTT, filesystem cache and independent operators remain outside
this cohort. Numeric identity preservation is separate from checkpoint quality
and cryptographic review. Independent cohorts are not multiplied.

## Reproduction

Use separate installed control/candidate wheels with matching dependencies and
the existing prepared-frontier public artifacts. New output paths are required.

```sh
target/venvs/prepared-client/bin/python -m scripts.probe_prepared_boundaries \
  --output target/evidence/prepared-boundaries-reproduction.json

PLLM_RESEARCH_ARTIFACTS=target/prepared-frontier-artifacts \
target/venvs/prepared-client/bin/python -m scripts.probe_prepared_path \
  --probe responses --change-scope client-fusion \
  --baseline-python target/venvs/prepared-kernels/bin/python \
  --candidate-python target/venvs/prepared-client/bin/python \
  --experiment benchmarks.research.prepared_frontier:serial \
  --prompt-file target/prepared-frontier-artifacts/prompt.txt \
  --output target/evidence/prepared-client-fusion-reproduction.json
```

Native core tests cover overflow/bounds and separate integer primitives. The
installed-wheel tests include independent NumPy/Python-integer bit-packing and
dequantization oracles, malformed frames and ordinary compact, chunked,
row-residue, stage-packed, inventory and verified execution controls.
