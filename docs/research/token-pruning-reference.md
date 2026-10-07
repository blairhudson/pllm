# CipherPrune threshold/count and stable compaction reference

## Source and adaptation

Primary source: [CipherPrune, arXiv:2502.16782v2](https://arxiv.org/html/2502.16782v2),
sections 3.2–3.3 and Appendix A Figures 13–14. The pinned PDF has SHA-256
`cc6df60a5e12bcc65caec5a94a51a810d7b5cab25daf55c5b476949cb2c4dbb2`.
Paper text is specification/provenance only; its code is not a runtime dependency.

The paper's basic pruning opens the original keep mask. This reference follows
the optional oblivious-compaction branch instead: open the retained count, move
whole records stably without separately opening their original indices, and
disclose polynomial-degree decisions in compacted order after pruning. Learned
thresholds, private model weights and selected polynomial execution are absent.

PLLM substitutes up to two fresh one-use half-gates circuits for source OT-based
comparison, arithmetic/Boolean conversion and multiplexing. The trusted local
issuer owns supplied inputs and independent additive/XOR shares; the evaluator
has opaque labels without an output decoder. This is an in-process operator
reference, not a distributed CipherPrune system or a malicious-party protocol.

## Exact numeric and movement contract

Attention inputs are head/query/key Q16 integers in `[0,65536]`. For each key,
sum every head/query contribution and compare its exact nonnegative numerator
with `threshold_q16 × heads × tokens`, using strict greater-than. Summing shares
locally in a 32-bit ring is exact under the public bound. Dividing each share
first is invalid: for example, modulo 2³² shares `1` and `2³²−1` sum to zero,
but separately dividing them by two produces `2³¹−1`, not zero.

A fresh XOR mask protects each keep bit at the first circuit's output. Only the
count is intentionally public. The second circuit receives both keep shares,
the shared score and every shared 32-bit payload word. A dedicated marker bit
preserves arbitrary signed/float bit patterns; it never occupies a payload MSB.
For `n` tokens and `m` discarded records, the fixed adjacent-swap schedule uses
`m(2n−m−1)/2` swaps. Kept records stay in original relative order. Fresh output
masks leave each party's record share uniformly hidden from the other party;
the test client reconstructs only to check the independent oracle.

Empty output skips the second circuit. When nothing is pruned, original shares
are retained and both compaction and degree selection are skipped, following the
selected source branch. Otherwise scores travel with
records and the compacted degree bit is `score > reduction_threshold_q16`.
Public parameters require `0 ≤ threshold < reduction_threshold ≤ 65536`.

Count and degree disclosures are real information. Algebraically, a one-token
count equals its keep bit (normalized one-token attention fixes that bit).
A two-token normalized row changing from (1/2, 1/2) to (3/4, 1/4) changes the
disclosed retained count from zero to one at a strict 1/2 threshold.
A disclosed degree bit reveals the corresponding threshold
predicate. Identical public counts use identical public schedules for different
original keep locations, but this is not a proof that counts or side information
cannot reveal locations. The retained disclosure witnesses state this boundary.

## Admission, ownership and tests

The native implementation computes exact gates, inputs/outputs and label payloads
from public geometry. Before issuance, it checks *every* possible retained count
against circuit limits and a 64 MiB combined allocation estimate. The estimate
includes both share owners, circuit bookkeeping, local buffers and scratch;
external Python caller allocations are additional. No private result can trigger
an unadmitted second phase after opening the first count.

Non-cloneable circuit handles consume encodings, evaluation and decoding once.
Foreign labels/decoders reject and cancellation retires remaining local owners.
Share/mask and temporary decoded buffers use zeroizing ownership. No reusable
secret handle crosses PyO3, whose execution releases the interpreter lock.

Rust tests cover all small keep masks, stable complete-word movement, strict ties,
empty/full results, equal-count schedule identity, foreign labels/decoders and
full-width rejection. The SDK independently sums Python integers and filters
complete records; it verifies both count and degree decisions. Source-declared
row-stochastic attention is not inferred merely from bounded caller values.

## SDK and evidence

`pllm.passes.CipherPruneTokenPruning` uses ordinary immutable component discovery,
serialization and `benchmark_operator`/`replay_operator_benchmark`. Thresholds
belong to the component; tokens/heads and feature width belong to benchmark
geometry. Configuration v2 records that geometry without rewriting v1 reports.
`ModelPlan.apply` and decoder execution remain unavailable for this pass.

- Cost/disclosure report: `docs/evidence/research-token-pruning-cost.json`.
- Historical keep-all control: `docs/evidence/research-token-pruning-initial-cost.json`.
- Installed-wheel report: `docs/evidence/research-token-pruning-sdk.json`.
- SDK example: `examples/benchmarks/token_pruning.py`.

Reports charge client encodings, evaluator ciphertexts, constant labels, public
metadata and boundary labels separately. Share generation, circuit issuance and
local online timings are distinct. Zero inter-evaluator frames does not make
label delivery or eventual transport free. Full wire and measured peak memory
are unknown. Symbolic full-width shapes issue no material and execute no decoder.

The final cohort checks **381 exact output words in 15 fresh processes**, including
empty, partial and full retention. Keeping 19 of 39 full-width records projects
**631.09 MB of evaluator material** plus **106.60 MB of client encodings** to
retire **143,360 bytes** of combined share payload. Metadata and boundary labels
are additional. Keeping 75 of 150 projects 8.13 GB of evaluator material. These
costs describe PLLM's half-gates adaptation, not the paper's OT implementation.
Pre-count admission prices the worst permitted private retention count, so even
an eventual keep-all result cannot authorize oversized issuance.

The separately retained initial cohort executed a redundant second phase when
every record survived. The final source-faithful keep-all path skips compaction
and degree selection; partial-retention material is unchanged. Historical
measurements retain their original source hashes.

## Remaining decoder boundary

Qwen uses causal rotary/GQA attention and response-owned persistent KV. Pruning
the current token array alone cannot define original positions, logits ownership,
or which cached rows survive into later decode. CipherPrune additionally trains
its model/thresholds offline. No such source-locked trained Qwen candidate or
selected-polynomial implementation exists here. A held-out quality number would
therefore measure an invented incomplete decoder, so quality remains unknown.

Full-width resource admission, trained source/numeric identity, causal-state
mapping, protected conversion, distributed input encoding and complete transport
costs must precede an Experiment. This slice establishes bounded operator
correctness and negative full-layout costs, not paper-system performance or
client-memory savings.

## Reproduction

```sh
uv run --no-sync python -m benchmarks.research.token_pruning \
  --output docs/evidence/research-token-pruning-reproduction.json
uv run --no-sync python examples/benchmarks/token_pruning.py \
  --lanes 8 --samples 3 --output docs/evidence/research-token-pruning-sdk-reproduction.json
cargo test --locked --release -p pllm-core -p pllm-garble token_pruning
uv run --no-sync pytest -q tests/test_token_pruning.py
```

Both commands require a new output path. The example also runs from an installed
wheel outside the checkout, without Cargo or benchmark-module imports.
