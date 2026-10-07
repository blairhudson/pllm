# Mixed private selection and complete-record movement

## Source contracts and scope

This clean-room reference retains the selection contracts already reviewed in
`docs/research/mpcache-reference.md` and `docs/research/token-pruning-reference.md`.
It replaces full-record Boolean multiplexers with ring32 conditional moves;
garbled comparisons and explicit arithmetic/Boolean conversions remain charged.
It is a bounded local construction, not either paper's complete backend.

Two distinct consumers share the native `shared_selection` implementation:

- **Stable top-k:** canonical nonnegative finite float32 score bits, mandatory
  recent rows, stable index tie-breaking, and selected records returned in
  original order. Fixed bitonic networks rank and restore order. A record may
  contain an absolute position and every original K/V word; scores and indices
  travel with records without being disclosed to an evaluator.
- **Threshold pruning:** exact Q16 column-sum numerators, strict threshold
  comparison, public retained count, stable adjacent compaction and public
  compacted-order degree decisions. Empty output skips movement; keeping all
  rows preserves their shares and skips degree disclosure. Schedules depend only
  on public geometry and the explicitly disclosed count.

The SDK `CipherPruneTokenPruning(backend="mixed")` uses the second contract
through ordinary `benchmark_operator` and replay. Omitting the backend preserves
the original component identity. Both backends retain one numeric identity and
separate execution/configuration identities. The top-k entry point is an
internal measurement binding, not a provider-held retained-KV owner.

## Ownership, privacy boundary and admission

Each garbled predicate consumes independent party input shares using fresh
dealer-aided random-OT correlations. Its result remains additively shared. Each
record move multiplies the shared predicate by the difference of two records
using fresh ring32 Beaver triples, then updates both records locally. Reusing a
predicate across its record words does **not** reuse a triple or a triple mask.
Every party-local primitive consumes one-use material; malformed frames, replay,
foreign context and cancellation retire it. Count and degree disclosure frames
bind context, shape, direction and a fresh issuance. Framing is not authenticated
transport or malicious-result verification.

The local harness creates both initial share owners and issues material serially.
It opens only authorized disclosures and the final independent-oracle result.
Client-local clear score calculation is outside the protected selector; no
provider attention-score computation is implemented. Caller-supplied context is
not a compiler-bound state identity. A complete retained-state transaction still
needs owner/revision binding, authoritative scores and independent role transport.

Geometry admission occurs before sharing inputs or opening a count. All possible
retained counts must fit a **64 MiB combined live-allocation estimate**,
**8,388,608 product limit** and **512 MiB cumulative both-party key limit**.
Both share owners, padded sorting and returned-record overlap, bounded predicate
and product workspaces, and phase-accounting storage are included. External
Python inputs and oracle records remain additional owners. Serial issuance lowers
peak live material; it does not remove cumulative issuance, key delivery or
dependent peer exchanges.

Reports separate garbled-decision material, arithmetic-movement triples,
constant labels, public metadata estimates, complete peer frames and boundary
estimates. Input sharing, issuance and local evaluation have separate timings.
The measured local frames are neither full wire nor deployed WAN measurements;
process peak memory remains unmeasured.

## Validation and reproduction

`docs/evidence/research-shared-selection-cost.json` records **171,303 exact outputs
across 33 fresh processes**. Three processes run each of eleven frozen cases.

| Consumer / input rows / complete words / retained | Both-party keys | Peer frames | Serial peer rounds | Live-allocation estimate | Median issuance / online |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pruning / 4 / 16 / 2 | 85,824 B | 13,836 B | 19 | 9.97 MB | 10.10 / 3.83 ms |
| Pruning / 39 / 896 / 19 | 13.09 MB | 8.39 MB | 688 | 13.69 MB | 2.758 / 0.062 s |
| Top-k / 16 / 257 / 8 | 4.95 MB | 1.31 MB | 312 | 10.25 MB | 0.626 / 0.184 s |
| Top-k / 39 / 257 / 19 | 42.52 MB | 11.30 MB | 2,736 | 11.20 MB | 5.429 / 1.594 s |
| Top-k / 256 / 257 / 128 | 295.26 MB | 78.59 MB | 19,200 | 15.45 MB | 37.492 / 10.961 s |

The matched 4×16 half-gates partial-pruning control uses 523,280 bytes of combined
issuer/evaluator material and 60.81/26.02 ms issuance/evaluation. Mixed partial
pruning reduces that material **83.60%**, while adding peer exchanges. Keep-all
and empty controls regress: mixed keys are 55,776 bytes versus 21,872/23,920 bytes,
and both issuance and online time increase. Backend selection stays explicit.

The 39×896 full-width pruning invocation retires 143,360 bytes of combined record
shares. Its historical geometry projects 631.09 MB of evaluator material **plus**
106.60 MB of client encodings; that legacy layout cannot execute within admission.
The mixed shape executes, but 13.09 MB of keys plus 8.39 MB of peer frames still
greatly exceeds retired payload. At 150×896, the half-retained projection needs
184.17 MB of keys and 122.43 MB peer frames, but another possible count exceeds
the product limit; pre-count admission therefore rejects the entire invocation.

Top-k records contain 128 K words, 128 V words and one absolute position. At
256 rows, half retention retires **131,584 clear-state bytes** (twice that for two
shares) while costing 295.26 MB of keys, 78.59 MB peer frames and another 192.08 MB
of estimated public metadata. The full-record Boolean control projects 1.717 GB
ciphertext, 0.246 MB constants and 101.32 MB client encodings; the compact-index
control projects 8.662 GB ciphertext with the same client encodings. Boundary
label and metadata categories remain separate in the report. Mixed movement
passes full-width live-allocation admission but remains a poor state-reduction
tradeoff. These serial round counts are the implemented schedule, not a claim
that independent comparisons cannot be batched in a future construction.

Native checks cover every small keep mask, strict ties, stable complete-word
movement, padded non-power-of-two geometry, mandatory recent rows, fixed public
schedules, exact geometry/resource counts, pre-disclosure admission and one-use
frame failures. Python uses independent integer sums, filtering and stable
sorting; arbitrary high-bit words verify that no payload bit becomes a marker.

`benchmarks/research/shared_selection.py` uses fresh processes for matched
half-gates/mixed pruning and complete 128-feature K/V records. It includes
synthetic full-width execution and separately labeled unissued cost projections.
Historical selection and quality cohorts retain their own source identities.

```sh
uv run --no-sync python -m benchmarks.research.shared_selection \
  --output docs/evidence/research-shared-selection-reproduction.json
uv run --no-sync python examples/benchmarks/token_pruning.py \
  --backend mixed --lanes 128 --samples 3 \
  --output docs/evidence/research-shared-selection-sdk-reproduction.json
cargo test --locked --release -p pllm-garble shared_selection
uv run --no-sync pytest -q tests/test_shared_selection.py tests/test_token_pruning.py
```

Output paths must be new. The SDK example also runs from an installed wheel.
Synthetic selector parity cannot establish trained threshold quality, source
MPCache scoring quality, decoder execution, transport privacy or a client-memory
benefit. Original positions and complete K/V words are preserved, but this
reference does not attach the result to a live decoder cache.
