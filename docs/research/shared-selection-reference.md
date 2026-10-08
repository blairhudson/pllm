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

The local harness creates both initial share owners and issues material in bounded
batches; the default batch size remains one.
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
Python inputs and oracle records remain additional owners. Bounded issuance limits
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
tradeoff. These round counts describe the historical serial layout.

## Bounded phase batching

The explicit `CipherPruneTokenPruning(backend="mixed", batch_size=4)` or
`batch_size=16` coalesces independent predicate and record-movement frames.
Top-k uses disjoint bitonic layers; threshold compaction uses a public
dependency-preserving wavefront of the original adjacent moves. Private selected
positions never choose a batch. The disclosed retained count still determines
the compaction geometry.

Every member retains its own context, fresh OT/triple material and one-use
state. Failure burns the entire outstanding batch. Headers for each inner frame
and the outer envelope are charged separately from masked payload; material
is retired between batches. Larger batches increase the live-allocation estimate
and retain the same 64 MiB ceiling and pre-disclosure worst-count admission.
Numeric identity, selection results, cumulative keys and product counts remain
unchanged. This is bounded local exchange coalescing; provider transport and
measured WAN latency remain separate work.

`benchmarks/research/batched_selection.py` compares sizes 1, 4 and 16 in fresh
processes for complete top-k and threshold-pruning records. It checks an
independent oracle and retains every sample's complete costs and phase lengths.

`docs/evidence/research-batched-selection-cost.json` records **468,585 exact
outputs across 36 fresh processes**, three per geometry/batch case. All batch
sizes preserve complete words, selection order, masked payload, keys and product
counts. Extra outer envelopes increase serialized peer frames slightly.

| Consumer / rows / words / retained | Batch | Both-party keys | Peer frames | Peer exchanges | Live-allocation estimate | Median issuance / online |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Top-k / 256 / 257 / 128 | 1 | 295.26 MB | 78.59 MB | 19,200 | 15.45 MB | 37.480 / 10.991 s |
| Top-k / 256 / 257 / 128 | 4 | 295.26 MB | 78.86 MB | 4,800 | 23.33 MB | 37.520 / 10.998 s |
| Top-k / 256 / 257 / 128 | 16 | 295.26 MB | 78.77 MB | 1,200 | 58.29 MB | 37.505 / 10.998 s |
| Pruning / 39 / 896 / 19 | 1 | 13.09 MB | 8.39 MB | 688 | 13.69 MB | 2.760 / 0.062 s |
| Pruning / 39 / 896 / 19 | 4 | 13.09 MB | 8.40 MB | 197 | 22.45 MB | 2.759 / 0.063 s |
| Pruning / 39 / 896 / 19 | 16 | 13.09 MB | 8.40 MB | 78 | 57.64 MB | 2.763 / 0.064 s |

The 16-operation top-k batch removes **93.75% of exchanges**, while local online
CPU is almost unchanged and admission prices **3.77×** the live allocation.
It still spends 295.26 MB of one-use keys to retire 131,584 clear-state bytes.
At 39×896, the dependency-preserving pruning wavefront reduces exchanges 88.66%;
its retained-count dependency prevents a simple 16-fold reduction. Both remain
local operator results, without measured network latency, live provider state,
protected scoring or reviewed transport. Batching does not satisfy those gates.

## Public-bounded metadata conversions

The internal top-k benchmark additionally accepts `bounded_metadata=true`.
Complete records and conditional-movement products remain ring32. Only predicate
inputs use narrower rings justified by public invariants:

- Nonnegative finite float32 score bits fit 31 bits; negative zero, negative
  values, infinities and NaNs reject before sharing.
- Real/padded/recent priority takes values 0, 1 or 2, requiring two bits.
- Stable reverse indices use `max(2, floor(log2(rows)) + 1)` bits. In particular,
  256 rows need nine bits so the last real row stays above zero-valued padding.

Reducing each additive share modulo the selected power of two preserves the
reconstructed value in that ring. The fixed sorting network only permutes these
bounded values. No private score range or selected position determines a width.
Predicates return independently masked full-ring output shares, and every record
word keeps its original 32 bits. Fresh OT correlations, triples, party ownership
and one-use burns remain unchanged. Context binding includes the width contract;
the option rejects threshold pruning, whose separate score-sum domain is not
covered by these bounds. Omission preserves the historical full-ring layout.

`benchmarks/research/bounded_selection.py` compares both layouts at batch size 16
with complete 16/39/256-row, 257-word records. It checks an independent stable-sort
oracle and reconciles actual issuance and every peer frame against geometry-only
resource projections. The conservative live-allocation estimate retains the
existing maximum predicate workspace; smaller cumulative keys do not establish
smaller process peaks or provider-state feasibility.

The retained cohort in `docs/evidence/research-bounded-selection-cost.json`
checks **239,010 complete output words across 18 fresh processes**, three per
case. Sizes below are decimal MB; both layouts use batch size 16.

| Rows × words | Keys, full / bounded | Peer frames, full / bounded | Exchanges, both | Live allocation estimate, both |
| --- | ---: | ---: | ---: | ---: |
| 16 × 257 | 4.95 / 2.57 | 1.31 / 0.79 | 48 | 54.22 |
| 39 × 257 | 42.52 / 22.47 | 11.32 / 6.96 | 171 | 55.03 |
| 256 × 257 | 295.26 / 164.74 | 78.77 / 50.35 | 1,200 | 58.29 |

At full width, combined keys fall **44.21%** and peer bodies **36.08%**, with
unchanged outputs, products and exchanges. Recorded full/bounded local issuance
medians are 38.35/22.86 seconds and online medians 11.27/5.51 seconds. These are
in-process timings on the development host; part of the cohort overlapped an
independent FP32 quality diagnostic. The exact key/frame reductions do not
depend on those timings, and no isolated-host or WAN speedup is claimed.

The full-width bounded layout still issues 164.74 MB of keys and exchanges
50.35 MB to retire just 131,584 clear-state bytes. Protected scoring, live
owner/revision binding, distributed setup and provider transport remain absent;
the smaller layout does not authorize a decoder memory optimization.

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
uv run --no-sync python -m benchmarks.research.batched_selection \
  --output docs/evidence/research-batched-selection-cost-reproduction.json
uv run --no-sync python -m benchmarks.research.bounded_selection \
  --output docs/evidence/research-bounded-selection-cost-reproduction.json
uv run --no-sync python examples/benchmarks/token_pruning.py \
  --backend mixed --batch-size 16 --lanes 128 --samples 3 \
  --output docs/evidence/research-batched-selection-sdk-reproduction.json
cargo test --locked --release -p pllm-garble shared_selection
uv run --no-sync pytest -q tests/test_shared_selection.py tests/test_token_pruning.py
```

Output paths must be new. The SDK example also runs from an installed wheel.
Synthetic selector parity cannot establish trained threshold quality, source
MPCache scoring quality, decoder execution, transport privacy or a client-memory
benefit. Original positions and complete K/V words are preserved, but this
reference does not attach the result to a live decoder cache.
