# MPCache state-selection reference

This independent bounded reference implements static importance retention,
hierarchical min/max cluster scoring (MPCache equations 2 and 3), and an
adjacent-layer index-sharing policy. It uses the pinned Qwen2.5-0.5B W8A8
compiled decoder with twelve fixed public prompts and three same-token decode
steps per prompt. It is a plaintext numeric/state diagnostic.

## Historical selected-view results

All six configurations preserve prefill exactly: selection starts after
prefill. The full-cache control also preserves every decode logit and KV byte.
The native equations and stable index order pass **8,712** independent NumPy
checks. All retention policies use public fixed parameters, with no prompt
fitting or held-out tuning.

| Policy | Decode top-1 agreement | Worst logit error | Selected attention rows |
| --- | ---: | ---: | ---: |
| Full control | 36/36 | 0 | 181,944 |
| Static retention | 34/36 | 2.7692 | 137,232 |
| Dynamic sign-aware bound | 33/36 | 6.7432 | 52,626 |
| Dynamic alpha=0.6 summary | 32/36 | 6.5244 | 51,151 |
| Combined static/dynamic | 32/36 | 7.9503 | 42,424 |
| Combined with layer-index sharing | 31/36 | 8.2167 | 43,716 |

These are selected-key-row counts across the checked attention operations,
not bytes, FLOPs, or measured speedups. Static retention would reduce active
prefill KV payload from 61.51 to 46.25 MB over the twelve requests. The runtime
retains full backing KV and evaluates selected views: **no cache-storage or
process-memory saving was measured**. JSON oracle IPC also prevents treating
diagnostic elapsed time as a runtime performance measurement.

## Source and execution boundaries

- Static scores average the final sixteen prefill query rows and query heads;
  retain 75% of keys, including eight recent rows. Always-retained keys are the
  top 12.5%. Dynamic clustering uses 16-row and 4-row levels, keeping half the
  clusters at each level before restoring always/recent keys.
- Grouped-query heads are averaged for cluster queries. Cross-layer sharing
  begins at layer 3 and intersects original token positions with the next
  layer's retained set. These are explicit short-context adaptations, not the
  original LongBench or MPC workload.
- Original token positions bind key/value gathers; the current token always
  remains visible. Native tests cover sign-aware upper bounds, alpha-weighted
  summaries that are **not** upper bounds, deterministic ties, hierarchy,
  nonfinite/invalid inputs, and changed-query index leakage.
- Static/dynamic decisions use plaintext client-local values. Revealing chosen
  addresses to a provider would reveal query-dependent information. Protected
  top-k and oblivious gathers are absent from this historical oracle. It does not use the
  compiler-bound retained-state extension below.
- The existing semantic eviction transform is separate from this oracle. The
  production full-KV contract rejects truncated backing state; this probe does
  not bypass that check or activate a fallback Pipeline.

The numeric regressions and protected selection remain promotion gates for the
paper's transformed private-state graph. No whole-generation quality, independent-provider privacy,
matched WAN throughput, or whole-response resource claim follows.

## Physical retained-state primitive

The separate native `RetainedKvCache` now owns only selected float32 key/value
rows and their original token positions. The internal Python
`pllm.runtime.retained_state.RetainedLayerCache` binds this owner with the
interpreter lock released. It has no reference to the dense input arrays.
Construction copies selected rows; subsequent compaction allocates a smaller
owner, preserves retained bits and erases the replaced native buffers.

The public-to-the-caller bounds are at most 8,192 rows, 8,192 flattened features,
64 MiB of owned payload and an exclusive logical-position limit of 2^20.
Capacity includes future decode rows. Owned payload is exactly
`capacity * (8 * flattened_features + 4)` bytes; allocator/owner overhead,
constructor conversions, returned arrays and old/new compaction overlap are
additional costs. These counts are allocation accounting, not peak RSS.
Input key/value payload is capped at 128 MiB. The compiler-bound extension below
additionally prices all layers and transient owners together.

Appends start at the original prefill length, not the number of retained rows.
They reject rewinds, gaps, nonfinite values and exhausted capacity before
mutation. Gathers use original positions and return independent read-only
arrays. A separate causal-mask primitive compares absolute query/key positions,
including multi-row decode and grouped-query attention. Native cluster scoring
also accepts physically retained rows with original-position cluster identities;
its existing 1,024-feature selection limit remains separate from storage bounds.
Close/drop erases native state; already exported caller-owned copies have their
own lifetime.

The storage and equation tests cover source release, signed-zero bit preservation,
capacity-priced appends, transactional failures, physical compaction, min/max
selection against an independent scalar oracle, and attention against an
independent dense calculation. Full retention is a control. This is an isolated
client-local storage/operator slice; the historical checkpoint cohort above still
retained full backing KV. Its quality or projected memory results do not become
measurements of this new owner.

## Compiler-bound client-local decoder

`crates/pllm-compiler/src/retained_decoder.rs` now derives a separate
`pllm.retained_decoder.v1` contract from the ordinary semantic decoder schedule
and the static subset of `KvCacheEviction`. Native admission binds the source
plan, composition, policy, complete key/value ownership, attention dependencies,
absolute-position mask and per-layer commit point. It discovers these through
semantic roles, without model-specific node or checkpoint-path parsing.

The internal `pllm.runtime.retained_execution.retained_runtime` entry point
uses the existing checkpoint-bound model and executor. Prefill still evaluates
complete causal attention. It averages each layer's final observation queries
and all heads, chooses a stable importance top-k plus the mandatory recent
suffix, then retires dense backing arrays after attention consumes them. Decode
appends paired key/value rows to the retained owner and masks using original
positions. Dynamic cluster selection and adjacent-layer sharing reject.

The extension admits batch one, at most 512 input rows and 64 decode rows, at
most 128 exclusive dense KV layer owners, and a caller-bounded state/workspace
allocation ceiling of 2 GiB. Each native owner retains its 64 MiB cap. Admission
charges retained capacity, dense geometric prefill storage, input/export copies
and graph-derived live numeric workspace together. Immutable weights, Python
allocator/process overhead and checkpoint loading remain additional owners;
this ceiling is not a whole-process memory estimate.

The state binding includes the new contract digest. Response-local owner identity,
logical position and capacity are checked before each decode. Failed execution
retires all response-owned native buffers and blocks continuation until reset.
Full-KV snapshots, prefix reuse, transferred state and ordinary continuation
contracts reject this state. Unsupported verified, recurrent, windowed, shared,
changed-causal-reduction and protected nonlinear compositions fail admission.

Generated Qwen2, Qwen3 and bias-free Llama checkpoints pass full-retention
prefill/decode controls with every logit and KV bit preserved. Static retention
matches an independent dense-backing selected-view oracle, including original
positions, while weak-reference checks establish that discarded dense arrays
are released. Contract substitution, state-owner substitution, capacity exhaustion
and partial-layer failures are checked. An installed release wheel passes the
same decoder and retained-storage tests outside the source checkout.

The explicit `ClientLocalKv(retention=KvCacheEviction(...))` state component
now selects this static client-local contract in ordinary `MaskedLinearCpu` and
`ClientOnlyCpu` Experiments. Dynamic selection must keep all rows and adjacent-layer
sharing must be disabled. Native scheduling binds the policy before checkpoint
import and provider session admission. Local benchmark admission and network
placement price its complete state/workspace bound. The selected retention policy
also enters benchmark numeric matching.

Generated-checkpoint SDK prefill/decode and message-history follow-ups pass in
client-only and separate-process prepared roles, and the gateway full-retention
control preserves output. Completion and cancelled streams retire every native
owner. Follow-ups re-prefill client-held message history rather than retaining
sparse KV between responses. Private selection addresses remain client-local.
The bounded protected substrate below binds a private owner/revision, with its
plan/source context supplied by the caller. Provider-owned selection still needs
compiler-validated context, protected score calculation, persistent private state
and transport. The general dynamic and layer-shared MPCache transform remains
unavailable.

## Bounded protected top-k and record movement

`crates/pllm-garble/src/retained_selection.rs` separately implements client-garbled
stable top-k and oblivious movement of complete KV records. It reuses the independent
[half-gates reference](https://eprint.iacr.org/2014/756). It is a substrate for
selection work, not a reproduction of [MPCache's](https://arxiv.org/abs/2501.06807v2)
three-party scoring and eviction protocol.

The trusted client owns the complete private state and supplies finite nonnegative
float32 importance scores. Private inputs encode scores, absolute positions and
every key/value float bit. A fixed bitonic network sorts by mandatory recent-row
membership, score and stable original row order. It moves whole records rather
than opening indices for an ordinary gather. A second fixed network restores
ascending absolute positions. Signed zero in KV remains exact; score signed zeros
tie. Negative or nonfinite scores reject.

Only the client holds output decoding material. The evaluator receives one-use
opaque input labels and a program; its output contains opaque labels for positions
and KV. Public instruction/wire-address digests are identical across changed-score
fixtures, while the clear-address negative control changes selected rows.
This structural check is not a cryptographic privacy proof or a timing study.
Rows, width, retention/recent counts, logical length and public context are visible.

The native client owns the cache exclusively. Receipts bind a fresh random owner,
layer, revision, logical position, shape, policy and issuance. Context contains no
hash of private state or scores. Append, cancellation and close invalidate pending
receipts; wrong outputs consume the rightful receipt before decoding. Foreign-owner
attempts cannot burn another owner's valid query. Successful commit verifies exact
owned KV bits, compacts native storage and preserves remaining decode capacity.
Context hashes are caller declarations, not compiler validation or authenticated
transport. Malformed-context evaluation consumes its handle; caller cancellation
and handle destruction release the remaining client material.

The initial cohort bounded each selection to 16 rows and 16 flattened features.
All layouts allow one outstanding query per owner. A **64 MiB combined allocation estimate**, including old/new state
overlap, gates every circuit before issuance; some shapes inside the row/width
limits still reject. The evaluator never receives reusable secret state handles.
Tests cover stable ties, subnormal/maximum scores, exact signed-zero KV, padded
non-power-of-two networks, context substitution, stale/foreign outputs, cancellation,
append/compaction and admission failures.

`docs/evidence/research-retained-selection.json` records twelve fresh native
processes and **474 exact output words** checked by an independent Python oracle.
All three samples per shape use fresh material and different public synthetic
score fixtures. These are two-to-four-feature records, not checkpoint KV:

| Rows / features / retained | Client→evaluator material | Evaluator→client labels | Public metadata estimate | Median issuance / evaluation | Native peak RSS | State bytes before→after |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 / 2 / 2 | 82,208 B | 5,120 B | 141,018 B | 13.67 / 5.86 ms | 3.26 MB | 120→80 |
| 8 / 4 / 4 | 461,632 B | 18,432 B | 892,876 B | 79.60 / 35.33 ms | 10.34 MB | 360→216 |
| 16 / 2 / 8 | 1,078,528 B | 20,480 B | 2,059,216 B | 190.65 / 86.35 ms | 26.25 MB | 360→200 |
| 8 / 4 / 8 | 705,280 B | 36,864 B | 1,421,248 B | 124.81 / 55.76 ms | 21.76 MB | 360→360 |

Client→evaluator counts include all AND ciphertexts, constant labels and input
labels. The client encoding owners add 34,816–155,648 bytes across these shapes;
they are separately recorded, not remote traffic. Public metadata uses the
existing fixed-width instruction/index estimate. No transport codec is implemented,
so these are logical material bodies, not wire measurements. Fresh native process
CPU includes startup and all phases; per-phase times are local wall measurements.
Native process-lifetime RSS excludes the Python launcher, with no summed-peak claim.

The 16-row/two-feature case spends 1.10 MB of label/ciphertext material, plus
2.06 MB of estimated public instructions, to retire 160 bytes of native state.
A 16×16 case fails allocation admission; the initial implementation's feature
bound also excluded Qwen2.5's 128-feature flattened KV row. Complete full-width
state, protected score calculation, distributed roles,
independent review and a substantially cheaper selection construction remain gates.
This Rust-only substrate does not change the SDK retention policy or establish
checkpoint quality, whole-response savings or provider-held MPCache execution.

### Compact-index sorting and complete oblivious gather

The separate `Layout::CompactIndices` now sorts only private rank/index wires,
restores their original order, and scans **every input record for every retained
record**. Hidden one-hot predicates select the complete absolute-position and KV
bits. Neither indices nor query-dependent native addresses are opened to the
evaluator. The same owner, revision, one-use receipt and exact-bit commit checks
apply; layout substitution also rejects. A padding sentinel larger than every
valid slot prevents dummy rows displacing the last real power-of-two slot.

Geometry can be projected up to 512 rows and 8,192 features without constructing
a circuit or issuing material. Both layouts use the same preflight: 64 MiB combined
allocation, plus the Boolean builder's wire/input/output/AND limits. The projection
counts every input and output label, constant label, half-gates ciphertext, client
encoding, sorting/gather wire and public instruction estimate. It includes old/new
state overlap. Admitted circuits independently match the projected wire and gate
counts. A projected shape is not permission to bypass these limits.

`docs/evidence/research-retained-selection-indices.json` records **36 fresh native
processes and 4,041 exact output words**, with three changed-score fixtures per
shape/layout. Five geometries compare both layouts in the same run; two additional
compact-only shapes exercise 128-feature rows and a 32-row narrow state. Independent
Python integer/float-bit oracles agree, including retained positions. Public access
schedules remain identical across score variants. The full-record control is
remeasured rather than borrowing historical timings.

| Rows / features / retained | Layout | Labels + ciphertexts | Public metadata estimate | Median native CPU | Native peak RSS |
| --- | --- | ---: | ---: | ---: | ---: |
| 8 / 4 / 4 | Full records | 480,064 B | 892,876 B | 119.00 ms | 10.35 MB |
| 8 / 4 / 4 | Compact indices + gather | 492,480 B | 597,970 B | 120.64 ms | 8.78 MB |
| 16 / 2 / 8 | Full records | 1,099,008 B | 2,059,216 B | 280.77 ms | 26.23 MB |
| 16 / 2 / 8 | Compact indices + gather | 1,193,984 B | 1,603,440 B | 303.08 ms | 22.45 MB |
| 2 / 128 / 1 | Full records | 664,032 B | 698,608 B | 98.42 ms | 10.81 MB |
| 2 / 128 / 1 | Compact indices + gather | 927,776 B | 625,227 B | 163.73 ms | 11.29 MB |
| 4 / 128 / 2 | Compact indices + gather | 2,930,848 B | 2,668,711 B | 605.32 ms | 34.41 MB |

The last case fits a 65,574,576-byte conservative combined allocation estimate and
retires 2,056 bytes of state. Full-width **rows** therefore execute, but complete
checkpoint **state** remains out of bounds. Smaller sort records reduce public
instruction estimates; the full-scan gather increases labels/ciphertexts in all
five matched cases. Native CPU also rises. Metadata is not a serialized wire codec,
and a lower process peak in some tiny cases is not a decoder-memory measurement.

The pinned compiler-bound Qwen contract supplies 24 layers and 128 flattened KV
features. Exact geometry-only counts for half retention give:

| Input rows / retained | Full-record labels + ciphertexts per layer | Compact + gather per layer | Compact + gather, 24 layers | Compact allocation bound per layer |
| --- | ---: | ---: | ---: | ---: |
| 16 / 8 | 31.03 MB | 37.32 MB | 895.70 MB | 867.26 MB |
| 64 / 32 | 257.20 MB | 556.18 MB | 13.35 GB | 13.24 GB |
| 256 / 128 | 1.77 GB | 8.71 GB | 209.12 GB | 208.76 GB |
| 512 / 256 | 4.43 GB | 34.71 GB | 833.11 GB | 832.35 GB |

Every complete-state shape rejects before issuance. Whole-body figures multiply
logical material by layer count; allocation is a per-layer estimate, not measured
RSS. Public metadata, protected score calculation, state import and role transport
remain additional costs. The `rows × retained × KV bits` gather dominates at these
retention fractions. This screen rejects this construction for provider execution;
it does not reject other oblivious-routing or MPCache constructions. The client-local
SDK policy is unchanged, and no provider-state transport is activated.

## Fresh pinned-checkpoint allocation and quality

`docs/evidence/research-retained-decoder-qwen25.json` records four fresh-process
candidates over twelve new fixed public rainfall-log prompts, 203–209 input
tokens and four output positions (one prefill, three teacher-forced decode).
The source is pinned Qwen2.5-0.5B-Instruct W8A8 at revision
`7ae557604adf67be50417f59c2c2f167def9a775`. The ordinary full-cache W8A8 decoder
is the reference. Component parameters were frozen before this cohort: the
observation window is the final quarter of query rows, retention is 100%, 75%
or 50%, dynamic selection keeps all rows, and adjacent-layer sharing is disabled.
These choices differ from the historical selected-view cohort above.

| State policy | Decode agreement | Worst logit error | Median owned KV MB | Median active KV MB | Diagnostic peak RSS MB | Diagnostic CPU s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Ordinary dense | 36/36 | 0 | 6.29 | 5.06 | 1,229.85 | 126.45 |
| Retained owner, all rows | 36/36 | 0 | 5.16 | 5.06 | 1,216.36 | 138.56 |
| Retain 75% | 35/36 | 2.7737 | 3.90 | 3.81 | 1,214.58 | 137.63 |
| Retain 50% | 35/36 | 7.3612 | 2.62 | 2.53 | 1,230.27 | 137.51 |

All four preserve every prefill logit. Full retention preserves all 48 checked
logit vectors and all twelve final KV states exactly. Its lower owned allocation
comes from capacity sizing rather than discarded rows. The two sparse policies
change every checked decode logit vector and every final KV state; matching a
selected token does not make those states exact. Their quality regression remains
visible despite physically smaller KV owners.

Owned payload includes native positions and reserved decode capacity; active
payload counts only visible key/value floats. Process RSS is a separate measured
lifetime peak and includes all locally loaded body weights, loading, tokenization,
oracle arrays and twelve sequential requests. CPU covers the loading and
twelve-request diagnostic window, including per-request contract construction,
but excludes earlier interpreter/module startup. Every candidate observed zero
new swap. Process peaks remain approximately 1.21–1.23 GB and measured CPU rises;
the storage reduction establishes neither a whole-client RSS improvement nor a
speedup. There is no provider deployment, independent float32/LongBench quality
comparison, free-running generation trajectory or SDK request in this cohort.

Source, checkpoint, body, token-cohort, policy and native-binary hashes accompany
the report. The driver checks the same implementation hashes before and after
every candidate and keeps oracle logits/token selections only in temporary
diagnostic files. Reports retain aggregate errors, equality flags and hashes.

## Fresh prepared SDK whole responses

`benchmarks/research/retained_sdk.py` freezes four ordinary prepared Experiments:
dense, full retention, 75% retention and 50% retention. Each public prompt uses one
CLI invocation and private cohort salt, with a fresh client process per candidate,
two separate local provider children, native shared 100/40 Mbps access caps, greedy
selection and eight output tokens. All use the same pinned W8A8 checkpoint, paged
client/Preparation weights, demand inventory and packed/residue frames. Each
candidate receives a fresh private bundle cache; Hub/checkpoint and compiled-weight
caches are pre-positioned. Installed package/native hashes are frozen and checked
throughout each cohort. There is one response per configuration and context.

The short cohort has 52 input tokens; the admitted longer cohort has 379. Reports:
`docs/evidence/research-retained-sdk-qwen25/short.json` and
`docs/evidence/research-retained-sdk-long-qwen25/long.json`, with reproduction
metadata and native memory preflights beside them. All eight requests reach the
requested output cap, pass their individual runtime/privacy-counter checks and
observe zero new swap. Retention policies keep distinct numeric contracts: the
comparison-level `matched_numeric_contract` and `passed` flags are **false**, and
all rankings remain empty. These flags do not indicate failed individual requests.

| Input tokens | State policy | Client peak RSS MB | Sampled total RSS MB | Request seconds | Cold aggregate CPU seconds | Output text matches dense |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 52 | Dense | 264.90 | 1,349.44 | 75.93 | 54.77 | Reference |
| 52 | Full retention | 257.77 | 1,196.87 | 75.27 | 53.23 | Yes |
| 52 | Retain 75% | 244.60 | 1,185.28 | 75.22 | 53.87 | Yes |
| 52 | Retain 50% | 252.44 | 1,185.56 | 75.33 | 53.91 | Yes |
| 379 | Dense | 546.31 | 2,366.55 | 289.56 | 146.89 | Reference |
| 379 | Full retention | 524.85 | 2,302.97 | 289.63 | 148.06 | Yes |
| 379 | Retain 75% | 525.43 | 2,380.58 | 289.57 | 148.90 | **No** |
| 379 | Retain 50% | 522.58 | 2,313.08 | 289.94 | 147.64 | Yes |

Full retention lowers the observed client peak by 2.69% short and 3.93% long.
Sparse retention does not monotonically lower whole-client peak memory. The longer
75% response changes its generated text; the 50% response's matching text does not
override its earlier held-out logit/KV and selection mismatches. These are narrow
free-running output-digest checks, not a new representative quality cohort.

All policies use exactly **257.79/989.08 MB** of covered setup-through-first-response
application bodies for short/long, including **82.72/538.12 MB** online. Total native
TCP-stream counters are approximately **258.64/989.92 MB**, with at most 158/160
bytes of within-cohort framing variation. Sparse KV changes client-local attention,
not the number of remote projection rows. No covered-traffic reduction or material
latency improvement appears in this pair of single-response cohorts.

Client peaks include the in-process dashboard, relay and sampler. Aggregate RSS
is a simultaneous 50 ms sample, not the sum of independent process peaks. CPU
includes client/dashboard startup and provider process birth through the first
response; request latency excludes provider startup. TCP counts exclude physical
headers, ACKs, retransmissions and checkpoint distribution. Memory admission is a
conservative capacity estimate, distinct from all reported RSS measurements.
These co-located services do not establish independent operators or a 10× client
memory result.

The original 24-entry longer journal passed memory admission but failed before a
response: 418 correction rows × 9,728 grouped gate/up outputs = 4,066,304 elements,
above the residue codec's **4,000,000-element per-frame cap**. Its partial progress
record remains at `docs/evidence/research-retained-sdk-qwen25/long.json`; it supplies
no latency, memory or quality result. The separate 22-entry rerun stays within
that bound. Host-memory admission alone is not codec/workload admission.

## Reproduce

```sh
uv run python -m benchmarks.research.mpcache_reference \
  --output docs/evidence/research-mpcache-reference-reproduction.json
cargo test -p pllm-core cache_selection_reference
uv run pytest -q tests/test_mpcache_reference.py
```

For the physical state/operator slice:

```sh
uv run maturin develop --release --locked
cargo test --locked --release -p pllm-core --lib
uv run pytest -q tests/test_retained_state.py tests/test_semantic_state.py
```

For compiler-bound decoder controls:

```sh
uv run pytest -q tests/test_retained_decoder.py
uv run pytest -q tests/test_retained_sdk.py
```

Recorded checkpoint-cohort invocation (the output is immutable; choose a new
destination to replay it):

```sh
uv run --no-sync python -m benchmarks.research.retained_decoder \
  --output docs/evidence/research-retained-decoder-qwen25.json
uv run --no-sync pytest -q tests/test_retained_decoder_diagnostic.py
```

For the installed SDK cohorts (use the recorded wheel/package environment and new
output directories; each context keeps its own private salt):

```sh
python -m benchmarks.research.retained_sdk --context short \
  --output-directory docs/evidence/research-retained-sdk-short-reproduction
python -m benchmarks.research.retained_sdk --context long --journal-rows 22 \
  --output-directory docs/evidence/research-retained-sdk-long-reproduction
uv run --no-sync pytest -q tests/test_retained_sdk_cohorts.py
```

For the separate protected selection substrate (choose a new output to replay):

```sh
cargo test --locked --release -p pllm-garble retained_selection
uv run --no-sync python -m benchmarks.research.retained_selection \
  --output docs/evidence/research-retained-selection.json
uv run --no-sync pytest -q tests/test_retained_selection_reference.py
```

For the matched compact-index screen (choose a new output to replay):

```sh
uv run --no-sync python -m benchmarks.research.retained_selection_indices \
  --output docs/evidence/research-retained-selection-indices.json
uv run --no-sync pytest -q tests/test_retained_selection_indices.py
```

The canonical report is `docs/evidence/research-mpcache-reference-qwen25.json`.
Configuration/source/checkpoint/paper digests and every rejected claim remain
explicit. Native code: `crates/pllm-core/src/cache_selection_reference.rs`;
orchestration: `benchmarks/research/mpcache_reference.py`.
Physical ownership: `crates/pllm-core/src/retained_kv.rs`; native binding:
`crates/pllm-python/src/retained_kv.rs`; position-aware attention:
`python/pllm/runtime/semantic_attention.py`.
