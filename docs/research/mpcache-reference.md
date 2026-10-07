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
  top-k and oblivious gathers are absent. This historical oracle does not use the
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

This is an internal client-local research extension. Its selection addresses
remain inside the trusted client. Protected top-k and oblivious gather still
need an exact layer/state-revision binding before provider-owned selected state
can run. The changed-query address-leakage witness remains a required negative
control. The general MPCache transform and an ordinary Experiment selection
remain unavailable.

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
```

Recorded checkpoint-cohort invocation (the output is immutable; choose a new
destination to replay it):

```sh
uv run --no-sync python -m benchmarks.research.retained_decoder \
  --output docs/evidence/research-retained-decoder-qwen25.json
uv run --no-sync pytest -q tests/test_retained_decoder_diagnostic.py
```

The canonical report is `docs/evidence/research-mpcache-reference-qwen25.json`.
Configuration/source/checkpoint/paper digests and every rejected claim remain
explicit. Native code: `crates/pllm-core/src/cache_selection_reference.rs`;
orchestration: `benchmarks/research/mpcache_reference.py`.
Physical ownership: `crates/pllm-core/src/retained_kv.rs`; native binding:
`crates/pllm-python/src/retained_kv.rs`; position-aware attention:
`python/pllm/runtime/semantic_attention.py`.
