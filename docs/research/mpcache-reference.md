# MPCache state-selection reference

This independent bounded reference implements static importance retention,
hierarchical min/max cluster scoring (MPCache equations 2 and 3), and an
adjacent-layer index-sharing policy. It uses the pinned Qwen2.5-0.5B W8A8
compiled decoder with twelve fixed public prompts and three same-token decode
steps per prompt. It is a plaintext numeric/state diagnostic.

## Results

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
  top-k, oblivious gathers and a transformed sparse-state executor are absent.
- The existing semantic eviction transform is separate from this oracle. The
  production full-KV contract rejects truncated backing state; this probe does
  not bypass that check or activate a fallback Pipeline.

The numeric regressions, sparse-state contract and protected selection remain
promotion gates. No whole-generation quality, independent-provider privacy,
matched WAN throughput, or whole-response resource claim follows.

## Reproduce

```sh
uv run python -m benchmarks.research.mpcache_reference \
  --output docs/evidence/research-mpcache-reference-reproduction.json
cargo test -p pllm-core cache_selection_reference
uv run pytest -q tests/test_mpcache_reference.py
```

The canonical report is `docs/evidence/research-mpcache-reference-qwen25.json`.
Configuration/source/checkpoint/paper digests and every rejected claim remain
explicit. Native code: `crates/pllm-core/src/cache_selection_reference.rs`;
orchestration: `benchmarks/research/mpcache_reference.py`.
