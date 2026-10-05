# Five additional network-reduction methods

Target: one tenth of covered **all-link** bodies, retaining numeric fidelity,
prompt/activation privacy and low client compute/storage. Fresh responses,
conversation reuse and distribution horizons are separate cohorts.

| Method | Required gate | Outcome |
| --- | --- | --- |
| Canonical generated-prefix qualification | Every logit/KV bit matches fresh prefill; exclude pending tokens, cancellation and `store=False` | Passes 24 Qwen checkpoints and prepared/verified/offset SDK tests; explicit `ClientPrefixReuse(generated_prefixes=True)` |
| Cross-configuration state equivalence | Matching validated numeric, weight, tensor and verification contracts; exact suffix replay | Checked `CompiledRuntimeModel.transfer_snapshot`; automatic planner/cache migration remains separate |
| Masked worker output aggregation | Exact one-use reconstruction, all-link savings, low client CPU | Client downloads halve, peer traffic replaces them, client CPU rises about 5.3×; research-only |
| Token-local integer projection memo | Generic semantic dependency analysis, exact results, fixed public ownership | 27/58 hits; 179 kB cache; about 40% lower selected-kernel CPU; at most 0.41% arithmetic-body saving |
| Progressive head precision | Conservative bitplane bounds certify the exact float32 greedy winner; price private refinement | 16/16 queries certified at each starting precision; six-bit refinement projects 40.8 worker CPU seconds; research-only |

The matched three-request conversation costs **402.05 → 366.63 MB**, an **8.8%
additional saving** over the lean prepared/reuse/compressed-artifact control.
No 10× result is established. Its common first-response cost, 241.72 MB, already
exceeds the cohort's 40.20 MB tenfold target.

Rust owns hot arithmetic, secret expansion, codecs, one-use role handles, bounded
row memoization and head certificates. Python owns immutable SDK options,
orchestration, source locks, budgets and independent numeric controls. The new
metrics APIs are `GeneratedStateReuseProbe`, `StateCompatibilityProbe`,
`MaskedAggregationProbe`, `TokenLocalProjectionProbe` and `ProgressiveHeadProbe`.

Next gates:

1. Use checked state equivalence in a same-client placement switch; meter actual
   leases, artifact misses, transition CPU, retained memory and full-miss fallback.
2. Measure generated-prefix reuse on a longer fixed conversation horizon with
   client peak memory. This remains a workload-dependent route.
3. Require a constant-shape batched private-head retrieval with public capacity
   and whole-query CPU evidence before promoting progressive certificates.
   An observed maximum is not a capacity bound.

The relay and token-local placement do not justify a tenfold research branch on
their current all-link bounds. No percentages are added across unrelated cohorts.
See `docs/evidence/next-five-network-2026-10-03.md` for exact evidence and commands.
