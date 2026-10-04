# Combined compatibility and costs — 2026-10-03

## Cohort

Pinned `Qwen/Qwen2.5-0.5B-Instruct` revision
`7ae557604adf67be50417f59c2c2f167def9a775`, W8A8,
`causal_reduction="prefix_f32"`, temperature zero, top-p one. One fresh 39-token
request, an identical repeat and a 45-token extension; eight output tokens each.
Five CPU candidates, fifteen responses: matched workload, model/body identity,
and corresponding output digests. One sample per request. All roles co-located;
checkpoint already cached. Bundle caches isolated between candidates.

Aggregate and comparison checks: [JSON](combined-compatibility-qwen25-2026-10-03.json).
Individual reports carry configuration, native extension, relevant source hashes,
per-link ledgers, authoritative token counts and output digests. They contain no
prompt text, generated text, token IDs or private activation state.

## Results

Decimal MB, 24 generated tokens per candidate. Setup is charged once.
These per-token figures cannot be directly compared with the earlier 32-output
cohort: this shorter cohort amortizes prefill and delivery over fewer outputs.

| Candidate | Covered setup-inclusive MB | Online MB | Online MB/output token | Setup-inclusive MB/output token |
| --- | ---: | ---: | ---: | ---: |
| Prepared default | 616.34 | 231.07 | 9.628 | 25.681 |
| Prepared reuse + on-demand + compressed artifacts | 310.96 | 117.75 | 4.906 | 12.957 |
| Above + attention/first-layer union | 323.94 | 92.85 | 3.869 | 13.498 |
| Seeded/packed offset | 496.80 | 350.50 | 14.604 | 20.700 |
| Offset + reuse + compressed artifacts | 303.94 | 178.93 | 7.456 | 12.664 |

Prepared's lean stack saves **49.55%** of covered total and **49.04%** online,
without moving body weights client-side. Union placement saves **59.82%** online
versus prepared control, but its extra weight delivery raises covered total by
12.98 MB relative to the lean stack. The union adds 57.11 MB i8 weights, 0.24 MB
scales and 57.11 MB native snapshots (114.47 MB total); it retains 84.0% of
**body-linear MACs** remotely. These ownership counts are not peak memory or
whole-response compute fractions.

Offset reuse saves **38.82%** of covered total and **48.95%** online against the
already-seeded/packed control. Its online upload still pays for fresh complements;
both workers retain the body weights. Its raw/compressed public objects have
the same cache identities, including the committed output-residue layout.

Cold startup-through-first-response client CPU was 9.92 / 13.36 / 13.81 seconds
for the three prepared candidates, and 13.77 / 15.89 for offset control/stack.
Corresponding aggregate cold CPU was 53.65 / 53.07 / 52.41 and 44.40 / 52.48 seconds.
Compression and reuse therefore do not establish a client-CPU win. Request-time
totals were 40.01 / 44.27 / 43.04 and 48.22 / 48.33 seconds; those exclude role
startup and advance preparation, so they are not equal-scope end-to-end cold
latency comparisons. Client peak memory, full wire, independent operators and a
full-response compute cap remain unvalidated.

## Admission and parity

- Missing objects use bounded independently checked zlib frames; raw bytes alone
  enter the content-addressed cache. Raw/compressed cache hits interoperate.
- Both offset workers re-lower and acknowledge continuation before stage work;
  missing/forged acknowledgements, replay and cancellation close both sessions.
  Per-stage cumulative row budgets cover prefill and decode.
- Prefix/role unions share one ownership predicate across scheduling, bundles,
  GPU snapshots, provider stages and cost estimates; overlapping stages count once.
- Verified snapshots seal verifier strength and a client-minted inventory
  incarnation. Cache-enabled inventory allocates twelve extra failure bits and
  admits at most 4,096 inventory transitions; exhausted or forged lineage fails
  before new stage work. Baseline verification without that reserve cannot
  accumulate unbounded new inventories. No unchecked remote phase can seal state.
- Tiny Qwen2/Qwen3 CPU/Metal, remote-head and transport tests check exact logits
  and every KV value, invalid state, bad acknowledgements and reservation burns.
  Real output agreement above is narrower than full-logit or broad quality parity.

The [verified real-checkpoint attempt](combined-compatibility-verified-timeout-qwen25-2026-10-03.json)
hit the 180-second inventory-startup limit before any response. Its roles were
stopped; it is excluded from successful byte/CPU comparisons. Tiny verified
combination tests passed. There is no measured real verified-combination win here.

Offset benchmark startup now retains observed zero/delta transport counters;
its setup-inclusive ledger includes the single public boundary bundle download
without adding that bundle twice. Historical reports lacking startup observations
remain unknown. Metadata/control and checkpoint traffic outside the existing
covered-body ledger are still unmeasured.

## Reproduction

From the repository root with the release native extension installed and checkpoint
cached, run one bounded candidate at a time:

```sh
.venv/bin/python scripts/probe_combined_runtime.py --candidate baseline --max-output-tokens 8 --output docs/evidence/combined-compatibility-baseline-qwen25-2026-10-03.json
.venv/bin/python scripts/probe_combined_runtime.py --candidate artifact_stack --max-output-tokens 8 --output docs/evidence/combined-compatibility-artifacts-qwen25-2026-10-03.json
.venv/bin/python scripts/probe_combined_runtime.py --candidate union_stack --max-output-tokens 8 --output docs/evidence/combined-compatibility-union-qwen25-2026-10-03.json
.venv/bin/python scripts/probe_combined_runtime.py --candidate offset --max-output-tokens 8 --output docs/evidence/combined-compatibility-offset-control-qwen25-2026-10-03.json
.venv/bin/python scripts/probe_combined_runtime.py --candidate offset_stack --max-output-tokens 8 --output docs/evidence/combined-compatibility-offset-stack-qwen25-2026-10-03.json
.venv/bin/python scripts/probe_combined_runtime.py --summarise docs/evidence/combined-compatibility-baseline-qwen25-2026-10-03.json docs/evidence/combined-compatibility-artifacts-qwen25-2026-10-03.json docs/evidence/combined-compatibility-union-qwen25-2026-10-03.json docs/evidence/combined-compatibility-offset-control-qwen25-2026-10-03.json docs/evidence/combined-compatibility-offset-stack-qwen25-2026-10-03.json --output docs/evidence/combined-compatibility-qwen25-2026-10-03.json
```

The probe invokes the ordinary benchmark driver with the immutable experiments
in `examples/benchmarks/combined_runtime.py`. It is not a separate runtime or
an automatic optimiser. Planner proposals preserve numeric/privacy identities;
schema-v2 artifact cost records price encoded misses separately from raw residency.
Unknown GPU/compression CPU costs still cannot win a tie without supplied evidence.

## Validation

Non-slow/non-HE Python regression and integration shards passed after updating
obsolete rejection assertions and synthetic lifecycle fixtures. Native compiler
gates passed 16 schedule, three continuation and 16 placement tests; Clippy,
Ruff and whitespace checks passed. All 64 documentation checks and the 400-route
production build passed. A fresh Python 3.13 environment imported the built
wheel outside the checkout and resolved all three combined profiles, verifier
strength, native schedules/continuations and optimization proposals. No provider
role processes remained after the probes and integration checks.
