# Exact conversation-prefix control

The ordinary benchmark executes five public ordered requests (repeat, growth,
growth, unrelated negative control) on one live client per candidate. All four
candidates use pinned Qwen2.5-0.5B W8A8 and `prefix_f32`, request-sized one-use
inventory and artifact delivery. These are co-located serialized-body cohorts,
with public artifact cache residency recorded by the reports, not isolated cold
checkpoint-distribution or full-wire measurements.

| Placement / reuse | 8-output sequence bodies | 32-output sequence bodies |
| --- | ---: | ---: |
| Prepared / uncached | 2,153,968,921 | 2,444,745,241 |
| Prepared / exact reuse | 777,751,222 | 1,068,528,310 |
| Client attention / uncached | 1,860,069,319 | 2,100,029,479 |
| Client attention / exact reuse | 714,896,356 | 954,856,708 |

Covered setup-inclusive reduction is 3.01x at eight output tokens and 2.56x at
32. Output text digests and usage match the same-mode uncached control for every
request. This establishes transport/numeric reuse parity, not representative
task quality. The longer output exposes residual decode cost.

The first optimized response alone accounts for 575,018,908 / 626,246,236 bytes.
Consequently this five-request cohort cannot reach its 215,396,892 / 244,474,524
byte tenfold budget merely by eliminating later requests. Longer reuse horizons
and startup/preparation improvements must be priced explicitly.

`causal-partition-exhaustive-qwen25-2026-10-02.json` retains 88 split boundaries
plus three teacher replays over 12, 29 and 50 public tokens. Canonical execution
matches all last-row vocabulary logits and all retained KV values exactly;
legacy execution does not. No generated snapshot enters fresh cache keys.

Continuation admission now follows actual compiled step-input liveness, charges
grouped outputs together and retains branches until their last use, with views
charged as copies and a bounded scratch allowance. Independently restored and
geometrically allocated KV remain charged. The 2 GiB ceiling is unchanged; this
is declared-array admission, not peak RSS.

## Reproduce

```bash
python scripts/probe_conversation_reuse.py --output conversation-8.json
python scripts/probe_conversation_reuse.py --output-tokens 32 --output conversation-32.json
```

The script checks digest/usage parity and checkpoints partial results so an
interrupted long run is never mislabeled complete. Numerical source is locked to
`880f80ec61274d3c80e2a0c1336394b9e955d53ff98436a426a680a460a592e7`.

Validation: 63 focused Python continuation/cache checks, three native
continuation tests (branch/fused-output liveness, overflow and memory rejection),
strict compiler Clippy and Ruff. Both complete real cohorts are retained beside
this report. Repeat/growth are measured; response-owned conversation chains,
pressure/eviction and crash/cancellation cohorts remain distinct follow-up gates.
