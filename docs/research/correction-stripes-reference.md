# Correction output-stripe gate

The installed preparation target accepts `output_columns=0` for a full-stripe
control or a positive bounded column count for a striped candidate. Both keep the
same full row batch, weight values and original prepared-mask domains. A selected
native integer or digit8 kernel computes each output stripe; an independent
signed-i64 oracle verifies every completed correction outside the timed window.

Fresh local processes communicate over inherited private pipes. A fresh HMAC key
and context bind the semantic plan, stage, source weights, exact ring, shape and
one-use attempt. Push and acknowledgement namespaces differ. Ordered column,
sequence and accepted-frame commitments precede advancement. Incomplete state
cannot become READY. Malformed frames, replay, substituted acknowledgements,
deadline expiry and cancellation retire partial files and owned child processes.

This is a benchmark fixture, not an HTTP inventory extension. It does not export
reusable tickets or authenticate independent operators. The complete input and
output masks remain resident. Immutable weight snapshots retain the same total
payload. Native immutable output and writable subtraction can temporarily overlap
within one stripe, and encoded frames/receiver decode add separate owners.

Reports price initial loading/binding, parent-plus-receiver sample CPU, endpoint
peak RSS, phase counters and private correction disk. Pipe lengths include their
four-byte framing; bootstrap, push, acknowledgement and final accounting bodies
are separate. No TCP, TLS, WAN or physical-wire claim follows. Independent oracle
allocations occur after the captured parent peak.

## Replay

```sh
python -m benchmarks.research.correction_stripes --processes 3 \
  --output target/correction-stripes.json
```

The study uses pinned Qwen2.5 configuration-derived synthetic MLP gate/up weights,
16/150 rows, integer/digit8 kernels and full/256/1,024-column output layouts.
Each case has three fresh parent processes and one fresh receiver per sample.
Unsupported native acceleration remains a visible cell. Whole-response provider
admission, paging, publication scheduling and measured serving benefits remain
separate gates before any Pipeline promotion.

## Matched stage cohort

Thirty-six fresh preparation targets and their 36 receiver children check
**29,067,264 exact corrections**. Every cell completes under installed build
`0201f1d6947159b1c5c528ac9648a8154258f24e2cbba13d1344494f78138c0f`.
The full 150-row batch is preserved for every output layout. Medians below use
decimal MB and seconds; sample CPU includes both preparation and receiver startup.
Public weight loading and immutable binding remain separate recorded costs.

| Kernel / columns | Parent peak MB | Receiver peak MB | Combined sample CPU | Sample elapsed |
| --- | ---: | ---: | ---: | ---: |
| Integer / full | 120.52 | 69.83 | 0.24524 | 0.18599 |
| Integer / 256 | 87.26 | 37.21 | 0.25151 | 0.26840 |
| Integer / 1,024 | 90.13 | 40.09 | 0.24893 | 0.25436 |
| Digit8 / full | 124.32 | 69.88 | 0.15984 | 0.14145 |
| Digit8 / 256 | 89.96 | 37.09 | 0.16567 | 0.18473 |
| Digit8 / 1,024 | 92.09 | 40.04 | 0.15892 | 0.17308 |

With digit8 and 256 columns, parent/receiver peaks fall **27.64% / 46.92%**,
while combined sample CPU rises **3.64%** and elapsed time **30.60%**. The
1,024-column layout lowers those peaks **25.92% / 42.70%**, with **0.58% less**
sample CPU and **22.36% more** elapsed time. These are separately sampled endpoint
peaks; summing them would not measure simultaneous topology RSS.

The 150-row push bodies are **4,377,950 / 4,384,561 / 4,379,561 bytes** for
full/256/1,024 columns, with **497 / 9,736 / 2,739 acknowledgement bytes** and
separate control bodies. Each receiver writes the same **4,377,600-byte** private
correction payload. Both input/output masks and **8,716,288 bytes** of immutable
weights remain retained. Smaller stripes reduce peak correction/frame owners,
not the correction payload or mask inventory.

The 16-row controls also complete and remain in the evidence rather than being
extrapolated from 150 rows. **Decision:** output striping preserves the large-batch
digit8 advantage and lowers isolated endpoint memory, but serial acknowledgement
adds elapsed time. A bounded live-provider transport and matched full-response
study are still required before choosing this in an Experiment.

Evidence: `docs/evidence/research-correction-stripes-cohort.json`, with its
`.plan.json` and `.frontier.json` companions.

## Bounded credit-window follow-up

`BenchmarkTarget.preparation(..., output_columns=256, credit_window=1..8)`
permits several ordered pushes before awaiting acknowledgement. The sender keeps
only bounded frame commitments, advances acknowledged and issued positions
separately, and drains all outstanding stripes before sealing. Wrong, replayed or
reordered acknowledgements, over-credit calls, premature seal and cancellation
burn the complete outstanding ledger. Admission prices at most 64 MiB of in-flight
frame capacity in addition to masks and kernel workspace. The original one-credit
choice retains its frame identities. Receiver readiness still requires a complete
ordered seal; credits do not make partial inventories executable.

An initial 36-pair screen overlaps targeted test execution and is used only for
exploration. A subsequent non-overlapping confirmation runs 18 fresh
Preparation/receiver pairs at 150 rows and checks **26,265,600 corrections**.
All choices send exactly **4,384,561 push bytes** and **9,736 acknowledgement
bytes** per sample. Actual maximum outstanding stripes are 1, 4 and 8 respectively.

| Kernel / credits | Combined sample CPU, s | Elapsed, s | Parent peak, MB | Receiver peak, MB |
| --- | ---: | ---: | ---: | ---: |
| Integer / 1 | 0.413913 | 0.428142 | 89.24 | 37.19 |
| Integer / 4 | 0.453661 | 0.448451 | 89.52 | 38.04 |
| Integer / 8 | 0.443437 | 0.445406 | 88.90 | 38.04 |
| Digit8 / 1 | 0.249135 | 0.283593 | 91.14 | 37.29 |
| Digit8 / 4 | 0.272696 | 0.274670 | 96.24 | 37.98 |
| Digit8 / 8 | 0.249252 | 0.262450 | 93.49 | 37.42 |

Eight credits lower the digit8 elapsed median **7.46%**, with nearly equal CPU
and higher endpoint memory. Integer credits regress in this confirmation. This
closes the bounded ordered transport gate, with a modest local tradeoff; it does
not justify a general serving speedup or automatic promotion. Full masks and
weights remain resident. GPU batching, live HTTP inventory integration,
multi-client scheduling and whole-response evidence remain separate work.

Replay (fresh output path):

```sh
python -m benchmarks.research.correction_stripes --credit-screen --rows 150 \
  --processes 3 --output target/correction-credits.json
```

Evidence: `docs/evidence/research-correction-credits-confirm.json` and its plan and
frontier companions. Build: `b299b1629b9ad522f826a150fed754d67b64cd0a67057596718f50ab9302383e`.
Fifty-nine targeted target/transport checks pass, including outstanding-window
retirement and inherited-child cleanup.
