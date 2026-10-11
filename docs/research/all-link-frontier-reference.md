# Five all-link traffic opportunities

This screen starts from the maintained three-client Qwen2.5-0.5B digit8
confirmation, with 150 input and eight output tokens in each cold/exact-repeat
pair. It preserves the historical evidence. The new public-matrix codec has
its own installed-build cohort; placement, restart and publication calculations
remain projections, not measured changes to that serving configuration.

## Where the bytes go

The first complete confirmation pair reconciles **493,575,247 application-body
bytes** across directed links:

| Owner / direction | Decimal MB | Share |
| --- | ---: | ---: |
| Preparation → Inference corrections | 139.06 | 28.17% |
| Inference → Client online results | 139.12 | 28.19% |
| Client → Inference online inputs | 90.44 | 18.32% |
| Inference → Client public bundle | 124.91 | 25.31% |
| Client/Preparation control and seeds | 0.043 | 0.009% |

Startup inventory and warmups are empty for this demand-issued control. Each
directed body is counted once, not again at its receiving endpoint. Stage
attribution is reconciled against the existing protocol counters. TCP framing,
acknowledgements and other omitted transport costs do not become application
bodies through this screen. The previously reported approximately 496 MB
TCP-stream total has a different, broader scope.

## 1. Durable exact history: avoid replay after restart or eviction

The existing RAM prefix cache already wins when it hits. The opportunity is
preserving exact client-owned continuation through process restart, eviction,
and explicitly qualified branches, then reserving only newly executed rows from
live providers. The new `CheckpointSession` and context-graph substrates provide
bounded local storage mechanics; ordinary prepared SDK restoration remains a
separate integration and measurement gate.

For an explicitly exact-extending 150/288/512-input history with eight outputs
per turn, dense full-row arithmetic falls from **971 to 519 rows (46.55%)**.
The pending last sampled token is not already executed KV. This is a row-count
projection, not a 46.55% whole-wire result: initial bundle delivery, terminal
pruning, fresh session admission and metadata remain charged. It establishes
zero new saving when the current qualified RAM cache already hits.

Keep encrypted state local to the client. Remote backup or cross-device
distribution must charge its own upload/download, authentication and ownership.
The final active KV payload is 12.75 MB, plus 0.61 MB logits, history and storage
framing; disk, source reload, CPU and transient copies are additional costs.
One-use masks and tickets are never restored with decoder state.

**Next gate:** same-client, multi-process prepared conversation against both
fresh re-prefill and the current exact RAM-cache control. Compare complete
logits/KV, all directed bodies, storage I/O and all seven serving metrics.

## 2. Byte-aware stage ownership under a small client budget

Choose individual compiled linear groups by total bytes eliminated per added
client matrix cost. Eliminating a stage removes its corrections, masked inputs
and returned outputs together. This extends the existing role/prefix placement
choices with a per-stage byte objective; it does not authorize adding weights
without an explicit client policy.

An exact bounded subset search over the measured 96 stage groups prices every
new i8 matrix and float32 row scale once. It caps local work at **10% of dense
body MACs per fully executed row**. It finds a 35.83 MB public-payload selection
that removes 54.32 MB of stage bodies per cold/repeat pair:

| Repeated workload pairs | Net projected saving after new weights | Fraction |
| ---: | ---: | ---: |
| 1 | 18.49 MB | 3.75% |
| 4 | 181.43 MB | 11.34% |
| 16 | 833.21 MB | 13.83% |

The common public bundle is delivered once across the horizon. Each pair assumes
a new 150-token prefix and its exact repeat, not sixteen cache hits for one
unchanged prompt. Extra weight delivery is uncompressed; new manifests/framing,
CPU, RSS and actual pruned-schedule MAC admission remain unresolved. The selected
99.7-per-mille dense-MAC fraction is not a measured whole-request CPU fraction.
Paging may contain resident weights but still adds disk and page-reading work.
These estimates must not be combined with the independent codec result below
as if they were one measured configuration.

**Next gate:** native stage ownership bound to source, exact numeric plan, client
memory and executed MAC limits; matched installed SDK requests including new
artifact delivery. Preserve full logits/KV and predominantly remote computation.

### Measured SDK follow-up

The installed `ClientLinearRoles(layer_roles=...)` now binds explicit semantic
layer/role pairs through native scheduling, source/weight validation, provider
inventory, paged delivery, memory admission, SDK and gateway execution. Optional
selectors join the existing role/prefix union; absent groups and complete body
offload reject. Omitting them preserves historical component identities.

`scripts/benchmark_all_link_placement.py` uses the frozen byte ledger, then
rechecks the actual terminal-pruned native prefill schedule and ordinary decode.
Each phase independently keeps at least 90% of body linear MACs remote. This
stricter gate selects **38 of 96 groups**, with **34.33 MB** of new public weights
and scales, rather than the earlier 35.83 MB dense-row projection. Prefill/decode
retain **90.07% / 90.45%** remote body MACs. The 0.248-second local selection cost
is separate from response timings; it is an explicit research selector, not an
automatic placement default or a whole-request client-CPU bound.

`docs/evidence/research-all-link-placement-sdk-confirmation.json` contains three
fresh clients per choice, each with cold and exact-repeat **150+8-token** requests
on pinned Qwen2.5-0.5B W8A8. The common stack uses paged client/Preparation
weights, digit8 Preparation, demand/windowed issuance, sixteen-row duplex prefill,
packed/residue frames, prefix reuse, compressed artifacts and the indexed
tokenizer. Shared native 100/40 Mbps caps cover all service streams. Each client
starts its own two provider children and cold object cache under the exclusive
frontier lock. The companion `.plan.json` freezes the selection, factories, prior
and installed build.

All twelve outputs agree, every runtime/privacy check passes, directed stage
ledgers reconcile, and the guard observes zero swap growth. Source, numeric,
state, workload, output, privacy and measurement comparison identities match.
Generated Qwen2/Qwen3 SDK and gateway checks additionally preserve complete
logits/KV with resident and paged delivery. A separate pinned-Qwen three-prompt
prefill/decode check preserves every logit and active KV bit for the exact 38-group
selection and the earlier all-attention placement. A reload regression resets optional
placement metadata when a caller reuses a mutable source manifest; this fix is
validated separately after the frozen measurement build.

Medians across the three clients per choice, in decimal MB:

| Metric | Control | Selected | Change |
| --- | ---: | ---: | ---: |
| Cold/repeat all-link TCP streams | 496.05 MB | **473.68 MB** | **−4.51%** |
| Covered application bodies | 493.58 MB | 471.80 MB | −4.41% |
| Client lifetime peak RSS | 223.20 MB | 229.11 MB | +2.65% |
| Client cold-first CPU | 33.92 s | 35.21 s | +3.82% |
| Preparation birth-to-first-response CPU | 21.32 s | 20.87 s | −2.10% |
| Preparation sampled peak RSS | 239.34 MB | 233.87 MB | −2.29% |
| Cold online TTFT | 44.98 s | 41.24 s | −8.32% |
| Repeat online TTFT | 3.80 s | 3.84 s | +1.07% |
| Cold N−1 decode throughput | 1.97 tok/s | 2.65 tok/s | +33.99% |
| Repeat N−1 decode throughput | 2.19 tok/s | 2.48 tok/s | +13.02% |
| Complete cold request | 116.32 s | 119.13 s | +2.42% |
| Complete repeat request | 16.03 s | 16.36 s | +2.05% |

Placement removes **52.34 MB** of covered execution/control bodies but adds
**30.56 MB** of cold public delivery, leaving **21.78 MB** net application-body
savings. The newly charged client owns 34.18 MB of paged i8 weights and 0.153 MB
of scales; cache files, page reads and transient buffers remain additional owners.
Cold-first aggregate CPU is nearly equal, 94.82 versus 94.29 seconds. Complete
requests remain slower despite fewer dependent online exchanges.

The initial `research-all-link-placement-sdk.json` retains one control failure
after admission (`target_execution_failed`) and five completed cells. It is
excluded from ranking and is not pooled with the complete cohort. The successful
rerun retains the frozen selector rather than tuning against its timings.

This establishes a bounded traffic/client-resource tradeoff. Provider checkpoint
distribution, tokenizer-index distribution, physical wire, Internet RTT and
independent operators remain outside this measurement. Prefix hits, cold bundle
delivery and all-link corrections are charged as recorded; the result does not
establish a general tenfold saving or a lower-client-memory default.

Replay creates a new cohort and requires the admitted artifacts in
`target/prepared-frontier-artifacts`, as used by the prepared-serving study:

```sh
target/frontier-sdk-env/bin/python scripts/benchmark_all_link_placement.py \
  --processes 3 --output target/all-link-placement-replay.json
PLLM_RUN_CLIENT_LINEAR_PARITY=1 target/frontier-sdk-env/bin/python -m pytest \
  -o pythonpath= -q tests/test_client_linear_roles.py
```

## 3. Lossless bit-plane coding of public weights

Masked tensors are not ordinary compressible activations. Public W8 artifacts
are a separate target: the current pair spends 124.91 MB delivering its public
bundle. The probe independently tests plain level-1 zlib against a reversible
bit-plane transpose, with and without signed zigzag remapping, followed by the
same zlib level. No quantizer or model values change.

The complete pinned 151,936 × 896 tied matrix is quantized with the installed
W8 row quantizer. Three fresh processes per layout encode and decode its entire
136,134,656-byte matrix in row-aligned frames bounded by 1 MiB. Every decoded
byte and all workers' weight/scale commitments must agree. Eight-byte length
headers are charged. Quantization, checkpoint loading, parity scans and source
hashing are outside the isolated codec CPU windows, and are explicitly additional
work. This is a public matrix experiment, not the complete artifact protocol.

The installed build and script, helper, quantizer, store and compression modules
are frozen before execution; source hashes are checked around each worker and
the complete cohort. The shared frontier lock prevents overlapping model/timing
cohorts. A 768 MiB memory admission and swap guard cover every fresh worker.

All nine workers completed, checking **1,225,211,904 weight bytes** with identical
weight/scale commitments and **zero observed swap growth**. Separate medians:

| Layout | Header-inclusive matrix frames | Encode CPU | Decode CPU |
| --- | ---: | ---: | ---: |
| Current level-1 zlib control | 121.50 MB | 5.360 s | 0.664 s |
| Bit planes + level-1 zlib | 134.11 MB | 6.225 s | 2.619 s |
| Zigzag bit planes + level-1 zlib | 122.67 MB | 6.855 s | 2.948 s |

The candidates increase bytes **10.38% / 0.97%** and both codec CPU windows.
**Reject these two layouts.** Keep the current codec; another lossless layout
would need a new bounded source-locked control before any serving promotion.
No whole-response performance benefit follows from exact decoding alone.

## 4. Publish role-specific load-ready weights

The response ledger excludes both providers' original BF16 checkpoint
distribution. A trusted offline publisher could instead deliver the exact
already-quantized matrices/scales needed by each role. Inference still receives
public client-boundary artifacts for delivery; Preparation needs only its body
stages. The publisher retains the original checkpoint and pays compilation and
publication costs.

The probe counts original safetensors bytes, W8 matrix elements, row scales,
and float32 local tensors directly from the pinned checkpoint headers. These
are raw payload estimates, not a new delivery implementation. Separate
operators each pay their own object distribution; a common source URL is not
shared network traffic. Existing source-resident providers save **zero bytes in
the current response counters**.

For this checkpoint, two original safetensors transfers total **1,976.20 MB**.
The projected role payloads are **496.07 MB Inference + 359.04 MB Preparation
= 855.11 MB**: **1,121.08 MB / 56.73% less source payload** before the extra
publication/distribution owners above. This cannot be subtracted from the
493.58 MB response ledger, which never charged those original downloads.

**Next gate:** trusted source/quantizer/body-bound artifact import, identical
compiled stage commitments and exact decoder execution. Measure an actual
two-provider cold bootstrap including manifests, tokenizer, publisher storage,
publication CPU and amortization horizon.

## 5. Matrix-aware compact correlation generation

The prepared output correction `W·r−s` occupies 139.06 MB in the control pair.
A real programmable pseudorandom correlation generator could potentially replace
some of this material with compact independently expandable party data, while
keeping Inference unable to reconstruct the input masks. Seed/setup delivery,
public tables and both parties' expansion would all be charged.

Even **free corrections** remove at most **28.17%** of these recorded bodies;
354.51 MB remain. Reducing client downloads alone by relaying results cannot
improve this all-link sum. Sending a normal PRG seed for `r` to Inference would
disclose the client's activation when combined with its online `x−r`.

The primary-source lead is Li et al., [Efficient Pseudorandom Correlation
Generators for Any Finite Field](https://eprint.iacr.org/2025/169), including
circuit-dependent and matrix-triple extensions. It does not implement PLLM's
exact power-of-two rings or arbitrary public-weight correction contract.
The executable screen records a minimal mismatch: `65535 + 1` is zero in
wrap16, but 65536 in the field of order 65537. The field/ring boundary cannot
be silently substituted. The paper's own revision also flags
[parameter-specific QA-SD attacks](https://eprint.iacr.org/2025/892), so scheme
selection requires a current assumption/parameter audit, not just seed-size
arithmetic. No upstream implementation is imported or executed.

**Next gate:** independently implemented, bounded matrix-correlation algebra and
explicit exact-ring conversion, followed by full material/compute accounting,
one-use ownership and independent cryptographic review. Existing subset-table
and small-field secret-replication results do not satisfy this contract.

## Replay and scope

```sh
target/frontier-sdk-env/bin/python -m pytest -o pythonpath= -q tests/test_all_link_frontier_probe.py
target/frontier-sdk-env/bin/python scripts/probe_all_link_frontier.py \
  --processes 3 --output docs/evidence/research-all-link-frontier.json
```

Input evidence: `research-prepared-study-qwen25-confirm.json` and
`network-algebra-qwen25-2026-10-06.json`. Only the latter's matching quantized-body
stage shapes are reused; its timings and workload are not pooled with serving
measurements. Output evidence: `research-all-link-frontier.json`. Incomplete
cohorts remain explicitly incomplete and are atomically checkpointed after each
worker. None of these screens establishes a general 10× all-link improvement.

The 21 focused checks cover all signed-byte codec values and unaligned planes,
an independent exhaustive subset oracle, history/pending-token row accounting,
both operators' publication sizes, directed-ledger conservation and rejection of
unreconciled or different-body placement inputs. Focused lint and whitespace
checks pass. This is an exploration of five ideas with one newly measured codec
gate, not five new end-to-end serving benchmarks.
