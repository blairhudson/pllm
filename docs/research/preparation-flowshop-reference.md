# Bounded Preparation compute/upload scheduling

Investigation 4; research reference only. No SDK changes, checkpoint access,
provider runs, native builds, traffic/tier qualification or serving speedup claim.

## What changes, and what does not

`python/pllm/runtime/preparation_window.py:34` already issues up to four independent
jobs. Each client job includes remote compute, correction push, full ACK and local
verification import. Work remains reserved until acceptance. Source-order issuance
can block behind the next job when aggregate declared work will not fit. Finished
jobs are accepted in completion order, not source order. Client lines 2677–2687
seal only after all jobs return; exceptions cancel the remote inventory and every
locally accepted mask. Preparation server lines 505–553 consume the one-use stage
before compute and validate correction ACK identity/length before returning.
The HTTP handler permits an empty inference ACK body; this fixture instead
requires a positive full authenticated ACK. That is a fixture constraint, not a
claim that production has this stronger contract already.

Candidate changes **issuance permutation**, not width or overlap. The model gives
each retained slot independent compute, then one serialized upload/ACK lane. This
is a hypothetical bottleneck, not the source server's measured concurrency model:
current HTTP/channel concurrency, upload fairness, worker pools, native internal
threads and weight-cache interactions can differ. Source-order control uses exactly
the same model, byte budget and ACK ownership as each candidate.

Executable: `scripts/probe_preparation_flowshop.py`. Tests:
`tests/test_preparation_flowshop_probe.py`. No shared runtime edits.

## Bounds, lifecycle and arithmetic controls

- Width uses the existing 1–4 validation; budget uses existing 1–64 MiB validation
  (width one requires the original 16 MiB default). All jobs, including width one,
  must fit this reference's bound. This is deliberately stricter than production's
  width-one topology-admission exemption. Never split rows to bypass rejection.
- Work is the source helper's exact `4096 + rows*(16*in + 24*out)` plus authorized
  verification work when supplied through that helper. Default replay uses **no
  verification**. Every computing, queued-ready, uploading and awaiting-ACK job
  keeps its full work reservation and slot. Both strict lists stop at a non-fitting
  head; there is no secret bypass or uncharged result queue.
- `run_local` reserves each stage before worker submission. Complete binding
  includes session, fresh attempt, stage, plan, public weight digest, row/input/
  output geometry, ring and fresh seed commitment. Duplicate stages/attempts,
  mixed sessions, substitutions, replay and re-reservation are rejected.
- Receiver accepts only sequence `len(accepted)`; each HMAC ACK commits to sequence,
  complete binding, payload length and content under distinct push/ACK namespaces.
  Accepted completion order is its own contract: exact accepted-order seal, full
  expected stage set, and identical full ACK ledger are required for READY. It is
  not a license to alter source-stage IDs or later model execution order.
- Failure sets cooperative stop and burns receiver state **before joining**;
  all successful issued results, including accepted and late worker results, are
  burned. Pending jobs are canceled. Successful sender buffers are overwritten and
  released after ACK. A callback that raises must burn its own unpublished partial
  material, since caller cannot recover worker locals. Tests exercise this contract.
- Cancellation, total deadline, optional actual issuance-wait limit and actual
  per-stage ACK deadline are checked around blocking operations. Cooperative threads
  cannot enforce a hard deadline on a hung compute or push. Real gate requires
  killable owned processes and bounded network timeouts; this is an explicit limit.
- This is a test-local receiver with commitments, not a persistent correction
  inventory. It does not prove production authentication or party separation.
  Python overwrite/release is not cryptographic zeroization of allocator copies.
  Real receiver correction files, masks, weights, wire buffers, thread stacks and
  kernel workspace need separately charged owners; declared work is not RSS.
- Numeric fixture computes a two-row integer correction independently checked
  against handwritten `[-8,13,1,15]`. Fixtures use public deterministic masks only
  for arithmetic checking; fresh opaque bindings do not make these secret masks.
  Scheduling does not alter correction arithmetic, quantization, ring or row count.

## Schedule oracle and fairness

Event replay records issue/push/ACK times, accepted order, peak slots/bytes, upload
busy/idle time, stage deadline misses and issuance-wait violations. All replay jobs
arrive at zero. No arrival wait is silently folded into service. Full completion is
the last ACK, not first uploaded stage or TTFT.

`exhaustive` enumerates every issuance permutation for at most eight jobs and keeps
only deadline/wait-feasible schedules. Its optimum is **restricted to strict eager
list issuance and FIFO compute-completion upload**, not unrestricted global
scheduling with intentional idle time or arbitrary upload reorder. An independently
implemented integer-tick oracle checks 1,152 tiny schedules across widths 1–3 and
two memory envelopes, plus all six heterogeneous-work permutations. No classic
two-machine Johnson optimality theorem is claimed: parallel compute and slots held
through ACK invalidate that theorem. The Johnson-inspired priority is only a
heuristic; use exhaustive results as a small-screen oracle, never large-plan proof.

| Tiny synthetic fixture, width 2 / 2 MiB | Source | Heuristic | Meaning |
| --- | ---: | ---: | --- |
| `(compute, upload) = (1,1),(2,1),(2,2)` | 6 | 5 | Earlier upload feed lowers idle time |
| `(1,1),(2,1),(1,1)` | 4 | 5 | Reordering delays longer compute |
| `(8,1),(1,8),(2,4),(7,1)` | 18 | 21 | Source already hides long compute |

Times above are arbitrary **simulated** units, not measured seconds. Adding stage
1 deadline 3 to the second fixture makes heuristic infeasible while source meets
it. Requiring every issue by time 0 with three jobs/two slots is explicitly
infeasible; issue limit 2 admits a schedule. Finite closed lists cannot starve under
positive finite service: conservative issue-wait bound is the sum of all compute
and ACK-lane durations. This is not fairness for unbounded arrivals. A real online
scheduler must enforce per-inventory maximum age/deadline, reserve capacity or
age priorities, and reject infeasible admission rather than silently waive limits.
Late deadlines, underestimated service, bandwidth changes, memory head blocking,
compute contention, already-saturated uploads and cache locality can all erase or
reverse gains.

## Prior measured phases, explicitly simulated transport

Replay reads `docs/evidence/research-preparation-factory.json`, separate-layout
digit8 samples, three fresh processes per stage/row count and four inventory phase
records per process. Geometry comes from that configuration/compiler-derived
evidence: Qwen2.5-0.5B QKV `[1152,896]`, gate/up `[9728,896]`, down `[896,4864]`.
Weights were public synthetic i8, **not checkpoint values**. Two hypothetical
layers produce six independent jobs; this is not a complete model inventory.

Compute surrogate is median prior mask+kernel time. ACK-lane surrogate is median
prior authenticated encoding/receiver decode/private-file I/O plus encoded bytes
divided by **simulated** bandwidth plus **simulated** full ACK latency. Original
encoding measurement is in-process, not measured network service. The additive
decomposition, concurrency and phase medians are modeling assumptions; shader/
weight construction, request/authorization transfer, verification, seal, network
stack and compute-contention costs are not newly measured. Metadata and all prior
source provenance are retained in evidence; no private IDs/seeds/payloads are emitted.

| Rows | Work budget | Simulated B/s / ACK s | Source s | Heuristic s | Exhaustive best s |
| --- | ---: | --- | ---: | ---: | ---: |
| 16 | 16 MiB | 10,000,000 / .001 | .168263 | .168263 | .168263 |
| 16 | 16 MiB | 100,000,000 / .010 | .086546 | .086546 | .086546 |
| 150 | 64 MiB | 10,000,000 / .001 | 1.554762 | 1.580699 | 1.512490 |
| 150 | 64 MiB | 100,000,000 / .010 | .343599 | .349685 | .337268 |

Every time in this table is a **simulated replay**, not a flowshop measurement.
Each oracle enumerates 720 permutations. Best modeled large-row gains are 2.72%
and 1.84%; heuristic instead regresses 1.67% and 1.77%. Sixteen-row control has no
gain. Source/heuristic/oracle peak declared work at 150 rows is
64,681,984 / 66,981,888 / 66,981,888 bytes in the first transport model, and
64,681,984 / 66,981,888 / 58,380,288 in the second: same **cap**, not necessarily
same observed occupancy. No larger cap is granted to candidate.

Per-stage declared work is 675,840 / 3,969,024 / 1,593,344 bytes at 16 rows and
6,301,696 / 37,175,296 / 14,903,296 at 150 rows. The large gate/up job cannot fit
the default 16 MiB concurrent window; evidence uses an explicit existing 64 MiB
configuration. Do not infer that longer contexts fit: source tests price 295-row
gate/up at 73,107,456 bytes, beyond even that cap.

Prior correction-stripe research already demonstrates full ACK-before-seal,
ordered frame credits and cancellation cleanup. Its eight-credit digit8 local-pipe
gain (7.46%) is not this stage-reorder gain and cannot be added to it. Each reorder
keeps its prior stage's row/stripe/credit identity; transport credits remain a
separate experiment.

## Replay and evidence

From repository root:

```sh
target/frontier-sdk-env/bin/python -m pytest -o pythonpath= -q tests/test_preparation_flowshop_probe.py
target/frontier-sdk-env/bin/python scripts/probe_preparation_flowshop.py --output docs/evidence/preparation-flowshop-synthetic.json
```

These bounded commands use stdlib scheduling and small correctness threads only;
no extension import, native GEMM, checkpoints, model scans or transport cohort.
Estimated admission: 128 MiB host-process reserve, up to four tiny worker threads
for tests, zero model weights, zero GPU, zero network bytes. Estimate is not measured
whole-client RSS. Evidence contains source/probe/test/note hashes and prior data
hashes. Output JSON's own SHA-256 must be recorded separately to avoid self-reference.

## Real transport/cost gate and precise safe parent experiment

**Decision: do not promote heuristic.** A small modeled upper opportunity exists;
real Preparation completion and TTFT remain unknown. Current next gate is an owned
process adapter to actual stage compute plus actual authenticated correction
transport. `run_local` is not that adapter; no runnable native/provider adapter is
claimed or smuggled into the replay command.

Parent can safely prepare the gate as follows, under existing frontier lock,
host-headroom admission and swap watchdog, serially with other heavy experiments:

1. Run the two commands above to freeze replay. Take source, heuristic and oracle
   issue lists from the 16-row evidence first. Pin installed build, configuration,
   shapes, row count, kernel, verification policy, source weight identity and
   stripe/credit choices. Use six fresh independent attempts per arm; never share
   seeds or correction material between arms. First reject any stage dependency
   other than public immutable weights. Each arm uses width 4 / 16 MiB, exactly
   the same existing source work formula. The model's 150-row follow-up is a
   separately admitted width 4 / 64 MiB case, not a silent budget increase.
2. In a **test-only isolated parent harness**, pass each frozen permutation to
   existing `run_preparation_window`, preserving full existing `prepare`, `accept`,
   `abort`, `discard` and `cancelled` semantics. Do not patch SDK or production
   client/server. Bind a fresh authorized inventory and preserve source stage IDs;
   only the list passed to the window changes. Receiver seal must check every
   authorized correction and all ACKs. Use local pinned synthetic public weights
   first; checkpoint/native loading requires separate parent approval. Keep
   original model execution order untouched.
3. Start separate owned Preparation and receiver processes on loopback TCP with
   actual existing codec, authentication and bounded HTTP/channel timeouts.
   Exercise the real push→ACK→seal path, not `LocalLedger`. Start with one
   correctness sample per arm; inject one compute failure, bad/replayed ACK,
   canceled upload and premature seal. Every owned process must drain or be killed,
   every partial inventory burned, no READY on failure. Keep hard process timeout
   30 seconds and stop entire gate on pressure, swap growth or identity drift.
4. Parent admission should conservatively reserve **1 GiB total host headroom** for
   the initial tiny shape-only topology, in addition to normal system headroom;
   charge CPU immutable weights, installed kernel workspace, all masks, endpoint
   buffers, six receiver corrections, four worker stacks and process baselines.
   This is an estimate, not a tier assertion; installed memory planner may require
   more and takes precedence. At 16 rows the six i64 receiver payloads alone are
   3,014,656 bytes, and unique public i8 shape weights total 14,106,624 bytes before
   replication. Abort rather than downgrade admission. No retries under pressure.
5. Only after correctness, parent authorizes three fresh-process replicates per
   arm in alternating/source-balanced order. Measure per-stage mask/kernel/encode,
   upload start/end, full ACK, accepted ledger and seal; actual simultaneous
   computing/ready/upload bytes and slots; Preparation/receiver/client CPU and
   peaks; authorization/request/correction/ACK/seal bodies and TCP/TLS overhead
   separately. Do not sum separately sampled endpoint peaks as topology RSS.
   Compare source against **same-cap** strongest feasible candidate and keep losses.
6. If actual transport and contention support a repeatable completion improvement,
   authorize matched fresh complete-response provider samples separately. Preserve
   exact numeric identity and >=90% remote body MACs separately for prefill/decode.
   Report cold/base and output-dependent costs, client and Preparation CPU/RSS,
   TTFT, N−1 decode throughput and all-link traffic. Low/medium qualification is
   maximum whole-client replicate peak <=268,435,456 / <=2,147,483,648 bytes;
   conservative admission and host reserve do not substitute for those samples.

Real concurrent compute time, network arbitration, weight locality, actual
Preparation/client RSS/CPU, complete-inventory completion, TTFT, decode throughput,
physical wire and all-link costs are **unknown**. No measured serving or SOTA result
follows from this investigation.
