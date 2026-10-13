# Growing correction epochs: bounded public-policy screen

**Verdict: useful conditional row/frame frontier, not demonstrated all-link win.**
Adaptive schedules are not admitted by current `PreparedInventory`; its
`decode_epoch_tokens` is a fixed integer in `[1,64]`. No production changes,
model loads, providers, native checks or new CPU/RSS/timing measurements here.
[Evidence](../evidence/research-adaptive-correction-epochs.json) hashes implementation,
tests, runtime/compiler sources and retained completed cohort. Arithmetic oracle
checks session splitting only; it does not establish installed SDK parity.

## Proposed legal public state machine

Choose immutable epoch sizes before request execution using public output cap and
declared public workload stop counts only. Non-final epochs do not shrink; final
epoch clips to remaining cap. No prompt inspection, token values, entropy/logits,
private prefix identity or private per-request stopping prediction selects policy.
Survival to a predeclared boundary advances its index; no online optimization.
Existing public shape demand still needs actual prefill length, as fixed epochs do.

1. Admit unchanged source, composition, i8 weight/activation numerics,
   `prefix_f32`, native full-KV continuation and full response memory cap. Acquire
   existing prepared queue seat for the entire response, including pauses.
2. While globally idle, prepare fresh per-stage masks/corrections, reserve once,
   open fresh authenticated provider session, validate exact continuation ACK.
   First epoch of size `e` prepares `P` prefill rows plus `e-1` decode steps.
3. Preserve runtime KV, logits, history, RNG, sampling parameters and application
   event sequence. Sample EOS before emitting; EOS is neither output nor executed.
   Cap's final selected token is emitted but remains pending/unexecuted.
4. At an exhausted epoch with cap remaining, the last selected non-EOS token must
   execute in the next epoch. Complete old session, close/drain channel, burn
   unused reserved rows, release online exclusion, snapshot state; only then
   prepare new material while idle. Keep queue seat. Later size `e` prepares
   `prepared_stage_rows(1,e-1,continuation=True)`, not another prompt prefill.
   Fresh session/key/nonce sequence and exact native handshake are mandatory.
   Restore snapshot and execute pending token exactly once, without resampling.
5. EOS/cap/cancel retires session/channel/material. Cancellation before reservation
   discards issued rows; after reservation burns unused rows. ACK mismatch,
   constructor/preparation failure cannot recycle material. Unconfirmed retirement
   closes further admission. No overlap/background Preparation while online.

Replace fixed runtime modulo boundary with cumulative schedule boundaries and
first/next allowances in both preflight and session rotation; simply mutating `E`
mid-response violates existing composition identity. New schedule must be public,
serialized and admission-validated. Runtime/provider numeric, stage shape/artifact
and source commitments remain unchanged. No claim this extension exists today.

Exact maximum executed-history bound is `initial_extent + cap - 1`, where extent
includes any admitted prior history/pending input suffix. Fresh `P=36, cap=64`
needs at least 99 positions; retained early cell compiles 128. Expected early EOS
never licenses smaller KV/cache/snapshot/scratch/retained-response admission.
Each measurement cell needs fresh SDK client, providers, session/material scopes
and cold bundle cache, with identical full-cap bounds for candidate and controls.

## Source-derived accounting and limits

`semantic_stages.prepared_phase_rows` counts native schedule steps, terminal-only
prefill rows and continuation rows. `model_binding.prepared_stage_rows` applies
one-row authorization minimum. This screen isolates the source counting function
without importing runtime, and regresses aggregate Qwen profile against all twelve
completed early/full control cells: 96 remote stage rows per decode; initial
prefill `93P+3`. Profile is fixture/aggregate-verified, **not independently compiled
adaptive schedule parity**; other source/composition profiles require re-derivation.

Let `K = n+1` for EOS after `n < cap` visible tokens, otherwise `K=cap`.
Let `B` be the first cumulative epoch boundary with `B >= K`, and `J` its index.

* Issued/reserved rows: `93P+3 + 96(B-1)`.
* Claimed rows: `93P+3 + 96(K-1)`; burned rows: `96(B-K)`.
* Successful admissions/completions: `J`; pauses: `J-1`.
* Preparation stage requests, logical correction bodies and correction ACK
  exchanges: `96J` each. These are logical messages, not TCP packets or physical
  WebSocket frames. `preparation_server.py` packs/pushes once per stage, waits for
  ACK, then returns preparation ACK. ACK bodies can be empty on HTTP paths;
  exchange count does not assert nonempty body size.

Boundary EOS matters: EOS after four visible tokens requires five selections.
First size-four epoch must rotate to execute token four before discovering EOS.
Retained EOS-after-three cell needs four selections and burns zero with E=4.
Probe exhausts every EOS position including zero, and cap stop, for caps 1–64,
every fixed E, four geometric starts and optimized schedules on caps 8/16/32/64.
Evidence retains case counts and deterministic ledger hashes rather than huge
per-stop tables. `account(...)` replays any individual stop.

Exact row-residue payload with known per-stage public layout is
`sum_s ceil(rows_s * sum(widths_s) / 8)` per epoch, per
`crates/pllm-core/src/offset_transport.rs::packed_size`; packing spans all rows.
Correction envelope additionally includes MessagePack metadata, compressed inline
width layout and server timing. Real Qwen width vectors/envelopes are not supplied
by retained aggregate report, so adaptive payload/envelope bytes stay **null**.
No division of aggregate bytes by rows masquerades as exact per-row coding.
Preparation request/ACK bodies, inventory authorization, continuation admission,
completion/cancel/status controls, bundle delivery, online arithmetic and teardown
must all be charged on every relevant link. Control/ACK bytes and all-link TCP
bytes for new policies stay **null**, not zero. Successful ledger excludes failure
traffic; failure controls test burn/discard/exclusion, not unknown byte totals.

## Explicit public priors and conditional policies

Evidence evaluates caps 8/16/32/64 under early/full mixtures 80/20, 50/50 and
20/80 (early means EOS after `min(3,cap-1)` outputs), plus equal count at every
EOS position and cap. Counts are synthetic public assumptions, not measured
population frequencies. Optimize exact count-weighted objective
`issued stage rows + lambda * pauses`, for `lambda = 0,96,384,1536` **row
equivalents**, not bytes, seconds or RTT. DP searches all monotone schedules;
fixed baseline searches every integer E up to cap, not selected powers of two.
Independent enumeration proves DP optimum on all caps 1–9 for every prior/penalty.

For `P=36, cap=64, lambda=384`:

| Public prior | Best fixed schedule | Fixed rows / pauses | DP schedule | DP rows / pauses |
| --- | --- | ---: | --- | ---: |
| 80% early / 20% full | E=8 | 5098.2 / 1.4 | `[4,60]` | 4791 / 0.2 |
| 50% early / 50% full | E=16 | 7095 / 1.5 | `[4,60]` | 6519 / 0.5 |
| 20% early / 80% full | E=32 | 8784.6 / 0.8 | `[4,60]` | 8247 / 0.8 |
| Uniform EOS plus full | E=18 | 7165.89 / 1.33846 | `[18,18,18,10]` | same |

For 80/20, geometric `[4,8,16,32,4]` also issues 4791 expected rows, but 0.8
expected pauses and four full-cap pauses, versus DP's 0.2 and one. Fixed E=4
issues the same expected rows but three expected pauses and fifteen full-cap
pauses. DP expected logical correction exchanges are 115.2, geometric 172.8,
fixed E=4 384. No decimal message count occurs on an individual request.

At lambda=1536, 20/80 prior selects full-cap `[64]`; geometric loses. At cap 8,
binary early/full optimum `[4,4]` is already fixed E=4. Growing policy is not
universally useful. For binary early/full, `[4,60]` beats `[64]` in row objective
only when early probability `p > lambda/(5760+lambda)`. This is not byte breakeven.

**Worst-case no-domination:** at full cap every legal partition issues identical
rows, while multi-epoch policy adds admissions/messages/pauses versus `[cap]`.
At an intermediate EOS a larger epoch can burn more rows than small fixed E.
Thus no nontrivial growing policy dominates every fixed policy at every stop.
Uniform prior's optimized fixed answer is a useful counterexample, not failure of
the conditional early/full result. Public-prior misspecification remains a risk.

## Measured control versus unproven target

Retained completed cohort (SHA-256
`ccd5469f114bcfe7c5000a8debdc80c66586deb507c38d15399b6bfa31a05c62`):
early/control issued 9399 rows, early/E=4 3639, claimed 3639 both; full/control
and full/E=4 issued/claimed 5697. Early E=4 measured TCP streams 266.096→213.389
decimal MB (19.81% reduction). Full 8-output E=4 throughput fell 39.70%.
Those bytes/times are **fixed-policy historical measurements only**. They do not
forecast growing-policy latency, RTT, CPU or RSS, even where initial rows match.

Target remains gated: exact native logits/KV/history/pending/RNG across every
boundary, EOS and cancellation; installed fresh-session continuation ACK and
idle-only Preparation; all-link cold-client measured traffic including ACKs and
controls; measured remote body work >=90%; whole-client lifetime peak <=256 MiB
at full cap with all retained owners charged. Existing measured client RSS does
not prove adaptive memory, and conservative allocation admission is not RSS.
No new remote-work fraction or peak-RSS measurement exists; both are null.

Parent must first implement/admit public schedule in a separately authorized SDK
change, then run serialized native controls and matched fresh-client cohorts
using existing frontier lock/host-pressure/swap guards. Existing fixed-epoch SDK
tests are baseline, **not an adaptive execution gate**. No native adaptive replay
command is supplied because no admitted native adaptive mode exists.

## Exact lightweight replay

```sh
target/frontier-sdk-env/bin/python -m pytest -o pythonpath=. tests/test_adaptive_correction_epochs_probe.py -q
target/frontier-sdk-env/bin/python -m ruff check scripts/probe_adaptive_correction_epochs.py tests/test_adaptive_correction_epochs_probe.py
target/frontier-sdk-env/bin/python scripts/probe_adaptive_correction_epochs.py --output docs/evidence/research-adaptive-correction-epochs.json
```

Stdlib-only deterministic probe; no private inputs, heavy imports, model loads,
providers, builds, installs, worker timing or overlapping resource measurements.
