# Phase-split public projection placement: bounded reference

## Outcome and limits

Implemented `scripts/probe_phase_split_placement.py`, independent tiny Cartesian
oracle, seven correctness tests, and source-hashed synthetic evidence. No SDK,
Rust registration, compiled executor, checkpoint, source values, timings, or
commits changed. This is an executable **selection/cost reference**, not an
implemented phase-split serving path or measured optimization.

Within the default declared seven-group public universe, split and fixed
ownership select the same four attention-output groups. **Incremental split
benefit is zero** for initial prefill, cold/exact-repeat, and continuation.
An independent synthetic case demonstrates a strict positive opportunity:
prefill owns terminal+dense groups while decode owns only terminal group. This
does not establish such a win for the source model or a latency improvement.

| Workload | Fixed net arithmetic payload saving | Split saving |
| --- | ---: | ---: |
| Initial 150 input, eight outputs | 2,401,280 bytes | Same |
| Cold plus exact prefix hit, eight outputs each | 2,652,160 bytes | Same |
| Cold plus 20-row continuation, eight outputs each | 3,368,960 bytes | Same |

These are **unframed projections relative to all-remote body stages**, not
measured application-body or wire savings. Each decode executes N−1=7 rows.
Selection is exact only within the seven named candidates, not all 96 groups.

Selected union costs 3,225,600 cold raw i8/float32-scale payload bytes and
6,451,200 explicitly conservative resident-owner bytes (immutable native i8,
one overlapping Python i8 copy, two scale owners). Its all-remote row removal
is 5,626,880 arithmetic bytes initially: cold pays 3,225,600 once; subsequent
seven-row decode adds 250,880 bytes avoided. One selected attention-output group
costs 806,400 raw bytes and avoids 8,960 arithmetic bytes per executed row,
requiring **90 executed rows** to break even in this narrow pricing model.
Decode-only placement therefore needs long/repeated output horizons; cold
distribution can dominate short requests. Reusing the union across phases
amortizes distribution without doubling weight charge, but still executes local
CPU work. No CPU or time win follows from the byte calculation.

## Public schedule and identity

Input is hash-locked retained config, 96 grouped stage shapes, weight/scale
digests, signed output bounds and archived directed ledger in
`docs/evidence/placement-frontier-screen-2026-10-01.json`. Native header-only
`lower_model` and `runtime_schedule` compile the retained config with existing
W8A8 `MaskedLinearCpu`, one CPU thread, terminal prefill pruning. No model
resolver or checkpoint is opened. A placeholder Model identifier carries no
architecture logic: groups derive from graph topology through
`semantic_stage_role` and `scheduled_stage_specs`, never model-name parsing.

Every body group must match retained dimensions/layer/role. Ordered operation,
input, weight and output-offset/width contracts must agree across phases.
Aliased body weights reject until explicit shared-owner normalization exists.
Numeric identity binds current plan, schedule, composition, ordered group,
source/checkpoint/body identities, signed bound and weight/scale digests. Existing
local bias, scale application, nonlinear, attention and state semantics stay
unchanged; no Q/K/V or gate/up sub-group ownership is introduced. Unsupported,
protected, ambiguous, or non-remote baseline schedules reject. Existing public
trust boundary stays fixed; newly distributed client weights require opt-in.

The historical 39-input/32-output directed ledger is conserved and retained as
**context only**. It is not divided by row count to invent phase measurements.
The separate all-link placement follow-up already measured fixed
`ClientLinearRoles(layer_roles=...)`: 38/96 groups, 90.07%/90.45% remote
prefill/decode, 34.33 MB additional payload, 229.11 MB largest reported client
peak, throughput improvement but slower complete requests. See
`all-link-frontier-reference.md`. Neither its timings nor its RSS qualify this
different phase split. Fixed roles, native CPU/Metal kernel selection and
`optimization_space` remain controls, not novel mechanisms.

## Planner, owners and fresh corrections

Each candidate has four legal research choices: remote/remote, client/remote,
remote/client, client/client. Native executability of the two split choices
remains explicitly false. Sparse exact multiple-choice DP includes separate
MAC budgets for **every prefill event**, ordinary decode per row, and union
resident owners. All 96 groups remain denominators, including noncandidates.
Integer `10 * local_macs <= body_macs` prevents rounding leakage. Exhausting
the configured state ceiling rejects rather than silently pruning to a beam.
Candidate ceiling is 12 internally, 10 at CLI; default seven also runs oracle.

Initial prefill uses one row for terminal-only groups. Continuation disables
that pruning, matching `prepared_phase_rows`; exact prefix hit uses zero new
prefill rows, not zero decode. Per-event prefill caps prevent mixing long cold
prefill with short continuation to hide a violation. Decode cap remains checked
even if a particular horizon has no decode rows, preserving placement legality.

Union client payload is charged once if either phase is client-owned. Providers
retain a group if **either** phase uses it remotely: provider union is the
complement of client/client, not the complement of client union. Replay shows
two-provider uncompressed raw payload floors separately. Existing full provider
checkpoint distribution stays fixed; unknown distribution/manifests cannot be
subtracted from the objective. Paged policy charges immutable shared file once,
two resident scale owners, page indices, and supplied I/O/kernel windows. Missing
paged window cost rejects selection. Default resident policy intentionally
charges overlapping copies rather than borrowing another build's sharing claim.

Unframed remote row price is `4*input_width + 2*ceil(wire_bits/8)*output_width`:
masked input, returned output and fresh Preparation→Inference correction.
Replay keeps prefill/decode online and offline corrections separate. Every
remote executed row has event/phase/stage/row-qualified one-use material;
ownership switches never reclaim or reuse previously consumed masks. Admission
burn-only minimum, control/seeds/manifests, transitions, failed attempts, actual
packed/residue encoding and framing remain unknown. Current row counts do not
assert a compiled ticket inventory implementation.

Client and Preparation CPU, transition CPU, phase latency, TTFT, throughput,
base CPU, complete all-link bodies/wire, and whole-client RSS remain `null`.
No unknown becomes zero. Optional supplied base/other owner costs only yield
an allocation floor; never measured RSS or tier qualification. Client head,
tokenizer, KV/cache, masks, transient buffers, stacks and overlap need complete
admission. Low/medium tier limits remain 268,435,456/2,147,483,648 bytes, qualified
only by largest matched fresh-process whole-client peak, with host headroom
separate. The 64 MiB extra-owner search budget is **not** a low-tier admission.

## Independent controls

1. Random tiny problems compare DP to independently enumerated four-choice
   Cartesian oracle, including both fixed and split policies and union memory.
2. Positive synthetic differing-phase choice and independent per-phase rejection.
3. Initial pruning versus continuation, and exact-hit zero-prefill lifecycle.
4. Resident/paged ownership, union once, provider retention, unknown costs and
   explicit state-limit rejection.
5. Tiny integer clear dot versus independently fresh masked/corrected modular
   dot through initial, suffix and hit phase switches. Exact signed reconstruction
   and unchanged post-dot float scaling/local bias agree. This does not test
   source-model quantization, full native logits/KV or cryptographic protocol.
6. Installed native public schedule/retained geometry consistency.

## Exact changes required before promotion

These are integration requirements, **not edits implemented here**:

- Add a versioned phase-qualified placement descriptor and native composition
  parsing/validation in `crates/pllm-compiler/src/lib.rs`, preserving old fixed
  descriptor/digests when absent. Validate each phase selector against real
  grouped semantic operations and explicit client-weight permission.
- In `decoder_runtime_schedule.rs`, pass different validated selector sets to
  the two `lower_phase` calls. Keep `linear_stage_signature` equality: it checks
  layer/operation/weight/output-width identity, not executor; do not weaken it.
  Check actual terminal initial prefill, full-row continuation and ordinary
  decode independently. Preserve grouping, state contracts and numeric ordering.
- Extend `network_placement.rs` cost/admission and Python planner bindings with
  union client/intersection fully-local provider semantics. Preserve bounded
  search, exact source/workload identity and fail-closed required unknown costs.
- Replace phase-blind `_client_owns_stage` dispatch in
  `python/pllm/runtime/transformer_engine.py` with schedule-selected ownership
  for initial prefill, decode and decode-path continuation. Source validation,
  `model_binding.py`, bundle descriptors/delivery, immutable native snapshots,
  GPU snapshots and paged authenticated owners must bind union client weights.
  Provider stage commitments/material must retain all remotely used phases.
- Update `semantic_stages.py` ownership/prepared demand and checkpoint-bound
  `preparation_protocol.py` demand together. Authorization, issuance, correction
  push, ticket indices and cancellation burns need phase-qualified bindings.
  Never expose or restore reusable masks/corrections with KV checkpoints.
- Extend `benchmark_memory.py`/decoder admission to price union snapshots,
  paged indices/files/tiles, phase buffers, transition overlap, in-flight inventory,
  scales, boundary/head/cache/tokenizer/stacks and import temporaries. Extend
  SDK/gateway resolution, source reload reset and bundle manifest checks only
  after native contract exists. Reject unsupported Metal/device execution rather
  than falling back while reporting phase split as implemented.
- Add per-phase measured CPU/latency and directed online/offline/cold ledgers;
  include distribution, fresh material, transitions and failures. CPU/Metal
  stage selection stays orthogonal, never evidence that split placement exists.

## Replay and serial gate

Small reference commands, no values/download/builds:

```sh
target/frontier-sdk-env/bin/python -m pytest -o pythonpath= -q \
  tests/test_phase_split_placement_probe.py
target/frontier-sdk-env/bin/python scripts/probe_phase_split_placement.py \
  --groups 7 --output target/phase-split-placement-replay.json
```

Output must not exist. Approximate reference envelope: one process, no children,
at most 100,000 DP states, metadata/native lowering only; reserve 512 MiB host
memory and 30 seconds. These are planning allowances, not measured serving
resource claims. Evidence binds probe/tests/checkout runtime/compiler/prior hashes
plus actual installed semantic/profile/native-extension hashes and exact native
schedule. No checkout shadowing for installed checks.

Parent serial promotion sequence under existing frontier lock/admission:

1. First implement changes above; freeze exact source/build/selector identities.
   Generated tiny native/SDK/gateway fixtures compare all logits and active KV
   bits across fixed control and split initial→decode→suffix→decode, exact hit,
   fork/restart, failure/cancel and zero-decode cases; resident and paged owners.
2. Source-bound, bounded three-prompt numeric checks on admitted pinned artifacts
   only. Record terminal-pruned initial and unpruned continuation MAC schedules,
   all weight/scale/bias commitments and fresh correction inventories. No values
   may load until parent grants serialized source-value gate.
3. New unique installed response probe after executor exists: three fresh clients
   per policy (all-remote, byte-selected fixed, phase split), each cold 150+8,
   exact-repeat 150+8 and exact 20-row suffix+8; include short/long-output
   amortization controls. Two provider children per client, one cohort at a time.
   Existing `benchmark_all_link_placement.py` is fixed-placement control only,
   **not a runnable split executor**. No honest split-response command exists yet.
4. Initially reserve existing fixed benchmark envelope (8 GiB whole-host cohort,
   900 seconds per bounded cell), and re-admit union/overlap owners before each
   source load. These are ceilings, not an 8 GiB client tier claim. Record largest
   fresh-process client RSS, client CPU, Preparation CPU/RSS, TTFT, N−1 decode
   throughput, complete latency, swap, every directed link, cold artifact/source
   distribution, paged I/O and failed work. Keep cold/base and per-token separate.
   Missing gate remains missing; no byte, tier or SOTA promotion from this note.
