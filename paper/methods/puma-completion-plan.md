# PUMA common-Qwen completion plan

Status: approved for implementation, including the protected checked-range
profile, with Sol 6.1 subagents on independent workstreams. Baseline: `5dd11004` and
[local-v11 validation](../artifacts/puma-sdk/local-v11/README.md).
Completion still requires the evidence gates below.

## 1. Concrete finish line

Finish the D11 **public-weight** adaptation on the pinned
`Qwen/Qwen2.5-0.5B-Instruct` source used by `examples/papers/common.py`:

1. Admit the complete 24-layer decoder, full vocabulary and all eight selections
   under the shared **64-input/eight-output envelope**. Four inputs/two outputs
   are the first execution gate, not the completion workload.
2. Execute through the ordinary three-worker SDK, gateway and CLI. Retain D6/D10
   Q18 arithmetic, D9 full-prefix recomputation, private feedback, stable argmax
   and final client-only token IDs.
3. Establish independent full-decoder integer correctness, including stochastic
   truncation, and failure/cancellation retirement. Final-token agreement alone
   does not establish hidden-tensor correctness.
4. Run the shared paper inputs through ordinary quality and response benchmarks,
   with three fresh-process performance replicates and complete role costs.
   Report quality failures and poor performance as results.
5. Publish the exact Experiment, build, source, workload, accepted/rejected counts
   and replay through the existing evidence machinery.

This qualifies an implemented, measured source-derived public-weight adaptation.
Private-weight full-Qwen capacity, the unavailable original evaluated graph,
representative task quality and independent cryptographic review have separate
qualification requirements. They cannot be inferred from these five checks.

## 2. Audited baseline blockers, including those after RMSNorm

| Blocker | Evidence / implementation at `5dd11004` | Required outcome |
| --- | --- | --- |
| Numeric admission | Public 4+2 source publication rejects selection 0, layer 1, step 29, RMSNorm; propagated Q18 residual interval is `[-343639410012, 343639400796]`. No worker starts. | One complete-model range strategy, validated across every layer and selection. |
| Lookup work | `public_lookup_work_floor` in `replicated_decoder/program.rs` prices 1,187 128-lane predicates per token, each using 28 ordinals. | Reduce actual execution costs and establish an admitted complete-program budget. |
| Frontend capacity | `replicated_decoder/sdk.rs` permits at most 32 input tokens; `lower_with_weights` permits at most 40 input-plus-output tokens. | Price and validate the shared 64+8 envelope across SDK, compiler, workers and memory admission. |
| Native capacity | 1,048,576 complete-program ordinals, 3,224,371,200 per-party AES blocks, 65,536 nodes, 64 MiB program storage and 1,048,576 values per secret owner are separate limits. | Check all limits, not only the first rejection. Every capacity change needs its own priced, tested envelope. |
| Weight execution | `public_source::multiply` scans immutable i64 weights with a 64 KiB buffer. Lookup invokes it separately for each token. | Bound and measure public file reads, both replicated-component products and live output owners. |
| Peer execution | `replicated_worker.py::run` uses HTTP advance/prepare/dispatch barriers; there is a 1,200-second attempt lifetime and a separate barrier-loop ceiling. | Bounded peer scheduling whose overhead and lifetime fit the intended request. Ordinals are not HTTP calls or network rounds. |
| Paper recipe | `examples/papers/puma.py` currently selects the default private-weight protocol with the shared 64+8 budget. | Explicitly select the qualified public-weight composition in the paper factory. |
| Correctness and measurement | Generated SDK controls pass; no complete pinned-Qwen execution, hidden-tensor oracle or matched response result exists. | Finish the full-source conformance and ordinary benchmark gates below. |

### Lookup-only cost calculation

The baseline schedule processes `sum(input_rows + selection)` lookup rows under
D9. With vocabulary 151,936 and width 896:

```text
tiles_per_lookup = 151936 / 128 = 1187
ordinals_per_lookup = 1187 * 28 = 33236
table_bytes = 151936 * 896 * 8 = 1089077248
```

| Geometry | Recomputed lookup rows | Lookup ordinals | Current ordinal ceiling multiple | Logical lookup file reads per worker |
| --- | ---: | ---: | ---: | ---: |
| 4+2 | 9 | 299,124 | 0.285× | 9.80 GB |
| 16+2 | 33 | 1,096,788 | 1.046× | 35.94 GB |
| 64+8 | 540 | 17,947,440 | 17.116× | 588.10 GB |

The 64+8 row is a hypothetical geometry extension of the current implementation;
the current frontend rejects it earlier. These are schedule calculations, not
serving measurements. Logical file reads are not physical disk traffic or network
bytes. At 64+8 the lookup additionally entails 147,025,428,480 ring64 multiply-adds
per worker across its two replicated components, before decoder body or head work.
Reusing a file scan can reduce reads without removing those products.

### Implementation update

The explicit checked-range, tiled lookup/selection and persistent-peer composition
has a coordinated installed wheel. Bounded native guards preserve valid operands,
retain sticky shared validity and mask all final tokens on failure. Generated
integer/AES oracles cover complete tensors and private feedback. The source-only
pinned 4+2 attempt advances to layer-eight attention before rejecting its product
bound; no full-checkpoint worker execution follows from this attempt.

P0's [complete accounting](puma-feasibility.md) finds a **no-go under the current
complete-request ceilings**, including node/storage and protected-work limits.
Compact equality reduces a measured 128-lane predicate from 69,768 to 19,920
encoded bytes, but the complete 64+8 graph still fails its directed-traffic floor.
These figures are native primitive measurements and public schedule calculations,
not whole-response results. The original engineering estimate remains conditional
on a successful revised feasibility gate.

Remaining implementation work must address the complete arithmetic preconditions,
bounded body execution with real live-owner pricing, and repeated private
comparison/conversion work. Shared templates may reuse public instructions; they
must never reuse MPC material. Capacity extensions require a separately priced
and tested envelope, rather than treating a larger constant as qualification.

## 3. Ordered delivery plan

Estimates are focused engineering days, not promised elapsed time. Budget
**8–13 days total if the feasibility gate passes**, with the first continuation
decision after one day. Host admission, checkpoint runs and review can extend
elapsed time.

| Milestone | Estimate | Dependency | Exit artifact / gate |
| --- | --- | --- | --- |
| P0. Complete cost and blocker inventory | 0.5–1 day | Existing implementation | Costed go/no-go decision for 4+2 and the shared 64+8 envelope. |
| P1. Complete-model numeric admission | 2–3 days | P0; approved checked-range contract | All layers and selections have checked primitive domains; positive and negative independent-oracle tests pass. |
| P2. Bounded bulk execution and capacity | 3–4 days | P0; P1 supplies the final numeric schedule | Complete 64+8 work/storage/transport admission; measured component scaling and generated three-worker regressions. |
| P3. Pinned full-Qwen conformance | 1–3 days | P1 and P2 | Actual 4+2, then 16+2, then boundary-workload execution with complete correctness and lifecycle evidence. |
| P4. Shared quality and response campaign | 1–2 days | P3 | Standard CLI reports, repeatable Experiment and updated capability coverage. |

P2's lookup and transport work can be developed against generated fixtures while
P1 is under review. Full-checkpoint execution waits for both.

### P0 — decide feasibility before more layer-by-layer repairs

Extend the existing native schedule/resource diagnostics and ordinary benchmark
reporting to expose:

- Ordinals, AES/PRSS consumption, primitive invocations and live tensor/program
  owners, separately from peer rounds, frame bytes and HTTP control exchanges.
- Lookup, body linear, RMSNorm, softmax, SiLU/multiply, complete head and protected
  argmax costs for every full-prefix selection.
- Public source import/snapshot storage and logical file reads, charged to each
  owner. Distinguish shape-derived lower bounds, conservative estimates and actual
  measurements; failed range admission cannot mint an executable plan.
- All known workload, source, value, node, program, work, queue and timeout caps
  in one report, rather than discovering them through successive wheel builds.

Use admitted generated cases through shared benchmark facilities to measure native
work and the existing three-party peer path. Add any missing measurement facility
to the SDK rather than a PUMA-only benchmark runner. Keep extrapolations explicitly
separate from measured full-model results.

**Decision:** identify a bounded implementation route for both 4+2 and 64+8, and
price its dominant CPU, communication and memory costs. Screen directed byte
floors under the study's shared 100/40 Mbps access caps against the current
1,200-second attempt lifetime. A known lower bound beyond that lifetime blocks
the current execution contract; unknown dominant costs do not count as a pass.
Report that decision before spending the remaining implementation budget.

### P1 — resolve the entire numeric graph once

First assemble a per-operator range map for all layers. Timebox a source-only proof
attempt to half a day within P1, using existing row/energy certificates, including
the joint gate/up and down-projection residual. Its goal is a complete graph result.

If those proofs remain insufficient, the approved completion route is a separately
identified **protected checked-range profile**. The subsequent instruction to
proceed with the complete plan approves this contract:

1. Freeze public source/workload-bound ranges in an immutable component/artifact.
   Select them from public constraints and public development data, never a private
   request. Workers validate the same profile and source commitments.
2. Before any operation whose safety relies on a range, evaluate membership inside
   MPC. Comparisons and selection themselves must be safe over the entire incoming
   domain, including failure cases and subtraction/wrap endpoints.
3. Privately select a proven-safe fallback operand for an invalid value **before**
   unsafe multiplication, accumulation or truncation. Keep a sticky shared validity
   bit. Checking an already wrapped result or merely rejecting at the end is
   insufficient.
4. Execute the same public schedule in valid and invalid cases. Only the client
   receives aggregate validity at the final acceptance boundary. Invalid requests
   return no model tokens or reusable state; no intermediate failure location or
   validity is opened to workers. Account for the client's final success/failure
   disclosure in the new contract.
5. For accepted requests preserve the approved Q18 equations, rounding law and
   tails. Charge fresh guard products, comparisons, frames and PRSS work under a
   distinct execution identity. Extra draws can change a particular stochastic
   realization; do not claim old/new bit parity without a coupled-tape argument.

Independent tests cover all boundary predicates, failed guards before unsafe work,
sticky failure through later layers, final acceptance and cancellation. Public
held-out range failures remain failures in quality reports; they cannot be dropped
or used to retune the profile retrospectively.

**Gate:** native numeric analysis validates all 24 layers and every requested
prefix, including final normalization/head/argmax, with fully proved primitive
domains. Diagnostics must not bypass resource checks to issue a runnable program;
complete source-publication admission requires P2 as well.

Primary files: `crates/pllm-core/src/puma/{tensor.rs,tensor_numeric.rs,work.rs}`;
`crates/pllm-compiler/src/replicated_decoder/{program.rs,source.rs,sdk.rs}`;
the existing Python protocol component, publisher and final-response acceptance.

### P2 — fix execution granularity, then justify capacity

Implement and measure these changes in order:

1. **Bulk private lookup:** add bounded public row/vocabulary tiles to the native
   executor. Form fresh shared predicates, stream public weights across several
   rows and accumulate complete exact embeddings. Keep complete vocabulary access
   independent of private IDs. Price selector, accumulator and file-buffer overlap.
   Preserve D9's recomputation; this is no cross-selection prefix cache.
2. **Predicate and local-operation batching:** fuse reusable local Boolean maps
   and batch independent primitive phases where the admitted payload permits.
   Prove any narrower predicate width from the public token domain. Preserve
   source PRSS fill semantics and fresh material. Measure CPU/AES/bytes/rounds as
   well as ordinals so a changed accounting granularity cannot impersonate a win.
3. **Bounded peer streams:** replace coordinator HTTP round trips per primitive
   with authenticated long-lived peer delivery and native phase/ordinal readiness.
   Retain bounded queues, backpressure, ordered acceptance and whole-attempt
   retirement. Slow peers must not receive a later phase before its prerequisites.
   Reuse role supervision, credentials and accounting; no second public runtime.
4. **Complete-program capacity:** derive the remaining node, value, storage, AES,
   sequence, frame and lifetime needs from the final schedule. If full-prefix
   schedule metadata duplicates dominate, use bounded public iteration/templates
   while charging every fresh selection. Extend only demonstrated resource
   envelopes, consistently across native admission, SDK, workers and benchmarks.

The 32-input/40-total frontend limits cannot simply be changed to make a failing
benchmark start. Nor can the request be split into independent budgets to erase
its complete-program cost. A larger envelope requires bounded allocations,
checked counters, measured work, failure tests and a distinct capacity contract.

**Gate:** the 64+8 envelope passes every capacity check; measured generated cases
reconcile schedule and transport counters; replay, cross-session substitution,
slow-peer, expiry and cancellation tests retire every owner. If the measured route
cannot fit the chosen request/resource budget, stop and record the blocker before
launching expensive checkpoint runs.

Primary files: native `replicated_decoder/{program.rs,runtime.rs,public_source.rs}`,
`puma/{tensor.rs,work.rs}`, the replicated Python worker/transport, shared role
supervision, `benchmark_memory.py` and `benchmark_targets.py`.

### P3 — actual checkpoint execution and independent correctness

1. Extend the existing independent stochastic-integer oracle to the complete
   semantic decoder, including streamed public products, attention, normalization,
   rotary, guarded activation, head and stable argmax. Do not call production
   arithmetic helpers from the reference.
2. In a test-only harness reconstruct party shares and compare every admitted
   layer boundary, complete logits, full-prefix attention K/V, selected IDs and
   feedback history against the independently replayed randomness. Exercise both
   generated adapters and the pinned checkpoint. This is conformance access, not
   a production hidden-tensor export or persistent-KV contract.
3. Run the ordinary three-process installed-wheel path at 4+2, then 16+2, then the
   64+8 boundary. Freeze public cases first, include nonzero/nonconstant data, and
   retain failures. Verify final-only disclosure through SDK and gateway.
4. Inject malformed/missing peer frames, stale sessions, mid-operation and
   final-acceptance cancellation, invalid ranges and partial source publication.
   Verify native buffers, authenticated files, pending replies and role processes
   retire before admission capacity is refunded.

Build/test instrumentation has a separate identity and cost from serving samples.
Final production checks use the installed wheel, source pins and ordinary role
supervisor. Host admission and swap monitoring remain active throughout.

### P4 — finish through the shared paper tools

- Make `examples/papers/puma.py` explicitly select the qualified public-weight
  component and any approved range/capacity profile. Keep its model, public inputs
  and workload envelope sourced from the shared paper files.
- Execute `pllm benchmark quality` on the shared prompts and same-token positions
  against the existing streamed FP32 reference. PUMA supplies final-token agreement
  through fresh one-selection requests; top-k and logit-error metrics stay unknown.
  This is a fidelity diagnostic, not a retrospective task-quality qualification.
- Execute `pllm benchmark run` with frozen shared cases and three fresh-process
  replicates. Include publication, client and three-worker CPU/RSS, private files,
  directed peer and client traffic, startup-through-cleanup time and time to final
  accepted output. Buffered output supplies no streaming decode-TPS measurement.
  Classify observed client memory from the largest complete-client replicate peak;
  topology RSS uses simultaneous samples, not a sum of independent lifetime peaks.
- Use one matched invocation/cohort for controls where supported. Retain different
  numeric, topology and acceptance identities and the comparison API's ranking
  restrictions. Source-present delivery, TCP streams and physical wire keep their
  separate scopes.
- Validate the exact published commands, standard replay and benchmark coverage.
  Update `IMPLEMENTATION-STATUS.md`, the PUMA contract and paper recipe links once.
  Use the shared evidence registry and existing build receipts, rather than a new
  runner or archive system for each admission error.

## 4. Completion control

Commit validated milestones, with focused native tests during development and one
installed-wheel regression pass per completed integration slice. Do not rebuild
and archive a full campaign merely because another individual layer admits.

P0 produces the first firm continuation decision. P1/P2 failure produces a precise
blocked status and the remaining work estimate; it does not silently narrow the
common workload, weaken arithmetic, or reset the agreed effort budget. P3 failure
is a correctness/lifecycle blocker. P4 can legitimately finish with poor measured
quality or speed, but those results cannot qualify a quality-matched or practical
performance claim. Completion is recorded only against the five checks in Section 1.
