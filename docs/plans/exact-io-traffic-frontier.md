# Exact-I/O traffic frontier

## Objective

Find substantial all-link traffic reductions for consumer Low-tier clients while
preserving the inputs and outputs of an unchanged existing-model control. The
user selected mathematical fixed-weight correlations, exact model adapters and
exact state/work reuse. Three Sol 6.1 workers investigate these tracks.

Lossy prompt compression, model distillation and task-quality agreement are
outside this investigation. A source-specific adapter may change representation,
but must reconstruct the original computation exactly. The historical learned
compression studies retain their original scope and evidence.

## Shared acceptance contract

1. **Same task.** Freeze checkpoint files, tokenizer/rendered input, numeric
   execution, workload bound, executed history and sampling configuration. A
   response's pending sampled token is not an executed state row.
2. **Exact reconstruction.** Preserve complete original logits, active state/KV,
   positions, selected outputs and sampler/RNG behavior. Alternative state needs
   an explicit invertible representation and exact continuation correspondence.
   Integer modular equality alone does not establish float32 decoder equality;
   real-arithmetic identities do not establish the original rounding order.
3. **Private execution.** Declare each role's view, security assumption and
   permitted disclosure. Plaintext private activations remain inside the trusted
   client. New topologies retain their own non-collusion contract. Fresh executions
   use fresh one-use material; cached computation never authorizes mask reuse.
   Software-private consumer execution is the target, without TEE dependence.
4. **Complete costs.** Count every directed link, including new worker traffic,
   setup, public artifacts, renewal, discarded material, misses and fallback.
   Separate online, first-client, recurring and source-cold lifecycles. Unknown
   quantities stay unknown; existing source-present reports do not price missing
   worker checkpoints or pre-positioned tokenizer distribution.
5. **Low client.** The target is 268,435,456 bytes for the complete client process,
   including SDK, transport, supervisor/relay where applicable, state, cache,
   verification, temporary overlap and any adapter. Logical storage and memory
   estimates cannot qualify measured peak RSS. External storage has bounded
   buffers, authenticated ownership and separately charged I/O. Keep at least
   90% of executed body linear work remote; report saved work and extra client
   expansion separately instead of hiding client work in that fraction.

For reuse, compare the same ordered input/request trace. For a new cryptographic
construction, identify whether the claim concerns input privacy, correctness,
malicious verification or only a toy algebraic property. A counterexample to one
construction is not a universal lower bound.

## Work ownership

| Track | Owned script/test stem | Research reference | Main question |
| --- | --- | --- | --- |
| Compact fixed-W correlations | `exact_correlation_frontier` | `exact-correlation-frontier-reference.md` | Can a specified generator remove correction delivery with cheap client expansion and bounded setup? |
| Exact existing-model adapters | `exact_adapter_frontier` | `exact-adapter-frontier-reference.md` | Can a lossless representation eliminate wide crossings without an expensive hidden nonlinear oracle? |
| Exact state/work reuse | `exact_reuse_frontier` | `exact-reuse-frontier-reference.md` | What additional work can be skipped beyond installed prefix, branch and restart reuse? |

Each worker owns its uniquely named script, tests, reference and new evidence.
The parent owns this plan, shared documentation, independent review, integration
and any source-loading or timing runs. Workers do not edit shared runtime/native
contracts, build/install wheels, commit independently or run serving cohorts.

## Required research deliverables

Each track examines three differentiated hypotheses, avoiding rediscovery of
previously rejected shortcuts. A useful result includes:

- A specified construction or sharply isolated missing primitive.
- A bounded executable positive control and independent oracle where feasible,
  plus adversarial negative controls for its key assumption.
- Exact role ownership, equivalence obligations and state/material retirement.
- A conditional all-link ledger and break-even threshold against the existing
  optimized control, not an intentionally weakened baseline.
- Explicit source/geometry provenance, rerunnable commands, source hashes and
  separately labeled synthetic versus recorded-checkpoint evidence.
- A concrete rejection criterion and next experiment or implementation gate.

Fresh 150+8 and 150+64 requests are separate analytical workload cells. Reuse
additionally distinguishes identical, divergent-branch, restart and unrelated
traces. No gains are multiplied across workloads, builds or private cohorts.

## Gates

1. **Construction:** exact algebra/state correspondence and honest role views.
2. **Budget:** complete specified costs leave a substantial net opportunity.
   Report the attainable ceiling even if it falls below a twofold reduction;
   an unsupported large target is not preferable to a sound smaller mechanism.
3. **Source feasibility:** independently validate checkpoint-specific assumptions
   with bounded metadata/source scans, after resource admission where necessary.
4. **SDK integration:** generated complete decoder and gateway parity, failure,
   replay and cancellation ownership; ordinary admission must price new owners.
5. **Serving evidence:** matched fresh-process controls, reconciled all-link
   bytes, all seven serving metrics, complete latency/CPU and largest client RSS.
   Confirm a winning screen independently before raising its evidence tier.

Small untimed synthetic checks may run independently. Checkpoint values, sustained
CPU/crypto work, native builds and serving cohorts run serially under the existing
host reserve, pressure/swap guard and exclusive frontier measurement lock.

## Existing evidence anchors

- [Fixed-W construction gate](../research/fixed-weight-correlation-gate.md):
  moving `W*r` to a tape or computing it at the client does not supply a cheap
  compact generator; public low-rank masks leak.
- [Correlation cartridge gate](../research/correlation-cartridge-frontier-reference.md):
  material, transform I/O and client expansion remain costs; existing correction
  removal ceilings alone do not imply a multi-fold whole-response improvement.
- [Installed exact decode reuse](../research/decode-frontier-sdk-results.md):
  first-use setup, authenticated local I/O, misses and fixed-epoch overhead remain
  charged. These installed capabilities are the comparator, not new proposals.
- [Certified row-codec screens](../research/private-boundary-row-codec-benchmark.md):
  initial 150+8 and 150+64 cohorts preserve exact state, but whole-traffic gains
  of 0.17% and 0.91% motivate mechanisms with larger scope. Their independent
  cohorts and source-present lifecycle cannot be pooled into a new control.

## Closure

The parent independently checks each construction, tests and evidence replay,
then publishes a decision table of surviving mechanisms, quantitative vetoes and
next gates. Bounded references establish no serving improvement or Low-tier
qualification by themselves. Validated major slices are committed with their
replayable evidence; historical records remain unchanged.

## Independent review gates

The first adapter reference passes independent bit-oracle and frozen-report
replay checks, but admits no complete model adapter. Its syntactic quotient,
exhaustive state lift and pre-cast rounding certificate remain separate bounded
constructions.

Two findings required revised reports:

- The installed prefix cache already publishes selected intermediate cuts through
  `put_prefixes`. Reuse comparisons must grant both arms those cuts, their actual
  admission rules and the same retention budget. The first sibling fixtures used
  already-selected boundaries, so their claimed incremental traffic savings are
   withdrawn. Corrected v2 grants both arms identical installed policy and reports
   zero incremental network savings in the screened scheduled-cut workloads.
   Actual Python cache differential checks cover synthetic block sharing,
   admission and retention. Identified v1 is preserved and rejected. An earlier
   stale replay reported by the auditor has not been recovered and is not claimed
   to be the same bytes as that preserved artifact.
- A public parity witness strengthens the sparse-noise gate. For public
  `A` with `n>d`, binary elimination gives a nonzero dual vector supported on at
  most `d+1` rows. With `t` uniform distinct noise coordinates and independent
  uniform ring values, its parity distinguishes chosen openings with total
  variation at least `binom(n-d-1,t)/binom(n,t)`. At the uncalibrated `d=128,t=27`
  sensitivity, this is at least 0.01406 for width 896 and 0.48301 for width 4,864.
   V2 tests the witness and rejects these parameters for every public A under
   the stated support/error law. The necessary noise-count thresholds 413/2387
   suppress only this lower bound; they are not sufficient security parameters.

Correlation headline budgets now use request-sized per-stage correction demand.
The fixed-256-row sensitivity remains separate. Known-payload cold controls are
352.77/483.27 MB at 150+8/150+64; the rejected sparse-image construction loses
both cold cells and retains unpriced complete costs. Correlation v1 has a frozen
source archive; reuse v1's unavailable source snapshot is explicitly distinguished.

The [consolidated decision](../research/exact-io-traffic-frontier-reference.md)
records all nine hypotheses, corrected comparisons and next gates. Integrated
bounded validation passes 119 tests plus 36 subtests and Ruff. No new serving
protocol, substantial incremental all-link improvement or Low-tier qualification
is established. The next mathematical experiment needs an exact, source-bound
small weight-circuit certificate; the adapter track needs a complete original
rounded-subgraph equivalence certificate. Reuse delta/receipt follow-ups concern
local CPU/I/O, not additional provider-traffic savings over installed full hits.
