# Hundredfold frontier: executable probes

These five completed investigations target architectural changes around existing open
checkpoints. A hundredfold improvement is a hypothesis on a named metric and
workload, not an admission requirement: useful incremental improvements and
reproducible negative results both advance the frontier.

The starting implementation is `5b691b35f0d39115d0fa06cb74c5473819b95db2`.
The [previous five slices](existing-model-frontier.md) remain separate evidence.

## Investigation contracts

| Investigation | Intended leverage | First executable gate |
| --- | --- | --- |
| Encrypted resident decoder | Fewer client boundaries and less network traffic | A bounded encrypted composition or boundary improvement, with independent clear parity and complete key/context/ciphertext costs |
| Model-specific correlation compiler | Less repeated Preparation work and material delivery | Fresh separated correlations or an exact incremental table/ownership improvement at compiler-derived stage widths, with setup and expansion charged |
| Multi-inventory Preparation factory | Better accelerator utilization and less host CPU | Independently bound inventories batched against shared public weights, exact corrections, bounded ownership and explicit waiting/service costs |
| Durable exact-context graph | Less repeated prefix work across restart and branches | Authenticated sharing or delta publication with exact continuation, independent branch ownership, rollback and real storage accounting |
| Existing constant-state checkpoints | Better long-context state complexity | Source-locked recurrent state/operation execution and independent parity; task-quality qualification remains separate from resource geometry |

Checkpoint changes have distinct source, numeric and task-quality identities.
Hybrid decoders with remaining full-attention layers retain their growing KV
costs. Full-state, generated-prefix and pending-token provenance contracts remain
in force. No experiment trains a replacement model or reuses one-use material.

## Execution and measurement

Five isolated workers prepared implementation and lightweight functional checks in
parallel. Model imports, GPU execution and timing/RSS cohorts then ran serially
on the shared host under the existing exclusive frontier measurement lock and
memory/swap guard. These probes reused the installed native build.

Each probe records the actual runtime/build, probe source, fixture or checkpoint,
numeric contract, control, candidate, owner and measurement window. Reports keep
setup, public tables, keys, native/GPU copies, disk I/O and rejected work visible.
Fresh processes measure process peaks; payload counts remain a separate metric.
Unknown measurements stay unknown.

Whole-response execution retains all seven serving metrics: client RSS/CPU,
Preparation CPU/RSS, TTFT, N-1 decode throughput and all-link traffic. Operator,
state-storage, synthetic and source-inspection results keep their narrower scope.
GPU work and setup amortization cannot disappear into a host-CPU ratio.

Controls use the strongest applicable existing implementation. A repeat-workload
candidate must beat existing continuation/cache behavior on the declared restart,
branch or capacity case. A batching candidate charges its waiting time. A new
cryptographic construction needs its own assumptions and independent review;
arithmetic parity alone does not establish privacy.

Validated slices receive separate implementation/evidence commits. Integration
and independent confirmation follow measured results rather than the size of the
original aspiration.

## Measured gates

All five probes have committed implementation, fresh-process evidence and exact
replay instructions. Their scoped results do not establish a hundredfold gain or
whole-response improvement.

| Probe | Measured result | Integrated commit |
| --- | --- | --- |
| Encrypted residency | 8× fewer online frames; 1.50× including setup, one client/key context | `683d65b` |
| Subset correlation compiler | Full-shape issuance 151.80–353.22× slower than native SIMD; reject this representation | `1a033ab` |
| Preparation factory | Down/150 Metal service −5.77%, CPU −13.32%; higher RSS and modeled individual completion latency | `712f95b` |
| Exact-context graph | Complete store bytes −8.06%/−9.40%; more restart reads, no CPU/RSS gain | `a3cfe91` |
| Recurrent ownership | At 1,024 synthetic steps, loop CPU −18.20%, process peak RSS −14.97%; state size unchanged | `39a3d4a` |

### Encrypted resident composition

[Six fresh workers](../research/encrypted-residency-reference.md) evaluate two
encrypted residual transitions and a head under one client/key context. Eight-lane
SIMD batching reduces online frames from 14.75 to 1.84 MB (8×); including the same
23.93 MB setup narrows the ratio to 1.50×. Median paired client-online CPU falls
9.70×, while setup-inclusive process CPU falls 3.73×. The client share of setup CPU
is unmeasured. All 221,184 decoded/padding slots pass the bounded CKKS oracle,
with worst output error below 4.50e-7 and zero observed swap growth.

This is a same-client synthetic throughput improvement. Batch-fill delay,
checkpoint numerics, encrypted KV, private token feedback and bootstrap remain
outside the executable contract; no per-stream autoregressive speed or complete
decoder claim follows. Reproduction and exact costs are retained in the linked
reference and `docs/evidence/research-encrypted-residency-cost.json`.

### Preparation-owned subset correlation compiler

[Forty-five fresh workers](../research/correlation-compiler-reference.md) check
114,624 exact output values and time 180 additional fresh correlations at five
compiler-derived synthetic geometries. Native SIMD wrap32 and exact digit8 are
measured controls, with setup/import/snapshot ownership charged. Full-shape
subset issuance is 151.80–353.22× slower than SIMD; every measured geometry lacks
a finite steady-state setup-amortization horizon against either native control.
Four-bit whole-body tables project 2.86 GB before other owners. Wider table
admission and deliberate storage-based timing exclusions remain explicit.

This rejects the bounded Python/NumPy subset representation as an optimization,
not all possible correlation generators. Fresh mask separation survives; a
production private generator still needs reviewed keys, one-use ownership and
measured expansion costs. Evidence:
`docs/evidence/research-correlation-compiler-serial.json`.

### Multi-inventory Preparation factory

[Two hundred sixty-four fresh workers](../research/preparation-factory-reference.md)
check 200,851,456 exact corrections across native integer, native digit8, Metal
and adaptive controls, with zero observed swap growth. Three-repeat down/150 Metal
batching lowers service 5.77% and process service CPU 13.32%. Construction plus
service falls 2.75%, while process RSS rises 118.55 to 120.78 MB. Kernel/subtraction
time rises; mask/copy and encoding provide the scoped improvement. Mean individual
inventory completion rises despite lower aggregate service.

Separate one-sample 1 ms/10 ms arrival cohorts retain modeled fill/backlog waits.
Sparse arrivals create singleton batches; adaptive threshold crossing regresses
against separate CPU controls. Small inventories prefer native CPU. No production
batch policy follows without live arrival, delivery and complete-response evidence.
Evidence: `docs/evidence/research-preparation-factory.json` and separate
`research-preparation-factory-arrival-1ms.json` / `-10ms.json` follow-ups.

### Durable exact-context graph

[Sixty-six fresh pinned Qwen workers](../research/context-graph-reference.md)
compare resident immutable-cache continuation, existing shared-page full-manifest
tiered storage and authenticated delta manifests. All twenty-four durable
branch/restart trajectories match full logits and final KV hashes, with zero
observed swap growth. Generated Qwen2/Qwen3/Llama gates additionally check every
active KV word across branch divergence, repeated reopening and anchor rollover.

Complete ciphertext/file stores fall 8.06%/9.40% at 128/512 prefix rows; allocated
filesystem blocks fall 6.44%/7.76%. Restart reads rise 27.95%/8.79%, branch CPU
rises 2.41%/0.92%, and process RSS does not improve. Prefix pages were already
shared by the control. Both complete attention and full-ledger serialization
remain charged. This validates bounded durable metadata savings; resident reuse
remains the stronger in-process control. Evidence:
`docs/evidence/research-context-graph.json`.

### Existing constant-state contracts and owned recurrence

[Eighteen fresh timing workers](../research/constant-state-reference.md) execute
8,064 Qwen-shaped synthetic gated-delta transitions, with nine exact-bit final
output/state commitment pairs against the existing copy reference. A separate
fresh worker checks 44 pinned Torch-oracle rows for gated delta and independently
implemented stable weighted-exponential recurrence.

At 1,024 steps, median loop CPU falls 1.231705 to 1.007533 seconds (18.20%) and
process peak RSS 53,952,512 to 45,875,200 bytes (14.97%). Both retain 2,097,152
state bytes; candidate state/scratch totals 4,259,840 bytes. Avoiding per-token
copy/export churn supplies the gain. Imports, setup and oracle costs remain
outside loop timing, and no checkpoint weights execute. Pure RWKV/Falcon state
estimates and source/import/quality gates remain separate; Qwen3.5 is hybrid and
retains growing attention KV. Evidence:
`docs/evidence/research-constant-state-measured.json` and its pinned source audit.

## Integrated validation

The merged regression gate passes **124 checks** across all five probes, generated
decoder branches and existing tiered execution. It uses the dedicated pinned
native-wheel environment with the new internal context-graph runtime loaded from
the source checkout. Generated Qwen2/Qwen3/Llama branch checks pass again after
making the test's next-token ownership explicit. Focused Ruff checks pass.

All frozen file identities across the seven retained evidence reports match the
merged source checkout and actual installed dependencies. Documentation regeneration
passes 85 Node checks, five paper-builder checks and all 431 content-page checks.
The production docs build also passes, generating all 433 static pages.
The cohort-specific references retain exact commands, source hashes, scopes and
all separate timing samples; this index does not pool their numeric identities.

## Result-driven follow-up gates

The [SDK follow-up](../research/frontier-sdk-reference.md) integrates exact
recurrence pairing into ordinary semantic decoder execution, adds installed
recurrence/checkpoint/Preparation-batch targets, and exposes same-client SIMD
through `PackedEncryptedLinear(batch_size=...)`. Local state and batching owners
have explicit lifecycle/resource admission; historical probe ratios retain their
original measurement windows. The remaining gates below still govern serving
performance and protocol promotion.

1. **Owned recurrence:** measure the integrated model-neutral executor on pinned
   full decoders, checking complete logits/state and whole-client resources.
   Generated parity, borrow lifetimes and failure retirement are covered by the
   SDK integration gate.
2. **Preparation batching:** test arrival-aware admission and independently completed
   inventories on the existing protocol; include waiting, transfer and CPU/GPU copies.
3. **Context graph:** construct deltas directly and qualify histories beyond resident
   cache before claiming lifecycle CPU or memory savings. Preserve caller-head custody.
4. **Encrypted residency:** establish an existing-checkpoint finite encrypted region
   with required nonlinear/range semantics before extending synthetic throughput claims.
5. **Correlations:** require a credible lower-work representation before another
   optimization attempt; porting the current table loop alone is not an evidence gate.

These are follow-up gates, not uncompleted work in the five-probe screen. No probe
activates a new Pipeline component or changes an existing numeric default.
