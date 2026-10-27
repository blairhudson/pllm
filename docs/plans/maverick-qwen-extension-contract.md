# MAV-QWEN-CONTRACT-JOIN

## Status and source anchors

Contract proposal for independent review, based on Root `23d691c6`.
Common-decoder extension direction is user-approved; this document implements
no Rust, source acquisition, preprocessing, verification or trained execution.
It is not a new FP8 interpretation, tolerance, security certificate or GKM route.

Source-backed premises (links identify existing definitions, not new authority):

1. [Model-neutral operations, graphs and DecoderPlan](../../crates/pllm-models/src/lib.rs#L250):
   operation IDs, producer inputs, shapes, attributes, prefill/decode and state edges.
2. [Existing declarative dense lowering](../../crates/pllm-models/src/lib.rs#L1530):
   Qwen2 has QKV bias and no Q/K norm; Qwen3 has the opposite declarations.
3. [D31/D32 exact numeric map](../research/maverick-numeric-source-recovery.md#32-exact-scalar-and-matrix-rules)
   and [original local schedule](../research/maverick-numeric-source-recovery.md#33-full-local-schedule).
4. [RAW retained source owner](../../crates/pllm-compiler/src/verified_source/raw_owner.rs#L37)
   and [allocation-free codec limitations](../../crates/pllm-compiler/src/verified_source/raw.rs#L1).
5. [Existing publication admission](../../crates/pllm-compiler/src/delegation_publication_inventory.rs#L440):
   actual AllocationPlan, ordered matrix/full-head coverage, explicit blocked status.
6. [Atomic request verification](../../crates/pllm-core/src/maverick_batch/request.rs#L21)
   and [D45 scope](../../paper/methods/DECISIONS.md#d45-maverick-atomic-two-batch-native-composition).
7. [Common checkpoint pin](../../examples/papers/common.py#L6):
   Qwen/Qwen2.5-0.5B-Instruct at `7ae557604adf67be50417f59c2c2f167def9a775`.

## Validated method contract over the complete graph

Proposed validation consumes the genuine immutable `DecoderPlan`, not a rebuilt
eight-operation template, model-name switch or caller-supplied semantic digest.
Both graphs must account for every operation, edge, state input/output and terminal
output. IDs are unique; each input resolves to an earlier producer or explicitly
declared graph/request/state input. Reject missing, foreign, cyclic or reordered
dependencies and inconsistent shape/layout/attribute/state declarations.

Derive roles from validated operator semantics, attributes and producer topology,
not names such as `layer.0.q_linear` or `model_family`. Record each role's source
tensor, ordered input/output coordinates, numeric rule and client/server boundary.
Unknown operators, unknown attributes or unsupported combinations REFUSE before
numeric work. An enum variant existing in `ModelOperator` is not method support.

- Delegated candidates: validated projection and full-output-head matrix products.
  Bind each complete logical dot, source orientation, row/token order and scale
  indices. Splitting publication/work must not introduce partial-dot BF16 rounding.
- Client-local candidates: token lookup, reshape, RMSNorm, RoPE, attention score/
  scale/mask/softmax/value operations, residuals, SiLU, multiply and bias addition.
  Each requires its own exact rule; local placement supplies no correctness proof.
- Last-token selection, greedy selection, feedback and KV append require explicit
  output/state contracts. No hidden delegated lookup, vocabulary truncation,
  attention substitution or silent host fallback for an unsupported operator.
- `CacheSuffix` is a separate functional role, not established by KV append alone.
  Bind ordered inputs `[append, input.positions, input.attention_mask,
  input.sequence_lengths]`, `axis=2`, `maximum_sequence`, and
  `semantics=visible_valid_prefix`; bind `absolute_write_positions_input`,
  `padding_mask_input` and `valid_lengths_input` to those exact roots, and preserve
  matching Key/Value `state_kind`. Visibility must include current valid entries
  without exposing future, padded or unwritten cells; the name permits no eviction.

QKV bias presence and Q/K norm absence follow actual declarations and edges.
Validate separate bias tensor dependencies inside linear attributes even when
they are not standalone operation nodes. No family-named Rust module is needed.
Retain full vocabulary, actual prefill length, decode position and grouped-query
head mapping. Prompt rendering and admission use actual source-rendered tokens;
neither eight tokens nor a measured common-checkpoint prompt count is established.

## Separate original and common numeric identities

Original D31+D32 identity remains the pinned FP8 Qwen3 candidate, not common BF16.
D31 uses `roundEven(256*x)` (eight fractional bits), canonical BabyBear residues,
complete-dot centered decoding and division by 65,536 directly rounded to BF16.
No intermediate F32 rounding, partial-sum conversion, saturation or tolerance.
Prove signed no-wrap and host reduction safety for every admitted complete dot;
integer range failure rejects, not a reason to change encoding or field.

D32 exactly promotes each stored BF16 block scale to F32 before the specified F32
multiply with decoded E4M3FN; preserve original 128x128 scale axes and orientation.
It recovers no earlier scale precision. These rules do NOT turn BF16 projection
weights into FP8 or authorize invented scales. Original BF16 head treatment does
not by itself authorize a BF16 rule for every common projection.

Common identity must separately bind revision, tensor extents/dtypes/aliases,
actual config, complete graph, numeric derivation and workload/state semantics.
Pin alone proves none of those tensor facts. Proposed common BF16-to-Q8 linear
derivation must name exact source-value promotion, rounding and complete-dot output
rule and receive numeric review; no re-quantization/import repair is implicit.

Original Norm uses epsilon F32 bits `0x358637bd` and cast-before-gamma order:
F32 square/sum/mean/epsilon/rsqrt, cast normalized value to BF16, then gamma multiply
with specified rounding. Original SiLU uses correctly rounded F32 of exact-real
SiLU then BF16; original attention has ordered, non-fused local reductions.
These are original-profile facts, not evidence stock common kernels match them.

Existing dense RoPE declares
`coefficient_profile=pllm.numeric.rope.q30.libm.v1`; preserve that explicit numeric
identity (and any declared `frequency_scaling` with its changed profile), rather
than silently treating it as original BF16 RoPE or approved common numerics.
Compatibility and coefficient/rounding derivation remain unresolved review gates.

Common derivation still owes independently checked source facts for epsilon value
and rounding, gamma dtype/order, biased-QKV cast/add order, SiLU and attention
rounding, RoPE frequencies/layout/positions, and KV stored dtype/append semantics.
Existing lowering supplies epsilon attributes, bias presence and producer topology,
not floating-point execution order. Do not invent numeric parameters or inherit
Qwen3 per-head norms when absent. Unsupported or unresolved rules block admission.

## Custody, admission and request-global acceptance

RAW22 join must retain the actual original immutable plan, independently provided
host pins, retained snapshot directory/reader, request-token owner, genuine producer
and original work ledger. Host pin provisioning is a trust premise, not publisher
authentication. Equal hashes, metadata, coordinates or cloned plans do not confer
ownership, authority or a replacement budget/root.

Before traversal/allocation, charge bounded graph visits, producer lookup/storage,
attribute parsing, contract backing, numeric/source passes and retained controls on
that original root. Price original buffers, directory/reader, contract, publication
windows, output/KV retention and construction/move/replay overlap together. Include
all-path cancellation/cleanup/retirement before refund; no implicit free traversal,
ledger reset, speculative allocation or inventory digest as an admission permit.
Logical counters do not close allocator/RSS, physical-byte, stack or backing holds.

Reuse `admit_maverick_batch_publication` for its actual geometry/coverage/allocation
checks, not as decoder/source/numeric authorization. It currently requires native
fixed geometry, one live window and complete matrix/head coverage; common shapes
may refuse. No tiling, cap increase, new owner or general decoder support follows.
Any future window plan must bind partitions, global source/scale indices, originals,
peak retained overlap and spent attempt state without resetting original limits.

Reuse `Client::begin_request_verification` and request-global finishing semantics:
fix ALL original replies/encodings before ANY primary challenges; fix ALL auxiliary
replies before ANY secondary challenges. Accept body/head only atomically after
every required check. No tentative output, public intermediate, cache commit or
token escape before global acceptance; failure/cancel retires private state.

D45 is two independent raw batches, not dependent decoder continuation. Publishing
an accepted intermediate to feed another batch cannot establish a request-global
join. Dependent layers require a separately reviewed private provisional lifecycle,
with all-original commitment timing reconciled before challenges. Existing API does
not supply that lifecycle; do not claim a complete decoder from an API-name join.

## Extension DAG and independent gates

1. Review this graph/numeric/security contract and freeze source-only changes.
2. After Astra review and compile gates, propose only
   `maverick_decoder_contract.rs` / `maverick_decoder_contract_tests.rs` in the
   existing compiler architecture; extend shared contracts, not parallel custody.
3. Validate original and common identities separately; derive common local/linear
   numeric rules from collected source facts. Unknown facts remain explicit gates.
4. Review dependent lifecycle, publication geometry/windows and complete resources.
5. Independently qualify numeric and security regressions with fresh lane grants.
6. Only then join the collected checkpoint through ordinary `Pipeline`/`Experiment`,
   genuine renderer, gateway and reusable benchmark experiment/output. Source
   collection, real execution and trained quality each need their own evidence and
   grants; recipe/common budgets are not measured fit or full-trained qualification.

Numeric review: check all producer edges/roles, dtype/alias facts, Q8 ties, complete
dots/no-wrap, exact scale promotion, epsilon/bias casts, SiLU, attention/RoPE and
prefill/decode/KV positions against independent exact references; no new tolerance.
Security review: keep dual-LPN assumptions, actual-code distance evidence, sparse
sampling, independence, request/query horizon, secondary checks and source-security
holds explicit and pending. Default attempt count is not whole-decoder security;
D31/D32 and native equations supply no missing certificate. No GKM substitution.

Finite future regressions: unknown operator/attribute; missing/duplicate producer;
foreign root/directory/token/plan; reordered dependency or reply; partitioned-dot
rounding or scale-index drift; wrong bias/epsilon/position; prefill/decode KV splice;
cancel/replay/tentative/cache escape; exact-budget and one-short refusal BEFORE work;
all-original/all-auxiliary barriers; no implicit model-family branch or fallback.

Open review questions: approve common BF16 linear and bias/cast derivations only
after source facts; select dependency-safe provisional lifecycle and geometry;
establish complete-root resource bounds and actual security horizon/evidence.
Until then, this is a source-supported join proposal, not an executable contract.
