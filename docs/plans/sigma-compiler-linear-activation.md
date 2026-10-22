# SIGMA compiler-certified linear-to-activation handoff

Status: **APPROVED FOR BOUNDED IMPLEMENTATION (D22)**. The user explicitly
approved the bounded composition and checkpoint/submatrix fixture policy. This
extends the local D20 boundary; it does not authorize a decoder, transport, new
general numeric profile, or checkpoint-wide range policy. D16–D20 remain unchanged. See
[the approved activation contract](sigma-full-activation.md).

## Existing authorization and approved extension

Declaration-only compiler binding of one existing D20 invocation can reuse
`SourceIdentity`, `EdgeDescriptor`, and `ConsumerUse`. Metadata does not mint
capabilities or prove a checkpoint range. Existing clear-root checks and
`LocalAssumption` remain in force. The existing float-boundary masked-wire
SiLU/multiply schedule is a different component, not this numeric profile.

The approved extension is one consuming, compiler-certified private matrix
complete-dot → exact GapARS12 → unchanged D20 GELU/SiLU composition.

## Fixed numerical and geometry boundaries

- Inputs and private weights: signed ring50/Q12.
- Complete sum `S = sum(X_i * W_i)`: ring50/Q24, with a source-bound public
  unwrapped interval contained in `[-2^48, 2^48)`. Accumulate the complete inner
  dimension; no partial-dot or per-term truncations.
- GapARS12: `Z = floor(S / 4096)`, ring50/Q12. The certified sum interval implies
  `Z` lies in `[-2^36, 2^36)`, inside D20's `[-2^37, 2^37)` input domain.
- D20 retains its ring38 projection, TR6, ring32/Q6 intermediates, clipping,
  deferred lookup/Select combination and ring50/Q12 output. No extra final shift,
  remask or masked sign extension.
- Each input, weight and output matrix has at most 256 elements; outputs have
  1–256 lanes. Retain the existing six-contraction `2^20` MAC ceiling and shared
  64 MiB logical allocation ceiling. These are not process-RSS guarantees.
- Bias is unsupported. Reject biased semantic operations rather than dropping
  or relocating their bias.

## Source and certificate obligations

Bind immutable checkpoint revision/content, tensor, original dimensions/layout,
selected coordinates, semantic operation, encoding and implementation identities.
Independently floor-encode finite source values to Q12; reject unsupported values.
Derive actual declared operand intervals and complete-dot bounds; do not select
invented bounds or infer public activation ranges from observed fixture extrema.

The current interval proof uses endpoint products times the full inner dimension.
Tighter per-coordinate certificates require independently reviewed shared
provenance support. Runtime private range checks or clipping are not authorized
by this approval.

The independent fixture selects row 0 and columns `[0,256)` from the original
`[4864,896]` output-input tensor `model.layers.0.mlp.gate_proj.weight` in
Qwen2.5-0.5B-Instruct revision `7ae557604adf67be50417f59c2c2f167def9a775`.
Its selected original BF16 bytes have SHA-256
`780a723ba14459a03b2e72508debd6d4551d6ca13da75534c0f456de84c680de`.
This is a **checkpoint-derived submatrix fixture**, not original-layer execution;
its definitions do not establish executed parity. A complete genuine bias-free
operation may be used only if its full original geometry fits admission.
Source-supported 50/12/mode1 mapping is not proof of the historical paper
evaluation's exact invocation.

The certificate is an opaque compiler-owned validated owner, not a deserialized
description or relabelled local range assumption. Hash actual supplied source
bytes against an independently expected pin; bind original dtype/endian, offsets,
geometry, ordered selection, semantic plan and operation, input/output edges,
activation and encoding identities. Compare every Q12 coefficient against the
canonical issuance material before randomness. Reject any declared bias, even
zero bias. A caller-chosen digest authenticates no publisher. Do not equate the
semantic `GeluTanh` operator with D20 GELU without an explicit supported mapping.

For half-open operand intervals `[lx,ux)` and `[lw,uw)`, evaluate the four endpoint
products using `lx`, `ux-1`, `lw`, `uw-1` in checked wide arithmetic. For complete
inner dimension `k`, certify `[k*min(products), k*max(products)+1)` inside the
quarter ring; derive the rescaled interval by signed floor at both inclusive
endpoints. Input domains are public declarations independent of fixture values;
the certificate proves composition conditional on that domain, not checkpoint-wide
activation ranges.

## Consuming ownership and admission

Consume original roots, private coefficients, matrix, rescale and activation
owners. Transfer matching bindings and original backing/leases without opening
secrets or re-encoding reconstructed intermediates. Keep the matrix singleton
with its actual guarded backing, including retained frames; retire both peers
on failure.

Preflight the complete concurrent owner inventory before entropy: roots,
coefficients, certificates, masks, native/wrapper owners, frames, tables and
transition overlap. Current typed matrix/arithmetic entry points release a
preflight reservation before native issuance reacquires capacity. Add reserved
entry points and split the original reservation; nesting those existing calls
does not establish continuous aggregate ownership.

Native matrix/arithmetic cancelled or completed wrappers must retain metadata
charges until their structs drop, as D20 does. Retained matrix frames keep the
actual singleton guard; unrelated final activation outputs do not inherit it.
Keep lease construction private to native ownership support and preserve the
dependency direction `pllm-compiler → pllm-garble`.

## Acceptance and explicit nonclaims

Use normal tests with independent checkpoint loading/encoding, wide-integer
complete-dot and signed-floor equations, and an independent D20 table oracle.
Cover source/tensor/layout substitution, partial dots, per-term floors, overflow,
quarter-ring endpoints, negative floor boundaries, wrong tables and unsupported
bias. Check exact/one-byte-short admission, zero-draw rejection, consuming use,
cancellation/fault retirement, retained frames and singleton lifetime.

Retain fixture parity separately from source-model quality. Evidence is a
reusable `.py` experiment and ordinary output, not a new receipt harness.

Remaining decoder work includes full-sized contractions and bias, residual and
fanout ownership, protected normalization/attention, embedding/full head, stable
private selection and feedback, whole-graph ranges, source/material delivery,
authenticated transport/replay retirement and ordinary SDK execution. Existing
D9 final-only fixed-count feedback approval is unchanged; persistent KV is not
implied.
