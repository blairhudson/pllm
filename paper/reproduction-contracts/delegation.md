# Independent source-fidelity audit: delegation contracts

Audit date: **17 October 2026**. Canonical specification:
[paper/methods/delegation-contracts.md](../methods/delegation-contracts.md).
This file is an independent audit of that draft, **not a second implementation
contract**. The canonical file was read, not edited.

**Verdict:** the canonical draft's core GKM and Maverick constructions, numerical
distinctions, source conflicts, and conditional security claims match the
independently read primary sources. No replacement protocol is needed to correct
the reviewed algebra. Model-level numerical fidelity and security admission
remain open for specific reasons below. The approval ledger needs narrowing:
paper-required correctness/security is not an optional adaptation decision.

Scope: source/documentary review only. No implementations, model loads,
workloads, builds, upstream execution, or benchmarking. Small arithmetic checks
of printed inequalities are documentary calculations, not protocol execution.

## 1. Evidence identity

The audited canonical snapshot has SHA-256
`abf9fac8c702fffb2354159ccbbe34e6ef1586df516cad066c9b6b2eadd9be70`.
Canonical section/line references below apply to those bytes; a revised draft
requires rechecking the relevant finding.

| Independently read source | Exact identity | SHA-256, recomputed locally |
| --- | --- | --- |
| [GKM PDF](../../papers/r19-open-weight-verifiable.pdf) | [ePrint 2026/1849](https://eprint.iacr.org/2026/1849), cached unversioned snapshot. Exact revision correspondence remains unspecified; do not call it v1. | `e1863980621e0bbd2d446b43f2d9415318ebeec400b0e6c818685752f6448e1d` |
| [Maverick PDF](../../papers/r24-maverick.pdf) | [arXiv:2609.10264v1](https://arxiv.org/abs/2609.10264v1), 9 September 2026, confirmed by PDF stamp. | `c9570fdeb51a26f9d707bcc7bf485f2001e7d3e15ebdf34815123935a1e8e2a0` |
| [GKM code lock](../artifacts/source-review/2026-10-16-gkm-qualification-v1/code-lock.json) | `kanav99/dvllm@271637be7196d26a5433ca1e62b1a5357e769a5b`, tree `e742e67ed278aba39a440a3d78c782134fe8732c`, 1 October 2026; twelve selected files, not an artifact-wide audit or evaluated-build identity. | `0575902701e9eae051965a2b1cd651e0ef30dcb70349248739cfe5e2b46c04ab` |
| [GKM primary locators](../artifacts/source-review/2026-10-16-gkm-qualification-v1/primary-locators.json) | Physical PDF page extracts, with individual hashes. | `57b9c3807d9542b89d292dc6906cdcd6d333d55568373297ecbf32d3b7e1b6a2` |

Both complete primary PDFs were independently read before the canonical draft.
The [coverage audit](../TEN-METHOD-COVERAGE.md),
[qualification card](../COMPETITOR-QUALIFICATION.md), and
[older two-offset plan](../../docs/plans/two-offset-boundary-comparator.md)
were also read. The older W8A8/ring32/DPF proposal is not authorization.

No Maverick author-code commit was established by the inspected local archive,
PDF, or fetched primary abstract record. The canonical draft correctly avoids
claiming code is unavailable or attributing PLLM's reference behavior to authors.

## 2. Findings and closure

### F1 — GKM original arithmetic and verification: confirmed

**Canonical §§2.2–2.3, lines 100–203.** Primary locators: §3/§4 encoding,
PDF p.6; Fig.1, p.7; Fig.2/batching, pp.7–9; Figs.3–5, pp.9–11;
§5 implementation, p.12.

The canonical requirements are correct:

- Nearest-integer `Q(v)=round(2048*v)`, signed int16 weights, ring64 shares,
  signed reconstructed products, rescaling by `2^-22`. The formal p.6 rule
  takes precedence over p.4's informal “truncate” and the inspected casts.
- Dense XOR one-hot sharing over **float embedding bytes**, followed by row
  SHA-256 verification. Fig.3 step 2's self-reference is a typo: p.10 explicitly
  supplies `x1=x XOR x0`. The same two workers serve lookup, body, and full head.
- secp256k1's **subgroup scalar order**, not its coordinate-field modulus and
  not `2^64`, governs setup scalars and verifier accumulators. Signed integers
  must lift consistently into that scalar field. In particular, using
  `2^64-1` instead of `-1` as an exponent is wrong.
- Verify the reconstructed, non-overflowing integer relation. Arbitrary wrapped
  individual-share products do not satisfy that integer relation without a
  separate wrap-correction argument. Fresh uniform scalar coefficients follow
  response fixation; accumulate modulo the subgroup order.
- Setup secrets stay unavailable to malicious servers. Privacy needs
  non-collusion; integrity may tolerate both servers colluding under the
  honest-owner/discrete-log/hash assumptions. The client must reject before
  accepting output or reusable state.

**Closure clarification:** the code lock's `include/client.hpp:263–279` performs
`float(out)/(2048*2048)`. The eventual exact float32 reference should explicitly
record conversion **before** division. Canonical lines 119–125 correctly leave
the ordering in the numeric manifest; a prose formula `z/2048^2` alone is not
that manifest. Halfway ties are unspecified by the paper. Nearest ties-to-even
is a source-compatible deterministic completion, not evidence of author behavior.

The row-bound condition in the draft is a sufficient conservative admission
test. Ring evaluation may wrap shares; the intended unmasked product must remain
in the signed range. Testing a centered residue after an already-overflowed dot
product cannot establish this prerequisite. No source-preserving fix requires
W8A8, ring32, DPF, or omission of verification.

### F2 — GKM inspected bugs versus substantive changes: separate them

**Canonical §§2.2–2.3 and approval rows G1–G4.** The discrepancy evidence is
accurate, but its classification should govern the approval queue:

| Evidence at the pinned author commit | Paper-faithful correction; not an optional protocol choice |
| --- | --- |
| `src/quantize.cpp:123–145`, `include/client.hpp:211–279`: truncating casts. | Use the formal nearest-integer definition; record tie convention and range policy. Reproducing author-cast numerics would be a separately requested diagnostic, not the default. |
| `src/llama3.2q.cpp:627–640`, `include/verification.hpp:59–64`: `rand()`-derived seeds and an int64 batching coefficient. | Secure independent entropy, uniform subgroup-field coefficients, and fresh post-response challenges. Preserve source AES share compression without copying weak initialization or narrowing the challenge distribution. |
| `include/client.hpp:377–399`: mismatch prints and row is cached; `include/verification.hpp:147–165`: failed checks print. | Fail closed; no bad embedding use/cache, accepted output, or continued state. These are inspected-path concerns, not a verdict about every wrapper or the abstract protocol. |
| `src/llama3.2q.cpp:662–695`: generation before final check; verifier loading/queue drain precede printed verification timer. | Keep generated state tentative until verification. Charge the entire verification lifecycle; no accepted-time claim based on that narrow timer. |
| Signed share arithmetic and copied asynchronous verifier jobs. | Defined modular operations, field accumulators, bounded ownership, and drain-before-release. Language overflow or abandoned work is not part of the paper protocol. |

**Actual changes requiring approval** include per-token verified streaming instead
of request-end batching, observable lookup-cache omissions, changed numerics or
lookup construction, weakened verification, new roles, and an altered workload.
Secure randomness and abort are mandatory prerequisites; do not ask whether the
user wants them. Protocol serialization and tie-rule documentation are ordinary
specification completion unless evidence shows a substantive numerical departure.

### F3 — Maverick original numerics and workload: confirmed, still incomplete

**Canonical §3.5, lines 404–468.** Primary locators: §7.1, p.11;
§7.2.1/Fig.1, pp.12–14; §7.2.2/Table 6/Table 7, p.14; §7.3/Table 10,
pp.15–17.

Correctly identified source profile:

`FP8 weights → nearest_integer(256*W) → BabyBear MVM → demask → centered integer
→ divide by 65536 → BF16 → local nonlinear computation`.

BabyBear `q=2,013,265,921` is **the paper's own evaluation field**. It is not
itself a PLLM substitution. The i8-only wrapper, per-row W8A8 semantics, seeded
code ensemble, and changed sparsity are the existing reference differences.
The p.13 “all arithmetic” sentence cannot override p.14's explicit BF16 boundary.

Useful exact bound to add to the numeric admission record:

- Centered field range is `[-1,006,632,960, 1,006,632,960]`.
- After `2^-16` rescaling, endpoints are exactly `±15360`.
- This is the range of the **intended unmasked quantized dot product**, not a
  range restriction on masked field products. Canonical's strict bound is a
  conservative admission policy; equality at the mathematical endpoints is
  representable. Modular verification cannot detect an unintended numerical wrap.

The draft correctly marks missing FP8 format/scales/checkpoint, activation
encoding, halfway rules, nonlinear precision/order, and KV representation as
unresolved. Equal input/weight scale 256 follows naturally from the squared
rescale but is not an explicitly supplied activation algorithm. Neither source
fidelity nor exact author-build parity can be recovered by guessing E4M3/E5M2,
using the existing float32 executor, or equating an i8 oracle with FP8.

The original complete-model experiment is **Qwen3-4B, eight prompt tokens to
verified next-token logits**. `8/0.9583 ≈ 8.35` counts prompt tokens, not
generated tokens. The canonical draft correctly refuses to infer a measured
persistent KV/decode loop. References to GPT-2/Gemma-3-270M describe DeepProve,
not additional Maverick evaluated models.

### F4 — Maverick 31 versus 41: conditional issue closed by direct calculation

**Canonical §3.3 item 2, lines 323–332: confirmed.** Primary locators:
Lemma 5.1/Theorem 5.2, p.7; Theorem 6.1, pp.10–11; §7.1, p.11.

For known relative distance `delta=1/2`, field `q=2,013,265,921`, and two
independent repetitions, the printed lemma gives

`Pr[accept incorrect] <= (2^-t + 1/(q-1))^2`.

At source-evaluated `t=31`, this is
`9.261495891187284e-19 ≈ 2^-59.9053885704`, below `2^-40`.
The theorem's stated sufficient formula yields `t=41` when `kappa=40`; that
formula is sufficient, not necessary. Thus “31 differs from the formula” does
**not** mean 31 must be repaired to 41. Keep source 31/two-check profile and cite
the lemma; retain the old PLLM 41 profile under its existing distinct identity.

This closes only the **conditional sparsity calculation**. Code-distance failure
and privacy are separate. For Protocol 2 the conditional error is the sum of
primary and secondary errors. If both certified distances and both parameter
sets were the same 31/two-check choice, the sum would be about `2^-58.905`.
That is a calculation, **not a claim that the evaluated secondary parameters
are pinned**. Shared primary challenges over a fixed batch need no extra factor
equal to batch size; separate batches/attempts still need lifetime accounting.

### F5 — Distance/certification: blocker retained, stronger primary lead found

**Canonical §3.3 items 1/3–5: confirmed with an additional research lead.**
Maverick Appendix B, pp.23–24 specifies the randomly scaled RAA ensemble and
quotes a sufficient field condition `q > 2^31`. BabyBear fails that condition.
It follows only that **this quoted condition does not certify BabyBear**.

Narrow primary record fetches on 17 October found:

| Record | Source-supported lead | Raw HTML SHA-256 |
| --- | --- | --- |
| [ePrint 2026/524](https://eprint.iacr.org/2026/524), Akhiani–Zhang; revision listed 29 July 2026 | Abstract discusses relative distance `1/2` over a **31-bit prime field**. It is not this implementation's concrete certificate. | `f1d95a041183fe070eeecadc984aaad397bc949cce475c4e5f29c53e3b54b1df` |
| [ePrint 2026/1075](https://eprint.iacr.org/2026/1075), Khabbazian; revision listed 20 August 2026 | For scaled RAA, record states `r>=4`, `0<delta<=1/2`, `q-1>=gamma*N` imply a `Theta(N^(1-r))` failure tail, with parameter-dependent constants. This is broader than a standalone `q>2^31` condition. | `6aa99fbb02d6639a2be5ddb6f9906fb82b4acf144184fa5435795039bca468cc` |
| [Maverick v1 abstract](https://arxiv.org/abs/2609.10264v1) | Confirms source version/date; no author-code pin established in the inspected record. | `1e0b9f5c9a21617ed8510c51647f070994eaaa60d6af4828853a096da3bb1fea` |

The attempted `2026/1075.pdf` fetch returned **HTTP 403**. No theorem/constants
were verified from that PDF. Abstract asymptotics do not certify finite dimensions,
actual sampled instances, SHA-256 seeded ensembles, or dual-LPN. Nevertheless,
these leads make an automatic larger-field proposal premature.

**Required closure evidence:** exact primary theorem/version and finite-parameter
bound or reviewed distance certificate for every actual privacy, verification,
head, and auxiliary code; correct public P/Q/Qaux validation; unbiased/admitted
code selection; actual exact-weight noise and multi-query security accounting.
Count a reused code's bad-selection event once, and per-attempt verification
errors over its lifetime. Passing corruption tests cannot replace this record.

### F6 — Batch protocol and public-state offload: math confirmed; access warning needed

**Canonical §3.4 and §6.2, lines 357–402/679–710.** Protocol 2, pp.8–9,
confirms `Qstack: Bn×N`, `Gprime: Bn×Nprime`, and
`Qaux=Qstack^T*Gprime: N×Nprime`, with both response-fixation barriers.
Its extra round, server projection work, reply vectors, and auxiliary storage
must be counted. The draft is correct. A direct Protocol 3 operator is not the
complete optimized system. The systematic-code storage saving requires an actual
systematic secondary code, not relabeling RAA.

**Add this constraint to M6 before any storage adaptation:** public matrices do
not make access indices public. Fetching only columns `P[:,supp(ex)]` from the
evaluator reveals privacy-noise support. For `t_x<n`, an adversary can form a
nonzero linear functional annihilating those selected columns of `Gx`; applying
it to `xhat` reveals that functional of `x`. Revealing all of `ex` recovers `x`
directly. Thus remote secret-index sparse fetches are not source-faithful paging.
Verification challenge accesses also need their prescribed disclosure timing.

Client-local mapped public artifacts can avoid this transcript leak under the
declared trusted-client/side-channel scope, but storage policy, validation,
page/disk traffic, RSS, full acquisition, and cold/warm accounting remain explicit.
The `64mn` P+Q payload identity and old Qwen2.5-shaped 22.90 GB calculation do
not prove original-system infeasibility. They do require honest ownership and
resource admission rather than transferring private work to an unreviewed helper.

## 3. Common GPT2-class workload assessment

The canonical draft leaves the common-model choice open (G5/C1). Independent
source reading establishes **no architectural or algebraic prohibition** on a
GPT2-class model: both constructions accept public linear maps and client-local
remaining operations. LayerNorm, GELU, learned absolute positions, causal
attention, and biases can remain local without replacing delegation protocols.
Current missing LayerNorm/position/import support, recorded in the
[shared contract](README.md#shared-model-and-execution-requirements), is an
implementation gap, not a source impossibility.

Numerical admission remains conditional:

- GKM must fit all quantized linear/head weights into int16 at scale 2048
  (roughly magnitude below 16), preserve float embeddings, and bound unmasked
  integer products in signed64. Model size alone supplies no such proof.
- Maverick must resolve its actual FP8/activation/BF16 schedule and fit intended
  quantized products in the centered BabyBear range. New head/auxiliary dimensions
  additionally require code certification. No relevant checkpoint bounds were
  measured in this audit.

The original numerical domains can therefore support a **conditional common
checkpoint/prompt workload**, but they do not specify one bit-identical numeric
computation. Use exact correctness against each method's own clear oracle and a
predeclared quality-matched cross-method contract. Do not lower scales, clip,
change fields, split/requantize dots, or inherit W8A8 merely to force a match.

Source-supported anchors remain Llama-3.2-1B or Gemma-3-1B for GKM (smallest
listed evaluated models), and Qwen3-4B for Maverick (only complete-model evaluation
here). Llama-3.2-1B is the cheaper GKM starting point supported by the inspected
client lineage. Different source-model cells are correctness anchors, not a
matched ranking. GPT2 is a proposed new workload for these papers, not an
already source-evaluated alternative.

## 4. Narrow the approval ledger

Canonical §5 currently mixes required fidelity, missing evidence, and actual
adaptations. Recommended closure:

- **G1:** formal nearest encoding, signed lifting, and bounds follow the paper;
  do not ask the user to choose them against unsafe casts. Record source-compatible
  tie/float completions explicitly. Escalate a genuinely changed numerical
  realization, not ordinary defined-arithmetic implementation.
- **G2/G3:** source dense XOR/original numerics/two servers/verification are already
  mandated. Remove as blocking choices unless proposing an actual departure.
- **G4:** secure entropy, failure propagation, fresh stream ranges and channel
  binding are requirements. Retain the genuine acceptance-window/leakage decision.
- **M1/M2:** retaining source BabyBear/full ensemble/31 checks needs no substitute
  selection. Certification and full secondary parameters remain evidence gates.
  Correctly labeled algebra tests are normal implementation stages, not permission
  to replace a full system or publish an uncertified security claim.
- **M3:** missing primary precision facts are a discovery blocker. Obtain them;
  if unavailable, present one complete, concrete numeric fallback for approval.
  “Approve unspecified FP8/BF16” is not an actionable decision.
- **M5:** source Protocol 2, validation, and private standard mode are requirements.
  Boost is an explicitly priced source variant; verification-only cannot fill a
  private cell. Their existence does not require a protocol-substitute question.

### Only blocking concrete adaptation decisions

These are recommendations, not approvals. Their blocking scope is stated so
unrelated source-correct work is not held behind speculative alternatives.

| Decision | Concrete recommendation | What it blocks / impact |
| --- | --- | --- |
| **Common workload/reference track** (G5/C1) | Qualify Llama-3.2-1B and Qwen3-4B source anchors; then select one pinned GPT2-class checkpoint under each original numeric domain, with per-method exact oracles and predeclared quality non-inferiority criteria. | Common-model integration and ranking. New workload does not justify changed arithmetic or identical-logit claims. |
| **Full decode and acceptance window** (G4/M4) | Maverick: client-owned causal state, verify each next-token execution before accepting/sampling/feedback. GKM: source-style request-end batching first; per-token verified streaming only as a separately approved profile. | Persistent Maverick generation extends the measured source task; GKM smaller windows change amortization, first-accepted-output latency, and state ownership. |
| **GKM CPU-first execution profile** (C1) | Use an independent Rust CPU implementation for the first complete correctness/profile cell; preserve int16/ring64 and all protocol steps. GPU-performance reproduction requires a later independent CUDA cell. | Backend/hardware measurements. CPU timing cannot stand in for the original A100 execution. This does not block mathematical primitive/oracle design. |
| **Maverick local public-state storage policy** (M6) | If the full model exceeds resident budgets, use client-controlled file-backed/mapped P/Q/Qaux with bounded resident windows and full setup/I/O/RSS charging; never reveal secret-dependent accesses to the evaluator. | Model-scale admission and resource results. Storage estimates alone neither authorize remote offload nor establish tier eligibility. |

No concrete FP8 fallback is selected here: primary details are missing. A future
complete fallback manifest is the appropriate numerical approval request, before
its implementation. A future field/code/security change likewise requires its
own specific proposal; none is needed merely to preserve the reviewed source.

## 5. Audit closure and next order

| Gate | Audit result |
| --- | --- |
| Source versions/hashes and primary locators | Confirmed for the cached bytes; GKM revision correspondence remains explicitly unknown. |
| GKM int16/ring64/2048, dense float-row PIR, secp256k1/hash verification | Confirmed. Formal rounding, secure entropy, scalar sampling and fail-closed acceptance must take precedence over inspected code departures. |
| Maverick Protocols 2/3, FP8/256/BF16, original eight-token workload | Confirmed as source requirements; precise numeric/import/state and evaluated secondary-code schedule remain unresolved. |
| 31 versus 41 conditional check | Closed mathematically in favor of retaining evaluated 31/two-check profile. No distance/privacy certificate follows. |
| Actual-field/code/preprocessing certification | Open; new primary-record lead above, full-text fetch blocked. No source infeasibility finding. |
| Canonical complete conformance gate (§§4.3–4.4/6.3) | Adequate documentary specification: independent arithmetic/model oracles; full pre-sampling logits, active KV, history/pending token/RNG; prefill/feedback/EOS; accepted-state continuation; cancellation/replay/retirement; complete lifecycle ownership. None executed or passed by this review. |

**Cheapest faithful order:** resolve the narrow approval/evidence items; freeze
numeric/source/material and accepted-state identities; build the canonical GKM
verified primitive capsule and independent oracle; compose its complete original
model through both workers; resolve Maverick precision and code certificates;
compose source Protocol 3 plus mandatory Protocol 2 and original next-logit task;
then add the approved full feedback extension/common workload. Only qualified
complete requests proceed to a later matched benchmark campaign.

Reusable Rust/compiler/runtime units identified by the canonical draft remain
appropriate: typed int16-ring64/field arithmetic, dense byte-table XOR, published
curve/hash verification, RAA encoders and two-level verifier, public/private
material ownership, explicit topology admission, and tentative-to-accepted decoder
state. None can be replaced by a similarly named component. This audit adds
closure evidence and the access-pattern constraint rather than a competing API.

Both complete serving ranks and client-tier eligibility remain unresolved.
Missing implementation, unsupported local geometry, and an unclosed sufficient
security bound do not establish failure of either original source system.
