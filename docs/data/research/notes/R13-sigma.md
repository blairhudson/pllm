# R13 · SIGMA: Secure GPT Inference with Function Secret Sharing

**Priority 13 · 2024 · two_online_comparison · source checked 2026-09-14**

Authors: Kanav Gupta, Neha Jawalkar, Ananta Mukherjee, Nishanth Chandran, Divya Gupta, Ashish Panwar, Rahul Sharma.  
Primary source: https://petsymposium.org/popets/2024/popets-2024-0107.php  
Full 19-page primary paper is cached as `papers/r13-sigma.pdf` and verified against the SHA-256 in `paper-library.json` (`f911b715b0e0523fa4d77f21c28c6a0440c5be781e67b5b30419f20e505a6ebf`). This is specification material; upstream code is not a PLLM dependency or a locked runtime artifact.

## What the source contributes

SIGMA evaluates shared transformer state with two online parties and one input-independent trusted dealer. Section 4.2 replaces insecure local truncation with faithful arithmetic right shifts: the known-gap variant requires a proven input range, whereas general signed inputs need a truncate-reduce and secure sign-extension. Section 5 uses fresh DPF-backed FSS keys for SiLU, GeLU, exponentiation, reciprocal, and rsqrt, including private attention and normalization. Appendix L shows dealer key transfer is part of full cost; the reported Llama2-7B key size exceeds 255 GB.

This short source summary is separate from the proposed PLLM design below. The source has not been reproduced merely by adding this card.

## Native implementation scope

Reproduce the original FSS helpers and complete reference workload. Register softmax, activation, normalization and truncation independently with exact phase contracts.

Data flow: **Two online share-holding roles + dealer keys → secure transformer helpers and source workload.**

Register the following independently versioned native components:

- `protocol.fss`
- `gate.fss_nonlinear`
- `prepare.fss`

Dependencies: None; foundational work item. Public compilation/model-data scans and all online evaluation happen in Rust or reviewed native accelerator libraries called from Rust. Python selects immutable parameters and displays public results. It does not implement a timed alternative or silently repair unsupported native execution.

## Required fixture and benchmark specification

Original model/token accounting first; sustained autoregressive decode is a separate workload, not inferred from one forward-pass time.

Mandatory paper-specific gates:

- FSS key freshness and widths
- Secure truncation and private attention
- Two-party rounds and fresh key consumption

Each fixture locks input/output representations, ring/field parameters, numeric graph, party count and material lifetime. Compare an optimized implementation with identical functionality and applicable privacy assumptions; record original-artifact numbers and adaptations separately. A source-advertised speedup is not an expected threshold for different hardware.

## Privacy, correctness and proof obligations

Dealer/corruption-set proof and specific helper composition, not a single-evaluator claim.

Two online parties and the dealer model must remain explicit. No implicit substitution into the one-evaluator default.

Provide the corrupted party's permitted view, known plaintext/public inputs, randomness model, allowed leakage and attack budget. Formal or empirical results have explicit scope. Passing functional tests cannot grant a production privacy badge. A changed hash, field, layout, actor role or precision creates a documented adaptation obligation.

## What PLLM already has

Resident-share and spectral archives are different constructions, not SIGMA reproductions.

The bounded, test-local path now uses one-use two-party point-function keys for
comparisons up to 10 bits and independent exact carry correction for signed
ring-2⁶⁴ truncation. Its hybrid uses an 8-bit FSS comparison for the low
part **of a wider shift**, then one-use Beaver carries for the remaining bits;
checked 16- and 24-bit shifts reconstruct exactly. Its two
authenticated party channels pass a Q8 causal-attention prefill-to-decode
check with KV shares retained separately. For a 10-bit comparison one key is
187 bytes **per party per element**, with another 128 bytes of optional local
expansion; key issuance and expansion are offline costs. Missing keys abort,
and consumed keys cannot be reused. This is not a protected compiled decoder,
SIGMA's 12-bit/GPU optimized DPF, a model-quality result, or proof of
independent operators. A separate semantic-schedule cost gate for the pinned
30+1-token Qwen2.5 plan counts only one gated product and exact 8-bit rescale
per intermediate MLP element. Even that optimistic subset needs at least
735,436,800 correlated-body bytes **per party** and 168,975,360 online
all-link opening-body bytes, versus 95,805,056 recorded online body bytes for
the matched two-worker offset comparator. This reference therefore fails its
256 MiB-per-party material budget and the network comparator before protected
attention, norms, setup framing or checkpoint distribution. See
`docs/evidence/shared-resident-mlp-cost-gate-2026-09-27.json`; these are
projected lower bounds for this construction, not measured wire bytes or a
universal bound on two-party inference. Compiler admission remains closed.

The original experiments and limitations are under `legacy/`. This handoff adds contracts and research tasks, **not a completed native reproduction of this paper**.

## Reproduction gates

`R13.acquire → R13.specify → R13.reference → R13.native → R13.assure → R13.benchmark → R13.document`.

The public research backlog records the remaining implementation and validation work. Native integration requires the actual PLLM command, source locks, raw measurements, and assurance report.

## Developer-facing explainer requirement

The eventual method page needs a worked operator example, a role diagram in prose or graphics, source section links, why/when to use it, configuration parameters, unsupported cases, numeric envelope, failure behavior and source-vs-PLLM differences. Generate tables from evidence rather than copying old performance ratios. Show `not implemented`, `adapted`, `reference only` or `reproduced at scope X` honestly.

## Default policy

`eligible_default = false` until implementation, full coverage, source/security review and deployment gates are met. The track restriction remains binding even after successful reproduction. Experimental or comparison methods cannot silently enter the preferred single-evaluator/no-HE profile.
