# R24 · Maverick: Private and Verifiable LLM Inference Made Practical via Matrix-Vector Multiplication Delegation

**Priority 24 · 2026 · single evaluator · source checked 2026-09-16**

Authors: Ben Merbaum, Mohammad Amin Raeisi, Wenhao Wang, Charalampos Papamanthou, Katerina Sotiraki, Fan Zhang.  
Primary source: https://arxiv.org/abs/2609.10264v1

## What the source contributes

Maverick delegates the matrix-vector multiplications that dominate LLM inference. Its construction combines transparent preprocessing, information-theoretic batch verification, and LPN-based pseudorandom input masking. The paper evaluates an end-to-end Qwen3-4B prototype and separates online-mask, precomputed-mask, and verification-only client configurations.

## PLLM perspective

The delegated linear operation is close to PLLM's prepared public-weight path, but the protocols and claims are not interchangeable. PLLM currently uses one-time masks and preloaded `W·r-s` corrections under an honest-but-curious, non-colluding preparation/inference assumption. Its Freivalds verification requires private, authenticated, one-use projections. Maverick instead uses a public coded-matrix preprocessing result, fresh private sparse checks, and dual-LPN input masks. These mechanisms have different offline work, trust, ring and memory contracts.

PLLM's first Rust reference isolates the *verification* algebra: a 6–10-row Walsh-linear code over the odd BabyBear field, with at least half-distance, public encoded weights, and private 41-sparse challenges sampled only after the response is fixed. Two independent repetitions and a bounded claim count detect checked single-coordinate forgeries. This small code expands exponentially and is deliberately **more expensive than direct matrix multiplication** at its supported dimensions. A separate bounded 1–128-row numeric RAA reference implements the paper's repeat/permute/scale/accumulate encoding and checks that `Gᵀ(Mx) = (GᵀM)x` against field matrix multiplication. Its transpose computes `Ge` from sparse coefficients, and a numeric test checks that subtracting offline `MGe` recovers the field product. These coefficients are deterministic test data, not a secure one-use sampler. The reference carries **no distance certificate** or soundness claim; it does not implement batch recursion or reviewed dual-LPN parameters. No input privacy, practical full-model verification, or real-checkpoint parity follows from these isolated checks.

A compiler-schedule-bound resource gate now counts the source's separate rate-1/8 BabyBear `P = M Gx` and `Q = Gyᵀ M` matrices without loading weights. On the checked Qwen2.5-0.5B-shaped configuration, 96 remotely owned W8 body stages contain 357,826,560 one-byte weight elements. The two four-byte-field preprocessed matrices need **at least 22,900,899,840 bytes** (21.33 GiB), before the batch-verification auxiliary matrix, source model, code state, and any copies. That exceeds both the 2 GiB client-residency budget and the 6 GiB compiled-cache limit; one stage alone may require 557,842,432 bytes. This is a *public configuration and schedule* calculation, not a measured checkpoint import, disk transfer, CPU cost, or a claim that streaming/amortization cannot work. The corresponding executable composition remains disabled until an explicit lifecycle and cost policy can account for the extra state.

A faithful delegated option needs an independently validated code with full-width distance/assumption parameters, a complete finite-field numeric contract against the existing W8A8 rings, published weight/preprocessing commitments, stage/session admission, and matched Qwen checkpoint accounting. The current reference cannot be selected through `Experiment`, serving or benchmark search.

## Status

Source identified. Bounded coded-verification and RAA numeric references plus a whole-schedule preprocessing-storage lower bound implemented; code-distance review, dual-LPN privacy, batch verification, real-model execution, security review and matched benchmark remain pending.
