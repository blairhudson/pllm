# R24 · Maverick: Private and Verifiable LLM Inference Made Practical via Matrix-Vector Multiplication Delegation

**Priority 24 · 2026 · single evaluator · source checked 2026-09-16**

Authors: Ben Merbaum, Mohammad Amin Raeisi, Wenhao Wang, Charalampos Papamanthou, Katerina Sotiraki, Fan Zhang.  
Primary source: https://arxiv.org/abs/2609.10264v1

## What the source contributes

Maverick delegates the matrix-vector multiplications that dominate LLM inference. Its construction combines transparent preprocessing, information-theoretic batch verification, and LPN-based pseudorandom input masking. The paper evaluates an end-to-end Qwen3-4B prototype and separates online-mask, precomputed-mask, and verification-only client configurations.

## PLLM perspective

The delegated linear operation is close to PLLM's prepared public-weight path, but the protocols and claims are not interchangeable. PLLM currently uses one-time masks and preloaded `W·r-s` corrections under an honest-but-curious, non-colluding preparation/inference assumption. It does not implement Maverick's verification or LPN masking construction.

An independent PLLM reproduction should first specify the paper's exact matrix-vector, masking, and verification interfaces; implement bounded native reference primitives; and compare them with the same Qwen3-4B workload and party/resource accounting. Similar data flow is not evidence of implementation, correctness, privacy, or verifiability.

## Status

Source identified. Reimplementation, assurance, and matched benchmark are planned and have not been executed.
