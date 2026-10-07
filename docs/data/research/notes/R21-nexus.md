# R21 · NEXUS: Secure Transformer Inference Made Non-interactive

**2025 · single-evaluator candidate · source acquired 2026-09-14**

Primary paper: https://www.ndss-symposium.org/wp-content/uploads/2025-868-paper.pdf  
Artifact: https://github.com/zju-abclab/NEXUS at `e15058c7f8f58fcf037c4709211109cb1be7d1e0` (GPL-3.0; external reproduction only).

NEXUS uses RNS-CKKS throughout non-interactive, semi-honest transformer forward inference: coefficient/slot matrix packing, polynomial nonlinear work and bootstrapping. Sections II–IV of the pinned PDF contain no garbled-circuit boundary. The earlier HE–GC summary was incorrect. Table IV reports BERT-base and Llama-3-8B one-output forward workloads, including encrypted argmax. It does not establish private Qwen token lookup, KV-cached autoregressive decode or private token feedback across a complete response.

PLLM independently implements the published contracts before measuring forward workloads. Upstream source is a specification/oracle input, never a runtime dependency. Any Qwen decoder extension is a separately named adaptation with new correctness, quality, resource, and privacy evidence. `eligible_default = false` until complete operator and feedback coverage is executable and reviewed. The inaccurate planned `NexusHeGarblingConversion` placeholder is replaced by the still-unavailable `NexusCkksBootstrap` contract; SEAL does not supply that bootstrap implementation.
