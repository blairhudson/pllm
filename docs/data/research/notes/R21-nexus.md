# R21 · NEXUS: Secure Transformer Inference Made Non-interactive

**2025 · single-evaluator candidate · source acquired 2026-09-14**

Primary paper: https://www.ndss-symposium.org/wp-content/uploads/2025-868-paper.pdf  
Artifact: https://github.com/zju-abclab/NEXUS at `e15058c7f8f58fcf037c4709211109cb1be7d1e0` (GPL-3.0; external reproduction only).

NEXUS combines approximate homomorphic encryption with lightweight garbled circuits for non-interactive, semi-honest two-party transformer forward inference. Its artifact benchmarks BERT, RoBERTa and ViT. It does not establish private Qwen token lookup, KV-cached autoregressive decode, output-head selection, or token feedback.

PLLM must first reproduce the published forward workloads in isolation. Any Qwen decoder extension is a separately named adaptation with new correctness, quality, resource, and privacy evidence. `eligible_default = false` until complete operator and feedback coverage is executable and reviewed.
