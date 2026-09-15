# R23 · MPCache: MPC-Friendly KV Cache Eviction for Efficient Private LLM Inference

**Registry alias R23 · 2025 · three-online-party comparison · source acquired 2026-09-14**

Primary paper: https://arxiv.org/abs/2501.06807v2  
Artifact: https://github.com/zwxandy/MPCache at `55e401b17db3c47dd6920b9a92ad937c7e9bb452` (MIT).

Publication source: `pllm.source.mpcache.arxiv-2501.06807v2`  
Upstream artifact lock: `pllm.upstream.mpcache.55e401b17db3c47dd6920b9a92ad937c7e9bb452`  
PLLM method: `pllm.method.mpcache-structural-adaptation.v1`

MPCache evaluates static and dynamic KV-cache eviction in a three-party SecretFlow/SPU setting. Its LLaMA experiments provide a direct stateful attention comparison, but the artifact focuses eviction and operator costs rather than establishing PLLM's complete protected model boundary.

PLLM uses the artifact as an external oracle only. Current Rust work is a structural `DecoderPlan` adaptation, not a reproduction of MPCache protected execution. Fidelity, protected execution, assurance, matched benchmark, and promotion remain unchecked. Three online parties and incomplete boundary evidence do not satisfy the preferred complete-model privacy contract.
