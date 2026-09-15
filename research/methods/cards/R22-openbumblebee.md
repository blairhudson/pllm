# R22 · BumbleBee: Secure Two-party Inference Framework for Large Transformers

**2025 · two-online-party comparison · source acquired 2026-09-14**

Primary paper: https://eprint.iacr.org/2023/1678  
Artifact: https://github.com/AntCPLab/OpenBumbleBee at `47c5560d069543591f3b0176eb530ede28e33fc3` (Apache-2.0).

BumbleBee provides additive-share transformer inference and specialized nonlinear protocols. The artifact includes LLaMA-7B generation, but reevaluates the full prefix rather than retaining a KV cache. Both computational parties remain online.

Reproduction must count both parties, preprocessing, communication, and repeated-prefix work. Its nonlinear components are useful comparisons, but its topology cannot silently satisfy PLLM's one-evaluator profile.
