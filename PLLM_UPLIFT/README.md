# PLLM uplift package

**Developer handoff · 14 September 2026 · integration proposal plus executable assurance fixtures and native source**

Start with [START_HERE.md](START_HERE.md). This package consolidates the supplied PLLM experiments and design into a Rust-first development programme with **20 prioritized research reproductions**.

It is **not** twenty completed protocol implementations or a production private-LLM release. The existing PLLM repository was not available for inspection. No branch, dependency lock, public model checkpoint or production runtime has been modified or benchmarked here.

The preferred deployment is a small client, an offline-only model-aware Preparation service, and **one online Inference evaluator**. No HE or second online model worker enters that profile. Other protocols are explicit comparison cohorts.

## Contents

| Path | Purpose |
|---|---|
| `docs/ARCHITECTURE.md` | Rust-owned compiler, runtime, execution and metrics; Python developer API |
| `docs/RESEARCH_PORTFOLIO.md` | Ranked 20-paper programme and reproduction policy |
| `research/cards/` | One detailed implementation/reproduction card per paper |
| `research/recipes/` | Machine-readable acquisition, implementation and evidence gates |
| `research/top20.json`, `research/references.bib` | Canonical source records and citations |
| `docs/NATIVE_INTERFACES.md` | Types, traits, storage/FFI boundaries, crate ownership |
| `docs/SECURITY_ASSURANCE.md` | What is provable, falsifiable, empirical or assumed |
| `security/formal/` | Ten SMT-LIB finite-specification checks and counterexamples |
| `docs/BENCHMARKS.md` | Multi-party launch contract, compiler search, quality and complete cost |
| `docs/MODEL_ADAPTERS.md` | Qwen2.5-0.5B starting point and evolving architectures |
| `docs/MIGRATION.md`, `backlog/tasks.json` | Repository-safe migration, ordered work and acceptance criteria |
| `docs/PAPER_PLAN.md` | Original-artifact comparisons, ablations and publication claims |
| `native/` | Five dependency-free Rust crates, including exact kernels and a benchmark binary |
| `bindings/`, `python/` | PyO3/maturin binding and thin Python DX |
| `configs/`, `schemas/` | Strict policies and executable package configurations |
| `tests/`, `tools/`, `evidence/` | Contract tests, solver results, negative controls and repository inventory |
| `legacy/` | Ten original artifacts copied byte-for-byte, readable reports and hashes |

## Validation boundary

The Python contract suite, native Z3 specifications and public attack fixtures are executed in this handoff. The Rust/Cargo/maturin toolchain was unavailable in the authoring environment: native sources, Rust unit tests and wheel CI are supplied, but **not compiled or timed here**. `evidence/build_status.json` and `evidence/test_results.json` record the exact status. Earlier benchmark numbers remain archived, source-reported evidence; none is a result of this new native implementation.

No benchmark pass proves full cryptographic privacy. A solver result proves only its declared model/property. A reproduced attack refutes only the targeted construction/assumptions. Administrative non-collusion and cryptographic hardness remain explicit obligations.
