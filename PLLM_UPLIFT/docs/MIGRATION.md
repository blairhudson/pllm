# Migration into the current PLLM repository

## Repository status

This handoff has not inspected the current repository, branch, lockfiles or package version. It therefore provides additive modules, contracts and ordered acceptance gates—not a patch that assumes paths or renames existing APIs. The previous artifacts are preserved in `legacy/`; their tests/results are historical until rerun.

## Public compatibility

Preserve the current distribution/import/CLI/API names, particularly `pllm-inference`, `pllm`, `pllm._native` where present, `pllm serve`, and the supported Responses-compatible client interface. Inventory any older aliases before removal. The production gateway is trusted-client code; an API-compatible plaintext request must never be sent directly to an untrusted inference server because a protocol backend was unavailable.

The staging `pllm-uplift-lab` distribution and `pllm_uplift` namespace deliberately do not shadow production. After acceptance, merge modules into the existing Cargo/PyO3 workspace, reuse its supported version locks and delete the staging namespace. Do not overwrite a working `pyproject.toml`, Cargo workspace or server entry point with the sample files.

## Ordered changes

| PR | Scope | Merge gate |
|---|---|---|
| U01 | Read-only inventory; archive manifests; identify actual tested baseline and API coverage | Baseline artifacts and missing numeric locks are explicit; existing tests unchanged |
| U02 | Add source/method/assurance schemas and generated method docs | All records validate; no unsupported reproduction/security labels |
| U03 | Native type/representation/material contracts and adapters around existing kernels | Old graph remains numerically identical; type errors fail closed |
| U04 | Port remaining runtime hot path from Python into Rust: allocation, transport, scheduling and clocks | Fresh-wheel same-process and separated-role parity, no per-gate Python calls |
| U05 | Common benchmark launcher and role/view capture interfaces | Accurate direction/phase counters and preparation-disconnect gate |
| U06 | Add native source primitives/conversions through reviewed construction plans | Original-artifact comparison, parameter checks and source/assumption lock |
| U07 | Full Qwen archival baseline under new harness | Real model, recovered numeric manifest, quality baseline and explicit client-heavy profile |
| U08 | Complete single-evaluator regions then decoder/state/feedback | No hidden client operations; all conversions/preparation counted |
| U09 | Regional compiler search and alternative methods | Better held-out frontier versus best fixed backend at the same contract |
| U10 | Independent reproduction and paper publication bundle | Separate security review, quality and full-system benchmarks |

Integration may parallelize research implementations after contracts stabilize, but cannot skip their source/proof gates. Twenty repository folders containing stubs are not twenty implemented methods.

## Compatibility adapters

Wrap each historical experiment behind its original numerical and role contract. Preserve source-reported metrics without renaming them as new native results. Derive stable identifiers for model graph, quantization, method, conversion, kernel, material schema and source version. Add adapters incrementally and remove the old execution path only after baseline equivalence and recovery tests.

The old report's server-consumed tickets are not enough for client freshness. Move assignment authority to the party that can create/reuse the masking input, with a supported durable state mechanism. Model state rollback must not roll back cryptographic material. Similarly, old tags/fingerprints cannot be relabeled as a full malicious-security mechanism.

## CI and release checks

Core Rust tests; scalar/SIMD/device parity; PyO3 wheel tests on supported platforms; Python API compatibility; native-boundary profiling; format/semantic/numeric/model conformance; source and method schema validation; formal and adversarial regression checks; fault/retry and resource counter checks; deterministic public artifacts; license/security review of vendored sources.

Pin workflow dependencies to reviewed immutable revisions in the production repository. The supplied CI uses conventional major action tags as a starting template, not a hardened supply-chain configuration. The staging Rust source was not compiled in the authoring environment; the first integration gate is an actual toolchain build/test before reuse.

## Completion definition

A complete uplift has a usable DX, native execution, at least one full-model eligible plan, reproducible comparative methods and an assurance report whose unproved assumptions are visible. “All top20 reproduced” is a separate portfolio milestone requiring every source's original artifact or faithful equivalent evidence, not merely passing common interface tests.
