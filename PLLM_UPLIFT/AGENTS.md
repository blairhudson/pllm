# Instructions for integrating this handoff into PLLM

Read `START_HERE.md`, `docs/MIGRATION.md`, `docs/ARCHITECTURE.md`, `docs/SECURITY_ASSURANCE.md` and the current repository's own contributor instructions before editing production code.

## Do not assume the staging package is the current runtime

The handoff did not inspect the current repository. Preserve its working distribution/import names, CLI and supported Responses client API. Inspect its Cargo/Python lockfiles, native extension, actual model adapter, numeric manifests, inventory and tests. Do not overwrite its workspace or publish the staging package as a replacement.

## Hard target constraints

One online inference evaluator. Model-aware Preparation is offline-only. The Client ideally has no model and does minimal encode/decode work. HE, multiple online workers and hardware-backed modes are separately authorized comparisons, never implicit fixes for a missing operation. A client-local activation/head changes the topology and must be labeled.

## Native boundary

Rust owns model-data scans, compilation hot paths, arithmetic, cryptographic operations, allocation, transport, scheduling, measurement and adversary instrumentation. Python supplies convenient public configuration and result APIs. Pure Python numeric functions belong only in readable test oracles or explicitly labeled experiments. No per-gate/scalar FFI loop and no NumPy performance fallback.

## Evidence requirements

The twenty research cards are reproduction targets, not installed methods. Acquire full sources, commits and licenses; record source-to-code differences and proof applicability. Never claim a completed paper reproduction from an adapted equation. Do not invent expected benchmark numbers, source hashes, implementation status or passing tests. Rust code in this handoff was supplied but not compiled in its authoring environment: run the actual toolchain and fix errors before using it.

Formal checks prove only their explicit model and assumptions. Keep model-to-code refinement separate. Failed attacks do not prove privacy; successful attacks must identify the exact contract they violate. Do not let the attacker access honest-party secrets from the test driver. No scalar privacy score or unconditional `secure=true` output.

## Build sequence

Follow U01–U05 before parallelizing advanced methods. Port or adapt the small native core under existing naming conventions, compile and test, and preserve the archived full-model baseline before removing its path. R03/R04 underpin native garbling; R02/R05/R06 conversions/scaling need their own source gates. Add complete regions before a full decoder, and measure private attention, KV, selection, rescaling and replenishment—not just public matrix kernels.

## Completion report

For every change, report files changed, commands actually run, passed/failed/unrun tests, evidence artifacts and remaining blockers. Distinguish source-provided, compiled, operator-tested, full-model-tested, faithfully reproduced and security-reviewed. Do not mark the entire uplift complete because schemas or interface tests pass.
