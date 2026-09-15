"""Deterministic coding-agent guidance for PLLM research work."""

from __future__ import annotations


def render_agents_guide() -> str:
    """Render repository-root guidance for a coding agent doing PLLM research."""
    return """# PLLM Research Agent Guide

## Goal

Find strong private-LLM methods, add needed reusable PLLM Python and Rust components, test correctness, privacy, and model quality, then run a matched benchmark against state-of-the-art reimplementations in PLLM.

Treat source discovery, implementation, evidence, and publication as separate stages. Do not infer runtime support or research success from a paper, recipe, schema, source record, or partial implementation.

## Architecture and locations

- `python/pllm/` is the only installed Python namespace. It owns public Python APIs, CLI surfaces, model and protocol orchestration, runtime integration, and the PyO3 facade.
- `crates/pllm-core/` owns reusable integer matrix execution, codecs, quantization, masking, and scalar/native parity paths without a Python dependency.
- `crates/pllm-types/`, `crates/pllm-models/`, and `crates/pllm-compiler/` own shared contracts, semantic decoder IR, and compilation. Add model-neutral semantics here instead of paper-specific runtime paths.
- `crates/pllm-assurance/` and `crates/pllm-bench/` own reusable assurance and benchmark machinery. Method-specific crates must still expose ordinary component and plan contracts.
- `research/methods/` contains source records, source locks, method records, cards, and citations. `research/recipes/` contains inert planned workflows, not executable research.
- `research/reference/<method-id>/` is the location for clean-room test references when needed. Production code must not import it.
- `research/assurance/` contains models and fixtures. `research/evidence/` contains accepted or historical scoped evidence; failed and negative results remain recorded.
- `schemas/` owns machine-record contracts, `tests/` owns Python composition and conformance tests, and `benchmarks/` contains focused benchmark drivers and controls.

## Find, Add, Test, Compare

1. **Find.** Identify the strongest relevant methods from primary papers. Record exact versions, citations, source access, assumptions, claimed scope, and license status. A paper figure is source-reported evidence, not a PLLM result.
2. **Add.** Specify method semantics from publications and approved public vectors. Implement reusable behavior in the Python or Rust owner that already owns the protocol, representation, kernel, compiler phase, or runtime contract. Do not build, link, vendor, translate, or import upstream implementations into PLLM.
3. **Test.** Test equations and reference behavior, native parity, edge and failure cases, privacy boundaries and leakage assumptions, complete compiler/runtime coverage, and generation quality on the declared model and workload. Fail closed on unknown or unsupported contracts; never substitute plaintext, Python, another topology, or a weaker protocol.
4. **Compare.** Reimplement eligible state-of-the-art baselines through the same PLLM component and plan interfaces. Benchmark complete locked candidates under one immutable cohort, report uncertainty and every resource axis, and retain failures, regressions, blocked work, and negative results.

## Matched benchmark rules

Use the same semantic task, model and tokenizer revisions, graph, workload, numeric policy, quality acceptance criteria, privacy and integrity contract, role topology, leakage assumptions, hardware allocation, resource limits, preparation and cache policy, warmups, repetitions, stopping rule, and failure accounting. Count all roles, phases, conversions, setup, preparation, retries, latency, throughput, memory, CPU, network, disk, quality, and privacy evidence. Changed scope creates a separate cohort and cannot support a matched speedup claim.

Paper tables and figures are not PLLM measurements. Kernel timings, synthetic shapes, and historical study data are not full-model PLLM results. Keep failed, unsupported, unavailable, and negative outcomes addressable; do not delete them, silently retune them, or average only successful runs.

## Verification

Run relevant focused tests while developing, then use the repository commands applicable to the change:

```bash
python3 research/validate.py
cargo test -p pllm-core
cargo clippy --workspace --all-targets -- -D clippy::correctness
uv run ruff check python/pllm scripts tests
uv run pytest
uv run python scripts/check_repository.py
uv build
uv run python scripts/check_distributions.py dist
```

For Rust edits, also run `cargo fmt --all`. After native binding changes, rebuild with `uv run maturin develop --release` and run `PLLM_REQUIRE_RUST=1 uv run pytest -m rust`.

## Missing capability and review gates

PLLM currently has no generic command that runs a full-model matched benchmark against state-of-the-art reimplementations. Existing records and focused benchmark tools do not supply that capability. Until a complete typed implementation, locked cohort, full-model execution path, quality evaluation, and comparison launcher exist, record the result as blocked or unsupported and make no full-model performance claim.

Human review is required for security conclusions, privacy-contract changes, disclosure-sensitive failures, benchmark comparisons intended for public use, and all public or publication claims. Automated checks organize evidence; they do not approve claims.
"""
