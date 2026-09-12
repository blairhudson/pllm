# Source provenance

This repository was originally reworked from the supplied
`pllm-rust-maturin.zip` archive. Its SHA-256 digest is:

```text
4c43e9e54d9fe3eccc2dbfb26e4511240240689f6cc7ba95b5adb933169945d7
```

The current source version is `0.17.0a1`. The workspace has one installed Python
namespace, `python/pllm`, a Python-independent Rust core, and a PyO3 binding built
with Maturin. Git history after the archive migration records the seeded relay,
offline prepared inventory, prepared transformer execution, and long-context
prefill changes independently.

The current public-weight application path is not the archive's BFV lifecycle
experiment. It creates a committed seeded inventory offline, pushes one-way
`W*r-s` correction batches to inference, seals the inventory `READY`, and keeps
online traffic between client and inference. Later changes added one-time row
reservation and burn, `u16`/`u24`/`u32` rings, compact batched prefill, persistent
decode transport, and local public token-boundary matrices.

The current loopback benchmark dashboard is operational instrumentation for that
three-role path. Its schema-versioned SQLite history stores sanitized, insert-only
run summaries for local model/context comparison. A retained nine-run
Qwen2.5-0.5B study at revision `277d19f` is published as
`research/evidence/current-runtime-2026-09-11.json`; its one-host loopback scope
does not support general performance, price, energy, or quality claims.

Rust arithmetic and codecs evolved from the supplied implementation, with matrix
state encapsulated behind the standalone core API. Python still owns model and
protocol orchestration. SEAL/TenSEAL remains an external dependency only for
explicit BFV and confidential-weight compatibility paths and research.

Historical research measurements remain under `research/` with their original
scope. They are not relabeled as current Rust or public seeded-inventory results.
Generated site and paper assets have their own build provenance; current-runtime
measurement claims trace to the retained JSON evidence rather than generated prose.

`verification/` retains initial migration evidence. [VALIDATION.md](VALIDATION.md)
records checks run against later source states; configured CI jobs are requirements,
not evidence that a particular revision passed remotely.
