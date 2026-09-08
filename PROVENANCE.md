# Source provenance

This repository was reworked from the supplied `pllm-rust-maturin.zip` archive.
Its SHA256 digest is:

```text
4c43e9e54d9fe3eccc2dbfb26e4511240240689f6cc7ba95b5adb933169945d7
```

The current source version is 0.17.0a1. The changes consolidate the Python
implementation into `python/pllm`, divide the Rust core and PyO3 binding into a
Cargo workspace, add lazy public exports, and align the build, tests, deployment,
Fumadocs and release workflows.

The Rust algorithms were carried forward from that archive. Matrix state was
encapsulated when exposing the core as a standalone crate. The HE scheme and
model arithmetic were not replaced. Historical research results are preserved
with their original scope. The paper was converted to canonical Pandoc Markdown
without relabelling any historical result as Rust performance.

The `verification` directory retains the initial migration evidence. Current
local results are listed in `VALIDATION.md`; previous release outputs are not
relabeled as current checks. Cross-platform wheel builds and publication remain
enforced by the supplied workflows.
