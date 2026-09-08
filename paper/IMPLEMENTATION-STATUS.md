# Implementation status

The manuscript records the preparation and inference lifecycle experiments from
before the Rust workspace migration. Its measurements retain that scope.

The current implementation is a mixed Python and Rust package, version
0.17.0a1. The source lives in `python/pllm`, `crates/pllm-core` and
`crates/pllm-python`. The Python numeric reference and real HE protocol tests
were run after the namespace migration. Rust compilation and performance were
not measured in the authoring environment.

See `../VALIDATION.md`, `../verification/` and
`../docs/content/docs/server/native.mdx`. A workspace change is not evidence of a
new cryptographic result, model quality result or throughput result.
