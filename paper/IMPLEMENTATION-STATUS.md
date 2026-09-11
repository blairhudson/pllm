# Historical and current implementation boundary

Paper measurements belong only to executable study under `research/lifecycle`.
Its project metadata identifies version `0.14.0+inference.study1`, derived from
persisted PLLM 0.14 research source. It uses Python, PyTorch, TenSEAL, and the
artifact's C++ SEAL binding. Target-sized weights are synthetic.

Current package is mixed Python/Rust PLLM 0.17.0a1. Its source lives under
`python/pllm`, `crates/pllm-core`, and `crates/pllm-python`. Current package
tests, protocol behavior, native-kernel performance, serving interfaces, and
security properties are not evaluated by paper experiments.

Repository migration is not evidence of a new cryptographic result, checkpoint
quality result, or throughput result. Reuse of historical numbers for current
runtime requires a fresh, versioned reproduction.
