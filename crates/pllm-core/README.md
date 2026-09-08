# PLLM core

Numeric operations for private inference. This Rust library has no Python
dependency. It provides immutable matrices, a persistent executor, exact modular
arithmetic, quantization, wire codecs and cryptographic mask sampling.

```bash
cargo test -p pllm-core
```

The library does not implement homomorphic encryption. The external HE backend
handles encryption, context validation and decryption. These kernels operate
on validated integer buffers supplied by the calling protocol.

The scalar implementation is a correctness reference. AVX2 and NEON paths are
selected at runtime when available. Do not compile portable wheels with the
build host's native CPU flags.
