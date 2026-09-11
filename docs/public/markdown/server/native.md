# Rust execution core

Maturin packaging, exact integer kernels, thread policy, and evidence limits.


PLLM builds `pllm._native` with Rust, PyO3, and Maturin. Python retains checkpoint
import, model orchestration, protocol lifecycle, and Responses behavior. The
current public path uses Rust integer arithmetic and no HE library. SEAL/TenSEAL
remains external for optional BFV and confidential-weight modes.

## Workspace

`crates/pllm-core` contains the Python-independent numeric implementation.
`crates/pllm-python` binds it as `pllm._native`. Maturin packages that extension
beside `python/pllm`; no second Python namespace is installed.

| Operation | Current implementation |
| --- | --- |
| Public prepared online matrix multiplication | Rust exact wrapping arithmetic, runtime AVX2/NEON selection |
| `u16`, `u24`, `u32` wire encoding | Rust little-endian compact codecs |
| Client mask/reconstruction loops | Rust integer kernels |
| Weight and activation quantization | Rust float32, ties-to-even rounding |
| Random mask material | Operating-system entropy and rejection sampling |
| Legacy BFV encryption/evaluation/decryption | External SEAL through TenSEAL |

The compiler copies each matrix into an immutable native snapshot and reuses a
persistent Rayon executor. Python/native conversion is explicit, not zero-copy.
The current snapshot adds one signed byte per weight while source tensors remain
available for metadata and preparation loading. Include both storage and conversion
cost in measurements.

## Build and inspect

```bash
uv sync
uv run pllm build
```

`pllm build` inspects the installed extension. Rebuild after changing Rust with:

```bash
uv run maturin develop --release
uv run pllm build
```

```python
from pllm.native import capabilities

print(capabilities())
```

Capabilities report implementation, crate version, runtime SIMD availability,
Rayon support, matrix storage, and absence of a GPU backend. Missing native code
is an error by default. `PLLM_KERNEL_BACKEND=python` explicitly selects the NumPy
correctness reference; `PLLM_REQUIRE_RUST=1` prevents that substitution in native
validation.

## Threads

The default executor uses available CPU cores capped at 32. Override it with
`--engine-threads` for a service or `PLLM_NATIVE_THREADS` for direct construction;
valid values are 1 through 32. One thread can outperform a larger pool for small,
dependent stages, but it is not the source default. Benchmark the intended matrix
shapes and concurrency rather than assuming core count predicts latency.

Wheels use runtime feature detection instead of `target-cpu=native`. AVX2 and
ARM64 NEON use optimized paths; unsupported hosts use portable scalar code.

## Benchmark boundaries

```bash
uv run python benchmarks/rust_migration.py --cpp-reference --threads 1
uv run python benchmarks/client_native.py --output results/client.json
uv run pllm benchmark dashboard
```

Kernel runners verify outputs and separate compilation from calls. Synthetic
matrix shapes are not model generation. The dashboard measures its real local
three-role run, but one machine and checkpoint are not universal throughput.
Historical C++ or BFV measurements are not current Rust seeded-inventory results.

See [Validation](/docs/reference/compatibility) for current feature evidence and
[Performance](/docs/server/performance) for complete-lifecycle measurement.
