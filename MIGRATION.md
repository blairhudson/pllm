# Migration to the PLLM workspace

The previous source tree had a `pllm` facade and a separate `he_openai`
implementation. This tree has one package, `python/pllm`. Public imports remain:

```python
from pllm import OpenAI, AsyncOpenAI
from pllm.native import MaskedGEMM, capabilities
```

Applications that imported internal `he_openai` modules must use the public PLLM
API or update those imports to `pllm.runtime`. The retired namespace is not
installed as an alias. This prevents two module identities for runtime state.

| Previous location | Current location |
| --- | --- |
| `pllm/` | `python/pllm/` |
| `he_openai/` | `python/pllm/runtime/` |
| `rust/src/kernels.rs` | `crates/pllm-core/src/kernels.rs` |
| `rust/src/codec.rs` | `crates/pllm-core/src/codec.rs` |
| `rust/src/lib.rs` | `crates/pllm-python/src/lib.rs` |
| Root Cargo package | Cargo workspace with two crates |

This is a repository and API namespace migration. It does not change the HE
scheme, model arithmetic, protocol security boundary or reference checkpoint.

Recreate the source environment with `uv sync --extra he`. After editing Rust,
use `uv run maturin develop --release`. `pllm build` reports the installed
backend. A missing extension is an error; `PLLM_KERNEL_BACKEND=python` is an
explicit reference mode for development, not a silent fallback.

The website remains Fumadocs under `docs`, and the paper source remains under
`paper`. GitHub Actions, distribution validation, tests, Docker builds and UV
scripts use the same source layout.

The dependency locks must be resolved from the registries and reviewed before
publication. This archive does not invent locks or claim a native binary build
that was not executed. See `VALIDATION.md` for the checks performed on this tree.
