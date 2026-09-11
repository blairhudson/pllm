# Migration to the current PLLM workspace

## Package layout

The previous source tree had a `pllm` facade and a separate `he_openai`
implementation. This tree installs one package from `python/pllm`:

```python
from pllm import OpenAI, AsyncOpenAI
from pllm.native import MaskedGEMM, capabilities
```

Code that imported internal `he_openai` modules must move to the public PLLM API.
`pllm.runtime` is implementation detail, not a compatibility namespace. The
retired package is not installed as an alias, avoiding duplicate runtime state.

| Previous location | Current location |
| --- | --- |
| `pllm/` | `python/pllm/` |
| `he_openai/` | `python/pllm/runtime/` |
| `rust/src/kernels.rs` | `crates/pllm-core/src/kernels.rs` |
| `rust/src/codec.rs` | `crates/pllm-core/src/codec.rs` |
| `rust/src/lib.rs` | `crates/pllm-python/src/lib.rs` |
| Root Cargo package | Cargo workspace with core and PyO3 crates |

Recreate a current public-path source environment with:

```bash
uv sync --locked
uv run pllm build
```

The `he` extra is no longer required for public-weight inference. Install it only
for explicit BFV, blinded confidential-weight, or research workflows. After
changing Rust, run `uv run maturin develop --release`. Missing native code is an
error unless `PLLM_KERNEL_BACKEND=python` explicitly selects the correctness
reference.

## Public protocol migration

Public-weight deployments now use `seeded-preparation` and must run both inference
and trusted preparation. The older per-stage online preparation flow is not a
fallback. Before accepting online work, the current flow:

1. Registers an inventory at inference with immutable model and stage commitments.
2. Authorizes it through preparation using a distinct credential.
3. Sends preparation one domain-separated root seed and row count per remote stage.
4. Pushes each prepared `W*r-s` batch to inference's fixed WebSocket endpoint.
5. Seals the complete inventory and requires status `READY`.

Online requests then reserve rows and contact inference only. Prompt prefill uses
compact packed batches; decode retains a persistent connection. Early completion,
cancellation, and failures burn all unused reserved rows. A restart or prepared
session idle expiry discards memory-only inventory. Operators should budget cold
preparation separately from warm response latency.

## Role-specific environment

Several similarly named credentials belong to different processes. Do not expose
one shared credential under every name.

| Process | Current environment | Purpose |
| --- | --- | --- |
| Inference | `PLLM_API_KEY` | Client-to-inference authentication |
| Inference | `PLLM_PROVIDER_PUSH_API_KEY` | Preparation push authentication |
| Preparation | `PLLM_API_KEY` | Client-to-preparation authentication |
| Preparation | `PLLM_INFERENCE_URL` | Fixed inference HTTP(S) origin |
| Preparation | `PLLM_PUSH_API_KEY` | Same push value configured at inference |
| Client | `PLLM_BASE_URL`, `PLLM_API_KEY` | Inference origin and credential |
| Client | `PLLM_PREPARATION_BASE_URL`, `PLLM_PREPARATION_API_KEY` | Preparation origin and credential |
| Client | `PLLM_PREPARED_INVENTORY_ROWS` | Minimum rows in each prepared stage batch |
| Local gateway | `PLLM_LOCAL_API_KEY` | Application-to-gateway authentication |

`PLLM_PREPARATION_API_KEY` is a client or Compose-level name. Inside the
preparation service process, its listener reads `PLLM_API_KEY`. Likewise,
preparation reads `PLLM_PUSH_API_KEY`, while inference reads
`PLLM_PROVIDER_PUSH_API_KEY` for the same channel credential.

## Model bundle changes

Public bundle schema 2 carries quantized token lookup and output-head matrices for
local execution. Tied embeddings reference one canonical matrix; untied boundaries
stay separate. This removes vocabulary-sized stages from the public remote path but
increases client bundle transfer and memory. Schema 2 token lookup quantization is
not numerically identical to independently quantized schema 1 lookup, so compare
quality against the matching clear quantized graph.

Confidential bundles do not expose these boundary matrices. Their blinded and
direct-BFV protocols remain optional compatibility paths with different latency,
dependencies, and security limits. They must not be used as performance evidence
for the current public path.

## Repository and deployment

Fumadocs remains under `docs/`; paper and historical research remain separate from
the wheel. Docker and systemd templates use distinct client-to-inference,
client-to-preparation, and preparation-to-inference credentials; Compose also
requires a local-gateway credential. Preparation needs the fixed correction
WebSocket route. Dependency locks are committed and should be consumed with
`uv sync --locked`, `npm ci`, and Cargo's locked mode where applicable.

See [SECURITY.md](SECURITY.md) for the trust model and [VALIDATION.md](VALIDATION.md)
for checks actually executed.
