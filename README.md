<div align="center">

# pllm

**Prepare once offline. Run masked inference online. Keep prompts on the client.**

[Quick start](#quick-start) · [Documentation](docs/) · [Architecture](ARCHITECTURE.md) · [Security](SECURITY.md) · [Validation](VALIDATION.md)

</div>

PLLM is one Python package built with Maturin. Rust executes quantized integer
matrices, wire codecs, masking, and reconstruction. Python owns model
orchestration, the client, provider roles, checkpoint import, and a Responses
interface.

The current public-weight path prepares one-time matrix corrections before chat.
Prompt text, token IDs, activation scales, attention state, sampling, and decoded
output stay inside the customer-controlled client. Preparation and inference each
hold the public transformer body, but receive different shares at different times.

```text
OFFLINE, before READY

Client                         Trusted preparation            Inference
inventory authorization  ───▶ validates model commitments ─▶ registers inventory
stage root seed + row count ─▶ expands r and s
                               computes W*r-s             ───▶ stores corrections
Client ◀──────────────────────── durable acknowledgements ─── Inference
Client ───────────────────────── seal request ───────────────▶ READY

ONLINE, after row reservation

Client                         Preparation                    Inference
ticket + x-r ────────────────────────────────────────────────▶ consume once
       ◀───────────────────────────────────────────────────── W*x-s
add s, center-decode           idle                            public-body GEMM
```

Correction payloads move one way, from preparation to inference. Acknowledgements
confirm durable acceptance but do not return corrections to the client. Prefill
uses one compact ticket vector and packed matrix per remote stage. Decode uses one
ticket per stage over a persistent client-to-inference connection. Preparation is
not on the online path.

Each stage selects the smallest exact `u16`, `u24`, or `u32` ring justified by its
signed output bound. Public client bundles include quantized token lookup and
output-head matrices, so those vocabulary boundaries execute locally. The
transformer-body matrices remain at the services.

## Trust boundary

The public path protects an activation from either service viewed alone only
under an honest-but-curious, non-colluding model:

- Preparation must follow the protocol, erase expanded masks, and not collude
  with inference. Self-hosting preparation keeps this trust inside the customer
  boundary.
- Inference is not trusted with plaintext, but it is assumed to execute the
  documented computation. Authentication does not prove correct model execution.
- Timing, stage names, tensor shapes, scheduling, and approximate sequence lengths
  remain visible.
- Inventory lives in memory. Restart or idle expiry discards it. Starting an
  execution reserves rows; cancellation, failure, or early completion burns all
  unused rows in that reservation.

This is not a malicious-security claim or a production security certification.
See [SECURITY.md](SECURITY.md) before moving a workload across machines.

## Quick start

Install [UV](https://docs.astral.sh/uv/) and the Rust toolchain selected by
`rust-toolchain.toml`. Python 3.11 through 3.13 is supported. From this checkout:

```bash
uv sync
uv run pllm build
uv run pllm --help
```

`uv sync` compiles `pllm._native`. The public seeded-inventory path does not need
TenSEAL or the `he` extra. `pllm build` reports the installed backend; it does not
compile code inside a running service.

Choose three distinct credentials, then start inference and trusted preparation
with the same local public checkpoint:

```bash
export PLLM_API_KEY="$(uv run python -c 'import secrets; print(secrets.token_urlsafe(32))')"
export PLLM_PREPARATION_API_KEY="$(uv run python -c 'import secrets; print(secrets.token_urlsafe(32))')"
export PLLM_PROVIDER_PUSH_API_KEY="$(uv run python -c 'import secrets; print(secrets.token_urlsafe(32))')"

uv run pllm serve ./models/checkpoint \
  --weights public \
  --model-id demo-model \
  --api-key "$PLLM_API_KEY" \
  --provider-push-api-key "$PLLM_PROVIDER_PUSH_API_KEY" \
  --local-files-only
```

In another trusted process with the same credential values:

```bash
uv run pllm preparation serve ./models/checkpoint \
  --model-id demo-model \
  --api-key "$PLLM_PREPARATION_API_KEY" \
  --inference-url http://127.0.0.1:8000 \
  --push-api-key "$PLLM_PROVIDER_PUSH_API_KEY" \
  --local-files-only
```

Configure and run the client:

```bash
uv run pllm configure \
  --server http://127.0.0.1:8000 \
  --api-key "$PLLM_API_KEY" \
  --preparation-url http://127.0.0.1:8001 \
  --preparation-api-key "$PLLM_PREPARATION_API_KEY" \
  --model demo-model
uv run pllm chat
```

`pllm chat` creates a `READY` inventory before accepting a prompt, reserves the
required rows for each response, and refills only while no response is online.
Loopback HTTP is for local evaluation. Remote inference and preparation origins
must be distinct HTTPS origins.

For a persistent command installed from this checkout:

```bash
uv tool install .
```

For one isolated invocation:

```bash
uvx --from . pllm --help
```

These commands install this source tree, not an unrelated project that may use the
same registry name.

## Python client

Prepare enough rows before starting a response:

```python
from pllm import OpenAI

prompt = "Explain private inference in plain English."
maximum = 128

with OpenAI() as client:
    required = client.prepared_rows_for_response(prompt, maximum)
    client.preprocess(count=required)
    response = client.responses.create(
        input=prompt,
        max_output_tokens=maximum,
    )
    print(response.output_text)
```

`AsyncOpenAI` and streaming are also exposed from `pllm`. Keep a client open to
reuse unreserved rows. A response never performs preparation after entering its
online phase.

For an ordinary OpenAI SDK or Agents SDK application, run the local gateway inside
the customer boundary:

```bash
export PLLM_LOCAL_API_KEY="$(uv run python -c 'import secrets; print(secrets.token_urlsafe(32))')"
uv run pllm sidecar --local-api-key "$PLLM_LOCAL_API_KEY" --port 8080
```

Point the application at `http://127.0.0.1:8080/v1`, not the inference provider.
The sidecar prepares its configured public model at startup. Larger later requests
may require an authenticated `/v1/preprocess` call while the sidecar is idle. See
the [OpenAI guide](docs/content/docs/client/openai.mdx).

## Live benchmark dashboard

Run the real client, preparation, and inference roles on loopback with the default
Qwen2.5-0.5B-Instruct checkpoint:

```bash
uv run pllm benchmark dashboard
```

The first launch may download the public checkpoint. The dashboard reports
observed OTLP process metrics, protocol bytes, inventory use, phase timing, time
to first token, and decode rate. Sanitized run records are stored by default in
`$XDG_STATE_HOME/pllm/benchmark-history.sqlite3` (or
`~/.local/state/pllm/benchmark-history.sqlite3`) for model/context comparison;
prompt and output text are not retained. Use `--history-db :memory:` for an
ephemeral session. For a fast transport smoke test with random tiny weights:

```bash
uv run pllm benchmark dashboard --tiny
```

Tiny output is not meaningful language, and neither dashboard mode is a universal
performance or security result. See the [dashboard guide](docs/content/docs/reference/dashboard.mdx)
and [performance guide](docs/content/docs/server/performance.mdx).

## Optional legacy paths

TenSEAL/SEAL remains available for confidential-weight blinded and direct-BFV
reference paths, plus historical research reproduction. These are not selected by
`--weights public`:

```bash
uv sync --extra he
```

Guarded query limits do not protect confidential weights from a modified client,
and the authenticated arithmetic preview is not a deployed malicious-security
protocol. Treat these paths as explicitly selected compatibility or research
modes, not stronger defaults.

## Development

```bash
uv sync --extra he --extra sdk
cargo test -p pllm-core
cargo clippy --workspace --all-targets -- -D clippy::correctness
uv run pytest
uv run ruff check python/pllm scripts tests
uv build
```

After changing Rust:

```bash
uv run maturin develop --release
PLLM_REQUIRE_RUST=1 uv run pytest -m rust
```

The Python numeric path is an explicit correctness reference, never evidence that
native code compiled or ran faster:

```bash
PLLM_KERNEL_BACKEND=python uv run pytest -m 'not rust and not he and not sdk'
```

Documentation content checks run independently of generated research assets:

```bash
cd docs
npm ci
npm run check:content
npm test
```

See [VALIDATION.md](VALIDATION.md) for checks actually executed and
[RELEASING.md](RELEASING.md) for artifact publication.

## Repository boundaries

`python/pllm` is the only installed namespace. `crates/pllm-core` is independent
of Python; `crates/pllm-python` exposes `pllm._native`. Fumadocs under `docs/`, the
paper under `paper/`, and historical evidence under `research/` are not installed
in the wheel. Historical measurements are not native Rust or current model
throughput results.

Code is licensed under [Apache 2.0](LICENSE). Checkpoint licenses are separate.
