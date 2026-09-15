<div align="center">

# PLLM

**A high-performance private LLM multi-party inference runtime and autonomous research harness.**

[Get started](#get-started) · [Learn](https://pllm.run/learn/) · [Research](https://pllm.run/research/) · [Architecture](ARCHITECTURE.md) · [Security](SECURITY.md)

</div>

PLLM is a Rust-first, Python-friendly system for private large language model
inference. It separates client data from remote computation across explicit
client, preparation, and inference roles. The same package provides a bounded
research workflow for independently implementing, testing, benchmarking, and
reviewing private-inference methods.

The project is built for high performance, but no universal speed or cost claim
follows from that goal. Results depend on the model, protocol, hardware, network,
and workload. See the evidence records for measured results and their limits.

## Why PLLM

Ordinary hosted inference gives one provider the prompt, context, model state,
and generated output. PLLM explores a different boundary: keep plaintext and
private state in the client-controlled environment, and distribute protected
computation across parties that are not assumed to share their views.

If this boundary can be made efficient and deployable, it could broaden the
supply of useful AI compute. Buyers could compare independent, regional, and
sovereign providers by price, location, latency, availability, and energy source
without granting each provider access to plaintext language. These are intended
economic and safety outcomes, not established production results.

## Two connected systems

### Multi-party inference runtime

Rust executes quantized integer matrix kernels, codecs, masking, and
reconstruction. Python owns model orchestration, client and provider roles,
checkpoint import, and the Responses API.

The current public-weight path prepares one-time matrix corrections before a
request. Prompt text, token IDs, private activation scales, attention state,
sampling, and decoded output stay inside the client boundary. Preparation and
inference hold the public transformer body, but receive different protocol values
at different times.

```text
OFFLINE, before READY

Client                         Preparation                    Inference
authorize inventory      ───▶ verify commitments         ───▶ register inventory
stage seed + row count   ───▶ derive one-time masks
                              compute W*r-s              ───▶ store corrections
Client ◀────────────────────── acceptance acknowledgements ─ Inference
Client ─────────────────────── seal inventory ─────────────▶ READY

ONLINE, after reservation

Client                         Preparation                    Inference
ticket + x-r ───────────────────────────────────────────────▶ consume once
       ◀──────────────────────────────────────────────────── W*x-s
add s and decode                idle                           public-body GEMM
```

Preparation is not part of the online request path. Starting a response reserves
one-time rows. Completion, cancellation, or failure burns every unused row in
that reservation.

### Autonomous research harness

PLLM also supports a reproducible research loop:

1. Lock the paper, source revision, artifact hash, and license.
2. Reimplement the method independently in PLLM's Rust and Python stack.
3. Test semantic fidelity, numeric behavior, and privacy assumptions separately.
4. Benchmark matched plans and preserve negative results.
5. Produce reviewable evidence and citation records without publishing claims automatically.

Upstream research implementations are specification and oracle inputs. They are
not vendored, linked, or used as PLLM runtime dependencies. Automated assessment
can organize evidence, but a person must review publication claims.

## Trust boundary

The prepared public-weight path assumes honest-but-curious, non-colluding roles:

- Preparation must follow the protocol, erase expanded masks, and not collude
  with inference. Self-hosting preparation keeps that trust inside the client
  boundary.
- Inference does not receive plaintext activations, but it is assumed to perform
  the documented computation. Authentication alone does not prove correct model
  execution.
- Timing, public model identity, tensor shapes, traffic shape, and approximate
  sequence length can remain visible.
- Two services controlled by one operator do not provide meaningful
  non-collusion.

This is not a malicious-security claim or a security certification. Read
[SECURITY.md](SECURITY.md) before moving a workload across machines.

## Get started

PLLM supports Python 3.11 through 3.13. Install the command from PyPI with
[uv](https://docs.astral.sh/uv/):

```bash
uv tool install pllm
pllm --version
pllm --help
pllm components list
```

For the Python SDK, add the same package to a project:

```bash
uv add pllm
uv run python -c "import pllm; print(pllm.__version__)"
```

The CLI currently inspects configurations, component descriptors, and research
records. It does not yet expose production service lifecycle or chat commands.
To exercise the real client, preparation, and inference roles on one machine,
run the development dashboard:

```bash
pllm dev dashboard --tiny --no-open
```

`--tiny` uses generated weights to test transport behavior. Its output is not
meaningful language and its measurements are not a production benchmark. Omit
`--tiny` to use the default public checkpoint.

For the next steps, use the authored guides:

- [Learn the system](https://pllm.run/learn/)
- [Use the CLI](https://pllm.run/cli/)
- [Use the Python SDK](https://pllm.run/sdk/)
- [Explore the research program](https://pllm.run/research/)
- [Check current support](https://pllm.run/sdk/reference/status/)

## Python API

The public package exposes planning, compilation, benchmark, assurance, and
runtime APIs. This example only lowers and inspects a semantic model plan; it does
not download weights or perform inference:

```python
import pllm

config = {
    "model_type": "qwen2",
    "hidden_size": 896,
    "intermediate_size": 4864,
    "num_hidden_layers": 24,
    "num_attention_heads": 14,
    "num_key_value_heads": 2,
    "vocab_size": 151936,
    "max_position_embeddings": 32768,
    "hidden_act": "silu",
    "rms_norm_eps": 1e-6,
    "rope_theta": 1000000,
    "tie_word_embeddings": True,
}

plan = pllm.lower_model(
    config,
    batch=1,
    max_input_tokens=128,
    max_new_tokens=32,
)
coverage = plan.coverage("research.single_evaluator")
print(plan.digest)
print(coverage.to_dict())
```

Semantic lowering, complete compiler coverage, runtime execution, model quality,
privacy evidence, and deployment support are separate claims.

## Development

```bash
uv sync --extra he --extra sdk
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
uv run pytest
uv run ruff check python/pllm scripts tests
uv build
```

After changing Rust code:

```bash
uv run maturin develop --release
PLLM_REQUIRE_RUST=1 uv run pytest -m rust
```

Documentation checks:

```bash
cd docs
bun install --frozen-lockfile
bun run check:content
bun test
bun run build
```

See [VALIDATION.md](VALIDATION.md) for the checks that have been run and
[RELEASING.md](RELEASING.md) for release instructions.

## Repository boundaries

`python/pllm` is the only installed Python namespace. `crates/pllm-core` has no
Python dependency; `crates/pllm-python` exposes `pllm._native`. The Fumadocs site
under `docs/`, papers under `paper/`, and historical evidence under `research/`
are not included in the wheel.

Code is licensed under [Apache 2.0](LICENSE). Model checkpoints have separate
licenses.
