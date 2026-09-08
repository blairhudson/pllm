<div align="center">

# pllm

**Private language model inference. Rust execution. Python integration.**

[Quick start](#quick-start) · [Documentation](docs/) · [Architecture](ARCHITECTURE.md) · [Paper](paper/) · [Contributing](CONTRIBUTING.md)

</div>

PLLM is one Python package built with Maturin. Its Rust core handles integer
matrix execution, wire encoding, quantization and masking. Python handles the
client, provider, checkpoint importer and Responses API. SEAL through TenSEAL
provides homomorphic encryption.

The application sends text to a client inside the customer's trusted environment.
That client prepares encrypted masks and sends masked tensors to the provider.
It keeps prompts, token identities, activation scales, keys and decoded output
local. The provider stores the large projection matrices.

```text
Application / OpenAI SDK
           │ plaintext Responses request
           ▼
PLLM client or local gateway
  keys · tokenizer · attention · private state · sampling
           │ encrypted preparation and masked integer tensors
           ▼
PLLM provider
  checkpoint importer · scheduler · Rust matrix execution
```

The current protocol assumes the provider follows the computation. Channel
checks do not prove correct provider execution, and guarded query limits do not
protect confidential weights from a modified client. See [SECURITY.md](SECURITY.md).

## Quick start

Install UV and the Rust toolchain specified in `rust-toolchain.toml`. Python
3.11 through 3.13 is supported. From this checkout:

On Homebrew, put the keg-only rustup proxies before any separately installed
Rust formula so the repository pin takes effect:

```bash
export PATH="$(brew --prefix rustup)/bin:$PATH"
```

```bash
uv sync --extra he
uv run pllm build
uv run pllm --help
```

Maturin compiles `pllm._native` during installation. `pllm build` reports the
installed backend; it does not compile code inside a running inference service.

Start a supported Hugging Face snapshot on the provider:

```bash
export PLLM_API_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
uv run pllm serve ./models/checkpoint \
  --weights public \
  --model-id private-model \
  --host 127.0.0.1 --port 8000
```

On the customer machine, use the same provider credential:

```bash
uv run pllm configure \
  --server http://127.0.0.1:8000 \
  --api-key "$PLLM_API_KEY" \
  --model private-model
uv run pllm chat
```

Replace the loopback address with the provider's HTTPS address when it is on a
separate host. The client must remain inside the customer's trusted environment.

For a persistent CLI installation from source:

```bash
uv tool install '.[he]'
```

For a temporary isolated invocation from this checkout:

```bash
uvx --from '.[he]' pllm --help
```

These commands install this repository, not an unrelated package that might
occupy the `pllm` registry name. Publication is a separate release step.

## Python client

The client reads the settings saved by `pllm configure`:

```python
from pllm import OpenAI

with OpenAI() as client:
    response = client.responses.create(
        input="Explain private inference in plain English.",
        max_output_tokens=128,
    )
    print(response.output_text)
```

`AsyncOpenAI` and streaming are also exposed from `pllm`. Importing the package
does not load the provider, model tensors, native extension or HE backend.

For the official OpenAI SDK or an Agents SDK application, start the local gateway:

```bash
export PLLM_LOCAL_API_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
uv run pllm sidecar --local-api-key "$PLLM_LOCAL_API_KEY" --port 8080
```

Point the application at `http://127.0.0.1:8080/v1`, not the remote provider.
See the [OpenAI guide](docs/content/docs/client/openai.mdx) and
[Agents guide](docs/content/docs/client/agents.mdx) for the supported API surface.

## One repository, one distribution

```text
pllm/
├── pyproject.toml             # UV project, Maturin build and Python dependencies
├── Cargo.toml                # Workspace version and shared Rust dependencies
├── rust-toolchain.toml
├── python/pllm/              # The only installed Python namespace
│   ├── __init__.py           # Public API, loaded on demand
│   ├── cli.py                # Installed pllm command
│   ├── client.py             # Python client API
│   ├── server.py             # Provider application factory
│   ├── native.py             # Compiled matrix API
│   └── runtime/              # Model graph and protocol implementation
├── crates/
│   ├── pllm-core/            # Rust library without a Python dependency
│   └── pllm-python/          # PyO3 binding built as pllm._native
├── docs/                     # Fumadocs site, static export and browser search
├── paper/                    # Pandoc Markdown source and manuscript build
├── tests/                    # Python, protocol, model and packaging tests
├── benchmarks/               # Native comparisons and profiling
├── research/                 # Historical studies and recorded evidence
├── deploy/                   # Docker and systemd deployment
└── .github/workflows/        # PR tests, native wheels, PyPI and GitHub Pages
```

There is no second Python distribution, no installed `he_openai` namespace and
no setuptools build. The C++ benchmark control is retained only under
`benchmarks/reference`; it is not the runtime backend.

## Development

```bash
uv sync --extra he --extra sdk
cargo test -p pllm-core
cargo clippy --workspace --all-targets -- -D clippy::correctness
uv run pytest
uv run ruff check python/pllm scripts tests
uv build
```

To rebuild the binding after changing Rust:

```bash
uv run maturin develop --release
PLLM_REQUIRE_RUST=1 uv run pytest -m rust
```

Native validation never substitutes the reference backend. For arithmetic and
protocol development without a native extension, select the reference explicitly:

```bash
PLLM_KERNEL_BACKEND=python uv run pytest -m 'not rust and not he and not sdk'
```

This still installs the project with Maturin. To run the same reference tests
before a native build in an already provisioned Python environment:

```bash
PYTHONPATH=python PLLM_KERNEL_BACKEND=python python -m pytest -m 'not rust and not he and not sdk'
```

Reference tests are not evidence that the Rust code compiled or ran faster.
The [validation record](VALIDATION.md) separates executed checks from checks
that require a connected native build environment.

## Documentation and paper

Install Pandoc and either Tectonic or TeX Live. The lightweight Homebrew setup is
`brew install pandoc tectonic`.

```bash
make paper
uv run --no-project --python 3.13 python scripts/prepare_docs.py
cd docs
npm install
npm test
npm run typecheck
npm run build
```

Fumadocs exports to `docs/out`. GitHub Pages supports both a repository path and
a configured custom domain. Browser search and Markdown downloads stay in the
static site. The website does not provide a prompt submission endpoint.

## Releases

Generate real dependency locks before the first release:

```bash
python scripts/lock_dependencies.py
```

Review and commit `Cargo.lock`, `uv.lock` and `docs/package-lock.json`. CI can
bootstrap a source checkout without locks, but publication and deployment require
committed locks. The release workflow builds platform wheels, tests them outside
the checkout and publishes using PyPI trusted publishing. See [RELEASING.md](RELEASING.md).

## License and research

Code is licensed under [Apache 2.0](LICENSE), with retained notices. Model
checkpoint licenses are separate. The [paper](paper/) records the experiments
and their scope; historical timings are not Rust benchmark results. Citation
metadata is in [CITATION.cff](CITATION.cff).
