# PLLM

<p align="center">
  <img src="https://raw.githubusercontent.com/blairhudson/pllm/main/docs/public/logo.svg" alt="PLLM" width="160">
</p>

<p align="center">
  Private LLM inference and reproducible experimentation.
</p>

<p align="center">
  <a href="https://pllm.run">Documentation</a> ·
  <a href="https://pypi.org/project/pllm.run/">PyPI</a> ·
  <a href="https://github.com/blairhudson/pllm">Source</a> ·
  <a href="LICENSE">Apache-2.0</a>
</p>

PLLM is a high-performance private LLM multi-party inference runtime. Its trusted
client boundary retains prompts, model state, masks, token boundaries, and output
while separate services perform prepared masked computation.

PLLM is also an extensible experimentation system. Typed component and provider
contracts, immutable experiments, constrained search, benchmarking, and evidence
records help turn credible private-inference research into reproducible results
that can be compared against eligible state-of-the-art baselines.

## Install

Install the base package:

```console
uv tool install pllm.run
```

The distribution is `pllm.run`; the Python import and command are both `pllm`.
Python 3.11 through 3.13 is supported.

Runtime variants and optional dependencies are documented in the
[installation guide](https://pllm.run/learn/installation/).

Follow the [first private request](https://pllm.run/learn/first-private-request/)
for a complete gateway example.

## What PLLM provides

- A trusted loopback gateway with Responses API and Chat Completions API surfaces.
- Prepared public-weight inference with distinct Client, Preparation, and
  Inference roles.
- Model-neutral semantic lowering and fail-closed compilation.
- Versioned component, provider, plan, benchmark, and evidence contracts.
- Grid and seeded random search over validated experiment spaces.
- Cohort-safe benchmark comparison without implicit cross-workload ranking.
- Python orchestration backed by bounded Rust kernels and typed native handles.

## Research workflow

1. Reimplement a method behind the smallest compatible component or provider
   contract.
2. Compose immutable experiments and reject incompatible candidates before
   execution.
3. Search bounded parameter spaces without bypassing correctness, privacy,
   quality, or runtime gates.
4. Benchmark the exact plan and preserve its workload, environment, metrics, and
   limitations.
5. Compare only equivalent cohorts with the strongest eligible baseline.

PLLM does not automatically translate papers into implementations, invent missing
evidence, or rank incomparable systems. Start with the
[Research guide](https://pllm.run/research/) and
[chronological paper bibliography](https://pllm.run/research/papers/).

## Documentation

- [Get started](https://pllm.run/learn/)
- [Understand the trust boundary](https://pllm.run/learn/concepts/trust-boundary/)
- [Use the Python SDK](https://pllm.run/sdk/)
- [Use the CLI](https://pllm.run/cli/)
- [Extend components](https://pllm.run/sdk/components/)
- [Search and benchmark](https://pllm.run/sdk/research/)
- [Check current support](https://pllm.run/sdk/reference/status/)
- [Read the research](https://pllm.run/research/)

## Security and status

The prepared public-weight protocol assumes Preparation and Inference follow the
protocol and do not collude. Local mode co-locates them for development and does
not establish non-collusion. The project does not claim protection against
arbitrary malicious providers or production security review.

PLLM is an alpha release. Complete protected whole-model composition and broad
multi-host, WAN, GPU, quality, energy, price, and adversarial evidence remain
incomplete. Read [current support](https://pllm.run/sdk/reference/status/) before
making deployment or research claims.

## Development

```console
uv sync --all-extras --dev
uv run pytest
cargo test
```

Documentation development and publication commands live in
[`docs/README.md`](docs/README.md). Release changes use
[`scripts/release.py`](scripts/release.py).

## Citation and license

Citation metadata is in [`CITATION.cff`](CITATION.cff). PLLM is licensed under
the [Apache License 2.0](LICENSE).
