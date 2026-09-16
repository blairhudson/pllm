# Command-line interface

Inspect PLLM records and run local benchmarks from the terminal.

[View canonical HTML](https://pllm.run/cli/)

Document ID: `pllm.docs.cli`  
Release: `0.1.0`  
Build: `sha256:4a93c61285a110010f1bafefa367e198ed52465071615e2d4d9a91a45f2d82e2`  
Source hash: `sha256:472e9ea33319126ca22bf8e2305291982f4e9d7e17a19589262cf1ae9cc386cd`

The PLLM command-line interface inspects public metadata, runs bounded local
benchmarks, and starts the gateway, inference, and preparation service roles.
Service commands do not imply production orchestration or make incomplete
compiled plans executable.

## Install the command

```bash
uv tool install pllm
pllm --help
```

This installs the `pllm` wheel from PyPI in an isolated tool environment. Use
`uv tool upgrade pllm` to update it.

## Choose a task

- [Run private inference](/cli/private-inference/) through the trusted
gateway, then connect an SDK or raw HTTP client.
- [Operate provider roles](/cli/provider-roles/) as independently
configured inference and preparation services.
- [Benchmark and diagnose](/cli/benchmarking/) with the headless runner
or loopback dashboard.
- [Inspect configuration and research](/cli/inspect-and-research/)
without starting a model.

Each guide links to exact parser-generated command help under
[Command reference](/cli/reference/).

## Available commands

- `pllm config` validates and exports public configuration.
- `pllm components` lists built-in component descriptions.
- `pllm benchmark run` runs the real client, preparation, and inference roles on
loopback and returns a text-free diagnostic record.
- `pllm research` lists research sources, methods, and recipes or assesses a
proposed publication record.
- `pllm gateway` starts the trusted loopback API gateway and can co-locate both
provider roles for development.
- `pllm serve` starts one inference or preparation role from explicit deployment
configuration.
- `pllm dev` contains development-only tools.

Run `pllm COMMAND --help` for the options accepted by a command. Use
[`--format json`](/cli/reference/) when another program will consume the
result.

## Inspect configuration

Validate a public configuration or export its canonical form:

```bash
pllm config show examples/pllm.yaml
pllm config export examples/pllm.yaml --output experiment.json
```

See [`config show`](/cli/reference/config/show/) and
[`config export`](/cli/reference/config/export/).

## Inspect components

List component descriptors or inspect one stable component identity:

```bash
pllm components list --format json
pllm components show pllm/cpu --format json
```

See [`components list`](/cli/reference/components/list/) and
[`components show`](/cli/reference/components/show/).

## Explore research records

Discover records in machine-readable form, then inspect records by stable ID or
registry alias:

```bash
pllm research sources list --format json
pllm research sources show R01
pllm research methods show pllm.method.mpcache-structural-adaptation.v1
pllm research recipes show R01
```

Discovery references: [`sources list`](/cli/reference/research/sources/list/),
[`methods list`](/cli/reference/research/methods/list/), and
[`recipes list`](/cli/reference/research/recipes/list/). Record references:
[`sources show`](/cli/reference/research/sources/show/),
[`methods show`](/cli/reference/research/methods/show/), and
[`recipes show`](/cli/reference/research/recipes/show/).

Assess a publication request without running a workflow, or generate coding-agent
guidance at repository root:

```bash
pllm research assess examples/publication-assessment.json --format json
pllm research agents --output AGENTS.md
```

See [`research assess`](/cli/reference/research/assess/) and
[`research agents`](/cli/reference/research/agents/).

## Run a benchmark

The default model is `Qwen/Qwen2.5-0.5B-Instruct`:

```bash
pllm benchmark run \
  --prompt-file prompt.txt \
  --max-output-tokens 24 \
  --repetitions 3 \
  --output benchmark.json
```

The command starts and stops all three roles. Use `--tiny` only for a random-weight
transport smoke test. Reports exclude prompts, generated text, token IDs,
activation payloads, masks, seeds, and credentials. They are single-host
diagnostics, not canonical research `EvidenceReport` records. See
[`benchmark run`](/cli/reference/benchmark/run/).

## Open the development dashboard

The dashboard is development-only. `--tiny` uses random weights for a local
transport smoke test:

```bash
pllm dev dashboard --tiny
```

See [`dev dashboard`](/cli/reference/dev/dashboard/).

## Continue

- [CLI reference](/cli/reference/) lists every current command and option.
- [Local gateway](/learn/integrations/local-gateway/) gives the exact public
serving commands and config boundaries.
- [SDK configuration](/sdk/configuration/) explains configuration files and
content identity.
- [Research methods](/research/methods/) explains the records returned by
`pllm research`.
