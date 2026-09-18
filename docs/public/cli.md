# Command-line interface

Inspect PLLM records and run local benchmarks from the terminal.

[View canonical HTML](https://pllm.run/cli/)

Document ID: `pllm.docs.cli`  
Release: `0.1.0`  
Build: `sha256:2e8eacabaf27e4c40f6b41814c3af934ab5186546825b569c6c9c1cb0a7b4db4`  
Source hash: `sha256:3d5aa8861a0179a01eaa09ff63d72f52a1aada96b0a8e83979b9608fbd0679e3`

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
- [Inspect configuration and components](/cli/inspect-and-research/)
without starting a model.

Each guide links to exact parser-generated command help under
[Command reference](/cli/reference/).

## Available commands

- `pllm config` validates and exports public configuration.
- `pllm components` lists built-in component descriptions.
- `pllm benchmark run` runs the real client, preparation, and inference roles on
loopback and returns a text-free diagnostic record.
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
- [Research papers](/research/papers/) and the [research backlog](/research/backlog/)
document source status and unfinished work.
