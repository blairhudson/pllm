# Command-line interface

Inspect PLLM configuration, components, and research records from the terminal.

[View canonical HTML](https://pllm.run/cli/)

Document ID: `pllm.docs.cli`  
Release: `0.1.0`  
Build: `sha256:1e873bcc571b7b7717ee66b62aff1ec3e361ead12088e6b707b77e88ee008328`  
Source hash: `sha256:6da475a0949f55f8cc565ff622cc3755862a03da1b822d70c88dfd4622c65f33`

The PLLM command-line interface inspects the package's public metadata. It does
not currently start inference services, compile executable plans, or run
benchmarks.

## Install the command

```bash
uv tool install pllm
pllm --help
```

This installs the `pllm` wheel from PyPI in an isolated tool environment. Use
`uv tool upgrade pllm` to update it.

## Available commands

- `pllm config` validates and exports public configuration.
- `pllm components` lists built-in component descriptions.
- `pllm research` lists research sources, methods, and recipes or assesses a
proposed publication record.
- `pllm dev` contains development-only tools.

Run `pllm COMMAND --help` for the options accepted by a command. Use
[`--format json`](/cli/reference/) when another program will consume the
result.

## Continue

- [CLI reference](/cli/reference/) lists every current command and option.
- [SDK configuration](/sdk/configuration/) explains configuration files and
content identity.
- [Research methods](/research/methods/) explains the records returned by
`pllm research`.
