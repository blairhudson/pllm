# pllm components show

Exact pllm components show help from the installed PLLM CLI.

[View canonical HTML](https://pllm.run/cli/reference/components/show/)

Document ID: `pllm.docs.reference.cli.components.show`  
Release: `0.1.0`

## Example

```bash
pllm components show pllm/cpu --format json
```

## Options

```text
usage: pllm components show [-h] [--format {human,json,jsonl}] [--quiet]
                            [--no-color] [--no-input] [--dry-run]
                            COMPONENT

positional arguments:
  COMPONENT             component identity, for example pllm/cpu

options:
  -h, --help            show this help message and exit
  --format {human,json,jsonl}
                        result format (default: human)
  --quiet               suppress non-error diagnostics
  --no-color            disable colored output
  --no-input            fail instead of prompting
  --dry-run             report work without persistent writes or service
                        startup
```
