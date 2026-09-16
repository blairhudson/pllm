# Reference

Look up exact CLI commands, Python APIs, schemas, components, research records, and current support.

[View canonical HTML](https://pllm.run/sdk/reference/)

Document ID: `pllm.docs.reference`  
Release: `0.1.0`  
Build: `sha256:dab1bcb88380ca9a4c2b79e3de085391f482bb97d841a9f0741979579c87dbfb`  
Source hash: `sha256:301c71676e2e7029758a8b7a9be75aaf317e6d07f58385b33f131cc76649a274`

- [Python API](/sdk/reference/python/pllm/) lists the public package exports.
- [Native API](/sdk/reference/native/) describes the Rust and PyO3 boundaries.
- [Schemas](/sdk/reference/schemas/) link to machine-readable contracts.
- [CLI reference](/cli/reference/) is generated from the real command parser;
task pages cover [configuration inspection](/cli/reference/config/show/),
[component inspection](/cli/reference/components/show/), and
[loopback benchmarking](/cli/reference/benchmark/run/).
- [Component catalog](/sdk/reference/components/) lists built-in components.
- [Research catalog](/research/records/method-catalog/) lists public research records.
- [Current support](/sdk/reference/status/) separates available, experimental, and unsupported behavior.

Run `uv run python scripts/generate_developer_reference.py --check` to verify committed inventories.

## Python SDK example

```python
import pllm
from pllm.components import ComponentRef

assert pllm.ComponentRef is ComponentRef
```

API: [`pllm.ComponentRef`](/sdk/reference/python/pllm/#objects-and-signatures)
