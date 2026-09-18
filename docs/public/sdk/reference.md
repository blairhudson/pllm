# Reference

Look up exact CLI commands, Python APIs, schemas, components, and current support.

[View canonical HTML](https://pllm.run/sdk/reference/)

Document ID: `pllm.docs.reference`  
Release: `0.1.0`  
Build: `sha256:8db228446618c1e95120afa32d874de6f8cba2e6e356c6b0c1157f0c5b7cc58e`  
Source hash: `sha256:ad1bcccd732d8bda3c513f4360146223fca6f12ec312a1e21c06aad9650bef89`

- [Python API](/sdk/reference/python/pllm/) lists the public package exports.
- [Native API](/sdk/reference/native/) describes the Rust and PyO3 boundaries.
- [Schemas](/sdk/reference/schemas/) link to machine-readable contracts.
- [CLI reference](/cli/reference/) is generated from the real command parser;
task pages cover [configuration inspection](/cli/reference/config/show/),
[component inspection](/cli/reference/components/show/), and
[loopback benchmarking](/cli/reference/benchmark/run/).
- [Component catalog](/sdk/reference/components/) lists built-in components.
- [Research papers](/research/papers/) and the [research backlog](/research/backlog/)
track source status and planned work.
- [Current support](/sdk/reference/status/) separates available, experimental, and unsupported behavior.

Run `uv run python scripts/generate_developer_reference.py --check` to verify committed inventories.

## Python SDK example

```python
import pllm
from pllm.components import ComponentRef

assert pllm.ComponentRef is ComponentRef
```

API: [`pllm.ComponentRef`](/sdk/reference/python/pllm/#objects-and-signatures)
