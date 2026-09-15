# Reference

Look up exact CLI commands, Python APIs, schemas, components, research records, and current support.

[View canonical HTML](https://pllm.run/sdk/reference/)

Document ID: `pllm.docs.reference`  
Release: `0.1.0`  
Build: `sha256:1e873bcc571b7b7717ee66b62aff1ec3e361ead12088e6b707b77e88ee008328`  
Source hash: `sha256:dd0833496a3391ffae5bffb22c86357a20d6621302e17cadd8d7e76395ffa35d`

- [Python API](/sdk/reference/python/pllm/) lists the public package exports.
- [Native API](/sdk/reference/native/) describes the Rust and PyO3 boundaries.
- [Schemas](/sdk/reference/schemas/) link to machine-readable contracts.
- [CLI reference](/cli/reference/) is generated from the real command parser.
- [Component catalog](/sdk/reference/components/) lists built-in components.
- [Research catalog](/research/records/method-catalog/) lists public research records.
- [Current support](/sdk/reference/status/) separates available, experimental, and unsupported behavior.

Run `uv run python scripts/generate_developer_reference.py --check` to verify committed inventories.

## Python SDK example

```python
from pllm.components import list_components

print([item.component for item in list_components()])
```
