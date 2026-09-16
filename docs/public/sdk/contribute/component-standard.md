# Component standard

Required identity, lifecycle, representation, role, capability, evidence, and limitation fields.

[View canonical HTML](https://pllm.run/sdk/contribute/component-standard/)

Document ID: `pllm.docs.contribute.component-standard`  
Release: `0.1.0`  
Build: `sha256:4a93c61285a110010f1bafefa367e198ed52465071615e2d4d9a91a45f2d82e2`  
Source hash: `sha256:f326459a3dbff7422c16892470d161e7f5d4b42a195957baf701aef71e696de4`

1. Choose a semantic category and stable authority-owned identity.
2. Define versioned parameters, lifecycle phase, input and output representations, host requirements, roles, artifacts, and evidence pointers.
3. Implement in the semantic owner, not a paper-named runtime path.
4. Add static discovery and identity tests without importing provider or native runtime code.
5. Add compatibility and operator-coverage tests.
6. Add runtime, benchmark, quality, assurance, and deployment evidence independently when each exists.

The canonical contract is [`design/component-standard.md`](https://github.com/blairhudson/pllm/blob/main/design/component-standard.md). Missing fields remain `not recorded`; empty evidence is not a pass.

## Python SDK example

```python
from pllm.components import get_component

descriptor = get_component("pllm/cpu")
record = descriptor.to_dict()
assert record["component"] == "pllm/cpu"
assert {"provider", "version", "category", "capabilities"} <= record.keys()
```

Built-in descriptors expose the public fields contributors must keep discoverable.

API: [`pllm.components.get_component`](/sdk/reference/python/pllm/#objects-and-signatures)
