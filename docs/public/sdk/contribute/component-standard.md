# Component standard

Required identity, lifecycle, representation, role, capability, evidence, and limitation fields.

[View canonical HTML](https://pllm.run/sdk/contribute/component-standard/)

Document ID: `pllm.docs.contribute.component-standard`  
Release: `0.1.0`  
Build: `sha256:bf56c232413fe9b57bc2befe009368956690afaf0d708914c7620c3b7d5d9531`  
Source hash: `sha256:44c34d2d0271fc8cd1aee38046a29ea349c4af180d8be68b0f3a48040c7f007a`

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
print(descriptor.to_dict())
```

Built-in descriptors expose the public fields contributors must keep discoverable.
