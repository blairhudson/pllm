# Upstreaming

Move validated research work into maintained semantic and runtime components.

[View canonical HTML](https://pllm.run/sdk/contribute/upstreaming/)

Document ID: `pllm.docs.contribute.upstreaming`  
Release: `0.1.0`  
Build: `sha256:1e873bcc571b7b7717ee66b62aff1ec3e361ead12088e6b707b77e88ee008328`  
Source hash: `sha256:38e74f7c36b8ba0ac3bebe8ba8a758da457404cc3c61066b479ff0e389ef92d7`

Upstreaming requires locked sources, a clean-room implementation record where applicable, fidelity tests, model-neutral semantics, component identity, compiler coverage, numeric analysis, matched benchmark evidence, scoped assurance, quality evaluation, limitations, and maintenance ownership.

A new publication claim also needs a stable method ID and evidence that isolates
the contribution from the baseline. Failed and negative results remain useful
research records.

## Python SDK example

```python
from pllm.components import get_component

component = get_component("pllm/kv-cache-eviction")
print(component.version, component.evidence)
```

Empty evidence remains `not recorded`; component presence alone is not an upstreaming claim.
