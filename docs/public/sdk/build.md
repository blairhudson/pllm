# Build

Define a reproducible PLLM system from configuration, models, components, and research methods.

[View canonical HTML](https://pllm.run/sdk/build/)

Document ID: `pllm.docs.build`  
Release: `0.1.0`  
Build: `sha256:1e873bcc571b7b7717ee66b62aff1ec3e361ead12088e6b707b77e88ee008328`  
Source hash: `sha256:f2b2505d1b1ceaf1097aa4b9ad31719954fc939291c5cb46bb33d370b15ee3a8`

- [SDK configuration](/sdk/configuration/) records what you intend to build.
- [Models](/sdk/build/model-adapters/) translate model-family settings into shared operations.
- [Model plans](/sdk/plans/) separate model structure from compiler and runtime support.
- [Components](/sdk/components/) select versioned capabilities.
- [Research methods](/research/recipes/build-method/) record provenance and plan changes.

## Python SDK example

```python
from pllm.components import list_components

print([component.component for component in list_components()])
```
