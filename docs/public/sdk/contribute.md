# Contribute

Add a model, component, research method, or evidence record without creating a one-off runtime path.

[View canonical HTML](https://pllm.run/sdk/contribute/)

Document ID: `pllm.docs.contribute`  
Release: `0.1.0`  
Build: `sha256:1e873bcc571b7b7717ee66b62aff1ec3e361ead12088e6b707b77e88ee008328`  
Source hash: `sha256:5888eb621e2fe9511afacef31f06a049d3c577ebbd65319865ecd5935ea0caf9`

Start with the [component standard](/sdk/contribute/component-standard/). Model
families use shared operations, research methods transform plans, providers expose
generic capabilities, and benchmark or assurance results attach to exact versions.

Keep identity, implementation, coverage, evidence, quality, privacy, and deployment
status separate. Do not add compiler branches named after a paper or hidden
registration that exists only in one runtime.

## Python SDK example

```python
from pllm.components import list_components

for component in list_components():
    print(component.component, component.category, component.version)
```

Contributors should preserve this side-effect-free discovery contract.
