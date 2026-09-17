# Python reference

The installed pllm namespace and its generated public API inventory.

[View canonical HTML](https://pllm.run/sdk/reference/python/)

Document ID: `pllm.docs.reference.python`  
Release: `0.1.0`  
Build: `sha256:43a7d6d03570b59100102980d9b739320473c2b1c5a1a31e49972f24b77b95e2`  
Source hash: `sha256:783bba56b0a47ef52112173d60c5a36c9520e1f1fc492a16c10a9648fb071011`

`python/pllm` is the only installed namespace. Public objects are imported on demand so metadata discovery does not load model weights, devices, providers, or runtime state.

See the generated [`pllm` package inventory](/sdk/reference/python/pllm/) for exact modules, symbols, signatures, and native runtime identities.

## Python SDK example

```python
import pllm
from pllm.config import Experiment

assert pllm.Experiment is Experiment
```

API: [`pllm.Experiment`](/sdk/reference/python/pllm/#objects-and-signatures)
