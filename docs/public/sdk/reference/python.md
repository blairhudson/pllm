# Python reference

The installed pllm namespace and its generated public API inventory.

[View canonical HTML](https://pllm.run/sdk/reference/python/)

Document ID: `pllm.docs.reference.python`  
Release: `0.1.0`  
Build: `sha256:55dedf191ed9de69d95eb69a68160a404194419bf2f721a57103e95fdce4a4e0`  
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
