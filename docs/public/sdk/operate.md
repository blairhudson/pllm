# Operate

Assign client, preparation, and inference responsibilities without weakening the privacy design.

[View canonical HTML](https://pllm.run/sdk/operate/)

Document ID: `pllm.docs.operate`  
Release: `0.1.0`  
Build: `sha256:65f4ad316621284cc28b60b1a825a6d9c6d1f108cbb5032aee0983de1e5c4966`  
Source hash: `sha256:701edae02def99edc0ffdf0514567fe24e04dac6e5d1ef1849c6f07e5b7a863f`

- [Client boundary](/sdk/operate/client-boundary/) keeps plaintext application data with the user.
- [Provider roles](/sdk/operate/provider-roles/) separate preparation and inference responsibilities.
- [Deployment](/sdk/operate/deployment/status/) describes current support and missing orchestration.

## Python SDK example

```python
import pllm

deployment = pllm.Deployment.local(root=".pllm/local")
assert deployment.to_spec() == {"kind": "local", "root": ".pllm/local"}
```

API: [`pllm.Deployment`](/sdk/reference/python/pllm/#objects-and-signatures)

This records local intent only; it does not start or authenticate services.
