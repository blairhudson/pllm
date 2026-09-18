# Operate

Assign client, preparation, and inference responsibilities without weakening the privacy design.

[View canonical HTML](https://pllm.run/sdk/operate/)

Document ID: `pllm.docs.operate`  
Release: `0.1.0`  
Build: `sha256:a3db51eca723684328314c5428e7e991d7b5bb20ff973c6d46dfe9a79aec3a08`  
Source hash: `sha256:6c6ea2aa33ae622e505697268d99b11df4ecd03aa7e728e52b180f260ac7d2be`

- [Client boundary](/sdk/operate/client-boundary/) keeps plaintext application data with the user.
- [Provider roles](/sdk/operate/provider-roles/) separate preparation and inference responsibilities.
- [Deployment](/sdk/operate/deployment/status/) describes current support and missing orchestration.
- [Serving integrations](/learn/integrations/) connect the trusted gateway to SDKs and coding agents.

## Python SDK example

```python
import pllm

deployment = pllm.Deployment.local(root=".pllm/local")
assert deployment.to_spec() == {"kind": "local", "root": ".pllm/local"}
```

API: [`pllm.Deployment`](/sdk/reference/python/pllm/#objects-and-signatures)

This records local intent only; it does not start or authenticate services.
