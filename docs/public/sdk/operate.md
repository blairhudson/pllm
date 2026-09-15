# Operate

Assign client, preparation, and inference responsibilities without weakening the privacy design.

[View canonical HTML](https://pllm.run/sdk/operate/)

Document ID: `pllm.docs.operate`  
Release: `0.1.0`  
Build: `sha256:1e873bcc571b7b7717ee66b62aff1ec3e361ead12088e6b707b77e88ee008328`  
Source hash: `sha256:3b50fc5e188402f2f53bc0c77e8ede831dbc5a2be5bada8c311c25f603ed58b9`

- [Client boundary](/sdk/operate/client-boundary/) keeps plaintext application data with the user.
- [Provider roles](/sdk/operate/provider-roles/) separate preparation and inference responsibilities.
- [Deployment](/sdk/operate/deployment/status/) describes current support and missing orchestration.

## Python SDK example

```python
from pllm import Deployment

deployment = Deployment.local(root=".pllm/local")
print(deployment.to_spec())
```

This records local intent only; it does not start or authenticate services.
