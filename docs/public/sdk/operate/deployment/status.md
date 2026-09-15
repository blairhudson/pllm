# Deployment

Understand current deployment support, required role placement, and missing orchestration.

[View canonical HTML](https://pllm.run/sdk/operate/deployment/status/)

Document ID: `pllm.docs.operate.deployment`  
Release: `0.1.0`  
Build: `sha256:65f4ad316621284cc28b60b1a825a6d9c6d1f108cbb5032aee0983de1e5c4966`  
Source hash: `sha256:2e99a4002ebccd03e7bff63f4c8a011f980a6815a236eb6076cf853e47487e69`

Deployment declarations currently support local public configuration. Runtime contains development application/service factories and loopback integration coverage, but generic remote `Deployment`, plan-locked role orchestration, operator identity provisioning, and production recovery are not established.

Docker and systemd artifacts are templates, not certified deployments. Never infer authentication from port, hostname, or role string. Secret role state must remain separate from exportable plan artifacts and benchmark records.

Unavailable CLI families: `prepare`, `run`, `chat`, `serve`, and `party serve`. Do not use old command examples. See [compatibility/status](/sdk/reference/status/) and repository [deployment notes](https://github.com/blairhudson/pllm/tree/main/deploy).

## Python SDK example

```python
import pllm

deployment = pllm.Deployment.local(root=".pllm/local")
assert deployment.to_spec() == {"kind": "local", "root": ".pllm/local"}
```

API: [`pllm.Deployment`](/sdk/reference/python/pllm/#objects-and-signatures)

Only local deployment configuration is supported; this does not perform orchestration.
