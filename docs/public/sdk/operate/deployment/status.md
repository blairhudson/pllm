# Deployment

Understand current deployment support, required role placement, and missing orchestration.

[View canonical HTML](https://pllm.run/sdk/operate/deployment/status/)

Document ID: `pllm.docs.operate.deployment`  
Release: `0.1.0`  
Build: `sha256:4a02d6d6c164f106cb6698f1e14f6ed79d2e45e68a44719d9f0aa417993b4f8a`  
Source hash: `sha256:320cf49735cbf4d8396953563062d172799336adc709cb3c3ad376b56da4aab9`

Deployment declarations currently support local public configuration. Public CLI commands start the trusted gateway and separate inference and preparation roles, but generic remote `Deployment`, plan-locked role orchestration, operator identity provisioning, and production recovery are not established.

Docker and systemd artifacts are templates, not certified deployments. Never infer authentication from port, hostname, or role string. Secret role state must remain separate from exportable plan artifacts and benchmark records.

Use the exact [gateway and role lifecycle commands](/learn/integrations/local-gateway/). They start processes; they do not provision hosts, TLS, identities, or non-colluding operators. See [compatibility/status](/sdk/reference/status/) and repository [deployment notes](https://github.com/blairhudson/pllm/tree/main/deploy).

## Python SDK example

```python
import pllm

deployment = pllm.Deployment.local(root=".pllm/local")
assert deployment.to_spec() == {"kind": "local", "root": ".pllm/local"}
```

API: [`pllm.Deployment`](/sdk/reference/python/pllm/#objects-and-signatures)

Only local deployment configuration is supported; this does not perform orchestration.
