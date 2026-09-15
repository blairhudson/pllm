# Deployment

Understand current deployment support, required role placement, and missing orchestration.

[View canonical HTML](https://pllm.run/sdk/operate/deployment/status/)

Document ID: `pllm.docs.operate.deployment`  
Release: `0.1.0`  
Build: `sha256:1e873bcc571b7b7717ee66b62aff1ec3e361ead12088e6b707b77e88ee008328`  
Source hash: `sha256:739fe6e092abaa92b71a58687a27c9cf30ab9a09bed61d46bb38bbfd901564a4`

Deployment declarations currently support local public configuration. Runtime contains development application/service factories and loopback integration coverage, but generic remote `Deployment`, plan-locked role orchestration, operator identity provisioning, and production recovery are not established.

Docker and systemd artifacts are templates, not certified deployments. Never infer authentication from port, hostname, or role string. Secret role state must remain separate from exportable plan artifacts and benchmark records.

Unavailable CLI families: `prepare`, `run`, `chat`, `serve`, and `party serve`. Do not use old command examples. See [compatibility/status](/sdk/reference/status/) and repository [deployment notes](https://github.com/blairhudson/pllm/tree/main/deploy).

## Python SDK example

```python
from pllm import Deployment

deployment = Deployment.local(root=".pllm/local")
print(deployment.kind)
```

Only local deployment configuration is supported; this does not perform orchestration.
