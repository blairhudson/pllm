# Deployment

Understand current deployment support, local role orchestration, and production gaps.

[View canonical HTML](https://pllm.run/sdk/operate/deployment/status/)

Document ID: `pllm.docs.operate.deployment`  
Release: `0.1.0`

Deployment declarations currently support local public configuration. `pllm.serve_local` and `pllm.runtime.build_roles` accept a typed `Model`, `Pipeline`, or `Experiment` and provide one bounded loopback topology for the local gateway, dashboard, and benchmark driver while keeping inference and preparation in separate operating-system processes. The handle owns environment-only role credentials, distinct ports, health checks, client routing, and deterministic shutdown. Generic remote `Deployment`, plan-locked multi-host orchestration, operator identity provisioning, and production recovery are not established.

Docker and systemd artifacts are templates, not certified deployments. Never infer authentication from port, hostname, or role string. Secret role state must remain separate from exportable plan artifacts and benchmark records.

Use the exact [gateway and role lifecycle commands](/learn/integrations/local-gateway/). They start processes; they do not provision hosts, TLS, identities, or non-colluding operators. See [compatibility/status](/sdk/reference/status/) and repository [deployment notes](https://github.com/blairhudson/pllm/tree/main/deploy).

## Python SDK example

```python
import pllm
from pllm.runtime import build_roles

deployment = pllm.Deployment.local(root=".pllm/local")
assert deployment.to_spec() == {"kind": "local", "root": ".pllm/local"}

model = pllm.Model.path("/srv/models/Qwen2.5-0.5B-Instruct")
topology = build_roles(model, correlation_mode="local-test")
assert topology.started is False
assert [status.role for status in topology.statuses] == ["inference", "preparation"]
```

`build_roles` only validates and builds the handle. Entering it, calling `start()`, or calling `pllm.serve_local(...)` starts both role processes and may resolve the model.

API: [`pllm.Deployment` and `pllm.serve_local`](/sdk/reference/python/pllm/#objects-and-signatures)

Only loopback development orchestration is supported; it does not provision hosts, TLS, identities, or non-colluding operators.
