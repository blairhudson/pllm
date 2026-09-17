# Deployment

Learn how to assign PLLM roles to authenticated services without changing privacy assumptions.

[View canonical HTML](https://pllm.run/sdk/operate/deployment/)

Document ID: `pllm.docs.deployment`  
Release: `0.1.0`  
Build: `sha256:5305ca7ad557c9bb6bc08f507fdb6c80bb709c09979323514d5730cb3edd75f9`  
Source hash: `sha256:ed02e65f51ae641c76f4e46fb24192ab873ab2807c994325afb77df69915a1e4`

A deployment assigns each protocol role to a process and operator. It records
endpoints, authentication, separate credentials, provider identity, storage,
capacity, expiration, observability, shutdown behavior, and trust boundaries.
Review [privacy and threat models](/learn/privacy-and-threat-models/) before
changing that placement.

The client must keep plaintext prompts, masks, private scales, model state, and
decoding. Client-to-inference, client-to-preparation, and
preparation-to-inference connections use different credentials. Changing the URL
of an ordinary model API does not make its unmodified SDK private.

A loopback [`pllm benchmark run`](/cli/reference/benchmark/run/) checks that the
services work together on one machine. It does not show that production operators
are independent. Deployment assurance must refer to the exact plan and environment
being evaluated. The [serving integration guide](/learn/integrations/local-gateway/)
lists the current gateway and provider-role lifecycle commands.

## Python SDK example

```python
from pllm.deployment import Deployment

deployment = Deployment.local(root=".pllm/local")
assert deployment.to_spec() == {"kind": "local", "root": ".pllm/local"}
```

API: [`pllm.deployment.Deployment`](/sdk/reference/python/pllm/#objects-and-signatures)

This creates configuration only. Generic remote role deployment is not supported.
