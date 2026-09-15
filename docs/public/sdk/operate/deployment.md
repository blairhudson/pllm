# Deployment

Learn how to assign PLLM roles to authenticated services without changing privacy assumptions.

[View canonical HTML](https://pllm.run/sdk/operate/deployment/)

Document ID: `pllm.docs.deployment`  
Release: `0.1.0`  
Build: `sha256:65f4ad316621284cc28b60b1a825a6d9c6d1f108cbb5032aee0983de1e5c4966`  
Source hash: `sha256:f514264025ce4ee91762bca2f05d29a025c59715d418b91de2c3c5fe3703fa5f`

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
being evaluated.

## Python SDK example

```python
from pllm.deployment import Deployment

deployment = Deployment.local(root=".pllm/local")
assert deployment.to_spec() == {"kind": "local", "root": ".pllm/local"}
```

API: [`pllm.deployment.Deployment`](/sdk/reference/python/pllm/#objects-and-signatures)

This creates configuration only. Generic remote role deployment is not supported.
