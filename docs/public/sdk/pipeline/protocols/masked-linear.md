# Masked linear

Prepared public-weight linear inference with seeded one-time masks and committed inventory.

[View canonical HTML](https://pllm.run/sdk/pipeline/protocols/masked-linear/)

Document ID: `pllm.docs.protocols.masked-linear`  
Release: `0.1.0`  
Build: `sha256:1e873bcc571b7b7717ee66b62aff1ec3e361ead12088e6b707b77e88ee008328`  
Source hash: `sha256:795f4e9cd825d2523af9d2a111eaf2ee1ebb540450a2a8404baaf8143238e801`

The prepared protocol separates client, trusted preparation, and untrusted inference roles. Offline, preparation computes `W*r-s` from domain-separated client seed batches and pushes committed corrections. Online, inference atomically consumes the ticket matching `x-r` and returns `W*x-s` for client reconstruction.

Inventory identity binds model, body, stage, weight, shape, quantization, ring, modulus, wire width, and attempt budget. Rows are one-use and burned on reservation failure, cancellation, replay, or early completion. Preparation is idle during online chat.

This contract assumes protocol-following non-colluding roles and public body weights. Confidential-weight and shared-state protocols are separate.

## Python SDK example

```python
from pllm.protocols.masked_linear import MaskedLinear

method = MaskedLinear()
print(method.to_spec())
```

This creates a configuration reference only; it does not allocate masks or start inference.
