# Masked linear

Prepared public-weight linear inference with seeded one-time masks and committed inventory.

[View canonical HTML](https://pllm.run/sdk/pipeline/protocols/masked-linear/)

Document ID: `pllm.docs.protocols.masked-linear`  
Release: `0.1.0`  
Build: `sha256:43a7d6d03570b59100102980d9b739320473c2b1c5a1a31e49972f24b77b95e2`  
Source hash: `sha256:ba67dc0c5a378cc0c277b5b680695625d31789ebb82b774a63b8ec0768a50da3`

The prepared protocol separates client, trusted preparation, and untrusted inference roles. Offline, preparation computes `W*r-s` from domain-separated client seed batches and pushes committed corrections. Online, inference atomically consumes the ticket matching `x-r` and returns `W*x-s` for client reconstruction.

Inventory identity binds model, body, stage, weight, shape, quantization, ring, modulus, wire width, and attempt budget. Rows are one-use and burned on reservation failure, cancellation, replay, or early completion. Preparation is idle during online chat.

This contract assumes protocol-following non-colluding roles and public body weights. Confidential-weight and shared-state protocols are separate.

## Python SDK example

```python
import numpy as np

from pllm import AuthenticatedMPC, TrustedPreprocessor

preparation = TrustedPreprocessor(modulus=65537, seed=3)
runtime = AuthenticatedMPC(preparation)
x = np.array([[3, -2]], dtype=np.int64)
weight = np.array([[2, 1]], dtype=np.int64)
masked_x = runtime.input(x, preparation.input_mask(x.shape))
correlation = preparation.linear_correlation(weight, x.shape)
result = runtime.centered(runtime.open(runtime.linear(masked_x, weight, correlation)))
assert result.tolist() == [[4]]
assert runtime.stats.linear_rounds == 1
```

API: [Python objects and signatures](/sdk/reference/python/pllm/#objects-and-signatures)

This exercises prepared masked linear arithmetic in the single-process reference
protocol; it does not provide role separation or deployment assurance.
