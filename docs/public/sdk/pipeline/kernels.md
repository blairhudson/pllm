# Kernels

Understand PLLM's native integer matrix kernels, memory costs, and CPU requirements.

[View canonical HTML](https://pllm.run/sdk/pipeline/kernels/)

Document ID: `pllm.docs.kernels`  
Release: `0.1.0`  
Build: `sha256:4358d00f593125ffb699b8a2329c69e607727783bf40149859925cbfa47327b5`  
Source hash: `sha256:32182996dd67509970655fcec9753669b156314e8e1aee10ccb742e796d9a484`

`pllm-core` implements bounded integer matrices, scalar reference code, runtime
AVX2 and NEON selection, codecs, quantization, masking, subtraction, and operating
system randomness. It copies validated weights into Rust once and reuses them.
Python-to-Rust buffer conversions still create copies.

Kernel records list required CPU features, threading, input and output formats,
supported operations, memory use, and evidence. Moving arithmetic to Rust does
not remove protocol messages, offline preparation, client state, or conversion
costs.

## Python SDK example

```python
from pllm import ConfigurationError, Cpu

kernel = Cpu(threads=4)
assert kernel.to_spec() == {"component": "pllm/cpu", "params": {"threads": 4}}
try:
    Cpu(threads=0)
except ConfigurationError:
    pass
else:
    raise AssertionError("Cpu accepted a non-positive thread count")
```

API: [Python objects and signatures](/sdk/reference/python/pllm/#objects-and-signatures)

This checks public kernel configuration and its thread bound; it does not execute a model.
