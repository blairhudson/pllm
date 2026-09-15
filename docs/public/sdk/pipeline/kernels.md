# Kernels

Understand PLLM's native integer matrix kernels, memory costs, and CPU requirements.

[View canonical HTML](https://pllm.run/sdk/pipeline/kernels/)

Document ID: `pllm.docs.kernels`  
Release: `0.1.0`  
Build: `sha256:bf56c232413fe9b57bc2befe009368956690afaf0d708914c7620c3b7d5d9531`  
Source hash: `sha256:9f5d2e6a0dae2ffae8b8dcf7614f6844cc317690fdc9c28e37b58e53682cebef`

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
from pllm.kernels import Cpu

kernel = Cpu(threads=4)
print(kernel.to_spec())
```

This selects public kernel configuration only; it does not execute a model.
