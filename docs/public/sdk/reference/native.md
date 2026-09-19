# Native reference

Rust crates, the Python binding, data ownership, threading, and ABI support.

[View canonical HTML](https://pllm.run/sdk/reference/native/)

Document ID: `pllm.docs.reference.native`  
Release: `0.1.0`

`pllm-core` owns validated integer matrices and native execution. `pllm-models` owns model-neutral semantic lowering. Research-method crates transform semantic plans. `pllm-types` owns canonical records; `pllm-compiler` validates executable plans. `pllm-python` contains only the PyO3 boundary exposed as `pllm._native`.

The Python binding copies bytes into validated integer buffers, releases the interpreter lock around core work, and returns immutable bytes. Matrices own their validated weights; dimensions and contents cannot mutate through public Rust APIs. An executor owns its persistent Rayon pool.

The stable Python ABI starts at 3.11. No third-party native plugin ABI is currently published.

## Python SDK example

```python
from pllm.kernels import Cpu

descriptor = Cpu.describe()
assert descriptor.distribution == "pllm"
assert "cpu" in descriptor.capabilities
```

This inspects built-in metadata without loading a third-party plugin.

API: [`pllm.kernels.Cpu`](/sdk/reference/python/pllm/#objects-and-signatures)
