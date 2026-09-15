# Native reference

Rust crates, the Python binding, data ownership, threading, and ABI support.

[View canonical HTML](https://pllm.run/sdk/reference/native/)

Document ID: `pllm.docs.reference.native`  
Release: `0.1.0`  
Build: `sha256:1e873bcc571b7b7717ee66b62aff1ec3e361ead12088e6b707b77e88ee008328`  
Source hash: `sha256:c51d239bc943c24e89c50523fef7bd90a3336506a1d9f8d2933dcb8c4a251080`

`pllm-core` owns validated integer matrices and native execution. `pllm-models` owns model-neutral semantic lowering. Research-method crates transform semantic plans. `pllm-types` owns canonical records; `pllm-compiler` validates executable plans. `pllm-python` contains only the PyO3 boundary exposed as `pllm._native`.

The Python binding copies bytes into validated integer buffers, releases the interpreter lock around core work, and returns immutable bytes. Matrices own their validated weights; dimensions and contents cannot mutate through public Rust APIs. An executor owns its persistent Rayon pool.

The stable Python ABI starts at 3.11. No third-party native plugin ABI is currently published.

## Python SDK example

```python
from pllm.kernels import Cpu

descriptor = Cpu.describe()
print(descriptor.distribution, descriptor.required_host_features)
```

This inspects built-in metadata without loading a third-party plugin.
