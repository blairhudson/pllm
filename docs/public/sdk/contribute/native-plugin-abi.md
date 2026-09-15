# Native plugin ABI

Boundary requirements for future independently distributed native capability providers.

[View canonical HTML](https://pllm.run/sdk/contribute/native-plugin-abi/)

Document ID: `pllm.docs.contribute.native-plugin-abi`  
Release: `0.1.0`  
Build: `sha256:1e873bcc571b7b7717ee66b62aff1ec3e361ead12088e6b707b77e88ee008328`  
Source hash: `sha256:933475182fe1479487b1a8bcd85b2535544918912ddca77e5f99789b2d9bb85f`

The installed native module is currently built with PLLM. A stable third-party native plugin ABI is not published.

Any future ABI must version ownership, allocation, byte layout, alignment, panic isolation, threading, cancellation, capability negotiation, artifact identity, host-feature detection, and error translation. Python entry points must remain lightweight and side-effect-free during discovery.

## Python SDK example

```python
from pllm.kernels import Cpu

print(Cpu.describe().distribution)
```

This discovers the built-in distribution only. No third-party native plugin ABI is published.
