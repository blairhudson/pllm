# Native plugin ABI

Boundary requirements for future independently distributed native capability providers.

[View canonical HTML](https://pllm.run/sdk/contribute/native-plugin-abi/)

Document ID: `pllm.docs.contribute.native-plugin-abi`  
Release: `0.1.0`  
Build: `sha256:87cbf81718b764da3fb4871c21f12e5943efd717a5d5854e5a0cecf969808585`  
Source hash: `sha256:18ab487287c0c85bacbb36f1101ad6d418351edc3235b1e2d75f046c387ca673`

The installed native module is currently built with PLLM. A stable third-party native plugin ABI is not published.

Any future ABI must version ownership, allocation, byte layout, alignment, panic isolation, threading, cancellation, capability negotiation, artifact identity, host-feature detection, and error translation. Python entry points must remain lightweight and side-effect-free during discovery.

No public Python API loads third-party native plugins because no stable plugin ABI is published.
See [current support](/sdk/reference/status/) for the supported built-in boundary.
