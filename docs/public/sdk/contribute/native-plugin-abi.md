# Native plugin ABI

C-compatible ownership, handles, status, and whole-region execution contracts for native providers.

[View canonical HTML](https://pllm.run/sdk/contribute/native-plugin-abi/)

Document ID: `pllm.docs.contribute.native-plugin-abi`  
Release: `0.1.0`

PLLM Native Plugin ABI 1 is published by `pllm-plugin-api` and `crates/pllm-plugin-api/include/pllm_plugin.h`. It is independently versioned from the Python API, PyO3 module, component standard, and wire protocols. A plugin exports `pllm_plugin_v1`; the host and plugin exchange size/version-tagged C vtables.

The ABI uses explicit-width scalars, borrowed byte slices, plugin-owned output buffers with an explicit release callback, and opaque `(slot, generation)` handles. Status values are integer wrappers so unknown foreign values do not create invalid Rust enums. No Rust `Vec`, `String`, trait object, allocator ownership, or unwind crosses the boundary. Implementations translate panics to `PLLM_STATUS_PANIC_VALUE` and process whole region or batch payloads rather than scalar callbacks.

The plugin vtable exposes synchronous whole-region execution and explicit asynchronous submit, poll, and cancel operations. Host callbacks expose event delivery and cancellation checks. Vtable size checks permit compatible append-only extension inside ABI 1; an unknown ABI version or undersized table fails closed. Provider manifests lock native artifact digests, targets, and ABI version before any future runtime load.

A Rust `cdylib` fixture and a C11 header fixture exercise symbol lookup, version negotiation, configuration, descriptor output, opaque binary region execution, cancellation, buffer release, destruction, and stale-generation rejection. This establishes the ABI contract and conformance harness, not trust or process isolation.

No public Python API loads third-party native libraries yet. Runtime host integration, role authorization, and process-isolation policy remain unavailable. PyO3 compatibility does not establish plugin ABI compatibility. See [current support](/sdk/reference/status/).
