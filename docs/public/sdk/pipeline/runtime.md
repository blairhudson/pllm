# Runtime

Understand which live state a PLLM runtime owns and how it differs from a reproducible plan.

[View canonical HTML](https://pllm.run/sdk/pipeline/runtime/)

Document ID: `pllm.docs.runtime`  
Release: `0.1.0`  
Build: `sha256:1e873bcc571b7b7717ee66b62aff1ec3e361ead12088e6b707b77e88ee008328`  
Source hash: `sha256:8d7c5abffcae5b43e0e3396117d712c7074a108770221f7806f10ab8b6a6aa9d`

A runtime session owns changing state such as inventory reservations, connections,
scheduling, the KV cache, native executors, and cancellation. This state is not
part of an immutable configuration or plan digest.

The Python binding validates input buffers, releases the Python interpreter lock
while Rust runs, and returns immutable bytes. Python coordinates models and
protocols. Runtime graph support and checkpoint loading remain separate from
model planning.

Before execution, the runtime rechecks plan identity, component support, weights,
and required assurance records. The CLI does not yet provide generic runtime
lifecycle commands.

## Python SDK example

```python
from pllm import PrivacyMode

mode = PrivacyMode.parse("public")
print(mode.value, mode.protocol)
```

Selecting a privacy declaration does not create a session or prove deployment support.
