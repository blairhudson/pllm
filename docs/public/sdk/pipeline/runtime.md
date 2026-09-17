# Runtime

Understand which live state a PLLM runtime owns and how it differs from a reproducible plan.

[View canonical HTML](https://pllm.run/sdk/pipeline/runtime/)

Document ID: `pllm.docs.runtime`  
Release: `0.1.0`  
Build: `sha256:43a7d6d03570b59100102980d9b739320473c2b1c5a1a31e49972f24b77b95e2`  
Source hash: `sha256:a52d372d8d7c272ac7b753f8d4e3e37bc6de7fd73126cc3e0a75d8543cf76c3a`

A runtime session owns changing state such as inventory reservations, connections,
scheduling, the KV cache, native executors, and cancellation. This state is not
part of an immutable configuration or plan digest.

The Python binding validates input buffers, releases the Python interpreter lock
while Rust runs, and returns immutable bytes. Python coordinates models and
protocols. Runtime graph support and checkpoint loading remain separate from
model planning.

Before execution, the runtime rechecks plan identity, component support, weights,
and required assurance records. The CLI provides explicit gateway, inference,
and preparation lifecycle commands; it does not provide generic plan-locked
orchestration. See [local gateway](/learn/integrations/local-gateway/).

## Python SDK example

```python
import numpy as np

from pllm import AuthenticatedMPC, TrustedPreprocessor

runtime = AuthenticatedMPC(TrustedPreprocessor(modulus=65537, seed=4))
opened = runtime.open(runtime.public_value(np.array([7, 8], dtype=np.int64)))
assert opened.tolist() == [7, 8]
assert runtime.stats.opening_rounds == 1
```

API: [Python objects and signatures](/sdk/reference/python/pllm/#objects-and-signatures)

The opening updates live runtime counters. This in-process arithmetic simulator is
not a deployed multi-party session.
