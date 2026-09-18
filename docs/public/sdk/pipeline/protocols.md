# Protocols

See how PLLM defines parties, messages, privacy assumptions, and failure behavior.

[View canonical HTML](https://pllm.run/sdk/pipeline/protocols/)

Document ID: `pllm.docs.protocols`  
Release: `0.1.0`  
Build: `sha256:358faf6bdfe0705a92c0f87f5cdd73fcee9a6ff7cd651f912f232815241ebaf7`  
Source hash: `sha256:bc5ed0e0d4d053194e1dd12b98569bfd7e1a97a4bdad39bb09416ec7ecae5887`

A protocol defines its parties, offline and online phases, inputs, outputs,
messages, authentication, replay behavior, one-time material, visible information,
trust assumptions, and failure behavior. Model adapters and deployment policy are
separate concerns.

Start with [privacy and threat models](/learn/privacy-and-threat-models/) before
using these implementation interfaces to select a protocol.

Current protocols include [masked-linear inference](/sdk/pipeline/protocols/masked-linear/)
and experimental [garbling components](/sdk/pipeline/protocols/garbling/). Check execution
support, security review, assurance, and deployment status separately; apply the
[research evidence](/research/evidence/) rules to any supporting records.

## Python SDK example

```python
import numpy as np

from pllm import AuthenticatedMPC, AuthenticationError, TrustedPreprocessor

mpc = AuthenticatedMPC(TrustedPreprocessor(modulus=65537, seed=2))
value = mpc.public_value(np.array([5], dtype=np.int64))
try:
    mpc.open(value.tamper_client(value_delta=1))
except AuthenticationError:
    tamper_detected = True
else:
    tamper_detected = False
assert tamper_detected
```

API: [Python objects and signatures](/sdk/reference/python/pllm/#objects-and-signatures)

This checks authenticated-opening failure behavior in the single-process reference
protocol; it does not start separate protocol parties.
