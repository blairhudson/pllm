# Preparation

Prepare one-time masked material before a private inference request starts.

[View canonical HTML](https://pllm.run/sdk/pipeline/preparation/)

Document ID: `pllm.docs.preparation`  
Release: `0.1.0`  
Build: `sha256:8db228446618c1e95120afa32d874de6f8cba2e6e356c6b0c1157f0c5b7cc58e`  
Source hash: `sha256:5b7483893fd1b386f018e4e615558233140714467317a5a63dbb7c6931e82cc9`

Preparation components define how one-time material is created, batched, committed,
transferred, confirmed, erased, expired, and invalidated. They also identify who
owns the material and how much can be stored. Protocol and kernel choices remain
separate.

The [parties and offline work](/learn/parties-and-offline-work/) overview explains
the role boundary. For public masked-linear inference, the client sends seed batches
to preparation. Preparation computes corrections, and inference stores them in a
sealed inventory. The client refills inventory only while idle. A restart or idle
timeout discards inventory held in memory.

Prepared material is part of the system cost. Benchmarks must state whether
preparation and loading are included in cold and warm measurements so the resulting
[research evidence](/research/evidence/) keeps that scope visible.

## Python SDK example

```python
import numpy as np

from pllm import TrustedPreprocessor

preparation = TrustedPreprocessor(modulus=65537, seed=7)
mask = preparation.input_mask((3,))
reconstructed = (mask.shared_mask.client.value + mask.shared_mask.server.value) % 65537
assert np.array_equal(reconstructed, mask.clear_mask_for_client)
```

API: [Python objects and signatures](/sdk/reference/python/pllm/#objects-and-signatures)

This creates and checks reference one-time material in process. Production use
requires a reviewed correlation protocol and separate roles.
