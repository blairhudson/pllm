# Get started

Install PLLM, inspect a private inference plan, and run the local development benchmark.

[View canonical HTML](https://pllm.run/learn/start/)

Document ID: `pllm.docs.start`  
Release: `0.1.0`  
Build: `sha256:65f4ad316621284cc28b60b1a825a6d9c6d1f108cbb5032aee0983de1e5c4966`  
Source hash: `sha256:4e0e240713691f8d0e7e10d0088ba4135eb2d569b1ccb846a2a977c245aaca98`

1. [Install PLLM](/learn/start/installation/) with a supported Python version.
2. [Inspect a private inference plan](/learn/start/first-private-request/) without starting a service or downloading model weights.
3. [Run the local development benchmark](/learn/start/first-local-benchmark/) through the client, preparation, and inference roles on one machine.

The CLI currently inspects metadata. Its `dev dashboard` command is a developer
tool, not a production interface. The Python package also exposes planning,
compilation, benchmarking, assurance, and runtime APIs. Check the support status
for each API instead of assuming that every model and protocol combination is
available.
