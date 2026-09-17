# Get started

Install PLLM, inspect a private inference plan, and run the local development benchmark.

[View canonical HTML](https://pllm.run/learn/start/)

Document ID: `pllm.docs.start`  
Release: `0.1.0`  
Build: `sha256:bab6f73b33765ac11862794abf645d4eb324a2fd52abc43cc2a9cd21c09e27c7`  
Source hash: `sha256:db6bff62bf12c5a9077a2acd49221129fdf9098bfc5f614ddd3980ef60cfe5f3`

1. [Install PLLM](/learn/start/installation/) with a supported Python version.
2. [Inspect a private inference plan](/learn/start/first-private-request/) without starting a service or downloading model weights.
3. [Run the local development benchmark](/learn/start/first-local-benchmark/) through the client, preparation, and inference roles on one machine.
4. [Connect an SDK or coding agent](/learn/integrations/) through the trusted loopback gateway.

The CLI currently inspects metadata. Its `dev dashboard` command is a developer
tool, not a production interface. The Python package also exposes planning,
compilation, benchmarking, assurance, and runtime APIs. Check the support status
for each API instead of assuming that every model and protocol combination is
available.
