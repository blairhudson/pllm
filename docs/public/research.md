# PLLM research

The mission, protocol, and evidence for a global market in private AI compute.

[View canonical HTML](https://pllm.run/research/)

Document ID: `pllm.research`  
Release: `0.1.0`  
Build: `sha256:dab1bcb88380ca9a4c2b79e3de085391f482bb97d841a9f0741979579c87dbfb`  
Source hash: `sha256:a5114b036fab6557de2065aa227526449d4764b77cad960c0621eab75e85c94a`

# Build private inference methods that can be reproduced and compared.

**PLLM is an autonomous research harness for high-performance private LLM inference.**  It tracks sources, guides independent Rust and Python implementations, tests fidelity and privacy assumptions, benchmarks matched plans, preserves negative results, and prepares evidence for human review.

[Read Whitepaper →](/research/whitepaper/)
[Explore the research registry →](/research/)

## Hosted inference combines computation with data access.

An ordinary inference provider receives the language it processes. This boundary limits where sensitive workloads can run, concentrates demand among providers that buyers already trust, and excludes other compute capacity.

### Every seller becomes a custodian

Plaintext inference requires the infrastructure operator to receive the client's prompt, model state, and generated result.

### Trust narrows the supply side

Buyers choose providers partly by who may see their data, not only by price, performance, location, or availability.

### Capacity cannot compete freely

Regional infrastructure, sovereign capacity, and independent machines cannot serve many private workloads under the ordinary boundary.

## Separate computation from plaintext data.

Private inference methods aim to let providers process protected values instead of plaintext client data. The exact protection depends on the selected protocol, implementation, and trust assumptions.

01
### Expand eligible supply

Providers can contribute useful compute without receiving plaintext client language.

02
### Increase buyer choice

Capacity can compete on price, latency, location, availability, and service quality.

03
### Use distributed capacity

Protected workloads can reach infrastructure that a plaintext trust boundary excludes.

04
### Limit direct data collection

A compatible compute service does not need plaintext client language as its input.

## Follow the work from source to evidence.

The research record keeps provenance, implementation, assurance, benchmark evidence, and publication review separate.

[1Source registryWhich original publications, versions, artifacts, and licenses ground each method?Inspect sources →](/research/sources/)
[2Private inferenceHow can providers execute useful model work without seeing client data?Trust and assurance →](/learn/understand/privacy-assurance/)
[3PLLM Research PaperWhat system, security boundary, and implementation details support the demonstrated result?Read the paper →](/research/paper/)
[4ContributeHow does a clean-room reimplementation advance through matched evidence and publication gates?Contribution guide →](/research/clean-room/)
