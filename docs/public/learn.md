# Learn

Understand PLLM's private inference runtime, research harness, trust assumptions, and evidence.

[View canonical HTML](https://pllm.run/learn/)

Document ID: `pllm.docs.learn`  
Release: `0.1.0`  
Build: `sha256:43a7d6d03570b59100102980d9b739320473c2b1c5a1a31e49972f24b77b95e2`  
Source hash: `sha256:acc4f065b47dc1a39090ff541e2e4fceef219c3f7fd071539604a7139b5fb3a5`

PLLM is a Rust-first, Python-friendly runtime and research harness for private
large language model inference across multiple parties. Its goal is to let a
client use remote compute without giving any one compute party the complete
request, model state, and output.

PLLM combines two systems:

- **The inference runtime** assigns work to the client, preparation service, and
inference service while preserving an explicit trust boundary.
- **The research harness** turns papers and hypotheses into independent Rust and
Python implementations, then records provenance, plans, tests, benchmarks,
limitations, and publication review separately.

PLLM is being built for high performance, but performance depends on the exact
model, protocol, hardware, network, and workload. Privacy depends on the selected
protocol, correct implementation, and stated non-collusion assumptions. Treat
the project's economic and safety benefits as goals; use evidence records for
measured results and current support for executable coverage.

Start here if you need to decide whether a PLLM workflow matches your privacy,
performance, or deployment requirements:

1. [Privacy and threat models](/learn/privacy-and-threat-models/)
2. [Parties and offline work](/learn/parties-and-offline-work/)
3. [Masked linear inference](/learn/masked-linear-inference/)
4. [Arithmetic and Boolean garbling](/learn/arithmetic-and-boolean-garbling/)
5. [Numeric semantics and model quality](/learn/numeric-semantics-and-model-quality/)
6. [Reading research and evidence](/learn/reading-research-and-evidence/)

These pages explain the concepts. The [reference](/sdk/reference/) section lists
exact APIs and schemas. [Current support](/sdk/reference/status/) reports which
combinations are available. [Integrations](/learn/integrations/) shows how the
trusted local gateway serves OpenAI-compatible SDKs and coding agents.
