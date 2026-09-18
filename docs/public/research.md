# PLLM research

PLLM original research, technical paper, tracked papers, and implementation backlog.

[View canonical HTML](https://pllm.run/research/)

Document ID: `pllm.research`  
Release: `0.1.0`  
Build: `sha256:a3db51eca723684328314c5428e7e991d7b5bb20ff973c6d46dfe9a79aec3a08`  
Source hash: `sha256:3a6c188317f497b69d71c7046e87e765ba05105799dc8eae573a1d9cd34981ec`

# Research for high-performance private LLM inference.

**PLLM is a high-performance private LLM multi-party inference runtime and autonomous research harness.**  We independently reimplement useful methods, compose compatible components, and benchmark complete plans to determine what improves the system.

[Read the whitepaper →](/research/whitepaper/)
[Read the research paper →](/research/paper/)
[Explore papers →](/research/papers/)

## A runtime and a research system.

The runtime keeps plaintext, model state, masks, and decoding inside the client boundary while separate services perform prepared masked computation. The research harness turns pinned papers and hypotheses into reusable components, immutable plans, assurance records, and matched benchmark evidence.

### Reimplement independently

Research papers are specification and provenance inputs. Implementations are built in PLLM's Rust and Python stack rather than imported as runtime dependencies.

### Compose by capability

Like-for-like cache, numeric, protocol, preparation, kernel, and placement components live together behind stable typed contracts.

### Compare complete plans

Correctness and privacy remain hard gates. Latency, throughput, traffic, memory, preparation, quality, energy, and cost remain separate measurements.

## Three research records.

Each record serves a different reader. The whitepaper explains why PLLM exists, the technical paper defines the implemented system and its limits, and the paper catalog tracks the methods that may improve it.

[01PLLM original researchA concise, business-friendly whitepaper on private inference, the multi-party boundary, and the autonomous research loop.Read the whitepaper →](/research/whitepaper/)
[02PLLM research paperThe current protocol, compiler and component architecture, security limits, implementation status, and retained evidence.Read the technical paper →](/research/paper/)
[03PapersOne PLLM-focused page per tracked source, with its contribution, boundary, status, original source, and implementation links where they exist.Explore the catalog →](/research/papers/)
[04Reimplementation backlogThe ordered work needed to turn tracked methods into validated components and comparable plans.See what comes next →](/research/backlog/)
