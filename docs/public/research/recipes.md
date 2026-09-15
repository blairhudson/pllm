# Research workflows

Build, reproduce, compare, and review private-inference research with explicit stop rules.

[View canonical HTML](https://pllm.run/research/recipes/)

Document ID: `pllm.docs.recipes`  
Release: `0.1.0`  
Build: `sha256:1e873bcc571b7b7717ee66b62aff1ec3e361ead12088e6b707b77e88ee008328`  
Source hash: `sha256:b7887f22b7b4cbcf4772d0e90ee8c050def31abce640e84b3c20e0a8bd75630a`

A recipe records exact sources, method and component versions, model and workload
limits, environment requirements, steps, expected files, evidence, and failure
policy. Listing a recipe never runs it.

Recipes support reproducibility without making third-party repositories runtime
dependencies. PLLM implements methods independently in Python and Rust. Original
artifacts remain sources and comparison material only.

Run `pllm research recipes list` to inspect current records. Recipe execution is
not supported because sandboxing, artifact identity, and result records are not
yet available together.

Use the pages in this section to choose a workflow. Build and reproduction work
must stop rather than silently substituting an upstream runtime, weakening the
privacy boundary, or presenting unmatched measurements as comparable evidence.
