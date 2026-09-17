# Research workflows

Build, reproduce, compare, and review private-inference research with explicit stop rules.

[View canonical HTML](https://pllm.run/research/recipes/)

Document ID: `pllm.docs.recipes`  
Release: `0.1.0`  
Build: `sha256:bab6f73b33765ac11862794abf645d4eb324a2fd52abc43cc2a9cd21c09e27c7`  
Source hash: `sha256:534dab86e03973791ceafb51bcff973c9d3946dfd9a66f6fd2417ab1b1df78ca`

A recipe records exact sources, method and component versions, model and workload
limits, environment requirements, steps, expected files, evidence, and failure
policy. Listing a recipe never runs it.

Recipes support reproducibility without making third-party repositories runtime
dependencies. PLLM implements methods independently in Python and Rust. Original
artifacts remain sources and comparison material only.

Use the pages in this section and the [research backlog](/research/backlog/) to
inspect current workflows. Recipe execution is not supported because sandboxing,
artifact identity, and result records are not yet available together.

Use the pages in this section to choose a workflow. Build and reproduction work
must stop rather than silently substituting an upstream runtime, weakening the
privacy boundary, or presenting unmatched measurements as comparable evidence.
