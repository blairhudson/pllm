# Research workflows

Build, reproduce, compare, and review private-inference research with explicit stop rules.

[View canonical HTML](https://pllm.run/research/recipes/)

Document ID: `pllm.docs.recipes`  
Release: `0.1.0`  
Build: `sha256:4a93c61285a110010f1bafefa367e198ed52465071615e2d4d9a91a45f2d82e2`  
Source hash: `sha256:8dcdbfb46c9671fe35d041794c5bf655dc22e4cb845b7580d8ff3be9fc192cd2`

A recipe records exact sources, method and component versions, model and workload
limits, environment requirements, steps, expected files, evidence, and failure
policy. Listing a recipe never runs it.

Recipes support reproducibility without making third-party repositories runtime
dependencies. PLLM implements methods independently in Python and Rust. Original
artifacts remain sources and comparison material only.

Run [`pllm research recipes list`](/cli/reference/research/recipes/list/) to inspect
current records. Recipe execution is not supported because sandboxing, artifact
identity, and result records are not yet available together.

Use the pages in this section to choose a workflow. Build and reproduction work
must stop rather than silently substituting an upstream runtime, weakening the
privacy boundary, or presenting unmatched measurements as comparable evidence.
