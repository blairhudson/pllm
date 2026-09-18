# Garbling

Reference arithmetic and Boolean garbling components with explicit maturity and composition boundaries.

[View canonical HTML](https://pllm.run/sdk/pipeline/protocols/garbling/)

Document ID: `pllm.docs.protocols.garbling`  
Release: `0.1.0`  
Build: `sha256:4358d00f593125ffb699b8a2329c69e607727783bf40149859925cbfa47327b5`  
Source hash: `sha256:e15bacdb7ac3166d7522e886b0b3ecf3c9b5c68fc40354f61fc4579da3a80ce2`

Garbling is a family of representations and protocols, not one interchangeable backend. PLLM records arithmetic projection gates, Boolean half-gates, lookup tables, and weighted paths separately because their domains, costs, proofs, and conversion obligations differ.

Current clean-room arithmetic work is reference-only and unreviewed. It is excluded from executable compiler profiles until protocol, transport, coverage, and assurance gates pass. See [research reproductions](/research/recipes/reproductions/).

No public Python API currently exposes garbling configuration or execution. Follow
[research reproductions](/research/recipes/reproductions/) for experimental evidence;
the unsupported SDK state is not an execution fallback.
