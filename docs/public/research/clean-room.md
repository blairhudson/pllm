# Contribute a clean-room reimplementation

Add an independent PLLM implementation with clear provenance, tests, evidence, and review.

[View canonical HTML](https://pllm.run/research/clean-room/)

Document ID: `pllm.docs.research.clean-room`  
Release: `0.1.0`  
Build: `sha256:4a02d6d6c164f106cb6698f1e14f6ed79d2e45e68a44719d9f0aa417993b4f8a`  
Source hash: `sha256:d4e56f432856ff6a9766a7e2bb8bd1367507e5523cfaa6986de76904a33753b7`

1. Add a source record with a stable ID and URL, exact version or digest, authors, citation, access method, and license review.
2. When lawful, preserve the original artifact outside the runtime. Never vendor or import it as a PLLM dependency.
3. Define the PLLM method and explain how it differs from the source.
4. Write an inspectable reference from the paper, public specification, and approved test vectors. Do not translate the original implementation.
5. Record matching results, mismatches, unavailable cases, and scope.
6. Implement reusable behavior in the component that owns the relevant protocol, kernel, preparation step, or compiler phase.
7. Apply the method through shared plan interfaces and reject unsupported coverage.
8. Run the relevant assurance checks and strongest comparable benchmark. Include limitations and negative results.
9. Complete the [publication review](/research/publications/) before promoting a claim.

Use canonical [paper records](https://github.com/blairhudson/pllm/tree/main/docs/data/research), [schemas](https://github.com/blairhudson/pllm/tree/main/schemas), the [component standard](https://github.com/blairhudson/pllm/blob/main/design/component-standard.md), and the [documentation standard](https://github.com/blairhudson/pllm/blob/main/design/documentation-standard.md).
