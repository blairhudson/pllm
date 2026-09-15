# Contribute a clean-room reimplementation

Add an independent PLLM implementation with clear provenance, tests, evidence, and review.

[View canonical HTML](https://pllm.run/research/clean-room/)

Document ID: `pllm.docs.research.clean-room`  
Release: `0.1.0`  
Build: `sha256:65f4ad316621284cc28b60b1a825a6d9c6d1f108cbb5032aee0983de1e5c4966`  
Source hash: `sha256:718c6d57b6ace2bfab13252ba2156ffcd0b02dc2fe8cad719747f7910a64db52`

1. Add a source record with a stable ID and URL, exact version or digest, authors, citation, access method, and license review.
2. When lawful, preserve the original artifact outside the runtime. Never vendor or import it as a PLLM dependency.
3. Define the PLLM method and explain how it differs from the source.
4. Write an inspectable reference from the paper, public specification, and approved test vectors. Do not translate the original implementation.
5. Record matching results, mismatches, unavailable cases, and scope.
6. Implement reusable behavior in the component that owns the relevant protocol, kernel, preparation step, or compiler phase.
7. Apply the method through shared plan interfaces and reject unsupported coverage.
8. Run the relevant assurance checks and strongest comparable benchmark. Include limitations and negative results.
9. Complete the [publication review](/research/publications/) before promoting a claim.

Use canonical [research records](https://github.com/blairhudson/pllm/tree/main/research), [schemas](https://github.com/blairhudson/pllm/tree/main/schemas), [research standard](https://github.com/blairhudson/pllm/blob/main/design/research-standard.md), [component standard](https://github.com/blairhudson/pllm/blob/main/design/component-standard.md), and [documentation standard](https://github.com/blairhudson/pllm/blob/main/design/documentation-standard.md).
