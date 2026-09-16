# Reproduce

Compare an independent PLLM implementation with the published method it follows.

[View canonical HTML](https://pllm.run/research/recipes/reproduce/)

Document ID: `pllm.docs.measure.reproduce`  
Release: `0.1.0`  
Build: `sha256:4a93c61285a110010f1bafefa367e198ed52465071615e2d4d9a91a45f2d82e2`  
Source hash: `sha256:d9fbc27aa2bfdf87004f68582309774a19c516a5295ab704982270f3ad3bb316`

1. Lock source publication, artifact, license, environment, and workload.
2. Implement reference from publication specification and approved vectors without upstream code.
3. Compare reference against equations, vectors, and lawful locked oracle outputs.
4. Compare [native component](/sdk/components/) against clean-room reference over declared domain and failures.
5. Compose exact plan and run [scoped assurance](/sdk/research/assurance/).
6. [Benchmark](/sdk/research/benchmarks/) against strongest eligible baseline in one matched cohort.
7. Publish mismatches, unavailable cases, failures, and limitations.

Recipes exposed by `pllm.research` are read-only descriptions with status
`not_executed`. Inspecting a recipe never runs its commands. See
[clean-room contributions](/research/clean-room/) and the
[research catalog](/research/records/method-catalog/).
