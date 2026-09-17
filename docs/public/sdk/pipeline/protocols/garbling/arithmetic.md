# Arithmetic garbling

Mixed-modulus labels and projection gates over bounded arithmetic values.

[View canonical HTML](https://pllm.run/sdk/pipeline/protocols/garbling/arithmetic/)

Document ID: `pllm.docs.protocols.garbling.arithmetic`  
Release: `0.1.0`  
Build: `sha256:43a7d6d03570b59100102980d9b739320473c2b1c5a1a31e49972f24b77b95e2`  
Source hash: `sha256:98e36edf694d4ce71dcd3b8e1c983037a2934867080c0780ac96d96b557b5724`

Arithmetic garbling represents values with labels per modulus. Free compatible arithmetic can avoid tables; nonlinear projection reconstructs signed values jointly from a coprime residue bundle and emits output labels. Gate material is shape-bound, strictly serialized, and consumed once.

Reference exhaustive correctness, tamper rejection, plan commitments, and one-use process-local burn checks establish implementation behavior only. The compiler can select the exact experimental Q7 SiLU projection profile. Cryptographic review, durable replay protection, complete-model coverage, and deployment assurance are not recorded.

No public Python API currently exposes the arithmetic-garbling evaluator. See
[research evidence](/research/evidence/). The evaluator remains internal to the
native compiled-plan boundary and is not part of the stable Python runtime facade.
