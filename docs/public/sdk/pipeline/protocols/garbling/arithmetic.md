# Arithmetic garbling

Mixed-modulus labels and projection gates over bounded arithmetic values.

[View canonical HTML](https://pllm.run/sdk/pipeline/protocols/garbling/arithmetic/)

Document ID: `pllm.docs.protocols.garbling.arithmetic`  
Release: `0.1.0`  
Build: `sha256:bab6f73b33765ac11862794abf645d4eb324a2fd52abc43cc2a9cd21c09e27c7`  
Source hash: `sha256:5d8e8cd921ebfd04bf47b410e73c52dbaa629702ba30cfa0582ef136aa064247`

Arithmetic garbling represents values with labels per modulus. Free compatible arithmetic can avoid tables; nonlinear projection reconstructs signed values jointly from a coprime residue bundle and emits output labels. Gate material is shape-bound, strictly serialized, and consumed once.

Reference exhaustive correctness, tamper rejection, plan commitments, and one-use
process-local issuance checks establish implementation behavior only. The compiler
can select the exact `research.single_evaluator` profile; the
baseline profile cannot activate it. Cryptographic review, cross-process or durable
replay protection, complete-model coverage, and deployment assurance are not
recorded.

No public Python API currently exposes the arithmetic-garbling evaluator. See
[research evidence](/research/evidence/). The evaluator remains internal to the
native compiled-plan boundary and is not part of the stable Python runtime facade.
