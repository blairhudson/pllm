# Reproduction checklist

A checklist for obtaining a source, reproducing a method, and producing comparable evidence.

[View canonical HTML](https://pllm.run/research/recipes/reproduction-checklist/)

Document ID: `pllm.docs.agents.reproduction-checklist`  
Release: `0.1.0`  
Build: `sha256:a919906eb7ec6dc3da86474b11b2e5fee671ce20bf3e0b394186df1a41084acd`  
Source hash: `sha256:2eab3e1105ed23d135a0cc281a16c34066635d9eb162df73041ef66b9eb5d32f`

1. Lock paper, revision, artifact metadata, licenses, and hashes.
2. Record claims and assumptions without importing implementation semantics.
3. Implement the method clean-room in model-neutral PLLM abstractions.
4. Validate primitive correctness, tamper and boundary behavior, and numeric fidelity.
5. Validate semantic model and operator coverage independently.
6. Compile and execute only profiles whose capabilities are complete.
7. Produce matched benchmark, quality, and assurance records with failures retained.
8. Compare against declared baselines on every relevant resource axis.
9. Publish limitations and distinguish reproduction from PLLM adaptation or novel contribution.
