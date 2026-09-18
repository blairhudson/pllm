# Arithmetic and Boolean garbling

Compare arithmetic and Boolean garbling without treating one primitive as a complete private runtime.

[View canonical HTML](https://pllm.run/learn/arithmetic-and-boolean-garbling/)

Document ID: `pllm.docs.learn.arithmetic-and-boolean-garbling`  
Release: `0.1.0`  
Build: `sha256:4a02d6d6c164f106cb6698f1e14f6ed79d2e45e68a44719d9f0aa417993b4f8a`  
Source hash: `sha256:2f1b7dae3f98797d91052ff1b0a42c16c417ab6b186e55dcfe52dd1f25029f19`

Arithmetic garbling works with modular integer values. Boolean garbling works
with bit-level circuits, often using half-gates and free-XOR-compatible labels.
Lookup tables and weighted paths are ways to represent a computation; they do not
provide identical protocol or security guarantees.

Moving values between arithmetic and Boolean forms changes range, bit width,
truncation, and communication requirements. A fast primitive does not make a
complete transformer executable. Correct reference code is also not a substitute
for cryptographic review.

Read [garbling protocols](/sdk/pipeline/protocols/garbling/) and
[conversions](/sdk/build/conversions/) for the exact interfaces and current limits.
