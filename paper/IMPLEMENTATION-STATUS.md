# Implementation and evidence status

The paper describes the current public-weight runtime:

- client-owned plaintext, token boundaries, nonlinear state, and sampling;
- offline Preparation computation and acknowledged correction upload;
- sealed one-use inventory at Inference;
- packed prefill and persistent decode transport;
- native exact-ring matrix execution; and
- immutable text-free dashboard records.

The retained study is `research/evidence/current-runtime-2026-09-11.json`. It
contains nine warm Qwen2.5-0.5B loopback runs from source revision `277d19f`,
with exact public prompts, model identifiers, raw measurements, and limitations.
It is the evidence source for the paper tables.

The study does not establish WAN or GPU performance, energy use, token price,
market operation, model quality, malicious security, operator independence, or
secure erasure. Historical BFV results under `research/lifecycle` are separate
and are not reused as current-runtime evidence.
