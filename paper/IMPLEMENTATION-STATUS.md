# Implementation and evidence status

The paper describes the current public-weight runtime:

- client-owned plaintext, token boundaries, nonlinear state, and sampling;
- offline Preparation computation and acknowledged correction upload;
- sealed one-use inventory at Inference;
- packed prefill and persistent decode transport;
- native exact-ring matrix execution; and
- immutable text-free dashboard records.

The retained study is `docs/evidence/current-runtime-2026-09-11.json`. It
contains nine warm Qwen2.5-0.5B loopback runs from source revision `277d19f`,
with exact public prompts, model identifiers, raw measurements, and limitations.
It is the evidence source for the paper tables.

The study does not establish WAN or GPU performance, energy use, token price,
market operation, model quality, malicious security, operator independence, or
secure erasure. Historical BFV results in the evidence archive are separate and
are not reused as current-runtime evidence.

The current model-neutral compiler binds baseline untransformed Qwen2 and dense
Qwen3 schedules. Pinned Qwen2.5-0.5B and Qwen3-0.6B checkpoints pass separate
**clear native-kernel** prefill-to-decode functionality tests. Qwen3-0.6B's
W4A4 path differs from the FP32 reference on two checked prompts, and
its W8A8 path differs on a short prompt. These tests do not establish real-model
generation quality, a prepared-protocol benchmark or protected full-decoder
execution; see `docs/evidence/qwen3-0.6b-reference-probe-2026-09-24.json`.
The independent, opt-in two-prompt prefill quality benchmark records 0/2 W4A4
and 1/2 W8A8 top-token agreement with the pinned float32 model. Its local
compiled clear-kernel scope and cohort digests are retained in
`docs/evidence/qwen3-0.6b-reference-quality-2026-09-24.json`; it is not a
provider, full-generation or model-quality claim.

`docs/evidence/slalom-freivalds-tiny-baseline.json` and
`docs/evidence/slalom-freivalds-tiny-verified.json` retain one completed run
each on the same generated tiny workload. They demonstrate optional verified
transport functionality only: no warmup, real-model check, PlanLock digest,
performance conclusion, or malicious-Preparation guarantee follows. Neither
cohort can be merged with the historical Qwen2.5 performance study.
