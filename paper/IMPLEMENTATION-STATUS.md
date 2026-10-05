# Implementation and evidence status

The paper describes the current public-weight runtime:

- client-owned plaintext, token boundaries, nonlinear state, and sampling;
- offline Preparation computation and acknowledged correction upload;
- sealed one-use inventory at Inference;
- packed prefill and persistent decode transport;
- native exact-ring matrix execution; and
- immutable text-free dashboard records.

The rewritten October 5 papers use distinct case studies:

- `docs/evidence/wan-tps-qwen25-2026-10-04.json`: one matched four-candidate
  Qwen2.5 cohort under enforced 100/40 Mbps access and 40 ms added round-trip
  delay. Seed-first dispatch improves offset decode throughput 1.94×; prepared
  overlap improves request throughput 17.3%. Online bodies do not decrease.
- `docs/evidence/slalom-prepared-topologies-cold-cpu-2026-09-26.json`: one
  cold response per topology. Prepared aggregate CPU is 34.66 s versus the
  two-worker control's 28.65 s; verified prepared uses 142.83 s.
- `docs/evidence/qwen3-4b-client-paged-2026-10-05.json`: one 16+8-token
  prepared response, 421.31 MB client/dashboard lifetime peak RSS, no new swap.
  It has no matched resident-client control or independent quality comparison.
- `docs/evidence/preparation-memory-qwen3-4b-2026-10-05.json`: two fresh
  processes per mode, 144 stages and 71 rows per stage. Paged Preparation reduces
  median peak RSS 6.24×, with matching correction content, 12.41% more load-plus-
  issuance CPU and 3.63 GB private snapshot disk. This is an isolated probe.

The papers introduce the agent-facing research workflow; they do not evaluate
autonomous discovery rates. Local link emulation is not an Internet deployment.
Full wire, independent providers, broad quality and tenfold whole-system gains
remain separate gates. Historical runtime and BFV records stay in the evidence
archive and are not combined with these cohorts.

The model-neutral compiler binds untransformed Qwen2, dense Qwen3 and the
text-only Qwen3.5 decoder, among other checked adapters. The public prepared,
client-owned, two-worker offset, and Freivalds-verified prepared choices bind
the same compiled stages; verified stages require one-use verifier material
before session execution. The provider checks plan and body commitments at
session start. Proprietary protocols remain separate. Pinned Qwen2.5-0.5B and
Qwen3-0.6B checkpoints pass separate
**clear native-kernel** prefill-to-decode functionality tests. Qwen3-0.6B's
W4A4 path differs from the FP32 reference on two checked prompts, and
its W8A8 path differs on a short prompt. These tests do not establish real-model
generation quality, a prepared-protocol benchmark or protected full-decoder
execution; see `docs/evidence/qwen3-0.6b-reference-probe-2026-09-24.json`.
The independent, opt-in two-prompt prefill quality benchmark records 0/2 W4A4
and 1/2 W8A8 top-token agreement on each of the pinned Qwen2.5-0.5B and
Qwen3-0.6B checkpoints. Its local compiled clear-kernel scope and separate
checkpoint/cohort digests are retained in
`docs/evidence/qwen2.5-0.5b-reference-quality-2026-09-24.json` and
`docs/evidence/qwen3-0.6b-reference-quality-2026-09-24.json`; it is not a
provider, full-generation or model-quality claim.

`docs/evidence/slalom-freivalds-tiny-baseline.json` and
`docs/evidence/slalom-freivalds-tiny-verified.json` retain one completed run
each on the same generated tiny workload. They demonstrate optional verified
transport functionality only: no warmup, real-model check, PlanLock digest,
performance conclusion, or malicious-Preparation guarantee follows. Neither
cohort can be merged with the historical Qwen2.5 performance study.

`docs/evidence/slalom-prepared-topologies-2026-09-26.json` separately records
one pinned real Qwen2.5 W8A8 response per four selectable role graphs, with
the same fingerprint, input/output counts and covered online bodies. The
verified path sent no extra online stage bodies but added 92.81 MB of covered
initial material; offline CPU, full wire and full-response compute-cap
admission remain unmeasured. The generated Qwen3.5 path now also has a
separate pinned official-checkpoint prefill/decode diagnostic and two-child
request at `docs/evidence/qwen35-4b-text-reference-2026-09-26.json`;
W8A8's 2/4 same-token selection agreement does not establish generation quality.
