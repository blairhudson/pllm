# Network reduction with a small trusted client

Keep pretrained weights and the admitted numeric contract. Price every client,
worker and preprocessing link; record cold public distribution separately from
online bodies. Count client compute and retained/temporary storage explicitly.
All five methods use native hot paths and existing SDK/benchmark conventions.

| Method | Hypothesis and admission gate | Status |
| --- | --- | --- |
| Seeded additive ingress | A fresh seed given only to the mask worker replaces one tensor upload; exact reconstruction, role/context binding and cancellation remain intact | Implemented as opt-in `TwoOnlineOffsetLinear(input_encoding="seeded")` |
| Public-bound row residues | Per-output weight bounds permit smaller exact power-of-two rings without activation-dependent sizes | Next: native codec, bound commitments, combined decoder parity and measured bytes |
| Private paged embeddings | Two non-colluding public-table workers return one hidden page, avoiding a client vocabulary-weight snapshot | Next: native DPF/XOR page execution, exact bytes, independent party state and scan-cost gate |
| Compact private head retrieval | A small public index plus privately retrieved original rows reduces client head storage and arithmetic | Next: fixed query budgets, held-out winner coverage and exact certification; approximation alone cannot authorize exact decoding |
| File-backed exact kernels | Authenticated immutable public pages cap live weight memory while preserving exact integer products | Next: native bounded paging, corruption/race checks and measured CPU/memory tradeoff |

## 1. Seeded additive ingress

Native AES-256 counter expansion binds fresh stage seeds to session, ticket,
semantic plan, composition, model/body/weight commitments, dimensions and ring.
Worker B receives the seed; worker A receives only `x - PRG(seed)` in that ring.
Both still execute the public matrix. Client reconstruction and all local
nonlinear/state operations are unchanged. Single-worker privacy now additionally
assumes the PRG; the two workers must not collude. Reusing a seed/context across
different inputs is prohibited. The seed is never a compression of both shares.

Canonical benchmark (one cold sample per configuration, pinned Qwen W8A8,
39 input and 8 output tokens):

| Measurement | Raw offset | Seeded offset |
| --- | ---: | ---: |
| Online application bodies, both workers | 147,671,296 B | 117,305,152 B |
| Worker B input bodies | 30,756,544 B | 390,400 B |
| Client CPU, startup through first response | 11.300 s | 11.211 s |
| Aggregate CPU over that scope | 37.166 s | 35.911 s |
| Full response latency | 12.588 s | 13.989 s |

Online bodies fall **20.6%**, and generated-output digests match. This is a
single co-located comparison, not a latency improvement or representative quality
study. The arithmetic is lossless; broad generation quality remains the original
W8A8 gate. Client peak RSS and full wire remain unmeasured. Body linear weights
remain wholly remote. No new client model weights are introduced.

Evidence: `docs/evidence/offset-seeded-qwen25-2026-10-02.json`.
Reproduce from the repository root:

```bash
.venv/bin/pllm benchmark run \
  --experiment examples/benchmarks/offset_transport.py:raw \
  --experiment examples/benchmarks/offset_transport.py:seeded \
  --trust-python --temperature 0 --capture-output-digest \
  --max-output-tokens 8 --warmups 0 --repetitions 1 --timeout 120 \
  --output /tmp/offset-seeded.json --quiet
```

The same ordinary Experiment path serves the SDK and gateway. Scoped checks
cover all signed-i8 values, rings 16/24/32, context separation, role mismatch,
malformed envelopes, replay and two-child tiny Qwen2/Qwen3 requests.
