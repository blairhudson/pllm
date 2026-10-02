# Network reduction with a small trusted client

Keep pretrained weights and the admitted numeric contract. Price every client,
worker and preprocessing link; record cold public distribution separately from
online bodies. Count client compute and retained/temporary storage explicitly.
All five methods use native hot paths and existing SDK/benchmark conventions.

| Method | Hypothesis and admission gate | Status |
| --- | --- | --- |
| Seeded additive ingress | A fresh seed given only to the mask worker replaces one tensor upload; exact reconstruction, role/context binding and cancellation remain intact | Implemented as opt-in `TwoOnlineOffsetLinear(input_encoding="seeded")` |
| Public-bound row residues | Per-output weight bounds permit smaller exact power-of-two rings without activation-dependent sizes | Implemented as opt-in `output_encoding="row_residues"`; combines with seeded ingress |
| Private paged embeddings | Two non-colluding public-table workers return one hidden page, avoiding a client vocabulary-weight snapshot | Bounded native SDK probe: exact pages, one-use party state and real-table scan costs measured |
| Compact private head retrieval | A small public index plus privately retrieved original rows reduces client head storage and arithmetic | Native SDK screen tested weight-only and public-query-calibrated indices; neither earned exact decoding admission |
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

## 2. Exact row-wise residues and combined transport

For quantized weight row `w_j` and declared input maximum `a_max`, compute
`B_j = a_max * sum(abs(w_j))` and `k_j = max(1, bit_length(B_j) + 1)`.
The response carries each party's output modulo `2**k_j`. Because that modulus
divides the input sharing ring, reconstruction remains `W*x mod 2**k_j`; the
strict signed bound makes its centered decoding unique. Bias and dequantization
remain at their existing client boundaries. Widths depend only on public weights,
never on input values, private scales or observed output magnitudes.

Both workers derive widths independently from their committed weights. A
digest-bound public bundle includes the layout, and both admission responses must
acknowledge its digest. Responses bind ticket, stage, shape and layout. Native
decoding rejects truncation, trailing bytes and nonzero terminal padding and
reconstructs directly from two packed inputs, avoiding expanded client share
arrays. There is no new model-weight ownership at the client.

Four-way 39+8 ablation, before AES batching:

| Encoding | Online bodies | Full response |
| --- | ---: | ---: |
| Raw | 147,671,296 B | 11.618 s |
| Seeded input | 117,305,152 B | 15.526 s |
| Row residues | 142,371,332 B | 12.343 s |
| Both | 112,005,188 B | 16.013 s |

All four share the same W8A8 body fingerprint and output digest. Row residues
alone save 3.6% on this checkpoint; both save 24.2%. The CPU regression motivated
native eight-block AES batching. A one-million-element mask/subtraction probe
fell from 0.129 s to 0.035 s CPU, with counter encoding checked against an
independent AES implementation.

The final batched implementation then passed a fresh 39+32 paired run:

| Measurement | Raw | Seeded + row residues |
| --- | ---: | ---: |
| Online bodies | 227,079,424 B | 172,726,628 B |
| Client CPU, startup through first response | 18.240 s | 17.035 s |
| Aggregate CPU, same scope | 55.884 s | 56.172 s |
| Full response latency | 25.956 s | 24.210 s |

That is **23.9% fewer online bodies**, identical output digest, and lower client
CPU in this single pair; aggregate CPU is approximately unchanged. No 10× claim
follows. Full wire, client peak RSS and independent operators remain unmeasured.
These encodings preserve the existing W8A8 arithmetic; they do not improve its
agreement with an unquantized checkpoint. Tiny two-child Qwen2/Qwen3 checks also
compare prefill/decode logits against the existing compiled arithmetic.

Evidence: `docs/evidence/offset-row-residues-qwen25-2026-10-02.json` (four-way,
scalar expansion), and `docs/evidence/offset-combined-batched-qwen25-2026-10-02.json`
(final paired implementation). Reuse the command above with `:raw` and `:combined`
and `--max-output-tokens 32` for the latter cohort.

## 3. Private vocabulary pages

`pllm.metrics.PrivatePageLookupProbe` runs a native binary-tree DPF/XOR-PIR
reference over immutable public records. Each server owns one key share, a table
snapshot and a bounded process-local replay ledger. SHA256 commits the table and
shape; fresh query IDs, role fields and exact response lengths bind decoding.
Keys have 128-bit seeds plus independent control bits. AES expansion and complete
depth-first table scans run in Rust without the GIL. Cancellation and malformed
decoder inputs consume the relevant one-use state.

The pinned Qwen table contains 151,936 W8 rows, each with 896 signed bytes and
four original scale bytes: **136,742,400 bytes per worker**. Three fresh queries
per layout return exact original records, including their scale encodings:

| Records per page | Query + reply bodies | Client CPU median | Both workers CPU median sum |
| --- | ---: | ---: | ---: |
| 1 | 2,706 B | 0.091 ms | 0.7791 s |
| 8 | 15,204 B | 0.087 ms | 0.1053 s |
| 32 | 58,336 B | 0.085 ms | 0.0357 s |

Every query scans **273.48 MB** across the two workers. Larger pages reduce tree
expansion at the price of returned bytes; the within-page offset remains local.
The client wire/returned-record payload for the eight-row layout is 23,304 B,
excluding language/runtime overhead. That is a storage accounting category, not
peak RSS. The test fixture owner retains source bytes for the independent oracle.

For 70 embedding uses (39 prefill + 31 decode rows), multiplying the checked
eight-row result gives 1.064 MB of bodies and about 7.37 s worker CPU, before
transport or the rest of inference. This projection is not a measured response.
One-row queries alone would add about 54.5 s worker CPU: small messages do not
establish a compute win.

The ordinary local lookup already has zero online bytes, and a tied local head
still retains the same matrix. Whole-client weight removal therefore depends on
the next head-boundary gate. This reference is not a selectable Experiment;
authenticated distributed lifecycle and independent cryptographic review remain
open. It assumes semi-honest, non-colluding servers under AES-based DPF privacy.

Reproduce: `.venv/bin/python scripts/probe_private_pages.py --pinned --output
/tmp/private-pages.json`. Evidence: `docs/evidence/private-pages-qwen25-2026-10-02.json`.

## 4. Compressed head index with private original-row retrieval

`pllm.metrics.PrivateHeadRetrievalProbe` builds a native index from public
weights, optionally using separate public calibration activations. Rank 32/64
occupies 9.95/15.04 MB native payload, versus the original 136.74 MB row/scale
table. Rust performs query projection, A8 coding, GEMM, bounds and fixed-count
ranking with the GIL released. Exact source weights remain unchanged: selected
rows must be retrieved privately and re-evaluated.

For public basis `U`, exact coefficients `P = W U`, quantized index `P_hat`,
coordinate `z = U^T x` and coded coordinate `z_hat`, use:

```
W x = P_hat z_hat + W(x - U z) + (P - P_hat)z + P_hat(z - z_hat).
```

Public row-norm/coefficient-error bounds, private query residuals and conservative
floating-point allowances upper-bound omitted scores. This identity does not
require exact floating-point orthogonality. Certification requires the exact
winning score to strictly exceed every omitted bound. All checked omitted rows
satisfy their bounds; independent Python oracles verify native ranking and ties.

The initial weight-only index found 6/32 and 7/32 original W8A8 winners at rank
32/64 with 128 candidates, with zero certificates. A second index used **80 public
calibration positions** and **32 fresh confirmation positions** (eight prefills,
24 teacher-forced decode positions):

| Rank | 1 candidate | 8 candidates | 32 candidates | 128 candidates | Certificates |
| --- | ---: | ---: | ---: | ---: | ---: |
| 32 | 3/32 | 13/32 | 19/32 | 22/32 | 0/32 at every budget |
| 64 | 6/32 | 14/32 | 19/32 | 27/32 | 0/32 at every budget |

These are different discovery/confirmation cohorts, not a paired improvement
claim. Both compare same-hidden original W8A8 heads; upstream float32 supplies
hidden trajectories. They do not measure whole-generation quality.

One selected original page was retrieved exactly. At 32 records per page, fixed
private budgets charge **58,336 B per candidate slot**, including padded duplicate
queries: 1.87 MB per output token at 32 candidates, or 7.47 MB at 128. Skipping
duplicates or stopping on a private certificate would expose input-dependent
activity and is not admitted. Index scoring alone costs about 1–2 ms client CPU,
versus roughly 10.6 ms for the full native head; retrieval and exact candidate
scoring are additional. Payload counts are not peak RSS.

**Veto:** smaller index and cheap ranking do not preserve tested winners or meet
the private retrieval budget. This SDK probe cannot replace the full head or
select an Experiment. The next method tests exact file-backed execution.

Evidence: `docs/evidence/head-retrieval-qwen25-2026-10-02.json` and
`docs/evidence/head-retrieval-query-calibrated-qwen25-2026-10-02.json`.

```sh
OPENBLAS_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 .venv/bin/python scripts/probe_head_retrieval.py --pinned --query-calibrated --output /tmp/head-retrieval.json
```
