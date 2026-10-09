# Prepared TTFT and client tensor lifetimes

## Contract and diagnosis

The ordinary prepared topology keeps Preparation offline, with one online
Inference provider and provider-owned body linear stages. This investigation
uses pinned Qwen2.5-0.5B W8A8, `prefix_f32`, paged client/Preparation weights,
indexed tokenization, demand issuance and packed/residue arithmetic frames.
Native relays enforce shared 100 Mbps download / 40 Mbps upload per party with
the calibrated 8 KiB reads and unchanged 16 KiB burst.

The instrumented client profile attributes **44.7 of 48.1 prefill seconds** to
remote exchanges. Those exchanges include provider work, serialization and
pacing; they do not isolate network wait. The optional `--array-owners` profile
observes returned NumPy allocation roots through weak references, without
retaining their values. It excludes unreturned scratch and is not a complete
allocation inventory or a process-RSS measurement.
The complete instrumented report is
`docs/evidence/research-prepared-prefill-client-profile-qwen25.json`; runtime
checks pass and profiled source hashes remain unchanged.

The semantic executor already removes dictionary entries after their final use,
but loop-local aliases kept grouped gate/up allocations alive during subsequent
remote exchanges. Linear-step temporaries now belong to a per-step function
frame. Continuation validation and hybrid-state copying also release their
temporary tensor aliases. Numeric operations and scheduling order are preserved.

Weak-reference regressions check reclamation before the next linear exchange
for generated Qwen2, Qwen3 and Llama W8A8 prefill/decode. All three fail against
the previous installed wheel and pass against the candidate. The broader
installed-wheel execution, binding, state, transport and memory suite passes
135 tests, with pytest's checkout import override explicitly disabled.

## Isolated ownership ablation

The implementation-only cohort alternates two baseline and two candidate fresh
clients; each performs the same cold/repeat sequence. Dependencies and native
libraries match, and `runtime/semantic_executor.py` is the only changed installed
Python file. All eight responses and the canonical comparison pass, with zero
observed swap growth. Evidence:
`docs/evidence/research-client-owners-qwen25.json`.

The following medians use two independent clients per build. Lifecycle phases
stay separate; process-lifetime RSS covers both requests.

| Priority metric | Previous ownership | Scoped ownership |
| --- | ---: | ---: |
| Client lifetime peak RSS, MB | 230.59 | 231.01 |
| Client cold-first CPU, s | 23.397 | 23.548 |
| Preparation cold-first CPU, s | 17.077 | 17.156 |
| Preparation sampled peak RSS, cold / repeat MB | 197.94 / 200.20 | 195.59 / 197.24 |
| Online TTFT, cold / repeat s | 50.449 / 50.428 | 50.405 / 50.485 |
| N−1 decode tokens/s, cold / repeat | 3.734 / 3.785 | 3.808 / 3.767 |
| Both requests' all-link TCP streams, MB | 830.94 | 830.94 |

Peak RSS spans 227.31–233.87 MB before and 228.13–233.90 MB after. Complete
cold/repeat request medians are 112.820/86.566 versus 112.724/86.548 seconds.
Covered application bodies remain exactly 829,000,681 bytes per sequence;
TCP-stream control-length variation is at most eighteen bytes. Aggregate
cold-first CPU changes from 67.66 to 68.20 seconds. These samples show no
meaningful whole-client memory or latency improvement despite deterministic
retirement of the unnecessary tensor owners. They do not explain the earlier
pacing cohort's RSS increase.

## Existing duplex-prefill screen

`benchmarks/research/prepared_prefill.py` compares whole-stage HTTP prefill with
the existing four- and sixteen-row duplex WebSocket options. Every fresh client
executes a cold request and an exact repeat, each with 150 input and eight output
tokens. There is no KV reuse. One parent owns the comparison salt, and each
client has an isolated artifact cache. All six responses and the canonical
comparison pass, with matching output digests.
The original canonical report is retained in
`docs/evidence/research-prepared-prefill-screen-qwen25.json`.

| Priority metric | Whole stage | Four rows | Sixteen rows |
| --- | ---: | ---: | ---: |
| Client lifetime peak RSS, MB | 228.62 | 222.94 | 217.81 |
| Client cold-first CPU, s | 24.33 | 22.90 | 22.38 |
| Preparation cold-first CPU, s | 16.80 | 16.77 | 16.19 |
| Preparation sampled peak RSS, cold / repeat MB | 200.56 / 205.03 | 194.41 / 194.82 | 196.31 / 192.04 |
| Online TTFT, cold / repeat s | 50.315 / 51.070 | 41.134 / 41.117 | 41.253 / 41.445 |
| N−1 decode tokens/s, cold / repeat | 3.699 / 2.934 | 3.511 / 3.642 | 3.737 / 3.639 |
| Both requests' all-link TCP streams, MB | 830.94 | 836.55 | 832.33 |

Sixteen-row prefill has similar TTFT to four-row prefill with fewer framed
messages. Its complete cold/repeat requests take 103.703/77.472 seconds versus
113.106/87.858 seconds for whole-stage prefill. Covered all-link application
bodies increase from 829.001 to 829.261 MB; chunk framing is charged. These are
one cold/repeat sequence per configuration and an exploratory transport choice.

Client RSS covers each fresh process through both requests. Cold-first CPU
includes process startup through the first response. Preparation RSS is the
largest OTel sample in each request window, including its opening and closing
samples; earlier startup and shorter peaks can be missed. Online TTFT starts
after setup and demand preparation, while complete request time includes both.

## Repeated combined-stack confirmation

A separate alternating baseline/candidate, candidate/baseline cohort compares
the old executor with whole-stage prefill against scoped tensor owners plus
`MaskedLinear(prefill_chunk_rows=16)`. Two fresh clients per choice each execute
one cold and one exact-repeat response. Pacing, native libraries, dependencies,
source, numeric body and verifier contract match; the script checks the sole
installed Python change, `runtime/semantic_executor.py`, separately from the
explicit Pipeline choice. Evidence:
`docs/evidence/research-prepared-prefill-combined-qwen25.json`.

All eight responses and the canonical comparison pass, with identical output
digests and zero observed swap growth. Medians below use the same scopes as the
ownership-only table; each client lifetime peak covers its complete sequence.

| Priority metric | Whole-stage baseline | Scoped owners + sixteen rows |
| --- | ---: | ---: |
| Client lifetime peak RSS, MB | 238.47 | 225.85 |
| Client cold-first CPU, s | 23.547 | 22.560 |
| Preparation cold-first CPU, s | 17.073 | 16.754 |
| Preparation sampled peak RSS, cold / repeat MB | 196.80 / 197.86 | 198.26 / 193.98 |
| Online TTFT, cold / repeat s | 50.546 / 50.449 | 41.306 / 41.412 |
| N−1 decode tokens/s, cold / repeat | 3.757 / 3.814 | 3.686 / 3.672 |
| Both requests' all-link TCP streams, MB | 830.938 | 832.326 |

Online TTFT falls **18.28% / 17.91%**, client lifetime peak RSS **5.29%** and
client cold-first CPU **4.19%**. Complete request time falls from
**112.934 / 86.673 to 103.712 / 77.508 seconds** (**8.17% / 10.57%**).
Client peaks span 236.63–240.30 MB before and 225.71–226.00 MB after. This is a
combined-stack measurement; the ownership-only cohort above did not establish
an RSS improvement and cannot be multiplied into these savings.

Decode medians regress **1.88% / 3.73%**, and TCP-stream traffic rises **0.17%**.
Covered application bodies rise from **829.001 to 829.261 MB** across the two
responses, including the chunk framing. Preparation elapsed time is nearly
unchanged at **33.554 / 33.722 versus 33.608 / 33.523 seconds**; its cold-first
CPU falls 1.87% while sampled peak RSS changes in opposite directions across
cold and repeat phases. Cold-first aggregate CPU is nearly unchanged,
**68.201 / 68.369 seconds**, as is sampled simultaneous topology RSS,
**1,467.32 / 1,466.27 MB**. Those small differences do not establish a
Preparation or whole-topology optimization.

This bounded repeated cohort supports opt-in prefill responsiveness with modest
client savings. It does not establish a universal winner, a decode speedup,
representative model quality, independent operators, physical-wire costs or a
tenfold memory improvement. Public-index and checkpoint distribution remain
outside the response counters.

## Reproduction

Build the public artifacts using the existing
[prepared-frontier instructions](prepared-frontier.md#reproduction). Use fresh
output paths and matching installed-wheel dependencies. The implementation
probe checks that only `runtime/semantic_executor.py` changes and that native
library hashes match; repetitions alternate baseline/candidate order and create
a fresh client for each sample.

```sh
PLLM_RESEARCH_ARTIFACTS=target/prepared-frontier-artifacts \
target/venvs/prepared-pacing/bin/python -m pllm benchmark run \
  --experiment benchmarks.research.prepared_prefill:control \
  --experiment benchmarks.research.prepared_prefill:chunk4 \
  --experiment benchmarks.research.prepared_prefill:chunk16 \
  --factory --trust-python --backend native --wan --isolate-candidates \
  --compare-kernels --capture-output-digest --temperature 0 \
  --prompt-sequence-file target/prepared-frontier-artifacts/repeat-sequence.json \
  --max-output-tokens 8 --warmups 0 --repetitions 1 --timeout 1800 \
  --output target/evidence/prepared-prefill-reproduction.json

PLLM_RESEARCH_ARTIFACTS=target/prepared-frontier-artifacts \
.venv/bin/python -m scripts.probe_prepared_path \
  --probe responses --change-scope client-owners \
  --baseline-python target/venvs/prepared-pacing/bin/python \
  --candidate-python target/venvs/prepared-client-owners/bin/python \
  --experiment benchmarks.research.prepared_prefill:control \
  --response-repetitions 2 \
  --prompt-file target/prepared-frontier-artifacts/prompt.txt \
  --output target/evidence/client-owners-reproduction.json

PLLM_RESEARCH_ARTIFACTS=target/prepared-frontier-artifacts \
.venv/bin/python -m scripts.probe_prepared_path \
  --probe responses --change-scope client-owners \
  --baseline-python target/venvs/prepared-pacing/bin/python \
  --candidate-python target/venvs/prepared-client-owners/bin/python \
  --experiment benchmarks.research.prepared_prefill:control \
  --candidate-experiment benchmarks.research.prepared_prefill:chunk16 \
  --response-repetitions 2 \
  --prompt-file target/prepared-frontier-artifacts/prompt.txt \
  --output target/evidence/prepared-prefill-combined-reproduction.json
```

The 9.29 MB public tokenizer index is pre-positioned. Its publication and
distribution, checkpoint downloads, physical wire, Internet RTT, filesystem
cache and independent operators are outside these response measurements.
The Qwen3-4B preflight requires 7.61 GiB against 7.07 GiB of available post-reserve
headroom and rejects before execution. This cohort therefore supplies no new
4B response result or 10× client-memory claim.
The rejection is retained in
`docs/evidence/research-prepared-prefill-qwen3-admission.json`.
