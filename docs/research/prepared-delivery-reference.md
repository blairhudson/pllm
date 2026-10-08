# Prepared correction buffers, decode compression and public artifacts

## Contracts

Two small runtime changes retain the compiled numeric identity and the offline
Preparation / one online Inference topology:

- Preparation subtracts output masks directly in uint32, then narrows to the
  validated power-of-two ring. This replaces two widened input arrays and an
  int64 remainder tensor. Correction bytes, fresh masks, acknowledgement before
  sealing and one-use inventory semantics remain exact.
- Public prepared WebSockets decline per-message deflate. Their arithmetic
  frames already contain fresh masked words and negotiated packed/residue
  layouts. Other channel users keep their existing compression setting.
  Authentication, sequence checks, frame bounds and duplex cancellation remain
  in the ordinary channel implementation.

Public-artifact pre-encoding is a separate local prototype. It writes the
existing bounded zlib frames into a private file, verifies exact decoded raw
identity, and replays with a size and framed-content digest check. It charges
publication CPU and additional disk bytes. It has no provider cache lifecycle,
HTTP route or selectable SDK component.

## Fresh-process screen

`scripts/probe_prepared_delivery.py` runs three fresh processes per case, with
seven-call timing samples and separate allocation tracing. The retained cohort
runs without another heavy diagnostic. An earlier screen overlapping the quality
run is excluded from these timings. Evidence:
`docs/evidence/research-prepared-delivery-screen.json`.

### Correction subtraction

Each process checks 150 × 9,728 synthetic outputs at 16-, 24- and 32-bit rings
against widened arithmetic, plus arbitrary-precision wraparound edge oracles.
Across the three processes, **13,132,800 output values** match.

| Ring width | Widened CPU ms | Uint32 CPU ms | Widened / uint32 traced peak MB |
| --- | ---: | ---: | ---: |
| 16 | 4.178 | 0.439 | 23.348 / 5.837 |
| 24 | 4.254 | 0.429 | 23.348 / 5.837 |
| 32 | 4.427 | 0.365 | 23.348 / 5.837 |

These 89–92% CPU and 75% traced-allocation reductions cover subtraction alone.
They do not reduce the correction upload, matrix multiplication, mask expansion
or complete Preparation working set. The shared 40 Mbps upload floor for a
132.82 MB correction inventory is still about 26.56 seconds before control and
framing costs. Existing stage windows can overlap independent work within their
admitted allocation bound; eliminating a subtraction copy does not remove that
transfer floor.

### Decode compression

Synthetic public metadata, fresh tickets and uniformly masked words pass through
the actual WebSocket deflate extension and an independent decode check.
Each case contains 144 one-row frames.

| Random body bytes/frame | Deflate / plain CPU ms | Deflate / plain framed bytes |
| --- | ---: | ---: |
| 3,072 | 14.857 / 0.019 | 453,169 / 452,736 |
| 8,192 | 29.488 / 0.021 | 1,191,971 / 1,190,016 |
| 28,672 | 91.149 / 0.020 | 4,145,828 / 4,139,136 |

Compression increases these synthetic framed bodies slightly. The CPU figures
cover local extension encode/decode only: no socket transfer, client masking
header, HTTP/TCP, added RTT or whole-decoder cost is included. Real metadata and
frame mixtures need a separate complete-request measurement.

### Ready-to-serve public artifacts

The source-locked Qwen3-4B screen selects an 8.39 MB quantized public embedding
slice using shape-only information, plus the original 11.42 MB tokenizer. The
raw bytes are already resident in both controls; quantization and source import
are outside these encoding timings.

| Artifact | On-request encode CPU ms | Publication CPU ms | Verified replay CPU ms | Extra disk MB |
| --- | ---: | ---: | ---: | ---: |
| W8 matrix slice | 232.416 | 238.209 | 5.282 | 7.620 |
| Tokenizer | 74.759 | 77.154 | 1.634 | 2.534 |

Median replay CPU is about 44–46× lower than re-encoding these two objects.
Publication still performs essentially the original encoding work, and verified
replay has a 2.10 MB Python-traced peak. Source artifacts and prebuilt frames are
separate disk owners. The first cold request cannot count publication as free.
Local filesystem-cache replay is neither network serving nor whole-client RSS.

Publication plus the first verified replay costs 243.491 / 78.788 ms for the
matrix slice / tokenizer, **4.77% / 5.39% more CPU** than on-request encoding.
The ordinary cold/repeat workload fetches each missing object once, then reuses
the client's authenticated object cache. It therefore supplies no second remote
object consumer to amortize publication. A provider-owned prebuilt cache needs
a separately measured multiple-fresh-client reuse horizon, bounded cache
lifecycle and resource admission before promotion. This screen does not
establish savings for the historical 91.52-second bundle-loading profile.

## Matched installed-SDK response pair

`docs/evidence/research-prepared-refinements-qwen25.json` retains a separate
Qwen2.5-0.5B W8A8 150+8-token cold/repeat pair under shared native 100/40 Mbps
access caps. Both installed builds include client-boundary fusion. Their native
libraries and dependencies match exactly; only `runtime/client.py` and
`runtime/transformer_engine.py` differ. Both use the ordinary serial prepared
factory, isolated client artifact caches, paged client/Preparation weights,
indexed tokenization, demand issuance and packed/residue frames. Neither uses
KV reuse or the artifact-publication prototype.

All four responses pass the runtime checks, and the canonical single-salt
comparison passes numeric, verifier, workload and output matching. The scoped
host guard observes zero new swap.

| Metric | Fused control cold / repeat | Correction + WebSocket changes cold / repeat |
| --- | ---: | ---: |
| Complete request s | 127.451 / 103.734 | 137.632 / 100.385 |
| After-first-output tokens/s, seven outputs | 2.930 / 2.871 | 3.499 / 3.548 |
| Preparation elapsed s | 37.810 / 39.180 | 40.582 / 40.759 |
| Client run-window CPU s | 20.600 / 16.947 | 21.316 / 16.390 |
| Preparation run-window CPU s | 7.648 / 7.640 | 7.608 / 7.371 |
| Covered all-link application bodies, MB | 476.953 / 352.048 | 476.953 / 352.048 |

Decode throughput improves **19.41% / 23.60%**, but cold request latency rises
**7.99%** and repeat latency falls **3.23%**. Cold-first aggregate CPU rises
**69.34 to 70.64 seconds (1.88%)**. Correction subtraction is only a small part
of complete Preparation, whose elapsed time does not improve in this pair.

Preparation's largest recorded process peak falls **218.87 to 200.59 MB
(8.35%)**, while client lifetime peak RSS rises **230.16 to 238.39 MB (3.57%)**.
Sampled simultaneous topology RSS falls **1,583.87 to 1,468.02 MB (7.31%)**;
sampling can miss short peaks and double-count shared pages. These separate
ownership measurements do not establish an overall client-memory improvement.

Total TCP-stream bodies across both responses rise **830.670 to 830.938 MB**,
despite exact application-body equality. Real envelope metadata differs from
the synthetic WebSocket screen and can benefit from deflate. This is a small
traffic increase, not a byte-saving codec. Neither dependent exchange count nor
the 132.82 MB per-response correction upload is reduced.

This single sequence per installed build supports a scoped decode-throughput
result, not a representative cold-speed or memory win. The co-located rate
emulation adds no Internet RTT; physical wire, checkpoint distribution, public
tokenizer publication and independent operators remain outside these counters.
The earlier fusion pair is a distinct cohort and its ratios are not multiplied.

## Reproduction

```sh
.venv/bin/python -m scripts.probe_prepared_delivery \
  --output target/evidence/prepared-delivery-reproduction.json

PLLM_RESEARCH_ARTIFACTS=target/prepared-frontier-artifacts \
target/venvs/prepared-refinements/bin/python -m scripts.probe_prepared_path \
  --probe responses --change-scope prepared-refinements \
  --baseline-python target/venvs/prepared-client/bin/python \
  --candidate-python target/venvs/prepared-refinements/bin/python \
  --experiment benchmarks.research.prepared_frontier:serial \
  --prompt-file target/prepared-frontier-artifacts/prompt.txt \
  --output target/evidence/prepared-refinements-reproduction.json
```

The public checkpoint must be locally available and admitted; every output path
must be new. The source lock, exact tensor/slice geometry, raw/framed digests,
publication cost and all fresh-process samples remain in the report. Synthetic
masked words are test stand-ins, never reusable runtime material.
The response reproduction requires separate installed wheels with the retained
implementation differences and identical dependencies. Targeted installed-wheel
controls cover ordinary, compact, chunked, residue, verified, preparation and
inventory execution; they preserve malformed-frame and cancellation behavior.
