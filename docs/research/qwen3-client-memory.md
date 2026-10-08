# Qwen3-4B client-weight ownership and memory

The matched fresh-client SDK pair reduces client peak RSS **3.91×**, from
1,020.46 to 261.05 MB, with matching eight-token outputs and zero observed new
swap. Paging increases request time 8.97%. This is a one-response storage
comparison within the indexed-tokenizer stack; it does not establish 10× savings.

## Ownership correction

Resident client bundles originally kept their imported signed-i8 matrix after
creating an immutable native execution snapshot. Tied token lookup and output
head stages shared the imported matrix, but the native snapshot was a second
retained owner.

After native compilation, the bundle now rebinds every stage using that matrix
to its read-only native weight view. Tied lookup/head stages still share one
matrix; untied stages retain separate matrices. Source, weight and compiled
binding digests are unchanged. A caller retaining the original array still owns
that separate allocation. The first native import still overlaps the original
and snapshot, so retiring the imported owner does not itself remove the import
peak. Whole-topology admission retains its conservative import bound.

`tests/test_native_snapshot.py` checks tied and untied ownership retirement,
read-only views, exact stored weights and token lookup, output-head arithmetic,
unchanged compiled binding identity, and view lifetime beyond bundle retirement.
The existing generated-checkpoint paging suite additionally checks complete
prefill/decode logits and KV through prepared, verified and two-offset SDK
requests, including cache reuse and gateway execution. Those generated numeric
controls are separate from the real-checkpoint output-text comparison below.
`scripts/probe_client_weight_ownership.py` compares the ordinary bundle with a
control that explicitly retains the original owner. Each sample uses a fresh
process, synthetic Qwen3-4B-shaped weights and independent closed-form integer
boundary oracles. This isolated reference measures ownership and phase RSS, rather
than checkpoint import or a complete decoder.

Six fresh processes check 3,830,784 integer outputs. Retained matrix payload
falls from 777.91 to 388.96 MB, but median process peak RSS stays essentially
unchanged: 828.83 versus 828.93 MB. Source and native snapshots still overlap
during import; logical owner retirement does not guarantee immediate release of
allocator-held pages. Evidence: `docs/evidence/research-client-weight-ownership.json`.

## Matched SDK contract

`benchmarks/research/qwen3_memory_frontier.py` selects ordinary resident and paged
client weight storage. Both use the same pinned Qwen3-4B W8A8 body, indexed
tokenizer, paged Preparation, compiler-demand on-demand issuance, compressed
artifact delivery, packed requests and row-residue outputs. Body linear stages
remain provider-owned.

One benchmark invocation owns the private cohort salt and launches a fresh
client per candidate, with separate native Inference and Preparation children.
Each candidate executes one 16-input/eight-output-token response under shared
100 Mbps download / 40 Mbps upload caps per party. The 9,285,852-byte public
tokenizer index and checkpoint are pre-positioned. Client lifetime peak RSS and
sampled simultaneous client/provider RSS are different metrics; independent
process lifetime peaks are never added together.

This pair isolates storage within the indexed-tokenizer stack. It cannot be
multiplied by historical tokenizer reductions. Output-text agreement does not
establish full-logit/KV equality or representative generation quality. Physical
wire, checkpoint/index distribution, Internet RTT, filesystem cache and
independent provider deployment remain outside the measurement.

## Matched results

| Metric | Resident client | Paged client |
| --- | ---: | ---: |
| Client/dashboard lifetime peak RSS | 1,020.46 MB | 261.05 MB |
| Sampled simultaneous total RSS | 6,383.37 MB | 5,797.35 MB |
| Complete request | 142.665 s | 155.460 s |
| Preparation within request | 26.031 s | 27.799 s |
| Online execution | 34.676 s | 37.288 s |
| Client cold-first CPU | 24.766 s | 26.517 s |
| Aggregate cold-first CPU | 140.277 s | 140.519 s |
| Covered setup-inclusive application bodies | 556.31 MB | 556.31 MB |
| Covered online application bodies | 127.58 MB | 127.58 MB |
| Cold-first TCP-stream bytes | 557.81 MB | 557.81 MB |
| Maximum observed new swap | 0 | 0 |

Both candidates pass every reported runtime/audit check and consume all 3,267
issued stage rows exactly once. Application bodies match exactly; TCP-stream
totals differ by 199 framing/control bytes. Client peak RSS falls 74.42%, while
sampled total RSS falls 9.18%. Aggregate CPU is nearly equal, but client CPU rises
7.07%. The resident control includes the ownership correction above, so this
pair measures paging rather than the correction in isolation. Paging also owns
private file snapshots; its RSS excludes filesystem cache and is not a total
device-memory or disk-storage reduction.

The canonical report is `docs/evidence/research-qwen3-client-memory.json`.
Publisher cost and source fingerprints are retained in
`docs/evidence/research-qwen3-client-memory-publisher.json` and
`docs/evidence/research-qwen3-client-memory-configurations.json`. The Research
scorecard registers this pair as its own cohort. Generated-checkpoint parity,
real-checkpoint text equality and broad model quality remain separate claims.

## Held-out quality admission

The twelve public prompts in `benchmarks/research/data/paper-quality-prompts.json`
are separate from the memory benchmark prompt and were not used to fit this W8A8
body. `benchmarks/research/qwen3_memory_quality.py` selects the same resident
Pipeline and delegates to the ordinary compiled prefill/FP32 reference diagnostic.
It first reads checkpoint tensor headers and prices the reference separately
from the small client role.

The retained attempt rejects before loading tensor values: 398 stored tensors
need **16.09 GB** for FP32 weights alone, versus **13.84 GB** available after the
4 GiB host reserve. The complete conservative reference estimate is 22.26 GB,
including 25% slack and 2 GiB for loading/workspace. No quality score was produced;
the original host's existing swap is not admission capacity. The rejected report
is `docs/evidence/research-qwen3-client-memory-quality.json`. Retry with sufficient
headroom remains required; this slice establishes matched memory behavior, not
held-out model quality.

## Reproduction

Run from the checkout with the release extension and locked dependencies. The
artifact builder resolves the pinned source and creates the trusted public
tokenizer index. The benchmark performs whole-topology admission before loading
values or launching providers and aborts on reserve loss or new swap growth.
Use fresh report paths.

```sh
uv run --no-sync python scripts/probe_client_weight_ownership.py \
  --rows 151936 --columns 2560 --repeats 3 \
  --output target/evidence/client-weight-ownership.json
uv run --no-sync python -m benchmarks.research.qwen3_artifacts \
  target/qwen3-memory-artifacts
PLLM_RESEARCH_ARTIFACTS=target/qwen3-memory-artifacts \
uv run --no-sync python -m pllm benchmark run \
  --experiment benchmarks.research.qwen3_memory_frontier:resident \
  --experiment benchmarks.research.qwen3_memory_frontier:paged \
  --factory --trust-python --backend native --wan --isolate-candidates \
  --prompt-file target/qwen3-memory-artifacts/prompt.txt \
  --max-output-tokens 8 --warmups 0 --repetitions 1 --temperature 0 \
  --capture-output-digest --timeout 1800 \
  --output target/evidence/qwen3-memory-frontier.json
PLLM_RESEARCH_ARTIFACTS=target/qwen3-memory-artifacts \
uv run --no-sync python -m benchmarks.research.qwen3_memory_quality \
  --output target/evidence/qwen3-memory-quality.json
```
