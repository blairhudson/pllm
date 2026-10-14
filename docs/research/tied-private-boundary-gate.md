# Exact coupled private embedding and full output-head gate

## Contract and status

`scripts/probe_tied_private_boundary.py` is a bounded research cost and correctness
gate. It relocates **both** uses of Qwen2.5's tied W8 vocabulary matrix: native
packed128 private lookup returns the original i8 row and original float32 scale;
two native exact wrap32 head kernels return additive shares of every vocabulary
dot product. The client reconstructs signed integers and applies the original
activation/weight scales in the original float32 expression order.

No output vocabulary restriction, replacement quantizer, TEE, or plaintext token
query is permitted. The complete vocabulary, original sampler and RNG sequence
remain intact. Each worker receives only its own fresh DPF query half and its own
OS-random additive hidden-code share. Activation scales stay client-local. Head
scales are public and fetched in full; fetching individual scales by private token
index is not allowed. Workers need not receive scales to evaluate head integers.

**This standalone runner has no archived real-checkpoint measurement.** No SDK
complete-response traffic, latency, client RSS, checkpoint quality, or throughput
improvement follows from these boundary-only operators. The separately implemented
SDK composition and two matched cold-client cohorts are documented in the
[coupled boundary reference](private-boundary-sdk-reference.md). They use different
worker ownership and framing; their results do not measure this standalone runner.
This document supplies no measured performance result or Pipeline selection.

The preceding [lookup acceleration reference](private-lookup-acceleration-reference.md)
retains the client's tied head and therefore removes zero client matrix bytes.
The [round-two screen](client-offload-round2.md) projects tied-boundary costs from
three samples. This gate instead executes every admitted boundary row, including
generated-token lookups, with independent integer and numeric validation.

## Exactness and privacy

The pinned model is `Qwen/Qwen2.5-0.5B-Instruct`, revision
`7ae557604adf67be50417f59c2c2f167def9a775`: 151,936 vocabulary rows, hidden width
896, tied BF16 source, original per-row W8 and activation A8 quantization.
Publication streams at most 512 source rows at a time. Each lookup record contains
896 original i8 codes followed by the original little-endian float32 scale.
The full matrix is never sliced to a finite output language.

For each hidden row, fresh independent uniform `u32` share `a` hides the signed
quantized input `q`; the other share is `q-a mod 2^32`. Both workers own the full
public signed-i8 head. Their exact ring dot products add to the original integer
head. The signed bound is `896 × 127² = 14,451,584 < 2³¹`; centered reconstruction
is unambiguous. Reconstruction precedes float conversion and scaling:
`integer.astype(float32) * activation_scale * original_weight_scale`.

Lookup responses must match original source-derived record bytes and float32
embedding bits. Every head integer is compared against an independent signed
int64 dot oracle, chunked across **all** vocabulary rows. Every reconstructed
logit bit is compared with original `ClientBundle.local_linear`. Original
full-vocabulary sampling checks selected tokens and RNG continuation. Replays
must reject; every issued decoder is retired even on failure.

The threat model remains bounded semi-honest, non-colluding workers. Separate
fresh interpreters on one host are measurement roles, not independent operators.
Native framing is not authenticated transport or malicious-worker verification.
Reports retain identities, public geometry, aggregate costs and output hashes;
they omit token IDs, prompt contents, hidden vectors, queries, shares and replies.
Ephemeral local JSON/hex harness messages do contain each addressed role's share;
the trusted controller reconstructs them. No sensitive user prompt is used.

## Executable schedules and baseline

| Public prompt rows | Output heads | Generated-token lookups | Total embedding lookups |
| ---: | ---: | ---: | ---: |
| 16 | 8 | 7 | 23 |
| 150 | 8 | 7 | 157 |

The eighth selected token needs no feedback lookup because generation stops.
The seven earlier selected tokens are looked up with fresh one-use packed128
queries, even if token IDs repeat. Prompt queries use batches of at most 64;
150 rows execute as 64, 64, 22. No rows are extrapolated from three sample tokens.
Generated-token queries execute individually, with seven table traversals per
party; batching future feedback queries together would underprice causal decoding.
Issued and executed logical counts and per-party counts are separately recorded.

The local baseline executes original `ClientBundle.local_linear` eight times,
with a full native paged tied head and original W8A8 quantization. Local token
lookup uses original `ClientBundle.local_token_lookup` over that same snapshot,
with one prompt gather and seven generated gathers; it owns no duplicate lookup
record table. It is not an offloaded ring kernel mislabeled as local execution.

Each offload sample uses two fresh head-plus-prompt party interpreters, then two
fresh generated-lookup-only interpreters after client reconstruction and sampling.
Each process sees only one party's shares; all four setup costs are charged.
Generated-only parties have a lookup table but no head snapshot. This deliberately
simple serial schedule is **not** concurrent latency, a persistent online service,
or a per-token full decoder replay. Synthetic hidden vectors are fixed public
fixtures, independent of retrieved embeddings. Body execution, KV updates and
causal hidden-vector generation are absent from this real-weight runner.

Baseline precedes offload within each cell. Prompt-length order alternates between
replicates; baseline/offload order does not. Filesystem cache state is neither
flushed nor claimed cold. Fresh interpreter means process-cold, not disk-cold.

The tiny test suite separately substitutes **both** lookup and head in generated
Qwen2, Qwen3 and Llama decoders, under default and `prefix_f32` reduction. It checks
complete-vocabulary logits, active KV bytes, tokens, RNG state and generated-token
feedback. Additional tests independently cover signed Python integer dots,
original float32 order, random fresh shares, packed boundaries at 127/128/129/257,
original scale/record bytes, replay retirement, admission and ledger reconciliation.
These tests establish generated-model functionality, not real-checkpoint quality.

## Byte and resource ledger

Raw payloads are exact:

* Matrix: **136,134,656 bytes** (`151936 × 896`).
* Public head scales: **607,744 bytes** (`151936 × 4`).
* Lookup table per worker: **136,742,400 bytes** (`151936 × 900`).
* Each head-enabled worker additionally creates an independent private anonymous
  head snapshot of **136,134,656 disk bytes**. This is not equivalent resident RAM.
* Two persistent logical head-plus-lookup owners would retain **545,754,112 bytes**
  of table plus snapshot payload. The serial runner does not measure that joint RSS.
* At table import, Python input bytes overlap the native retained table. The
  temporary Python input is released before head creation. Native head metadata,
  transient tile buffers and process-lifetime peak RSS are reported separately.
* The harness shares one public staging head file and one record file. A real
  source-missing deployment retaining delivery files at both workers owes those
  extra disk copies too. Source checkpoint storage is not included in these
  derived-artifact payload totals and remains charged to source-owning roles.

The client matrix payload can move only when both embedding and head move.
Scales stay local; tokenizer, model body, KV state, sampler and remaining decoder
resources do not disappear. The original local client has one tied snapshot;
the harness controller retains public scales and reference outputs for validation,
so its RSS is not a deployment client-memory result.

Head online payload is `2 × 8 × (896 + 151936) × 4 = 9,781,248 bytes`.
Lookup bodies include native DPF query/reply framing; their lengths are summed
from every issued/executed party frame rather than guessed from a projection.
Head bodies count raw `u32` arithmetic bytes only: **no head protocol framing
exists in this runner**. JSON/hex pipes, HTTP, TLS, authentication, retries and
physical wire are excluded and marked unmeasured, never zero.

Two source cases are separate:

1. **Source present at both workers:** no derived table/head delivery payload is
   required, but public derivation and all retained source/derived storage still
   cost resources. The shared publisher is timed separately; independent worker
   source derivation is **not** timed and must not be presented as free. Client
   scale artifact and every online body remain charged.
2. **Source missing:** delivery owes two complete tables and two complete heads,
   **545,754,112 raw bytes**, before real transport. The report also prices exact
   bounded research zlib artifacts for both workers, plus client scales and online
   bodies. Delivering a different source checkpoint instead is another unmeasured
   strategy, not an authorized zero-byte substitution.

Research artifact framing is the existing bounded `artifact_lengths` format,
not SDK serialization. Matrix/scales, scales-only and worker table/head artifacts
are individually measured. Any apparent operator subtotal saving is not a full
cold-request or SDK response saving.

## Admission, identities and reproduction

The runner requires an already cached pinned checkpoint and installed native
extension. It never downloads weights, builds native code or installs a wheel.
`frontier_probe_support.fresh_workers` supplies the shared serial measurement lock,
post-reserve **2 GiB** host admission, pressure/swap watchdog, **240-second** process
timeout and process-group retirement. An additional **2 GiB** disk gate bounds
publication/snapshot work. Table admission is **256 MiB**; cohort repeats are
bounded to 1–3. Source checkpoint hashes, derived files, probe/support code,
quantizer/sampler/paged code, relevant Rust sources, Cargo inputs and installed
extension identities are locked and rechecked before report publication.
`BuildRef` is checked by the shared runner. Interrupted/failed runs publish no
complete report; a new output path is mandatory. Per-party setup, lookup/head CPU
and elapsed windows remain separate from client issuance, share construction,
reconstruction, sampling and independent oracle validation. Import startup is
included in lifetime peak RSS but not the explicitly timed setup window.

Targeted tiny tests (no checkpoint access):

```sh
target/frontier-sdk-env/bin/python -m pytest -o pythonpath= -q tests/test_tied_private_boundary_probe.py
```

Parent-scheduled real-weight run only, after source/build work stops:

```sh
target/frontier-sdk-env/bin/python scripts/probe_tied_private_boundary.py \
  --output target/tied-private-boundary-qwen-r1.json --repeats 3
```

Do not run concurrently with SDK cold-request measurements. Operator evidence
alone cannot establish SDK admission/lifecycle, complete-response parity or
whole-client resources. The separate SDK reference records its own completed
checks and cached-source costs. Independent non-colluding deployment, physical
wire and missing-worker source delivery remain separate measurement gates.
