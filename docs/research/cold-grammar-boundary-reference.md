# Cold grammar-specialized client boundaries

The subsequent [pinned checkpoint gate](public-grammar-boundary-reference.md)
executes complete tokenizer reachability, sliced-head/sampler parity and separate
native lookup-party processes. Its 73-row head is small, but 150 prompt lookups
cost 118.28 aggregate worker CPU seconds. The synthetic accounting below retains
its original scope and conservative lookup slots.

## Result and scope

**Conditional byte win, not a private serving implementation.** A public,
fixed grammar with a proven small reachable token union can replace a full
client output head with selected i8 rows and f32 scales. Arbitrary prompt
embeddings still require full-table private lookup. Two software workers must
be independently operated and non-colluding; no TEE is assumed. A single
provider controlling both workers defeats this privacy assumption.

The probe executes only small clear fixtures and protocol **accounting**.
It does not execute DPF, BFV, upstream code, real tensors, network or decoder.
There is no private embedding-provider consumer in the inspected ordinary
`ClientBundle` / `TransformerClient` boundary path. Existing research operators
are not evidence that this composition can serve requests privately.

The proposed 151936 x 896 geometry matches **Qwen2.5-0.5B**, not
Qwen3-0.6B (1024 hidden). Raw matrix is 136,134,656 bytes; scales add
607,744 bytes. Fair full tied boundary baseline is **136,742,400 bytes**.
256 rows plus scales cost **230,400 bytes**, before token map and lookup.

## Actually specified lookup option

Price repository's bounded native two-server XOR page PIR, not ideal PIR:
`crates/pllm-garble/src/private_pages.rs:1-39,106-179,249-358`.
Replicate an immutable public table at both workers. Each record contains
896 signed i8 codes then one little-endian f32 scale. Table descriptor binds
domain-separated SHA-256, count and width. Fresh cryptographic random query ID
and roots produce one DPF key per worker; XOR bound responses, recover codes
and scale locally, multiply to the declared embedding f32 representation.
Client knows token and row offset; workers must not learn either.

Native source is explicitly semi-honest and **unreviewed**. No security claim
is inferred from synthetic equality. Malicious workers can alter responses;
source header bindings are not verifiable PIR or authenticated transport.
Two-worker independence, transport authentication, replay ledger lifetime and
source integrity are admission gates, not solved here.

For V records, d=ceil(log2 V), R=H+4:

* Query each worker: 65-byte header + 16-byte root + 17d corrections + 1 bit
  byte = `82+17d` (388 at this geometry).
* Response each worker: 65-byte header + R = 965 bytes.
* Total both directions, both workers: **2706 bytes per lookup**.
* No per-client public-key/key-exchange setup for this DPF. Proposed descriptor
  delivery: two 48-byte records (digest32/count-u64/width-u64), **96 bytes**.
  This descriptor wire contract is proposed; native API returns Python tuple,
  not an existing serving transport.
* Each worker needs 136,742,400 table bytes; cold public replication across
  both workers costs **273,484,800 bytes** if counted on request's critical path.
* `evaluate_batch` admits 1..64 queries, same individual query/reply bodies,
  fresh ID per slot, up to 4096 lifetime IDs per server. It traverses table once
  per batch but does row-byte XOR work per lane. No invented amortized PIR
  query compression. Ledger exhaustion requires explicitly admitted rotation.

At fixed 16 prompt + 8 output/dummy lookup slots, online lookup bodies are
64,944 bytes. Add 230,400 head + 1024 global-token map + 96 descriptor setup:
**296,464 client boundary bytes**, before additional public metadata.
Client-only saving against full tied boundary: **136,445,936 bytes**.
Worker-cold global bytes: **273,781,264**, so **137,038,864 bytes worse** than
full client boundary. Workers must already possess matching public table for
claimed cold-client win; this distinction cannot be hidden in “setup free”.

For U reachable rows, M additional public grammar/identity metadata bytes,
L padded lookup slots, and C cold worker-distribution bytes, byte saving iff
`U*(H+4) + 4*U + M + 96 + 2706*L + C < V*(H+4)`.
Probe exposes strict integral break-even; native 4096-query lifetime still
applies even when arithmetic permits more. At U=V, even warm workers lose
because token map and lookup are extra. M is configurable: zero default means
grammar/identity metadata is already supplied or deliberately excluded, not
that a complete compiler manifest has zero bytes.

24 independent queries require 6,563,635,200 row-byte masked XOR operations
across workers. One 16-slot prompt batch plus eight serial decode slots
traverses 2,461,363,200 table bytes across workers, while XOR work remains
6,563,635,200 bytes. These are operation/traffic models, **not wall-time or
physical DRAM measurements**. Client can reuse locally delivered selected
rows as generated-token embeddings when tied identities are proven, reducing
real lookup need; this probe deliberately keeps all eight output slots as
private dummy/real queries for conservative constant-shape accounting.

## Privacy and length composition

Grammar, union and artifact selection must be explicitly public and independent
of private prompt-derived facts. Publishing a union derived from “answer using
these private entity names”, private enum values, or retrieved documents leaks
task data even if row retrieval is PIR. Probe rejects `visibility=PRIVATE`;
this is a policy assertion, not an automated proof of public provenance.

Tokenize prompt entirely client-side with exact pinned tokenizer, normalization,
added/special tokens, BOS/EOS policy and chat template. Count their download
bytes on both sides if common, or as additional specialization bytes if not
common. Never ship prompt text/token IDs to worker. Arbitrary input token IDs
are valid across full table, including tokens absent from public output union.

Use public prompt bound P and public output bound G. Fixed prompt batches
(split at 64 slots) include fresh indistinguishable dummy token lookups. Do not
pass padding into body as real prompt tokens. Attention validity, positions,
prefill row schedule and true cache length must stay private or be explicitly
disclosed by an independently admitted body protocol. Fixed embedding lookup
traffic **alone does not hide prompt length in ordinary decoder**. Alternative
explicit disclosure of actual prompt token count is valid only if declared;
lookup bodies then scale with disclosed count. Default accounting assumes P=16.
Overflow must reject locally, not leak an adaptive grammar or batch size.

Generation keeps grammar state, sampled token IDs, RNG, positions, logits and
cache client-private. Grammar masks update locally; do not fetch dynamic state
rows based on private output. EOS terminates client-visible output normally;
send public-bound dummy lookups/body work after EOS if termination is private.
Exactly how dummy work preserves a fixed protected body schedule remains open.
Accounting includes one conservative output lookup per slot, including last
slot, although normal decoder does not embed EOS or final emitted token.

## Numerical and generation contracts

Executable fixture has V=12, H=4, signed i8 rows, binary-exact f32 scales,
public three-state token grammar, globally mapped union and EOS. Independent
tests cover all 81 hidden vectors in {-1,0,1}^4, state masks, greedy and
stochastic distributions/selections, ties, excluded huge logits, nucleus
threshold behavior and row/scale reconstruction. Evidence stores 18 explicit
cases. Mask before softmax, not after unrestricted sampling. Full-head masked
distribution and selected-row distribution agree exactly under declared oracle.
Four additional clear toy-generation traces use prompt tokens entirely outside
output union, an explicitly non-transformer additive embedding/cache recurrence,
local state transitions, two uniform draws and normal EOS no-cache-update rule.
Full/sharded traces match including final cache and hidden state. This tests
boundary composition wiring, not real transformer/body/privacy correctness.

Oracle arithmetic is integer dot times f32 row scale evaluated in binary64;
it is **not SDK float32 quantization/dequantization bit parity**. Sampling follows
current SDK's `cumulative <= top_p`, always retaining maximum; this differs from
common nucleus “include crossing token” convention. Declared equal-score order
is ascending global ID. Fixed uniform inverse CDF is used to compare selection,
not NumPy `Generator.choice` replay. SDK's `argsort(...)[::-1]` stochastic tie
ordering is not a stable global-ID contract. Greedy ties select smallest global
ID. Exact existing SDK seeded stochastic sequence is therefore **not proven**.

Normal `TransformerClient.generate_steps` prepares prompt, samples logits,
breaks on configured EOS, emits decoded token, and advances cache with
`decode_step` (`transformer_client.py:1748-1821`). Current `_forward` returns
dense vocabulary f32 logits and applies output multiplier/softcap after head.
To keep existing sampler unchanged, reconstruct full vector with -infinity at
disallowed global IDs. This uses V*4=607,744 bytes transient client memory,
not head-download bytes. Match output multipliers, softcap, activation
quantization, accumulation ring, dequantization order and float32 cast exactly.
Head bias, when present, must also be sliced under same identity.
Do not apply finite softcap to -infinity placeholders before final mask.
Global logits/history processors must either have equivalent restricted
semantics or reject: arbitrary unmasked logits/logprobs, rank queries and
unconstrained fallback cannot be supplied from shard. Top-p order, RNG state,
EOS and multiple stop-token semantics must be pinned in compiler contract.

Body layer caches/shared KV, prepared-prefix snapshots, request identity,
positions, full hidden state and final normalization must remain unchanged.
Private lookup must return embedding codes/scales matching same compiled model
identity; lookup alone cannot replace protected body or its complete state.
No claim of clear-provider confidentiality, full generation, or cache parity.

## Source identity and missing compiler output

`transformer_engine.py:1020-1068` resolves tied head and lookup to same source
key when manifest declares tied embeddings; lookup uses transposed orientation.
`transformer_engine.py:2093-2121` exports one `tied_embeddings` object from head
quantized weight and assigns linear/embedding layouts. Current ordinary export
therefore permits **exact row slicing of this exported tied object**, not fresh
independent quantization of head/embedding tensors. Matching a source name alone
does not prove quantized-byte/scale parity across pipelines or checkpoints.
`scripts/probe_client_weight_ownership.py:45-82` supplies existing synthetic
ownership/lookup/head oracle, including shared snapshot identity. This study
reads, does not execute that implementation.

Current compiler exports full client matrix shape and scales. Client importer
checks shapes (`transformer_client.py:628-710`); ordinary token lookup expects
full table. Tied head cannot be remote-only (`transformer_engine.py:2110-2112`).
Missing compiler contract:

1. Public grammar source/provenance hash, tokenizer/template/special-token hashes,
   reachable-state token union proof (token byte strings, not characters), global
   IDs, EOS/stop semantics and grammar-state transition validator.
2. Checkpoint/source tensor digest, tied/untied declaration, exported quantized
   table/code/scale digests, orientation, shape, row-record byte format and head
   shard identity including full-to-shard mapping. Untied tables need separate
   identities; equal source labels cannot substitute for byte equality.
3. Private embedding provider role graph, bound descriptors, authenticated
   independent-worker transport, public batching/padding/disclosure policy,
   replay/cancellation/rotation behavior, worker memory and setup ownership.
4. Hidden/head numerical contract, dense masked-logit reconstruction or explicitly
   restricted result type, sampling/tie/RNG contracts and supported processors.
5. Normal generation/cache/body state contract, fixed private-length execution
   admission, stop behavior and exact replay compatibility.

`PrivateEmbeddingLookup` (`python/pllm/protocols/__init__.py:271-307`) is a
bounded BFV coefficient benchmark limited to V<=4096, not drop-in 151936-row
consumer. `python/pllm/metrics/operator_benchmarks.py:166-168` wires operator
benchmark only. Planned-method gate in `components/planned_methods.json`
explicitly leaves full-width execution, compiler transport and tied-head memory
admission open. No HE run or invented extrapolation here.

## Negative grammar cases

* Union=full vocabulary: byte saving negative even before worker provisioning.
* Private-derived union: rejected, irrespective of size.
* Arbitrary JSON strings: RFC 8259 permits broad Unicode content. Depending on
  tokenizer, tokens containing arbitrary ordinary text remain reachable inside
  strings and union may be nearly full. “JSON grammar” does not establish U=256.
  This probe records warning, not a real Qwen tokenizer union measurement.
* Missing token/EOS, inconsistent shapes, changed source identity, invalid scales
  or unsupported geometry fail closed. No silently expanding union on request.

## Replay and next gate

Run from repository root:

```sh
target/frontier-sdk-env/bin/python -m pytest -q -o pythonpath= tests/test_cold_grammar_boundary.py
target/frontier-sdk-env/bin/python scripts/probe_cold_grammar_boundary.py --output docs/evidence/cold-grammar-boundary-synthetic.json
target/frontier-sdk-env/bin/python scripts/probe_cold_grammar_boundary.py --replay docs/evidence/cold-grammar-boundary-synthetic.json
```

Evidence hashes probe, tests, note and inspected repository source bytes; no
timestamp, timing or machine-specific values. Replay verifies all known fields
and recursively preserves unknown extension fields. Source changes intentionally
invalidate exact replay. The evidence file itself is not recursively self-hashed.

**Best next gate:** pinned real tokenizer + public finite enum/task grammar,
compute conservative reachable token union including EOS/special tokens without
private task data. Then parent serially verify exported tied code/scale identity
and row-sharded masked float32 logits through existing numerical kernels; pin
SDK sampler behavior and cache replay before integrating private provider.
Do not start with arbitrary JSON or heavy BFV.

Resource estimates (not measurements): present probe uses 12x4 fixture, no
NumPy or native arrays, comfortably <1 MiB extra fixture data. Parent can verify
256x896 shard with 230,400 payload bytes, one hidden row and full f32 logits
607,744 bytes; additional arrays can stay <128 MiB with streamed/mapped raw
matrix access. A full copied tied matrix plus scales is 130.4 MiB and already
exceeds 128 MiB extra-array budget. Native PIR `Server::new` copies full table
per worker: two workers add ~260.8 MiB plus source ownership, masks and outputs;
do not allocate in this probe. At batch64, worker masks add 9,723,904 bytes and
responses ~57,600 bytes per worker. Parent must preflight and serialize any
checkpoint/DPF experiment; CPU scan/AES costs remain unmeasured.

## Primary references

* [Qwen2.5-0.5B-Instruct configuration](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/blob/main/config.json):
  V=151936, H=896, tied embeddings, EOS=151645. Mutable upstream inspected for
  this screen; checkpoint/tokenizer pin required for next gate.
* [Qwen3-0.6B configuration](https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/config.json):
  H=1024; distinguishes requested geometry from Qwen3.
* Boyle, Gilboa, Ishai, [Function Secret Sharing: Improvements and Extensions](https://eprint.iacr.org/2018/707)
  (CCS 2016, expanded report): two-party DPF foundation and concrete optimization;
  does not certify repository's clean-room implementation or this composition.
* [RFC 8259, sections 2 and 7](https://www.rfc-editor.org/rfc/rfc8259): JSON
  grammar and arbitrary Unicode strings; lexical structure is not small-token
  reachability proof.
