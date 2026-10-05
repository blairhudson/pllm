# Five client-offload experiments

These are new PLLM engineering screens, not claims of literature-first novelty.
The starting point is the pinned Qwen3-4B paged client, whose short-response
tokenizer/setup allocations dominate its small KV state. The earlier tokenizer
lifetime experiment changed ownership; this round asks whether public setup or
private computation/storage can move to another role.

| Candidate | Work moved | Privacy and numeric gate | Cost gate |
| --- | --- | --- | --- |
| Public precompiled BPE index | Vocabulary/merge parsing and index construction to an offline public compiler | The client pins a trusted compiled artifact; all prompt-dependent BPE and decoding queries stay local. Compare exact IDs and decoded text, including normalization and special tokens. | Fresh-process RSS and setup plus execution CPU; count the derived artifact and compiler cost. |
| AEAD-sealed immutable KV slots | Persistent KV storage to an untrusted blob service | Fresh response key; unique nonce; plan/session/layer/shape-bound AAD; append-only ordered slots. Fetch the complete public prefix, preserving float32 bytes and the existing attention reduction. | Price every upload and every decode reload; retain one layer plus crypto and attention scratch. |
| Encrypted mask tapes | Native mask expansion to offline Preparation | The existing non-colluding Preparation role already knows these masks. Only ciphertext goes to Inference; its decryption key stays with client/Preparation. Consume or burn each tape slot once. | Compare client AEAD decryption with native SHAKE expansion; charge both Preparation→Inference and Inference→Client tape bodies. |
| Shallow encrypted attention islands | QK and probability×V products to an HE evaluator with persistent encrypted KV | Only the client has the secret key; softmax stays local and refreshes ciphertext depth. Compare against an independent clear attention oracle. | Count KV ciphertexts, public evaluation keys, both online rounds and both parties' CPU. |
| Shallow encrypted SiLU islands | A fixed public polynomial activation to an HE evaluator | Fixed public interval and coefficients; client-only secret key. Check polynomial and encryption error separately. Approximate arithmetic cannot inherit the existing W8A8 numeric identity. | Price each fresh SIMD input/output across the semantic MLP schedule, before gated multiplication or any remaining decoder work. |

## Initial measured screen — 6 October 2026

Raw record: [five-probe Qwen3-4B screen](../evidence/client-offload-five-qwen3-4b-2026-10-06.json).
Sizes below use decimal MB. Tokenizer medians use two fresh processes per
candidate in resident/indexed/indexed/resident order; AEAD timings use three
local repetitions. The HE islands have one sample. No new host swap was observed.

| Candidate | Result | Decision for the short 16+8 response |
| --- | --- | --- |
| Public precompiled BPE | Peak process RSS **169.57 → 36.18 MB**; setup plus 16-case encode/decode CPU **177.49 → 16.09 ms**. All 182 token IDs and decoded text match. | Follow-up whole-client probe below confirms lower peak RSS. Prompt-dependent queries stay local; warm encode/decode rises **1.02 → 8.16 ms**. |
| Sealed KV | Synthetic KV bytes and attention match exactly. The 4B geometry has only **6.78 MB** logical KV at 16+8, while uploading slots and fetching each whole decode prefix adds **48.11 MB** of bodies. | Reject for this short-workload memory objective. At 4096+8 the same layout moves 1.21 GB logical KV but adds **9.67 GB** bodies. |
| Encrypted mask tapes | The largest stage's 16-row client expansion falls **3.834 → 0.378 ms** when replaced by decryption. Preparation expansion plus sealing takes **4.396 ms**. | Reject for WAN optimization: projected tapes add **329.04 MB all-link**, including **164.52 MB** client download, before request framing. |
| HE attention | One 8-row/16-wide encrypted QK/AV island uses **9.51 MB** KV upload and **8.53 MB** online bodies; client online CPU is **72.49 ms**, versus **0.00487 ms** clear attention. | Reject this layout. Even an unimplemented four-ciphertext-per-layer packing floor projects **322.88 MB** for seven 4B decode steps. Float32 parity fails; maximum synthetic error **0.00151**. |
| HE SiLU | A degree-15 public polynomial has **0.00869** sampled approximation error; encrypted evaluation reaches **0.04436** error against SiLU. | Reject this layout and numeric identity. Fresh SIMD input/output bodies project **1,213.39 MB** for the 4B 16+8 SiLU tensors alone. |

The tokenizer index is a **9.29 MB** derived public artifact, alongside the
**11.42 MB** original tokenizer source. Its separate public compiler costs
**0.529 CPU seconds** and peaks at **128.32 MB** RSS. These costs are moved,
not eliminated. The isolated client peak falls 78.67%; this is neither a
whole-client 4B memory measurement nor a 10× whole-response improvement.
The two HE islands share a **63.05 MB** public evaluation context, separately
charged from their online bodies. Their provider context contains no secret key.
HE CPU figures are in-process phase samples; the client phases also include
loading serialized input ciphertexts into the reference provider context.

KV and tape traffic are geometry projections from the pinned semantic schedule,
not measured network transfers. AEAD authentication, slot substitution,
single-use consumption, malformed-frame burn and retirement have executable
regressions. Encryption does not change the existing Preparation
non-collusion/erasure assumption or establish malicious-evaluator correctness.

## Complete capped 4B response — indexed tokenizer follow-up

Raw record: [matched fresh-process pair](../evidence/client-runtime-memory-qwen3-4b-indexed-tokenizer-2026-10-06.json).
Both candidates use the ordinary guarded prepared benchmark, paged client
weights, one response-owned tokenizer, a 16-input-token public capacity bound,
and an eight-output-token cap. The control uses the original in-memory BPE;
the candidate binds the offline index to the original bundle's tokenizer bytes.
Only the probe adapter changes; this is not a selectable Pipeline component.

| Measurement | Shared original tokenizer | Shared indexed tokenizer |
| --- | ---: | ---: |
| Client/dashboard process-lifetime peak RSS | 347.00 MB | 225.28 MB |
| Client/dashboard end-of-benchmark RSS | 346.72 MB | 225.03 MB |
| Largest tokenizer-construction high-water increase | 103.61 MB | 3.28 MB |
| Client CPU, benchmark startup through first response | 29.42 s | 28.81 s |
| Aggregate role CPU, same cold scope | 181.96 s | 180.10 s |
| Measured request duration | 70.77 s | 70.16 s |
| Covered setup-through-response application bodies | 571.58 MB | 571.58 MB |
| Covered online application bodies | 141.71 MB | 141.71 MB |
| Observed host swap growth | 0 | 0 |

Peak RSS falls **121.72 MB, or 35.08% (1.54×)** against the shared-tokenizer
control. Both requests pass every runtime/audit check and match the source lock,
Pipeline, model body fingerprint, salted prompt cohort, 16 input tokens, eight
output tokens, and output-text digest. No plaintext prompt or token-ID bytes
are sent; all prompt-dependent index lookups remain client-local. Output-text
agreement is not a full-logit/KV parity or generation-quality evaluation.

The public compiler separately uses **0.537 CPU seconds** and **128.43 MB** peak
RSS. Its **9.29 MB** artifact is pre-positioned locally, so its delivery is
excluded from the equal transport counters above and must be added to a cold
deployment. The original **11.42 MB** tokenizer source remains in the bundle.
This single ordered pair does not establish a latency or compute improvement;
provider work, OS file cache and filesystem storage are outside client RSS.
It does establish a whole-client allocation improvement for this capped,
cached-checkpoint response. Neither a resident-weight 4B control nor a 10×
whole-client result has been measured, and historical ratios are not multiplied.

The [64-token-capacity attempt](../evidence/client-runtime-memory-qwen3-4b-indexed-tokenizer-blocked-2026-10-06.json)
failed physical-headroom admission before a completed pair. A subsequent
[interrupted attempt](../evidence/client-runtime-memory-qwen3-4b-indexed-tokenizer-interrupted-2026-10-06.json)
was stopped after a threaded regression exposed SQLite connection ownership.
The successful rerun uses serialized SQLite and covers cross-thread use and
owner cleanup. Both unsuccessful attempts are retained and excluded from the
comparison. Admission reserves were preserved; both successful candidates use
the same narrower public workload bound.

## Boundaries

- The BPE experiment is a public, trusted-build artifact reference. A server's
  self-asserted digest cannot authorize a tokenizer. It uses native SQLite pages
  with bounded caches as a memory oracle, not a proposed production BPE kernel.
  SQLite queries must never be delegated: their keys contain private token text.
- Sealed KV storage does not offload attention arithmetic. Streaming or changing
  reduction order would need a separate numeric contract. The blob oracle has
  no network transport, restart/resume, durable anti-rollback or runtime state
  adapter. It leaks the same public ordered workload shape used in its projection.
- Mask tapes preserve offline Preparation and its erasure/non-collusion
  assumptions. They add incompressible traffic. The test-local AEAD owner is not
  a replacement for native inventory commitments, reservation and burn ledgers.
- HE evaluations are bounded synthetic islands using the existing TenSEAL/SEAL
  dependency. They do not establish real-checkpoint quality, malicious-evaluator
  security, compiler coverage or complete-response resource use. Ciphertext
  packing projections are explicitly tied to this layout and parameter set.
- None of these probes is an executable Pipeline choice. Reports retain only
  public source/configuration hashes, counts, parity outcomes and aggregate
  measurements. No private prompts, token IDs, masks, keys or activations are
  archived.

## Reproduction

From the repository root, with the pinned public Qwen3-4B configuration and
tokenizer already present in the shared Hugging Face cache:

```bash
HF_HUB_OFFLINE=1 uv run --no-sync python scripts/probe_client_offload.py \
  --output docs/evidence/client-offload-five-rerun.json
HF_HUB_OFFLINE=1 uv run --no-sync python scripts/probe_client_runtime_memory.py \
  --four-b --max-input-tokens 16 --tokenizer-index-ablation \
  --output docs/evidence/client-offload-indexed-runtime-rerun.json
uv run --no-sync pytest -q tests/test_client_offload_probes.py
```

The controller runs a separate public compiler process and four alternating
fresh tokenizer clients, then isolated KV, mask and HE workers. It preflights
2 GiB of physical headroom plus the ordinary host reserve, monitors pressure and
swap, bounds child duration, and retains an incomplete report if a gate fails.
The scratch directory is configurable with `--scratch-dir`.
The complete-response follow-up uses the ordinary whole-topology admission and
pressure watchdog, fresh client processes, isolated client bundle caches and one
shared private cohort salt. Its report binds the probe and tokenizer source-file
hashes. Provider roles still use the cached checkpoint and their ordinary loaders.

References: [Hugging Face Tokenizers](https://huggingface.co/docs/tokenizers/),
[cryptography AEAD contracts](https://cryptography.io/en/latest/hazmat/primitives/aead/),
[CKKS approximate arithmetic](https://eprint.iacr.org/2016/421),
and the repository's earlier
[whole-layer HE gate](../evidence/he-token-boundary-feasibility-qwen25-2026-09-30.json).
