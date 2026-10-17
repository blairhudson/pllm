# Complete two-offset private-boundary comparator

## Decision and promotion boundary

**Later study decision:** the author requires a source-faithful GKM
reimplementation for the ten-method framework study. The W8A8/ring32 and DPF
tracks below remain proposed adaptations. They need explicit approval and cannot
replace the original fixed-point, lookup, and verification contracts. This older
plan is not implementation authorization for those substitutions.

Plan date: **16 October 2026**. Start implementation **after the parent's current
publication-audit commit**; pin that resulting revision and recheck these seams.
This plan follows [COMPETITOR-QUALIFICATION.md](../../paper/COMPETITOR-QUALIFICATION.md)
and [COMPARISON-CONTRACT.md](../../paper/COMPARISON-CONTRACT.md).

Independently implement **complete** private lookup → batched decoder body → full
vocabulary head → client sampler/feedback, on two authenticated workers:

- **A:** GKM-derived exact-W8A8 adaptation with dense XOR-PIR and seeded shares.
- **C:** same complete computation/placement with packed-DPF lookup, a stronger
  performance architecture control, not a stronger-security claim.
- **O:** original fixed-point/ring64 and verified GKM variants remain separately
  unresolved. Neither A nor C reproduces O's numerics or verification. Upstream
  source remains specification/oracle text, never executed, linked or depended on.

Initial semantic admission: existing tied, bias-free vocabulary boundary; ordinary
full-KV CPU W8A8 decoder and client-local nonlinear operations. Admit by compiled
operators, shapes, weights and source commitments, never model-name branches.
Untied/auxiliary boundaries or unsupported operators remain explicit unresolved
cells. Generated family tests establish implementation correctness; only later
matched real-checkpoint evidence can qualify a publication comparison.

## 1. Minimal public and internal contracts

**Proposed additions**, preserving existing default serialized bytes:

```python
TwoOnlineOffsetCpu(
    model,
    linear=TwoOnlineOffsetLinear(input_encoding="seeded", dispatch="seed_first"),
    quantization=SymmetricPerRow(weight_bits=8, activation_bits=8),
    topology=TwoOnlineOffsetBoundaryRoles(),
    boundary=PrivateLookupAndFullHead(
        lookup_protocol="dense_xor_pir_v1",  # C: "packed128_xor_dpf_v2"
        head_dispatch="parallel",
    ),
)
```

- Add `TwoOnlineOffsetBoundaryRoles`, component
  `pllm/two-online-offset-boundary/v1`: exactly `client`, `worker_a`, `worker_b`;
  separation pair A/B; body, lookup, head, setup/delivery and retirement channels.
  Add `boundary` to `TwoOnlineOffsetCpu`; reuse `TwoOnlineOffsetLinear`,
  `PrivateLookupAndFullHead` and the ordinary SDK/Experiment path. No paper-named
  executor or new model frontend.
- Add `lookup_protocol` to `PrivateLookupAndFullHead`, defaulting to the existing
  packed protocol and omitted when default. Its existing head dispatch/kernel/
  output-codec options remain independently bound. Dense A uses seeded dense
  issuance below; raw dense issuance is a native correctness/control API.
- Compiler emits an explicit **v2 composite boundary/session contract**, with
  owners `worker_a/b`, protocol, source/checkpoint/config/plan/schedule/composition/
  topology identities, numeric/row bounds and resource floors. Composite head
  input encoding derives from `TwoOnlineOffsetLinear.input_encoding` and is
  recorded explicitly: seeded body implies seeded head. Existing prepared v1
  descriptor/session defaults retain their meaning and digest.
- Extend `TwoOnlineOffsetTransport` with a read-only `.boundary` owner; reuse
  `PrivateBoundaryTransport` reconstruction/codec logic through a private,
  typed `_from_offset_session(...)` binding. It borrows the parent's connections,
  exchange and lifetime; it cannot open an independent session. Runtime binding
  checks exact owner identity, not an arbitrary callback or `hasattr` capability.
- Native additions: `private_page_issue_dense(..., seeded=True)` returning two
  packets plus a one-use `PrivatePageDecoder`; `PrivatePageServer.evaluate_dense_batch`.
  Preserve `private_page_issue(..., leaf_bits=128)` and existing DPF packet bytes.

### Current hard gates to change together

`decoder_boundary.rs:160-195` requires prepared roles; compiler `lib.rs:2123-2206`
rejects the extra offset slot; `profiles/__init__.py:348-402,690-712` independently
restricts both combinations. `model_binding.py:368-409,456-466` requires a prepared
lease for boundary plus an exact offset transport for body. `client.py:1109-1159`
requires four boundary origins while offset forbids Preparation; `servers.py:1147-1153`
also demands the five-role graph. Change explicit composition branches, not the
meaning of the broad `private_boundary` flag everywhere.

## 2. Immutable owners, bundle and one lifetime

Each worker loads one ordinary public decoder engine. Body stages retain their
existing `CompiledMatrix` owners. Lookup uses the **existing native head matrix**
via `PrivatePageServer.from_matrix(head._matrix, original_scale_bytes, party)`;
context children clone `Arc` ownership, not vocabulary bytes. Existing seams:
`crates/pllm-python/src/private_pages.rs:26-48` and
`crates/pllm-garble/src/private_pages.rs:340-485`. Share only within a worker;
each independent worker still owns a complete public model replica.

Factor descriptor/admission derivation into reusable helpers over the loaded
engine; do not invoke the boundary-only checkpoint loader and load weights twice.
Keep actual-weight/scale/source checks from `private_boundary_worker.py:81-118`.
Audit existing NumPy/native loading duplicates and charge transient copies;
sharing the PIR table does not prove the engine has only one weight allocation.

Allow the offset thin bundle through `transformer_engine.py:2056-2120,2285-2312`.
Omit **both** vocabulary matrix payloads/aliases, including tied aliases and
artifact-object routes; keep tokenizer, small nonlinear coefficients, biases,
public scales and all weight/source identities. `ClientBundle` and
`model_binding.py` must reject unsolicited matrix payloads, missing scales and
local fallback. Lookup cache/deduplication stays disabled: repeated tokens and
generated feedback issue fresh queries. Preserve float32 lookup multiplication,
per-row rounding/scaling order, signed bounds and the compiled causal reduction.

**One response-owned state machine:** `new → opening → armed → active → terminal`.

1. Client chooses a fresh group nonce; opens both workers with compiled identity,
   public envelope and paired credential digests. Workers independently re-lower
   their loaded source and reserve capacity. They return distinct opaque local
   session IDs. No body or boundary work before both acknowledgements validate.
2. A bounded `arm` message binds the group, ordered pair of worker-session IDs,
   epoch, actual descriptor and schedule to both workers; then bind native lookup
   contexts. Use a new context domain including owners/protocol. Peer credentials
   remain digests; a worker receives only its own secret credential.
3. One coordinator per worker gates body and boundary routes with the same
   terminal/busy state, phase/operation sequence, per-stage row/call budgets and
   boundary counts. Body envelopes wrap existing stage packets; no independent
   route may evade the composite admission. Exclude boundary semantic operations
   from the body's stage allowlist so head budgets cannot be bypassed. Consume
   tickets/budgets before work.
4. Partial open/arm, malformed replies, timeout, replay, cancellation, EOS and
   close retire the whole composite owner. Index pending opens by the client
   nonce so cancellation also works after a lost/malformed ID acknowledgement.
   Do not refund material or reopen the same context. Completion requires no
   pending work and valid completion acknowledgements; EOS may leave unused
   reserved capacity, never an extra execution of the pending EOS token.
5. Drain seed/head futures and shield native jobs before releasing owners or
   capacity. Cancellation of an ASGI await does not stop native work; preserve the
   retirement discipline in `private_boundary_worker.py:162-178`. Standalone
   boundary close and parent close delegate to one idempotent retirement path.

## 3. Dense XOR-PIR: bounded native/wire specification

Use the existing immutable `Table::Vocabulary`/record accessor and context/replay
ledger. Put dense packet/PRG code in new
`crates/pllm-garble/src/private_pages/dense_xor.rs`; integrate through the existing
server and Python wrapper. No interleaved `V × (H+4)` table copy or Python scan.

- Record: original `i8[H] || f32_scale_le`; `W=H+4`, `Q=ceil(V/8)`. Same descriptor
  hash as packed lookup. Bounds: `1≤V≤262144`, `W≤65536`, table≤512 MiB, batch
  `1≤B≤64`, ≤4096 one-use queries/context. Checked arithmetic precedes allocation.
- For each query mint a fresh 16-byte query ID and independent 16-byte AES seed
  with `getrandom`. Derive `D=SHA256("pllm/dense-xor-mask/v1\0" || contextual_digest
  || query_id || LE32(V) || LE32(W))[:16]`; concatenate
  `AES128_seed(D XOR LE128(counter))`, counters starting at zero, truncate to Q
  bytes and clear unused final-byte bits. This uses existing AES/SHA-256 crates;
  it is a disclosed stateless per-query PRG binding, not the author's session PRG.
- Issuer flips bit `index` in this mask for A's raw complement; sends only the
  seed to B. Index never enters a packet. `seeded=False` emits both raw shares for
  independent algebra/control tests. Never derive masks from sampler RNG, public
  IDs alone or a seed shared with the complement recipient.
- New fixed **66-byte** header: magic[8], party:u8, mode:u8 (`0=raw`, `1=seed`),
  V:u32le, W:u32le, contextual_digest[32], query_id[16]. Query magic `PLLMDQ01`;
  reply magic `PLLMDR01`. Query payload is exactly Q bytes or 16 seed bytes;
  reply echoes party/mode/binding and contains exactly W XOR-share bytes. Composite
  A admits mode 0 for A and mode 1 for B; no inferred/fallback encoding.
- Server validates **all** batch headers, parties, modes, exact lengths, zero tail
  padding, contexts and duplicate/consumed IDs before burning the batch; burns
  before evaluation. Evaluate every public row using byte masks, one table
  traversal for B queries, then XOR replies in the consuming decoder. Malformed
  batch/reply burns the enclosing session even if native prevalidation rejected.
- Dense working selectors ≤`B*Q` (2 MiB), outputs ≤`B*W` (4 MiB), plus bounded
  packet/serialization copies and replay state. Enforce the existing 16 MiB outer
  frame limit before reading/allocating; split lookup batches only on public
  geometry. Add distinct magic/length/layout rejection across dense and DPF.

The existing packed batch allocates up to `B*V` selector bytes
(`private_pages.rs:528-585`); resource estimates must distinguish these backends.
Native decoders consume on success **and** error; cancellation wipes temporary
seeds/masks and retires handles, without promising complete Python/OS erasure.

## 4. Seeded body, head dispatch and joint-view privacy

Retain `_TwoOnlineShareEvaluator` batching and `offset_codec.py` seed context:
one native complement to A, fresh 32-byte seed only to B, validated dimensions and
stage identity, including composition/session/operation binding. Keep prefill
rows batched per compiled linear stage; no tokenwise latency handicap. Any row
chunking must be public, numerically identical and priced in the schedule.

Composite head uses the same seed/complement primitive at **ring32**, with a
head-specific domain and separate fresh seed, followed by existing signed-bound
checking and ordered float32 dequantization. Serial and parallel head dispatch
are explicit options; parallel means one current-step pair, no speculative decode.
Preserve `existing`/`byte_dot` kernels and `u32`/uniform/certified-row replies where
admitted, with certificate costs and actual worker rescans. Compare A and C under
identical head/body settings; measure the strongest qualified options alongside
serial/raw controls. Price differences from the prepared candidate explicitly.

Review privacy for the **joint** lookup/body/head transcript of one static
semi-honest worker. A receives independently masked complements; B receives
independent masks/seeds, and C supplies one fresh DPF key per query. Composition
requires independent entropy and disjoint domains across operations, retries,
epochs and sessions—not separate privacy statements for three APIs. Honest
client/source admission, confidential authenticated transport and non-colluding
operators are assumptions; two local processes establish execution isolation only.
Leakage includes public model/geometry, lengths/batches, stage order, sizes,
timing and termination; no token IDs, true activation scales or reconstructed
outputs on worker links. A/C make no malicious-result verification claim.

## 5. Resource/envelope and measurement gates

- Replace the worker's ad hoc **64-input / 32-output** admission restriction
  (`offset_worker.py:378-391`) with the compiled bounded envelope, preserving
  lookup/head caps, 4M-element stage limit, 8192-call policy and checked memory
  floors. Required later cells are **150+8 and 150+64**; do not masquerade as a
  continuation/cache request to pass admission. Compute counts from executed
  schedule: ordinary boundary demand is `P+D` lookups and `1+D` heads for P
  prompt rows and D executed feedback steps, not D pending selections.
- Per worker charge unique body/head owners once, scale storage, source import,
  dense/DPF selectors, request/reply copies, codec work, retained contexts and
  admitted concurrent jobs. Budget simultaneous client head replies, full KV,
  tokenizer, sampler, SDK, supervisor/relay/audit and teardown. Admission estimates
  and observed complete-client peak are separate; no Low-tier inference from
  matrix omission alone.
- Attribute body/lookup/head diagnostics to the same **physical worker links**;
  do not add duplicate boundary-role traffic or process RSS. Include attempted
  uploads, observed partial replies, setup/arm/cancel/cleanup and coverage gaps.
  Reconcile application bodies separately from all directed TCP-stream bytes.
- Extend native network placement/cost floors for this explicit two-worker graph
  (`network_placement.rs:233-241` currently rejects any private boundary). Keep
  unpriced graph variants rejected. Local relay measurements do not certify
  independent deployment; live HTTPS cells need actual source/placement admission.
- Subsequent parent-owned serial campaign: equal aggregate provider and access
  budgets; source-present cold, warm-public/fresh-private and source-cold; complete
  latency, first text, prefill/decode, traffic, client/provider peaks/CPU and length
  crossover. Freeze build/source/workloads; retain failures and pressure attempts.

## 6. Ownership and dependency order

These are implementation work packages, **not permission to run agents or builds
now**. Files have one editor. Native/API schema agreement is a prerequisite to
integration; separate files do not make evolving contracts parallel-safe.

| Unit | Exclusive files / seams | Deliverable and dependencies |
| --- | --- | --- |
| **N — native/API** | `crates/pllm-garble/src/private_pages.rs`, new `private_pages/dense_xor.rs`; `crates/pllm-python/src/private_pages.rs`; `python/pllm/_native.pyi`; `crates/pllm-compiler/src/decoder_boundary.rs`, `decoder_runtime_schedule.rs`, `lib.rs`, `network_placement.rs`; `python/pllm/roles/__init__.py`, `roles/topology.py`, `profiles/__init__.py`, `components/__init__.py` | Native dense API, exact schemas/topology, compiled bounds/floors, serialization compatibility. Own native/registration tests and new `tests/test_dense_private_pages.py`. Freeze protocol/descriptor/session identity with W/C before they implement against it. |
| **W — worker/transport** | `python/pllm/runtime/offset_worker.py`, `offset_reference.py`, `offset_codec.py`, `private_boundary_protocol.py`, `private_boundary_worker.py`, `private_boundary_transport.py`, `private_boundary_head_codec.py`, `servers.py`; new `tests/test_offset_boundary_transport.py` | Shared authenticated coordinator, v2 parser/context/arm/retirement, native dense dispatch, seeded/parallel head, physical-link counters and two-child launcher. Depends on N's API/schema and C's loaded-engine/bundle interface. Own transport/fault tests. |
| **C — client/admission/testing** | `python/pllm/runtime/transformer_engine.py`, `transformer_client.py`, `model_binding.py`, `client.py`, `semantic_executor.py`, `benchmark_memory.py`, `benchmark_communication.py`, `benchmark_response_audit.py`, `benchmark_history.py`, `network_benchmark.py`; new `tests/offset_boundary_fixtures.py`, `tests/test_offset_boundary_decoder.py`, `tests/test_offset_boundary_sdk.py` | Thin bundle, exact owner validation, SDK/session/gateway path, resource and evidence accounting, complete conformance. Depends on N's schedule fields and W's lifetime/counter surface. Own fixtures and full-decoder tests; pure semantic hooks should need minimal changes. |

**Shared seams needing serialized review:** `profiles/__init__.py` feeds every
resolver; compiler `lib.rs` also owns resource/operator admission; W's
`private_boundary_protocol.py` must match N's contract and C's independently
checked reconstruction; C's engine bundle is consumed by W's `servers.py`;
`client.py` currently chooses boundary versus offset counters mutually exclusively.
Exchange interfaces first. Land **N's schema/API plus W0's strict Python v2
descriptor/parser surface** as one reviewed contract change; W0 must precede C's
binding work. Then N completes native code, C0 supplies the loaded-engine/thin-
bundle interface, W1 supplies worker/transport, and C1 integrates SDK/evidence and
full tests. Other units request shared-file changes through the named owner;
do not edit that owner's file concurrently. Final gate reviews all three together.
Parent owns builds/installations, real checkpoint execution, cohorts and publication
edits; do not edit frozen audit artifacts.

## 7. Acceptance tests and next gate

1. **Native/protocol:** independent literal dense-XOR oracle and existing packed
   oracle; original row/scale bits; V=1, byte/128-bit tails, maximum query/batch
   envelopes and overflow rejects. Wrong party/context/magic/mode/size/tail,
   duplicate/replayed queries, consumed decoder, cancel, allocation failure;
   shared-owner lifetime survives dropping Python matrix wrapper. No missing-native
   skip may count as acceptance.
2. **Generated complete decoders:** adapt patterns in
   `tests/test_private_boundary_decoder.py`, `test_private_boundary_bundle.py` and
   `test_offset_worker_process.py`, using `create_tiny_llama_checkpoint` for Qwen2
   (QKV bias), Qwen3 (Q/K norm) and Llama, multiple layers/GQA and varied geometry.
   Run A/C through real native products and two separately supervised HTTP workers,
   ordinary `Experiment` → SDK/gateway/stream lifecycle. Mocks serve fault controls
   only. Compare with the exact clear compiled decoder and an independent numeric
   reference; no swapped decoder or client-local body/head fallback.
3. **Witnesses:** byte-exact full logits **before sampling**, active K/V for every
   layer, positions/cache lengths, executed history, pending selected token,
   selections/RNG state and output. Test greedy and seeded stochastic sampling,
   repeated tokens/feedback, batched prefill/chunk boundaries, maximum admitted
   shapes, both head dispatch modes and supported codecs/kernels. Actual fixture
   logits/configuration exercise early/final EOS through the real sampler; EOS is
   not forwarded again. Same-session progression and explicitly admitted
   continuation preserve state; unsupported cross-request reuse rejects.
4. **Failure/ownership:** mismatched second worker/source/scales, half-open/arm,
   wrong credentials, substituted bundle, reordered/duplicate operations, broken
   response, HTTP timeout, cancellation during native work/seed-first/parallel
   head, paused stream/gateway disconnect and double close. Assert both sessions
   terminal, no late result delivered, no work owner recycled before drain, no
   secret packet retention in evidence, and honest partial-byte accounting.
5. **Promotion:** generated tests plus clean installed end-to-end replay establish
   executable A/C only. Parent then runs an admitted pinned **real Qwen2.5**
   complete-request conformance cell and matched publication campaign, serially
   after the audit commit/build. Toy/native-only success, isolated lookup/head
   timing or another candidate-only listing cannot close P2-A. Missing support or
   mismatched evidence remains unresolved, never a competitor exclusion.

**Next actionable gate:** freeze N/W/C's v2 schema, owner/lifecycle and bounded
dense-wire contract; implement the independently owned vertical slices in that
dependency order. This planning pass performs source reads and writes this plan
only—no runtime/native edits, builds, model loads, cohorts or commit.
