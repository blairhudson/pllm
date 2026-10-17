# Bounded verified decoder reference

This internal reference composes the [GKM capsule](gkm-delegation-capsule.md)
with the native semantic compiler and existing client-local float32 operators.
Generated Qwen2, Qwen3 and bias-free dense Llama fixtures execute complete
prefill and greedy decode. Each request verifies its complete linear window
before returning logits, final active KV, executed history or selections.

The governing [source contract](../../paper/methods/delegation-contracts.md)
and [CPU/common-model decisions](../../paper/methods/DECISIONS.md) remain in
force. This bounded reference does not qualify original paper checkpoints or
the full Qwen2.5 checkpoint. Both logical workers share one process.
Ordinary SDK integration is required for method completion under
[D5](../../paper/methods/DECISIONS.md#d5-ordinary-sdk-integration-is-required).
The native and generated-decoder milestones cannot satisfy that gate alone.
The subsequent [bounded SDK slice](gkm-sdk-reference.md) supplies party-local
worker processes and the ordinary SDK/gateway/CLI/benchmark path. This earlier
in-process reference retains its original evidence and limits.

## Compiler and source binding

`GkmDecoderContract::compile` lowers public configuration and workload through
the existing model-neutral graph. It reuses the decoder matrix inventory and
the ordinary scheduler's grouping authority. It binds ordered source tensors,
matrix members, output slices, biases, local coefficients, original token table,
full output head, prefill/decode steps and the complete verification window.
There are no model-name branches in the compiler or executor.

The native importer accepts exactly the declared finite little-endian float32
tensor bytes. It hashes actual bytes with the compiled contract before publishing
commitments. Tied source weights have one source name but separate original
float32-table and encoded-matrix owners. The trusted in-process publisher pins
the verification material. This does not authenticate caller-supplied bytes
against an official checkpoint lock.

The primitive and local boundaries are:

1. Dense XOR lookup returns original float32 embedding bytes. Signed zero is
   preserved and the native owner checks the selected row hash.
2. Linear weights and inputs use nearest-even multiplication by 2048. Weights
   must fit signed i16. Encoded inputs have public absolute bound `2^31`.
3. Each worker evaluates its ring64 share. The client reconstructs a signed
   integer, casts it to float32, then divides by `2048^2`.
4. Declared biases are added by the client after that division. Float32 RMSNorm,
   rotary positions, causal attention, SiLU, multiplication and residuals use
   shared semantic local operators. Qwen3 Q/K normalization is retained.
5. Full KV remains client-owned. The head uses only the final prefill row and
   every decode row. Complete logits drive client-local greedy selection.

The local profile is
`semantic.float32.dense-full-kv.client-bias-after-div22.v1`. It specifies this
conformance graph, not equivalence to W8A8 or to every upstream float kernel.
The independent Python oracle evaluates the model equations directly and uses
Python integer sums for each delegated product. It calls neither the semantic
executor nor native GKM numeric helpers.

## Acceptance and retirement

One fixed-length request contains its prefill and zero to eight decode steps.
The final greedy selection remains pending: it is not part of executed history
or KV. EOS does not shorten this conformance workload. Streaming, stochastic
sampling, prefix reuse and restoration are not admitted.

The compiler enumerates every lookup and linear row in order. Native ownership
checks the next operation and stage, with fresh challenges after frozen replies.
Only the final verification barrier emits an acceptance receipt. The receipt
binds source, contract, publication, request, window, claim count and sharing
selector. Tentative values are available only to the trusted local computation.

Full shares and the [seeded profile](gkm-seeded-shares.md) use the same numeric
contract. Only logical Worker One expands its independent linear/lookup AES
streams. Prefill rows are issued serially in this reference; this is not the
paper's batched-prefill performance implementation.

Wrong order, range, shape, Python argument type, incomplete verification and
explicit abort retire native material. Local failures and cancellation retire
the request, clear its KV arrays and release local coefficient references.
Returned accepted data consists of immutable bytes. Caller-owned copies and
Python exception temporaries do not acquire a secure-erasure guarantee.

## Admission and ownership

Admission precedes graph expansion, source conversion and secret issuance.
The source registry resolves architectural dimensions before applying the
caller-selected native expansion budget. Unpriced mappings reject this path.
Source-map cardinality is checked before values are read; only declared tensor
names are accessed. Configuration encoding bounds traversal before allocation:

| Owner or dimension | Bound |
| --- | --- |
| Configuration | 16 KiB; encoder nesting depth at most 64, with root depth zero |
| Prompt rows / decode steps | 1–32 / 0–8, also subject to complete-window limit |
| Matrix rows and columns | At most 256 each |
| Token table | At most 4,096 rows; the complete head also imposes the 256-row matrix limit |
| Matrix groups / window operations | At most 16 / 128 |
| Any declared tensor | At most 262,144 elements |
| Unique float32 source bytes | At most 8 MiB |
| Reference payload estimate | At most 64 MiB |

The reference estimate adds the existing capsule allowance, three source-payload
copies, six conservative graph/scratch/state owners and 16 MiB for metadata.
The capsule retains its shared publication/window leases and process-local
logical payload ceiling. Source loading, caller buffers, allocator overhead,
interpreter/libraries and process RSS remain separate. These are admission
estimates, not measured memory tiers or universal OOM guarantees.

## Validation and replay

The installed-wheel suite checks six family/tied-head fixtures in both sharing
layouts. Twelve complete decoder executions compare **2,112 float32 logit and
final-active-KV values**, plus all selections, executed histories and positions.
Nineteen Python tests additionally cover source changes, original signed-zero
lookup bytes, malformed arguments, geometry/range rejection and failure
retirement. Four native compiler/session tests exercise complete operation
ordering and independent integer/byte oracles.

Shared-executor regressions cover ordinary compiled bindings, state, retention,
private-boundary execution and generated hybrid SDK/gateway execution. The
initial run had 151 passes, two optional real-Gemma skips and four failures.
Two failures came from a test fixture missing its existing boundary-owner field;
two lacked the optional Transformers dependency. The repaired affected files
pass seven tests with the same two optional skips. The union covers 155 passing
shared-executor cases; failed records remain retained.

A separate formatted-source cached wheel passes all 174 installed Python
tests in one run, with the same two optional real-checkpoint skips. All 356
installed package files match that wheel. Forty native inventory, schedule and
decoder tests also pass. Its initial 120-second test timeout remains separate
from the completed retry, with the remaining owned role explicitly retired.

The subsequent independent review found two admission defects: source layer
counts reached graph construction before resource admission, and the Python
facade copied arbitrary source mappings before checking their size. Historical
functional results do not establish those repairs. The native repair adds a
source-mapped expansion budget before any graph-builder call; unpriced source
mappings reject. A sentinel-lowerer regression checks the ordering without
constructing an oversized graph.

The follow-up review found two shape-pricing defects in that shared budget API:
an ignored Qwen2 head-width field could underprice KV, and Qwen3's wider query
projection was missing. Source-faithful width resolution and complete projection
pricing close both findings. Sentinel controls reject both 256-element probes
before graph construction, while 512-element controls preserve ordinary plans
and check every resulting activation/state shape. The independent source/test
recheck found no remaining admission blocker in this bounded native path.

A third cached wheel validates all repairs in a fresh environment: **206 Python
tests pass**, with the same two optional skips. Its 51 decoder/admission cases
include 32 new facade regressions; the 155 shared-executor cases remain separate.
All 356 installed package files match the wheel before and after testing.
The native suite passes **44 tests**, including three source-mapping and five
decoder/session tests. Test-source snapshots remain unchanged. These results
do not replace the earlier functional evidence or constitute security certification.

Build and test records, source snapshots and the read-only replay verifier are
under `paper/artifacts/gkm-decoder/`. All three guarded cached wheel builds observed zero
swap-use or actual compressor-swapout growth. This is local build evidence, not
a clean-build attestation or a serving-resource measurement.

```sh
CARGO_BUILD_JOBS=2 cargo test --offline --locked -p pllm-compiler \
  --test delegated_decoder --test decoder_matrix_inventory \
  --test decoder_runtime_schedule -- --test-threads=1
python -I -B -m pytest -q -o pythonpath= tests/test_delegated_decoder.py
```

The Python command requires a wheel containing this internal binding. It must
not silently run against an older extension. The archive records the checked
wheel and test environment.

## Next gates

1. Complete independent cryptographic review and qualify the larger composed
   source, acceptance and memory contracts. The bounded engineering review does
   not automatically qualify full-width or distributed execution.
2. Admit full-width immutable source owners, public verification material and
   complete request windows with honest peak/copy accounting.
3. Bind an official checkpoint import and the complete local numerical graph;
   compare against that method's clear oracle and original-model quality gates.
4. Add source batched prefill, confidential authenticated transport, independent
   role processes and ordinary SDK/gateway lifecycle ownership.
5. Measure complete publication, cold delivery, client/worker resources and
   request costs under the frozen study contract.

The separately accepted Qwen source scan proves only source coefficient bounds.
It supplies no missing activation, local-kernel, full-width, transport or quality
evidence. No Pipeline, serving benchmark ranking or cryptographic certification
is enabled by this reference.
