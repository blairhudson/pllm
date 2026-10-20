# Bounded FuseFSS decoder and ordinary SDK

`FuseFssSourceCpu` selects a model-neutral ring50/Q12 decoder with a separate
trusted offline dealer and two online additive-share holders. The generated
GPT-2 and dense-RoPE SDK paths execute two fixed-count selections through three
actual child processes, including private feedback. SDK/gateway failure and
cancellation tests exercise the same native owners and authenticated HTTP path.

## Numeric and source binding

The two static profiles are
`fusefss.gpt2-complete-ring50-q12.v1` and
`fusefss.dense-rope-ring50-q12.v1`. Their approved equations are D13 in
[`paper/methods/DECISIONS.md`](../../paper/methods/DECISIONS.md).
Original F32/F16/BF16 bytes stream through bounded native import. Source,
tokenizer, geometry, numeric identity and program commitments precede private
input issuance. Public weight paths remain in source mappings; method lowering
consumes semantic operators rather than model-specific node names.

Every contraction accumulates Q24 products, floors once to Q12, then adds its
Q12 bias. Attention scale has a separate product and floor. Dense RMSNorm retains
the source table, unaveraged square sum plus integer one and separate affine
floors. RoPE floors each signed coefficient product before addition. These are
explicit method numerics, not FP32 checkpoint equivalence.

The opt-in `fusefss.dense-rope-checked-ranges-ring50-q12.v1` profile adds D14's
private RMSNorm guards. Each complete row must satisfy
`|x| <= floor_sqrt((2^49 - 2) / width)`; the bound is 792,648 Q12 codes at width
896. A failed guard selects an entire zero row before normalization. Secret
validity persists across every padded row and selection, and masks all final
tokens. The client raises `NumericDomainRejected` on final failure without
releasing earlier tokens or a failing location. Accepted requests retain both
D13 floors and tables. Other arithmetic obligations still require admission.
The checked-profile factory is `examples/papers/fusefss.py:qwen_checked_smoke`;
native generated parity does not establish trained-checkpoint acceptance.

Five frozen generated models cover tied/untied GPT-2, Qwen2, Qwen3 Q/K norm and
Llama structures. The independent semantic interpreter checks 918 complete
observations across three full-prefix selections each, including physical padding,
causal GQA, full heads, feedback and final IDs. The older 5,125 internal-register
exchange has a different observation contract and is not claimed as passing.
The strict semantic contract lives in
[`paper/reproduction-contracts/fusefss/decoder/SEMANTIC_EXCHANGE.md`](../../paper/reproduction-contracts/fusefss/decoder/SEMANTIC_EXCHANGE.md).

## Ownership and transport

The dealer receives no private input or masked online openings. It delivers only
each recipient's fresh material to that holder's encrypted, ordered one-use tape.
After acknowledged setup, the client retires the dealer before sending either
input share. Online peer exchanges are direct between holders; selection and
feedback stay shared. Only the final fixed-count IDs reconstruct at the client.

Native canonical codecs check context, role, schedule and exact lengths. Failed
imports burn claims; cancellation drains callbacks and HTTP owners before capacity
refund. Tapes authenticate records, consume in order and reject replay. Returned
frames retain native leases; Python/HTTP copies have separate conservative owner
allowances. Tape deletion and reference retirement do not claim secure physical
erasure. Confidential authenticated remote channels and non-colluding operators
remain deployment assumptions; co-located benchmark children do not prove them.

The native live-material allowance remains 64 MiB. Request-long exact-key replay
ledgers now support bounded indexed heap or private-file storage; claims survive
individual kernel retirement. Tape admission derives record counts from complete
packet, metadata, framing and authentication-tag costs. Capsule retains its
256 MiB cumulative-packet and 1 GiB file ceilings. Explicit Decoder capacity has
a 32 GiB per-holder framed-tape ceiling and separately priced replay files.
Source, program, table/cache, callback and export owners retain separate bounds.
Shared immutable public tables do not permit mask/key reuse.

Decoder tapes use a distinct v2 encryption domain: the complete original metadata
remains AES-GCM additional authenticated data, supplied again by the native-bound
reader. Disk retains the 24-byte header, ciphertext and 16-byte tag per record.
Pricing is `min(cumulative_packet_bound, record_count * packet_bound) + 40 * record_count`;
metadata input and live-buffer bounds remain charged. Capsule and direct tape
callers retain inline metadata and the original v1 format by default.

The explicit `material_delivery="batch-v1"` choice coalesces up to 64 original
material records per authenticated HTTP body, bounded to 1 MiB including framing.
It reuses shared length-prefixed framing, native erasing packet owners and the
existing sequential tape append. Both holders acknowledge a complete batch before
the dealer advances; partial acceptance, lost acknowledgement or cancellation
retires the whole attempt. Larger admitted records use the existing singleton
path after queued records drain. Native queues and Python copies receive separate
charges under the existing aggregate ceiling. The checked SDK contract binds the
choice; numeric Program/source-body identities and default single delivery remain
unchanged. Batching does not merge one-use records or online peer exchanges.

`capacity="decoder"` streams Q12 coefficients and row statistics into shared-core
immutable files. Public contractions reuse `pllm_core::streamed_matrix`; the
complete product still precedes the original FuseFSS floor. Complete vocabulary
lookup reuses the consuming one-hot primitive, and tiled stable selection retains
every head row. Shared source/tokenizer validation permits unmapped reserved head
rows without removing them from selection. The existing tokenizer omits such IDs
from text; the selected IDs remain unchanged.

Shared integer matrix/row-energy certificates tighten public admission bounds.
Their provenance follows complete semantic rows through supported views and
operators. They do not remove a normalization's input-domain obligation or change
the source tables, epsilon, two affine floors, or private multiplication schedule.

## SDK and benchmark interface

`FuseFssSourceDecoder`, `TrustedDealerTwoShareRoles` and
`FixedCountPrivatePrefix` use ordinary component discovery, immutable
serialization and Experiment resolution. The typed profile selects all required
slots. Unsupported sampling, tools, retained KV and continuation reject.
Streaming API requests buffer text until complete final disclosure; EOS does not
shorten the selected count.

The reusable [generated factory](../../examples/papers/fusefss_smoke.py) and
[ordinary benchmark outputs](../../paper/artifacts/fusefss-sdk/README.md) exercise
both numeric profiles. The CLI, `BenchmarkTarget.response` and shared response
driver use the same role supervisor and resource admission. CPU/RSS is reported
per owner, dealer/holder/client application bodies are counted once per directed
edge, and tape I/O stays separate from network totals. Reports retain
setup-inclusive final disclosure and leave streaming TTFT/decode throughput null.
Metadata preflight and completed native program identity have separate scopes;
an absent role resource estimate remains unknown.

Worker phase counters separately measure dealer preparation, delivery and seal
callbacks, and each holder's native execution. Preparation outside callbacks
includes planning, issuance, encoding and bridge work. The reported maximum holder
duration is not the client online interval; client online time remains unknown.

A separate generated dense-RoPE sample records 3,875 material records and 4,513
peer frames per holder. Dealer preparation takes 5.396 seconds, including 5.279
seconds in delivery callbacks (97.82%). Maximum holder-native execution takes
3.683 seconds. The original ordinary report is
[`dense-phase-single.json`](../../paper/artifacts/fusefss-sdk/dense-phase-single.json).
Delivery callbacks include HTTP, receiver append and scheduling. This small
serial-HTTP sample supplies neither trained-Qwen timing nor a measured batching
improvement.

A subsequent fresh-client single/batch pair uses the same generated source,
numeric contract, private cohort and 1+2 workload. Dealer preparation falls from
5.393 to 0.390 seconds, delivery callbacks from 5.277 to 0.288 seconds, and full
request time from 9.680 to 4.796 seconds. Output digests agree; online bodies stay
7,681,488 bytes and each holder reads/writes the same 6,189,173 tape bytes. Covered
all-link bodies fall from 21,618,568 to 19,909,435 bytes through reduced delivery
envelopes and acknowledgements. Aggregate cold-first CPU falls from 11.022 to
6.956 seconds; maximum holder-native time rises from 3.565 to 3.656 seconds.
Both samples observe zero swap growth. This one-pair generated screen is separate
from trained-Qwen execution and independent confirmation. The ordinary report and
exact factories are linked from the [benchmark README](../../paper/artifacts/fusefss-sdk/README.md).

Decoder launch first admits a bounded source-qualification phase. One temporary
native owner supplies actual program and transport prices, then retires before
fresh whole-topology memory/disk admission and child launch. Workers independently
import and recheck the pinned source. Qualification time and CPU belong to ordinary
cold setup. A metadata-only preflight keeps conservative unknown transport costs;
it does not certify a whole request. Explicit Decoder deadlines may extend to
86,400 seconds, with remaining time reduced by imports and setup rather than
renewed by peer traffic.

Shared disk leases retain pending capacity and credit only verified physical
backing of live, same-device files, deduplicated by file identity. File length or
global free-space changes do not supply that credit; sparse replay capacity and
unwritten tapes remain pending. These are process-local admission reservations,
not filesystem quotas.

[`examples/papers/fusefss_conformance.py`](../../examples/papers/fusefss_conformance.py)
supplies the same frozen generated Qwen2 source for Capsule and Decoder. Ordinary
three-child SDK checks preserve three independent-oracle selections, private
feedback, more than 4,096 consumed tape records and cleanup. Decoder also exercises
file-backed source/replay owners and native-priced launch. Native tests compare
all 186 complete Qwen semantic observations; these are generated-model checks.

## Qualification boundary and next gate

No trained GPT-2/Qwen quality, matched paper performance or independent security
qualification is established. The static profile's earlier 1+1 source import
rejected numeric admission before worker launch. Its unresolved early obligation
is the next RMSNorm input after layer-zero MLP; later conditional propagation does
not discharge that obligation.

The separate D14 profile now imports and compiles pinned Qwen2.5 at 4+1 under
Decoder capacity, with zero observed swap growth. Its immutable source occupies
3,952,262,144 coefficient bytes plus row statistics. Native admission prices
2,602,128 kernels, 11,668,738 material records per holder and 13,424,643 peer frames
per holder. These are complete scheduled counts, not execution measurements.
Private range acceptance still requires the actual request to run. The initial
inline-metadata tape quote requires 71.53 GB of whole-topology disk including the
2 GiB reserve, and rejects the observed host before child launch.
External-AAD tape pricing reduces that quote to 47,630,190,304 bytes while
preserving all material records, complete metadata authentication and replay
storage. This is admitted storage arithmetic, not measured request execution.

The integrated revision passes 1,025 Rust library tests across core, garble and
compiler, 816 installed Python checks, and two additional three-child Decoder
tests. The latter consume more than 4,096 v2 tape records and enforce late
checked-range rejection followed by successful fresh-request recovery. These
tests retain their generated-model scope.

Trained execution, independent integer parity, quality and ordinary matched
benchmarks remain open. The common 64+8 envelope still exceeds sequence/selection
limits. The reusable pinned-source factory is
[`examples/papers/fusefss.py`](../../examples/papers/fusefss.py).
The [completion plan](../../paper/methods/fusefss-completion-plan.md) keeps these
remaining method gates explicit.
