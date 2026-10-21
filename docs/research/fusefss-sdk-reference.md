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

The separate `online_schedule="wavefront-v1"` option executes public dependency
groups of the original consuming primitives. At most 256 ready primitives,
2 MiB of decoded party material and a 1 MiB framed opening fit one group; the
unchanged aggregate 64 MiB ceiling includes schedules, registers, exports and
retained frames. Exact kernel resources may reduce the group further. Every
primitive retains its own context, original packet and fresh one-use material.
Malformed groups, replay, partial replies and cancellation consume the active
owner. Public schedule construction and repeated metadata also count as work.

Bounded lane templates combine independent contractions, floors, activations,
normalization rows and attention work. Public matrix products retain complete
accumulation before their floor. Adjacent-pair private selection preserves the
lowest original index on ties and carries odd tails unchanged. Logical tree
depth and actual tile/exchange counts remain separate. Checked-range validity
stays secret and sticky through complete feedback and final masking. D15 records
the approved execution adaptation; numeric source identities stay comparable,
while Program, schedule and admission commitments bind the new order. Serial
execution remains the default. Full dense conformance fixtures use Decoder's
existing cumulative-work envelope; the new construction work can exceed Capsule
even when its live payload fits.

`online_schedule="tensor-v1"` additionally retains one immutable instruction and
register-description template per tile. Lane-specific private registers, keys and
wire contexts remain independent; logical instruction, register and work limits
still apply. Public dependency spans also retain one template; bounded virtual
lane/PC lookup reconstructs the same expanded dependency order without allocating
one ready-work record per lane. Sorting, address lookup and hashing remain charged.
A ready set may contain up to 4,096 primitives, constrained by the
same 2 MiB decoded-material, 1 MiB frame and 64 MiB aggregate bounds. Construction
prices the overlap between old and compact metadata before releasing either
owner, and whole-topology admission charges all three construction owners.
The ordered selection tree uses strict score comparison with left-on-tie payload
movement. Its compiler-owned ascending leaves and adjacent-subtree reductions
prove lowest-original-index ties, including odd tails, without privately comparing
index payloads again. The general arbitrary-index selector remains separate.
This selector and compact templates have a distinct Program/schedule identity;
the original Q12 numeric source contract remains unchanged.

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

A subsequent checked-range/batched Qwen attempt passes source and topology
admission and reaches online execution. After approximately 66 minutes in the
measured request window, macOS warning-level memory pressure triggers retirement
before final acceptance. The four-hour deadline remains unexpired; the pressure
observation alone does not attribute its cause to a process or allocation. The
[ordinary failed checkpoint](../../paper/artifacts/fusefss-sdk/qwen-checked-batch-pressure.json)
retains an incomplete status and no report. No token, numeric acceptance,
completed resource measurement or quality result follows. All role processes
retire. Online gate batching and callback-cost calibration are the next execution
gate; offline material batching does not reduce these peer exchanges.

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

## WAN execution gate

Full-model completion requires measured execution with both access-rate caps and
propagation delay. The shared native relay now covers the complete FSS graph:
client controls, dealer-to-holder material delivery and both online peer paths.
Each party shares its upload/download allowance across peers. Same-party traffic
remains local. `--link-conditions` selects a delay-only `LinkConditions` document;
`--docker-network-profile` remains an alias.

```sh
uv run --no-sync pllm benchmark run \
  --experiment examples/papers/fusefss_smoke.py:dense_batch_smoke \
  --factory --trust-python --backend native --wan \
  --link-conditions examples/benchmarks/wan_20ms.json \
  --prompt hello --max-output-tokens 2 --timeout 600 \
  --capture-output-digest --output target/fusefss-generated-wan.json
```

This selects 100 Mbps download/40 Mbps upload per party and 20 ms of added delay
per inter-party direction: 40 ms added request/reply RTT. Propagation is pipelined
after rate admission, not a serial sleep for every read chunk. A shared eight-MiB
queue allowance, charged within the existing 32-MiB relay owner, bounds delayed
reads. Bandwidth-delay products beyond that allowance reject before startup.
Cancellation drains queued owners; reports bind and validate the selected delay,
rates, role routes and queue bounds. Native loss and a second rate limiter reject.

This is local TCP-payload emulation, not Internet measurement. TCP handshakes,
kernel headers, retransmissions and checkpoint distribution remain outside its
scope. Setup material, client/relay work and all routed role bodies remain charged.
Buffered final disclosure still supplies no streaming TTFT or decode throughput.
The scalar online schedule is not WAN-qualified; tensor-wide dependent rounds,
balanced private selection and preprocessing volume remain execution gates.

The matched online-schedule factories are
`examples/papers/fusefss_smoke.py:dense_batch_smoke` and
`examples/papers/fusefss_smoke.py:dense_wavefront_smoke`. The pinned checked-source
candidate is `examples/papers/fusefss.py:qwen_checked_wavefront_smoke`. Selection
alone supplies no trained acceptance or WAN performance result.

A separate fresh-client matched pair runs both generated factories at shared
100/40 Mbps caps and 20 ms per direction (40 ms added RTT). Both produce the
same output commitment with zero observed swap growth; native and SDK controls
above supply the independent tensor/selection and rejection evidence.

| Measured scope, one sample per arm | Serial | Wavefront |
| --- | ---: | ---: |
| Full request, including offline preparation | 222.505 s | 34.583 s |
| Maximum holder native-run elapsed | 212.013 s | 24.251 s |
| Dealer preparation elapsed | 8.801 s | 8.469 s |
| Covered all-link application bodies | 19,909,435 B | 14,022,118 B |
| Covered online application bodies | 7,681,488 B | 1,716,046 B |
| Startup-through-response TCP streams | 24,273,664 B | 14,584,395 B |
| Client lifetime peak RSS | 112,312,320 B | 112,295,936 B |

This is a generated hidden-width-two, one-layer, three-token-vocabulary 1+2
workload. Its 6.43-fold request reduction and 8.74-fold holder reduction do not
project full-Qwen performance. Client RSS is effectively unchanged. Fresh
preprocessing remains charged; checkpoint distribution, physical TCP/IP overhead,
Internet measurements and independent operators remain outside scope. The
ordinary report is [`fusefss-wavefront-wan-pair.json`](../evidence/fusefss-wavefront-wan-pair.json).

```sh
uv run --no-sync pllm benchmark run \
  --experiment examples/papers/fusefss_smoke.py:dense_batch_smoke \
  --experiment examples/papers/fusefss_smoke.py:dense_wavefront_smoke \
  --factory --trust-python --backend native --wan \
  --link-conditions examples/benchmarks/wan_20ms.json \
  --prompt hello --max-output-tokens 2 --timeout 600 \
  --capture-output-digest --isolate-candidates \
  --output target/fusefss-wavefront-wan-pair.json
```

### Trained-source scheduling gate: wavefront

Source-only qualification of the pinned checked Qwen candidate at 4+1 completes
range construction but rejects the kernel walk at layer 23 SiLU: cumulative public
copies reach 34,371,192,850 bytes, above the unchanged 34,359,738,368-byte ceiling.
This includes the new public schedule construction; it is not a private-value
rejection or a completed request. No workers launch.

A separate 1+1 source/pricing control admits 29,334 peer exchanges per holder,
4,144,944 fresh primitive records per holder, and a 64,688,850-byte combined native
transport reservation under the 64 MiB ceiling. Native packet pricing is bounded
by 5,181,796,140 bytes per holder, before HTTP metadata; that bound is neither an
observed transfer nor a transfer lower bound. Complete source/tape/guard/reserve
disk pricing is 25,317,061,976 bytes. Qualification retires its source owner and
observes zero swap growth, without launching workers or executing private input.

At 40 ms RTT, the current sequential HTTP exchange schedule alone implies about
1,173 seconds (19.6 minutes) of latency waits for that 1+1 control. The logical
selection tree therefore does not make the bounded tiled schedule WAN-practical.
This motivated compact public schedule ownership and wider bounded tensor waves;
the subsequent qualification below measures that implementation separately.
Fresh preprocessing delivery remains a separate cost gate. These estimates do
not qualify trained acceptance, quality or serving.

### Compact tensor schedule: generated WAN pair

A new fresh-client pair compares `dense_wavefront_smoke` with
`dense_tensor_smoke` under one private cohort, the same generated 1+2 workload,
shared 100/40 Mbps caps and 20 ms propagation per direction. Output commitments
match, every report check passes, and both memory guards observe zero swap growth.

| Measured scope, one sample per arm | Wavefront | Tensor |
| --- | ---: | ---: |
| Full request, including offline preparation | 35.749 s | 34.732 s |
| Maximum holder native-run elapsed | 25.299 s | 24.503 s |
| Dealer preparation elapsed | 8.656 s | 8.387 s |
| Aggregate startup-through-first-response CPU | 9.769 s | 7.310 s |
| Covered all-link application bodies | 14,022,118 B | 13,916,321 B |
| Covered online application bodies | 1,716,046 B | 1,690,014 B |
| Startup-through-response TCP streams | 14,584,397 B | 14,467,175 B |
| Client lifetime peak RSS | 113,410,048 B | 111,804,416 B |

This separate pair observes a 2.84% request-time reduction. It does not isolate
template sharing from ordered-selection changes or establish a causal memory
gain. Its tiny generated geometry supplies no trained-model speed projection;
checkpoint distribution, physical TCP/IP overhead, Internet measurements and
independent operators remain outside scope. Buffered disclosure supplies neither
streaming TTFT nor decode throughput. The ordinary report is
[`fusefss-tensor-wan-pair.json`](../evidence/fusefss-tensor-wan-pair.json).

```sh
uv run --no-sync pllm benchmark run \
  --experiment examples/papers/fusefss_smoke.py:dense_wavefront_smoke \
  --experiment examples/papers/fusefss_smoke.py:dense_tensor_smoke \
  --factory --trust-python --backend native --wan \
  --link-conditions examples/benchmarks/wan_20ms.json \
  --prompt hello --max-output-tokens 2 --timeout 600 \
  --capture-output-digest --isolate-candidates \
  --output target/fusefss-tensor-wan-pair.json
```

The native composition suite passes 110 checks, including compact-versus-expanded
dependency order, 1,024-lane execution, fresh issuance and allocation-failure
retirement. The 118 decoder checks include all five frozen semantic fixtures,
checked-range rejection and stable ordered selection. The rebuilt installed wheel
passes 408 focused Python checks plus four three-process SDK/gateway checks.

### Trained-source scheduling gate: tensor

The pinned `examples/papers/fusefss.py:qwen_checked_tensor_smoke` 4+1 source
qualification completes public range construction but rejects the kernel walk at
the output head: cumulative public copies reach 35,178,175,330 bytes against the
unchanged 34,359,738,368-byte ceiling. This is a resource rejection before worker
launch, not a checked private-input rejection or accepted request.

A separate source/pricing control selects the same factory with
`budget__max_input_tokens=1`. Source pins, numeric profile, vocabulary and selection
count match the earlier wavefront control. Tensor pricing admits 19,229 peer
exchanges per holder (34.45% fewer), 3,081,399 fresh primitive records per holder
and a 61,851,153-byte combined native transport reservation. Complete native
material is still 7,451,946,472 bytes. The per-holder packet allowance is
4,168,085,820 bytes, excluding HTTP metadata; it is an upper-bound price, not a
measured transfer or traffic lower bound. Two encrypted tapes plus replay guards
require 9,136,331,688 bytes, with source-file ownership charged separately.
Qualification retires its source owner and observes zero swap growth; it runs no
private input and launches no workers.

At 40 ms RTT, sequential peer exchanges still imply about 769 seconds (12.8
minutes) of waiting before compute. Compact templates and the ordered tree do
not make this trained workload WAN-practical. Further round and fresh-material
reductions, four-input resource admission, trained acceptance and quality remain
open; the capacity ceilings have not changed.

The ordinary [generated response report](../evidence/fusefss-generated-wan.json)
records two final outputs in 222.124 seconds, with 7.766 seconds of dealer
preparation and 212.664 seconds for the maximum holder-native run. Covered
all-link bodies are 19,909,435 bytes, including 7,681,488 online bytes. All runtime
checks pass and observed swap growth is zero. This single transport qualification
run, which overlaps short regression checks, is a negative practicality result;
it supplies no trained-Qwen timing or matched speed comparison.
