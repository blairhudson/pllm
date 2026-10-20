# Bounded FuseFSS decoder and ordinary SDK

`FuseFssSourceCpu` selects a model-neutral ring50/Q12 decoder with a separate
trusted offline dealer and two online additive-share holders. The generated
GPT-2 and dense-RoPE SDK paths execute two fixed-count selections through three
actual child processes, including private feedback. SDK/gateway failure and
cancellation tests exercise the same native owners and authenticated HTTP path.

## Numeric and source binding

The two explicit profiles are
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

The native material allowance remains 64 MiB. F4 additionally bounds complete
requests at 4,096 primitive imports and each holder's cumulative tape packets at
256 MiB. Source, program, table/cache, callback and export owners retain their
separate bounds. Shared immutable public tables do not permit mask/key reuse.

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

## Qualification boundary and next gate

These are functionality and accounting checks on tiny generated sources. No
trained GPT-2/Qwen quality, matched paper performance or independent security
qualification is established. The full pinned sources exceed current native
source/value limits. Larger generated F3 controls also exceed F4's request-long
import count despite fitting native computation bounds.
An installed pinned-Qwen metadata check rejects the common 64+8 workload at the
request bounds; a separate 2+2 check rejects source expansion before tensor values
or workers are loaded.

Next work is a bounded monotonic schedule/replay owner, followed by complete
checkpoint source/range/work admission. It must preserve fresh material and
failure retirement; clearing a replay set or raising a cap alone is insufficient.
The [completion plan](../../paper/methods/fusefss-completion-plan.md) keeps these
remaining method gates explicit.
