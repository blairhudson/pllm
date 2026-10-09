# Compiler-bound native client-stage ownership

This bounded research slice tests the first gate/up stage of the
[frame-to-state arena proposal](../plans/frontier-tenfold-challenges.md#a-frame-to-state-arena-compiler).
The maintained prepared SDK recipe remains the control. The experimental response
target selects one semantic role and layer inside the complete client process.

## Ownership and execution

`NativePreparedLinear` validates the existing compiled checkpoint binding and
accepts only its declared remote-stage subset. Immutable native metadata binds
stage ID, dimensions, maximum rows, activation precision, exact wire/residue
widths, weight scales and bias. Python retains semantic scheduling, input
equalization, authenticated session transport, inventory coordination and
telemetry. Transformer-body weights remain provider-owned.

The native one-use query claims the existing lazy mask stream, performs the
existing quantize/mask/pack arithmetic, and retains output masks, row scales and
one output allocation. Responses are parsed through bounded borrowed MessagePack
fields and checked against ordered tickets, stage, row count, ring and residue
digest. Unmask/dequantize writes directly into the output owner without building
a decoded response-body copy. Float32 multiplication and bias order remain exact.
The final read-only buffer retains its Rust owner through NumPy views and zeroizes
only after the last view retires.

Each input and output matrix admits at most 4,000,000 values, with at most 4,096
rows, 128 chunk replies and a 16 MiB plus 8 KiB frame ceiling. Public stage geometry
is checked before mask claims. Query completion, malformed frames, replay,
reordering, incomplete completion and cancellation retire remaining private
state. Transport failure also closes the complete response lease, burning unused
reserved rows. Existing one-use mask domains and protocol frames stay unchanged.

Whole-topology admission retains the control's conservative one-active-stage
mask/conversion allowance and semantic live-root workspace. This owner uses the
same bounded stage geometry; it does not reduce the admitted memory estimate.

This is an internal stage owner, not a selectable Pipeline component, native
decoder instruction tape, shared tensor arena, protected nonlinear path or
Freivalds implementation. Unsupported compositions reject. Python, HTTPX,
tokenizer, model binding and full decoder/KV ownership remain in the measured
client; the small native owner does not replace their memory floor.

## Validation

- An independent NumPy/integer oracle checks exact input packing and output bits
  for 16/24/32-bit rings, A4/A8, one-row decode and chunked 17-row prefill.
- Truncation at every response byte, malformed fields, replay, reordered chunks,
  cancellation and transport failures verify one-use retirement.
- Generated Qwen2, Qwen3 and Llama checkpoints preserve every compared prefill
  and decode logit and active KV bit through separate Preparation/Inference
  children. Byte checks account separately for the variable-length provider
  duration field.
- Native tests and Clippy exercise bounded parsing and arithmetic; installed-wheel
  tests also cover the existing fused boundaries, lazy masks, benchmark targets,
  frontier studies and role cleanup.

The installed-wheel suite passes **133 tests**. Release-mode native checks pass
two stage-owner and four offset-transport tests; Clippy passes with warnings
denied. Documentation regeneration, **90 documentation checks** and all executable
SDK examples in those checks pass.

## Measurement contract

`benchmarks/research/client_stage_ownership.py` constructs a `FrontierStudy`
around `examples/prepared_sdk.py`. Its control and candidate differ only in the
explicit `client_stage.owner` response-target selector. The ordinary runner owns
fresh processes, build locks, private cohort salt, checkpointed evidence, memory
admission and shared native 100/40 Mbps pacing. Repeat cells execute a cold request
and exact-repeat request; sequence cells execute their ordered fresh or growing
contexts. Client artifact caches start empty. Public checkpoint
and indexed-tokenizer artifacts are pre-positioned.

Separate workload cells exercise shorter prefill, the existing medium prompt,
longer prefill, longer decode, unrelated fresh prompts and growing message
histories. Growing histories contain fixed public assistant turns and use ordinary
message-history re-prefill plus eligible prefix reuse; they do not reuse
response-owned continuation snapshots. Each retains all seven serving axes and
whole-response application-body counts. Stage-local timing records identify only
the selected stage's contribution and cannot replace whole-client measurements.
Different workloads do not rank together. Physical wire, checkpoint/index
distribution, filesystem cache and independent provider operation remain outside
these local measurements.

Replay the generated transport control:

```sh
python -m benchmarks.research.client_stage_ownership --tiny --processes 1 \
  --output target/client-stage-tiny.json
```

With the public artifacts from the maintained prepared study:

```sh
python -m benchmarks.research.client_stage_ownership --processes 1 \
  --artifacts target/prepared-frontier-artifacts \
  --output target/client-stage-screen.json
```

Use repeated `--workload` flags to select cells. `--confirm SCREEN --nominate
medium/native` constructs a separate three-process finalist/control cohort through
`FrontierStudy.confirm`; the study must still include every screening cell, even
when only one finalist is nominated. `--serial-issuance` explicitly selects serial
Preparation for both owners when a workload exceeds the concurrent work-window
bound. It creates a separate composition and does not enlarge that bound.
`--input-bound` selects an explicit public compiled token ceiling for both owners;
actual request sizing and whole-topology memory admission still apply. Reuse the
same bounds and issuance choice when confirming that screen.
Qwen3 artifacts from `benchmarks.research.qwen3_artifacts`
can be passed through the same `--artifacts` argument; their publisher record
supplies the pinned source and tokenizer identity. `--prompt-file` supplies the
medium public workload independently of the tokenizer artifact directory, so the
long-context cell need not inherit a short publisher example. Keep model cohorts separate.
Existing output files reject. Source/build/workload/output commitments
must pass before interpreting any memory ratio. Extending this owner across
stages or into local state remains conditional on measured savings.

## Initial checkpoint screen

The first pinned Qwen2.5-0.5B screen has one fresh client per owner and workload,
each running a cold and exact-repeat request under shared native 100/40 Mbps caps.
Its installed build is
`fc9c0e009ff459e74c662486503839895d7582f94d79da22a2da6ccff432842d`.
The three completed workload pairs preserve output commitments and covered
application bodies, with all ordinary runtime checks passing. The longer-prefill
pair fails before recording a response and supplies no memory or latency result.

All seven serving axes remain visible below. CPU and TTFT use seconds; RSS and
TCP-stream traffic use decimal MB. Client RSS is process lifetime high-water;
Preparation RSS is sampled inside request windows. TTFT and decode TPS summarize
both requests; traffic includes their measured setup. These scopes are unchanged
from the maintained study.

| Workload / owner | Client RSS | Client CPU | Prep CPU | Prep RSS | TTFT | Decode TPS | Stream MB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 46+8 / Python | 177.46 | 16.670 | 11.608 | 161.15 | 9.560 | 3.819 | 262.935 |
| 46+8 / native | 166.31 | 16.520 | 11.564 | 160.10 | 9.493 | 3.822 | 262.935 |
| 150+8 / Python | 213.78 | 21.322 | 16.482 | 212.37 | 22.518 | 3.738 | 496.046 |
| 150+8 / native | 218.60 | 21.598 | 16.103 | 214.58 | 22.550 | 3.666 | 496.046 |
| 46+32 / Python | 161.46 | 19.201 | 12.496 | 180.17 | 9.523 | 3.654 | 378.220 |
| 46+32 / native | 162.66 | 19.138 | 12.370 | 176.80 | 9.501 | 3.836 | 378.220 |

The apparent short-workload RSS reduction is **6.28%**; medium and longer-decode
RSS instead increase **2.25%** and **0.74%**. A single screen cannot distinguish
repeatable savings from process variation. Exact arithmetic compatibility does
not justify extending this stage owner across the decoder.

Evidence: `docs/evidence/research-client-stage-qwen25-screen.json`, with its
`.plan.json` and `.frontier.json` companions. The original failed longer-prefill
cells remain in that record.

## Independent short-workload confirmation

A separate cohort alternates **three fresh clients per owner**, each executing
the same **46+8 cold/repeat** workload and installed build as the screen. All
twelve requests pass ordinary runtime checks. Source, numeric, state, workload
and output commitments match, covered bodies are exactly **260,893,054 bytes**
per cold/repeat pair, and every host guard reports zero observed swap growth.

| Priority metric, median across three clients | Python owner | Native owner |
| --- | ---: | ---: |
| Client lifetime peak RSS, MB | 163.43 | 165.59 |
| Client cold-first CPU, s | 16.556 | 16.742 |
| Preparation cold-first CPU, s | 11.474 | 11.631 |
| Preparation sampled request-window peak RSS, MB | 159.86 | 161.35 |
| Online TTFT across both requests, s | 9.467 | 9.512 |
| N−1 decode tokens/s across both requests | 3.921 | 3.895 |
| Cold+repeat all-link TCP streams, MB | 262.935265 | 262.935266 |

Client RSS **increases 1.32%**, rejecting the initial 6.28% screen improvement as
a repeatable memory result. Complete cold/repeat latency is nearly unchanged:
**61.301 / 11.070 → 61.497 / 11.122 seconds**. Small stream-count differences are
control/framing bytes, not arithmetic-payload savings. The bounded native owner
preserves exact arithmetic but has not earned expansion into a native decoder
or automatic SDK selection. These samples establish no tenfold improvement.

Evidence: `docs/evidence/research-client-stage-qwen25-confirm.json`, with its
`.plan.json` and `.frontier.json` companions. Confirmation retains a fresh private
cohort salt; the frontier keeps the initial screen and confirmation distinct.

## Broader workloads and admission

These broader pairs use installed build
`5f9c28f75bc161b33058175a482d5fa7659155b0e5bdb4e0ffeb8b4dde0cd252`.

The separate fresh-prompt Qwen2.5 pair runs two unrelated requests with **46 / 41
input tokens and eight outputs each**, serial Preparation, and a 256-token
compiled bound. Both clients pass runtime checks with matching output commitments,
**300,432,386 covered application-body bytes** and zero observed swap growth.

| Workload / owner | Client RSS | Client CPU | Prep CPU | Prep RSS | TTFT | Decode TPS | Stream MB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Fresh sequence / Python | 164.69 | 16.640 | 11.489 | 149.85 | 12.249 | 3.802 | 302.601 |
| Fresh sequence / native | 163.32 | 16.647 | 12.491 | 152.81 | 12.258 | 3.802 | 302.601 |

The **0.84%** client-RSS difference is one pair, not a confirmed saving. All metric
units and windows match the earlier tables. This sequence has its own composition,
build and private cohort salt; it does not strengthen the rejected short-workload
memory claim by pooling samples.

The initial long/history pairs in the same broader screen fail compiled binding:
the 512-token maximum times the 9,728-column gate/up stage exceeds the existing
4,000,000-value residue-codec ceiling. This check uses the compiled bound, not
actual prompt length or chunk size. Serial Preparation and a host-memory pass
cannot bypass it. The original failures remain in
`docs/evidence/research-client-stage-qwen25-breadth.json`, alongside the completed
fresh-prompt pair and its `.plan.json` / `.frontier.json` companions.

A separate **384-token-bound** cohort admits long prefill and growing histories
under serial Preparation. Long prefill uses **288+8 tokens** for cold and repeat
requests; histories use **46 / 81 / 120 inputs with eight outputs each**. All ten
requests complete with matching within-cell source, numeric, state, workload and
output commitments, matching covered bodies, and zero observed swap growth.

| Workload / owner | Client RSS | Client CPU | Prep CPU | Prep RSS | TTFT | Decode TPS | Stream MB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 288+8 repeat / Python | 259.74 | 28.563 | 21.638 | 261.88 | 40.365 | 3.618 | 805.269 |
| 288+8 repeat / native | 276.35 | 28.489 | 21.525 | 240.09 | 40.336 | 3.709 | 805.269 |
| Growing history / Python | 175.90 | 16.544 | 11.745 | 156.89 | 14.435 | 3.851 | 452.788 |
| Growing history / native | 173.38 | 16.525 | 11.660 | 149.73 | 14.415 | 3.895 | 452.788 |

Long-prefill client RSS **increases 6.40%**; growing-history RSS falls **1.43%** in
its single pair. Their covered bodies are **802,307,858** and **449,457,217 bytes**,
respectively. These mixed exploratory samples add workload coverage, not a
repeatable memory benefit or a reason to extend the native owner. The response
target's message-context repair is exercised through real checkpoint roles here.

Replay this separately bounded cohort:

```sh
python -m benchmarks.research.client_stage_ownership --processes 1 \
  --artifacts target/prepared-frontier-artifacts \
  --workload long --workload growing --input-bound 384 --serial-issuance \
  --output target/client-stage-bounded-breadth.json
```

Evidence: `docs/evidence/research-client-stage-qwen25-bounded-breadth.json`, with
its `.plan.json` and `.frontier.json` companions. The original 512-bound failures
retain their own records.

Qwen3-4B retains separate admission-only evidence. The short 64-token-bound pair
requires **8.14 GiB** of conservatively estimated topology memory against
**5.90–5.99 GiB** of post-reserve headroom. The original long/history 512-bound
cells require **19.98 GiB**, against **5.90–6.01 GiB**. Every cell rejects before
role execution. These estimates are neither measured peaks nor an owner comparison.
Records and their plan/frontier companions:
`docs/evidence/research-client-stage-qwen3-short-admission.json` and
`docs/evidence/research-client-stage-qwen3-breadth-admission.json`.

The historical generated-checkpoint sequence screen is retained in
`docs/evidence/research-client-stage-sequences-tiny.json`. Its fresh sequence
completed; its message-history cells exposed a response-target string-only input
check. The target now reuses the ordinary bounded context validator, with
installed-wheel real-role regressions for both owners. That repair does not
rewrite the earlier unsupported observations.

## Feedback on the benchmark primitives

- **An ownership ablation needs the complete response target.** A stage-only RSS
  result omits the Python runtime, tokenizer, checkpoint binding and retained
  decoder state. The explicit selector changes one checked stage while preserving
  those owners, the topology and the numeric cohort.
- **Implementation choice belongs in case identity, not numeric identity.**
  The owner selection has its own target digest. Comparison binds the same
  semantic role/layer and still checks source, numeric, privacy, state, workload
  and output identities independently.
- **Local attribution cannot replace serving measurements.** Selected-stage CPU
  and elapsed time remain diagnostics alongside all seven complete-response
  axes. Small timing fields in existing frames can change byte counts without
  changing the arithmetic payload; parity tests reconcile that field explicitly.
- **Lifecycle is a first-class workload.** Repeat, unrelated sequence and growing
  message-history cells exercise different reuse conditions. Fixed public
  histories keep later inputs identical across candidates. They do not imply
  coverage of response-owned continuation or model-selected conversation turns.
- **Confirmation needs a new cohort.** The same study constructs fresh
  finalist/control cases through `FrontierStudy.confirm`; screen evidence alone
  does not authorize broader native ownership. Admission failures remain visible
  rather than prompting larger memory or material limits.
- **Allocation and protocol admission are separate gates.** The existing residue
  codec checks every remote stage at the compiled maximum prefill width. A
  512-token Qwen2.5 plan exceeds its 4,000,000-value ceiling even if the actual
  prompt or duplex chunks are smaller. A host-memory pass cannot establish codec
  coverage. Explicitly changing the public workload bound creates a separate case;
  it does not repair or replace the rejected observation.

The fixed research adapter is temporary worker-local instrumentation. A promoted
native decoder should instead have a compiler-checked owner/allocation contract;
this selector is not that contract or a general runtime-plugin interface.
