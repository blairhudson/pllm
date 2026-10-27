# FuseFSS implementation plan

Status: bounded source decoder and ordinary SDK integration implemented and
tested on generated checkpoints. Full trained-model qualification remains open.
Separate PUMA compact-execution work has not cleared its admission gate.

## Contract

Follow [D1, D3, D5, D8, D9, D13, D14 and D15](DECISIONS.md), the
[shared computation contract](shared-computation-contracts.md#4-fusefss) and
[frozen scalar specification](../reproduction-contracts/fusefss/README.md).
The implementation is independent PLLM Rust/Python code.

D13 selects two explicit ring50/Q12 profiles:

- `fusefss.gpt2-complete-ring50-q12.v1`: learned positions, centered LayerNorm,
  affine GELU and Conv1D-oriented public projections.
- `fusefss.dense-rope-ring50-q12.v1`: source-table RMSNorm, gated SiLU and
  split-half public RoPE. Negative rotary coefficients are applied before each
  product floor; floors cannot move across subtraction.

Both preserve nearest-even external encoding, source-defined internal floors,
two online share holders, a separate trusted offline dealer, complete-vocabulary
lowest-index argmax, fixed-count full-prefix feedback and final client-only IDs.
The dense-RoPE profile is a named adaptation. The paper's missing evaluated
descriptor manifest still prevents claiming its exact evaluated numeric identity.

## Gates

| Gate | Current evidence | Remaining work |
| --- | --- | --- |
| F0: source and feasibility | Pinned equations, public model metadata and explicit cost estimates | Reprice complete trained-model execution after each scaling change |
| F1: scalar | Typed descriptors, shared DCF, affine lookup and consuming products; independent frozen GELU checks | Independent security qualification |
| F2: protected gates | Complete SiLU, softmax, LayerNorm/RMSNorm and floor/conversion gates; 275 frozen cases | Full-model domains and efficient complete widths |
| F3: semantic decoder | Five generated GPT-2/Qwen2/Qwen3/Llama controls; 918 complete semantic observations, private feedback and final IDs | Trained-checkpoint integer/domain admission; original 5,125 internal-tensor interface is not claimed |
| F4: ordinary SDK | Immutable components, source import, three actual role children, encrypted one-use tapes, SDK/gateway/CLI, failure/cancellation and standard benchmark paths | Larger admitted workloads, independent deployment and security review |
| F5: model qualification | No passing full trained-model result | Original-model and pinned common-Qwen capacity, exact integer parity, quality and matched resource/performance cohorts |

The complete method remains unqualified until F5 passes. Earlier gates supply
only their stated scopes; generated benchmarks cannot fill a trained-model cell.

## Implemented paths

- Shared absolute-position model lowering lives in
  `crates/pllm-models/src/absolute_position_decoder.rs`; checkpoint names and
  orientations remain source-mapping data.
- `crates/pllm-compiler/src/fss_decoder/` binds public source, numeric profile,
  semantic operations, fixed-count execution, resource bounds and one-use roles.
- `crates/pllm-garble/src/fusefss_backend/` supplies the common typed gates,
  exact kernels, compact private one-hot lookup and bounded material codecs.
  Immutable source-table sharing removes repeated table construction; fresh
  masks, keys and request state are never cached.
- `pllm_core::{public_file,fixed_file,streamed_matrix,integer_bounds}` supplies
  shared immutable storage, streamed public contractions and integer certificates.
  PUMA and FuseFSS retain their separate source encodings, rounding and proofs.
  Shared tokenizer validation and process-local resource reservations remain
  outside method-specific arithmetic.
- `FuseFssSourceDecoder`, `TrustedDealerTwoShareRoles` and
  `FixedCountPrivatePrefix` compose through `FuseFssSourceCpu`.
- `python/pllm/runtime/fss_*` owns bounded source import, authenticated channels,
  one-use encrypted preprocessing tapes, offline-dealer retirement and draining
  cancellation. Local children establish role separation, not non-collusion.

## Reproduce

Build/install with the standard uv/Maturin commands in
[`CONTRIBUTING.md`](../../CONTRIBUTING.md). Use ordinary tests for correctness:

```sh
cargo test --release --locked --offline -p pllm-compiler --lib fss_decoder:: -- --test-threads=1
uv run --no-sync python -m pytest -o pythonpath=. tests/test_fss_sdk.py tests/test_fss_native_transport.py
```

[`examples/papers/fusefss_smoke.py`](../../examples/papers/fusefss_smoke.py)
constructs reproducible generated sources for both profiles. Its ordinary CLI
outputs are in [`paper/artifacts/fusefss-sdk/`](../artifacts/fusefss-sdk/README.md).
Use [`examples/papers/fusefss.py`](../../examples/papers/fusefss.py) for the pinned
common-Qwen configuration; selection does not waive its admission failures.

Benchmark reports retain dealer and holder costs, directed bodies, separate tape
I/O and setup-inclusive final disclosure. Buffered output supplies neither streaming
TTFT nor model decode throughput. No receipt archive or copied build/source tree
is required.

## Current blocking slice: joint exact signed-SiLU FSS specification only

Source-derived specification and independent numerical review are consistent:
a **SPEC STUB**, not a construction, security proof, implemented layout or achieved
cost. Only specification and independent review are authorized; no crypto implementation, keys, trained
inference, cap or sequence increase, or adoption is approved.
Corrected ordinary ChatML workload is **30 prefill + 2 output tokens**: it still
requires prefixes **30 and 31**, both over source cap 16, and remains over the
unchanged full-cost material budget. No silent sequence-32 change or cap rise.
All 1+1, 3+1 and 4+1 prompt/admission results below are **HISTORICAL**, unchanged;
`user: hi` = 3 there is not current ordinary corrected-30 ChatML fit or proof.

### Frozen numeric and composition contract (D14)

- Interpret canonical ring50 input `X` as signed Q12; accepted interval is
  `[-2^37, 2^37-1]`. Set `Y = floor(X/64)`; sign-extend before `abs(Y)`,
  especially at `Y = -2^31`. Define `D = [Y >= 0]`, `A = abs(Y)`,
  `B = [A < 1024]`, `J = min(A, 1023)`, `Z = D*X - T[J] mod 2^50`.
  Multiplicand is **original X**, never Y. Preserve negative-floor corners
  `(-65, -64, -1, 0, 63, 64)` and clamp changes at X `-65473/-65472`
  and `65535/65536`, although adjacent A=1023/1024 select the same index.
- Existing owners: [composition/gates.rs](../../crates/pllm-garble/src/fusefss_backend/composition/gates.rs),
  [composition/source_tables.rs](../../crates/pllm-garble/src/fusefss_backend/composition/source_tables.rs),
  [fixed_activation_tables.rs](../../crates/pllm-garble/src/fixed_activation_tables.rs).
  `fixed_activation_tables::silu_q12` is the rational-certified frozen oracle;
  no floating regeneration or new table. Reuse existing Descriptor/Plan/Party,
  tape/import, conversions, predicates and lookup owners, not parallel machinery.
- Public declared interval `InputRange` is **not secret membership proof**. Existing
  certificate or protected check must precede unsafe narrowing/arithmetic. D14 requires safe
  substitution, sticky validity and full schedule; client-only aggregate acceptance,
  no invalid tokens, reusable state or failing-location disclosure. Optional helpers
  `n,b` require fixed public descriptor and internal shares, never adaptive or clear disclosure.
- Shared masked opening already exists but queries fresh keys. `VectorComparison`
  is one-threshold/multi-payload, **not multipoint compression**. No compact key
  layout is asserted and no speculative compression is claimed.

### Construction, security and lifecycle obligations (missing)

- Supply concrete generation/evaluation equations and key format, plus joint
  simulator for one corrupted holder's keys, input/output shares, openings,
  prior/downstream conversions, and access/failure metadata. Simulator uses only
  prescribed holder input/output shares and public descriptor/leakage consistent
  with D14, not actual secret helper values or client-only aggregate acceptance.
- Trust: independent trusted dealer; at most one semi-honest online holder.
  Context binding is not authenticated transport. Output-ring masks must be
  full-ring uniform; joint masking of Booleans must be proved, not presumed.
- Fresh material per invocation/context/lane/epoch. Begun instances burn on
  failure/cancellation/replay; public preadmission without material invents no burn.
  Secret-invalid inputs still execute full schedule before aggregate acceptance.
- Independent numerical review found the target consistent. **NEW CONSTRUCTION
  SECURITY REVIEW NOT PERFORMABLE**: construction/layout/simulator absent; security has not passed.
  The reusable [public integer experiment](../../examples/papers/fusefss_silu_integer_campaign.py)
  and [ordinary regression tests](../reproduction-contracts/test_fusefss_silu_integer_campaign.py)
  cover finite floor/clamp/endpoints, frozen-table equality and public models of
  invalid safe substitution, sticky validity and full-schedule rejection.
  Neither was run in this source-only finishing pass; no new passing result is claimed.
  Public arithmetic/control checks are not protected membership or lifecycle evidence.
  The protected lifecycle/security campaign remains **UNRUN**: secret-invalid
  safe substitution and full-schedule rejection, failure/cancellation/replay burn,
  fresh-request isolation and joint-transcript security.

### Conditional cost only; no achieved bound

- Proposed projection **25,601,495,040 B/holder** is **UNIMPLEMENTED/UNAPPROVED**,
  not verified achieved bound. It leaves **8,758,243,328 B** of 32 GiB for
  **7,120,896 SiLU lanes plus all other operations**; **1229.936 B/lane** is only
  optimistic ceiling, not achieved gate cost or lower bound.
- Require complete ledger: projection/embedding/norm/attention/FFN products/RoPE,
  head/selection/checks/output masks, tape/frame/envelope, work and live owners.
  Retirement never refunds cumulative one-use bytes. Preserve **32 GiB material
  per holder**, existing **64 MiB native-resident budget**, separate host RSS guards,
  descriptors and work caps unchanged. No cost/admission substitution between them.

## Next blocking slice (historical record; not corrected-30 ChatML evidence)

Prioritize pinned Qwen benchmarking. Exact-key indexed replay owners and
framing-derived tape admission now support the larger generated Qwen SDK control,
including file-backed Decoder execution. Native tests preserve all 186 independent
Qwen observations; three-role tests preserve three selections and failure cleanup.

The static profile's earlier 1+1 checkpoint import rejects numeric admission
before workers launch. An offline Gram/Hadamard-lift diagnostic certifies the
layer-zero residual, but has neither full-layer coverage nor a bounded production
owner. Later conditional propagation does not certify earlier obligations.

The approved [D14](DECISIONS.md#d14-opt-in-checked-ranges-for-fusefss-qwen-completion)
profile now passes actual pinned-Qwen source, numeric and work admission at 4+1.
It preserves D13 arithmetic on accepted requests and rejects invalid requests only
at final aggregate acceptance. Shared private predicates/selection and their full
schedule are charged. Generated three-role tests cover late rejection without
token disclosure and fresh-request recovery. Actual checkpoint request acceptance
is still unmeasured; static proof improvements remain a separate follow-up.

Next: calibrate and reduce online execution overhead, complete the ordinary short
Qwen request, check independent complete integer results, then quality and matched
benchmarks.
The initial inline-metadata tape quote needs 71.53 GB including reserve; the host
rejects before child launch. External-AAD tape storage reduces the quote to
47.63 GB without removing authenticated metadata, material or replay claims.
Generated Decoder tests cover this format through actual role processes.
A separate small serial-HTTP benchmark attributes 97.82% of dealer preparation
to delivery callbacks. Explicit bounded delivery batching now preserves original
one-use records, complete acknowledgements and numeric identity through ordinary
SDK execution. A fresh-client generated pair lowers preparation from 5.393 to
0.390 seconds with matching outputs, tape bytes and online bodies. This remains
separate from trained-Qwen execution and independent performance confirmation.
A subsequent checked-range/batched Qwen attempt passes topology admission and
reaches online execution, but the host memory-pressure guard stops it after about
66 minutes, before final acceptance and before its four-hour deadline. The
[ordinary failed checkpoint](../artifacts/fusefss-sdk/qwen-checked-batch-pressure.json)
retains no completed report. All role processes retire. The cause of host pressure
is unassigned. Scalar online gates still incur millions of Python/HTTP callbacks;
measure a representative bounded fixture and reuse shared transport/scheduling
owners for dependency-safe batching before another long checkpoint run.
Launch uses bounded temporary source qualification, fresh exact
topology pricing and independently checked worker imports. Qualification cost,
all source/tape/replay disk, cumulative material and execution work stay charged.
The common 64+8 workload still exceeds sequence/selection bounds. Preserve the
approved arithmetic and role graph; obtain approval before substantial protocol,
numeric, state or trust adaptations.

### WAN requirement

Loopback completion does not close F5. Use the ordinary three-role benchmark
under shared 100/40 Mbps party caps and a declared inter-party delay, starting
with the 20-ms-per-direction profile in `examples/benchmarks/wan_20ms.json`.
The shared native relay now routes dealer delivery and both holder-peer paths;
its bounded propagation queues and report validation are transport qualification.

The scalar schedule has 13,424,643 peer frames per Qwen holder. Merely replacing
HTTP or batching offline delivery does not remove those dependent waits. D15's
explicit `online_schedule="wavefront-v1"` now groups independent tensor lanes and
uses balanced stable selection. Every lane still consumes fresh, context-bound
material and exact replay claims; frontier storage is priced before issuance
under the existing live limits. Public group and tile bounds still split logical
tree levels into multiple exchanges.

The core passes 105 composition checks; 115 decoder checks include all five
frozen semantic outputs and the checked-profile control. Four installed
three-process checks cover generated Decoder feedback, late rejection/recovery,
Capsule execution and buffered gateway output. A fresh-client matched generated
pair at 100/40 Mbps plus 40 ms added RTT lowers full requests from 222.505 to
34.583 seconds, with identical output commitments and zero observed swap growth.
The [ordinary report](../../docs/evidence/fusefss-wavefront-wan-pair.json) retains
that tiny source and separate cohort.

Pinned-source 4+1 wavefront qualification still rejects: repeated public schedule
construction pushes the kernel walk past the existing 32 GiB public-copy ceiling
at layer 23 SiLU. A separate 1+1 control admits but retains 29,334 sequential peer
exchanges per holder, roughly 19.6 minutes of waiting at 40 ms RTT before compute.
Its packet allowance is 5.18 GB per holder, an upper-bound price rather than a
measured transfer. No private request runs in either qualification control.
That result motivated compact public schedule ownership and wider bounded tensor
waves; increasing capacity ceilings is not the solution.

The subsequent explicit `online_schedule="tensor-v1"` shares instruction,
register-description and dependency-span templates while retaining separate
private registers and fresh material. Its compiler-owned ordered tree removes
redundant private index comparisons without changing lowest-original-index ties.
The composition and decoder suites pass 110 and 118 checks; the installed wheel
passes 408 focused checks plus four three-process SDK/gateway checks. The
[new generated WAN pair](../../docs/evidence/fusefss-tensor-wan-pair.json) observes
35.749/34.732-second wavefront/tensor requests, equal output commitments and zero
swap growth under the same caps and delay. Its separate cohort is not a trained
performance result.

Historical pinned tensor 4+1 qualification rejected at the output head, with
35,178,175,330 public-copy bytes against the unchanged 34,359,738,368-byte ceiling.
The separate 1+1 control admits 19,229 peer exchanges per holder and 7.45 GB of
complete native material. About 12.8 minutes of RTT waiting at 40 ms remains
before compute; packet allowances still reach 4.17 GB per holder before HTTP
metadata. No workers launch in these source-only controls. More dependent-round
and material reduction remains necessary before another long trained request.

With shared immutable public chunks, fresh installed-wheel source-admission tests
now pass at both 1+1 and 4+1 under the unchanged ceilings. The earlier cumulative
copy rejection is no longer the current source gate. These tests perform the
complete native numeric/resource walk but create no tapes or workers. A separate
opt-in test now prices native transport from the live qualification owner, retires
it, and requires fresh whole-topology memory/disk admission and reservations.
That gate, followed by an accepted private SDK request, remains distinct from
source admission. Shared tables do not establish fewer dependent rounds or fresh
private bytes; those still require repricing and measurement.

The separate complete-topology admission passes at 1+1, including ordinary
logical memory/disk reservations after source retirement. Before the one-hot
staging fold, the 4+1 case rejects during native transport pricing, before final
host/disk admission. A rebuilt-wheel diagnostic retry identifies
`aggregate_reservation`: 68,913,145
bytes against the unchanged 67,108,864-byte cap, exceeding it by 1,804,281 bytes.
The SDK reference records the complete public owner terms. Reduce actual
ownership/copy overlap rather than discounts or ceilings. No material, workers
or private requests run in either case. Trained serving and quality remain
unqualified.

The one-hot staging fold (`607813fd`) removes actual full-domain backing while
preserving individual-party output bytes and source work. Fresh installed-wheel
retries pass the 1+1 complete-topology control and clear 4+1 native transport
admission at 66,784,028 bytes, 324,836 bytes below the unchanged cap. The complete
pricing walk reduces each workload's combined reservation by 2,129,117 bytes;
per-party allocation savings alone did not establish that result.

The 4+1 retry now reaches final host/disk admission, where disk rejects:
45,517,613,936 required bytes, including the unchanged 2 GiB reserve, exceed
41,405,534,208 available bytes. Host memory admits at 4,068,474,880 required bytes
with normal pressure. Do not remove live owners, weaken reservations or raise
caps to force acceptance. Complete-topology admission must pass before worker
launch; no material, workers or private request ran in this retry. Logs remain
ordinary pytest output under ignored `target/fusefss-one-hot-fold-topology-{1,4}.log`.

The admitted 1+1 control does not fit ordinary SDK rendering of short prompts:
the pinned tokenizer encodes `user: hi` as three tokens. Preserve the Responses
role-prefix contract; raw-token or prompt-format bypass is not SDK acceptance.

The separate `qwen_checked_tensor_minimal_smoke` 3+1 factory fits that rendering.
Its guarded installed-wheel complete-topology check passes without changing
guards: native combined reservation 64,155,060 bytes, required host memory
4,068,474,880 bytes with the normal 25% margin, and required disk
37,476,300,632 bytes including the 2 GiB reserve. Qualification owners retire
before graph reservation. No workers, tapes, material or private request run
in this admission test (`target/fusefss-one-hot-fold-topology-3.log`). Actual
ordinary SDK execution remains a separate acceptance gate; 4+1 remains
disk-blocked on this host.

The subsequent ordinary 3+1 SDK benchmark stopped with exit 6 at
`macOS memory pressure is warning`. Its normal progress report marks the
candidate failed, with no completed result (`target/fusefss-qwen-3-plus-1-sdk.json`);
CLI output is the adjacent `.log`. Do not count the attempt as private-request
acceptance, quality or measured throughput. Keep pressure guards enabled and
resample live capacity before retrying.

Round reduction alone does not solve cold-request bytes. Reprice and measure the
complete fresh-material delivery, source/setup scope, online traffic and aggregate
resources before another long trained run. Require generated integer/feedback
parity and failure retirement first, then a bounded delay-injected response, then
pinned-Qwen acceptance, quality and matched WAN measurements. Keep warm/offline
pre-positioning separate from a setup-inclusive result.

See the [SDK contract](../../docs/research/fusefss-sdk-reference.md) and the
[historical scalar cost inventory](fusefss-decoder-feasibility.md) for separate
current and earlier scopes.
