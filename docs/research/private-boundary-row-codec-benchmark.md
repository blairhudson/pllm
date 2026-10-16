# Matched private-boundary head row codec

**Separate pinned Qwen 150+8 and 150+64 initial screens pass exact-state and
accounting gates.** Each uses six fresh clients and observes zero swap growth.
Row replies reduce head payload 8.31%; complete TCP traffic falls 0.17%/0.91%
and median decode throughput rises 2.41%/0.87% respectively. Full requests are
1.18% slower at eight outputs and effectively unchanged at 64; aggregate CPU
rises in both. These are incremental traffic/decode tradeoffs, not a new overall
SOTA result. Receipt-backed replay preserves each separate cohort. Results below
retain their original scopes and independent-confirmation gates.

The separate generated-source SDK smoke preserves 83,352 checked logit/KV values, complete state/output
commitments and zero observed swap growth. Actual row-codec frames, certificate
commitments and directed bodies reconcile, and receipt-backed replay reproduces
the saved summary exactly. This tiny workload establishes functionality, not a
serving-speed, cold-network or client-memory-tier result.

Original attempts, accepted v6 report/plan/receipt/summary, publication inputs,
build records and trusted verifier snapshots are retained in
[`private-boundary-row-codec-tiny-v1`](../evidence/private-boundary-row-codec-tiny-v1/).
Earlier attempts remain rejected. The archive's `index.json` records original
paths, exact SHA-256 values and external dependencies; copied snapshots are inert
provenance rather than automatically executed replay code.

The first pinned Qwen publication attempt rejected host admission before tensor
loading. A separate post-rejection diagnostic priced 6,690,704,256 bytes (6.23 GiB)
against 6,444,630,016 available post-reserve bytes (6.00 GiB), retaining the 4 GiB
host reserve. The exact admission-time snapshot was not recorded. That attempt
produced no Qwen certificate or response cohort.

After host memory was freed, a fresh admitted Qwen publication completed with zero
observed swap growth. Its 11,364-byte certificate matches the independent full-row
scan: 1/11,794/139,982/159 rows use 21/22/23/24 bits respectively. Each worker's
packed head payload is 435,362 bytes. Separate native pins bind the frozen 150+8
and 150+64 recipes; both metadata preflights pass. Publication itself costs
3.880755 process CPU seconds; publication plus independent complete source binding
costs 22.044972 CPU seconds in the recorded partial worker window. The worker's
1,045,004,288-byte lifetime RSS is an offline-publisher peak, not client RSS.
These costs do not measure artifact distribution or serving performance.

## Pinned Qwen 150+8 initial screen

Original report, plan, completion receipt, summary, publication attempts and
trusted source snapshots are retained under
[`private-boundary-row-codec-qwen25-v1`](../evidence/private-boundary-row-codec-qwen25-v1/).
Three fresh clients per arm use the recorded AB/BA/BA order, one private cohort,
the same installed build and shared native 100/40 Mbps access caps. Every request
reaches eight outputs; seven executed decode steps supply the N−1 rate. This
screen does not exercise an early-EOS comparison.

All values below are medians. Change is candidate minus control, relative to
control; higher decode throughput is preferable, lower other values preferable.
MiB means 2^20 bytes and MB means 10^6 bytes.

| Metric | Parallel uniform residues | Parallel certified row residues | Change |
| --- | ---: | ---: | ---: |
| Client peak RSS (MiB) | 218.781 | 213.750 | -2.30% |
| Client cold CPU (s) | 20.023 | 21.117 | +5.47% |
| Preparation cold CPU (s) | 15.046 | 15.771 | +4.82% |
| Preparation peak RSS (MiB) | 205.828 | 204.219 | -0.78% |
| Online TTFT (s) | 46.839 | 47.200 | +0.77% |
| N−1 decode (tokens/s) | 2.726 | 2.792 | +2.41% |
| All-link TCP streams (MB) | 365.221 | 364.590 | -0.17% |
| Covered application bodies (MB) | 363.670 | 363.039 | -0.17% |
| Full request (s) | 82.172 | 83.145 | +1.18% |
| Startup through cleanup (s) | 109.390 | 113.924 | +4.15% |
| Five-owner cold CPU (s) | 63.178 | 67.360 | +6.62% |

The complete client worker includes SDK, supervisor, dashboard, relay and state
audit. Its **largest** replicate peak is 218.95 MiB for control and 221.66 MiB for
candidate, within observed Low/Medium limits. Candidate median RSS is lower, but
one paired peak regresses and its maximum is higher: no causal memory improvement
is established. Conservative runtime admission retains its separate estimates.

Each worker's head payload falls from 474,800 to 435,362 bytes. Across two workers
and eight selections, exact head savings are **631,008 bytes**, entirely payload;
74-byte subframe headers and outer envelopes are unchanged. Paired total body/TCP
savings are 631,007 / 631,007 / 631,009 bytes. The corresponding +1 / +1 / -1
lifecycle-control deltas reconcile completely, with no unexplained delta.
Ordinary prepared arithmetic, material issuance and full state remain identical.

The 11,364-byte approved certificate is pre-positioned at client and both boundary
workers. Declaring all three missing adds a separate **34,092-byte setup budget**;
it is not measured network traffic. Separate publisher costs above remain charged
outside response counters. Checkpoint/tokenizer distribution, physical wire,
Internet delay and operator independence remain outside this source-present
screen. All three paired decode changes are positive, but independent confirmation
is still required; the longer workload retains a separate cohort and gate.

The first 150+64 attempt completed one control and rejected the other five cells
before provider launch: safe headroom was 5.04–5.50 GiB against the unchanged
6.03–6.04 GiB topology estimates. The raw attempt has no receipt or accepted
summary and supplies no paired performance or tier result. Its original files
remain under `150x64-rejected-v1/`; a fresh retry does not reuse that control.

## Pinned Qwen 150+64 initial screen

After memory was freed again, a fresh six-client retry completed under the same
guards and frozen recipes. All requests reach 64 outputs, with 63 executed decode
steps each. It preserves **486,555,648** checked logit/KV values and complete
state, position, selection, RNG and output commitments. All six guards observe
zero swap growth. Its private cohort is independent of the short screen and the
rejected long attempt; none supplies another's replicates or confirmation.

| Metric | Parallel uniform residues | Parallel certified row residues | Change |
| --- | ---: | ---: | ---: |
| Client peak RSS (MiB) | 219.500 | 217.484 | -0.92% |
| Client cold CPU (s) | 26.726 | 27.052 | +1.22% |
| Preparation cold CPU (s) | 17.500 | 18.143 | +3.68% |
| Preparation peak RSS (MiB) | 243.406 | 247.641 | +1.74% |
| Online TTFT (s) | 47.044 | 47.126 | +0.17% |
| N−1 decode (tokens/s) | 2.466 | 2.488 | +0.87% |
| All-link TCP streams (MB) | 553.614 | 548.566 | -0.91% |
| Covered application bodies (MB) | 548.441 | 543.392 | -0.92% |
| Full request (s) | 120.221 | 120.107 | -0.10% |
| Startup through cleanup (s) | 147.505 | 148.939 | +0.97% |
| Five-owner cold CPU (s) | 84.378 | 86.790 | +2.86% |

Largest complete-client peaks are 220.11 MiB for control and **217.83 MiB** for
candidate, within observed Low/Medium ceilings. Paired RSS changes still have
mixed signs, including a 13.48 MiB candidate regression in one pair; the lower
median does not establish a causal memory benefit. All three paired decode
changes are positive, but this is an initial screen, not independent confirmation.

Exact head payload saving is **5,048,064 bytes/request**, with unchanged headers,
envelopes, prepared arithmetic and lifecycle body counts. It accounts for every
application-body delta; no unattributed body delta remains. TCP savings are
5,048,055 / 5,048,064 / 5,048,055 bytes: transport framing retains its separate
nine-byte differences in two pairs. The same 34,092-byte declared certificate
setup budget and separately measured publisher costs remain outside response
counters. The source-present, pre-positioned-artifact and local-rate-emulation
limitations of the short screen apply independently here.

The archive's `150x64/` directory retains original v2 raw report, plan, receipt,
summary, progress and log. An independent read-only raw audit confirms source and
receipt bindings, complete state, resource medians and all byte partitions. The
two nine-byte TCP differences localize to Inference-to-Client framing. Its
accepted summary replays exactly:

```sh
target/frontier-sdk-env/bin/python scripts/summarize_private_boundary_row_codec.py \
  docs/evidence/private-boundary-row-codec-qwen25-v1/150x64/certified-head-qwen150x64-v2.json \
  --plan docs/evidence/private-boundary-row-codec-qwen25-v1/150x64/certified-head-qwen150x64-v2.plan.json \
  --receipt docs/evidence/private-boundary-row-codec-qwen25-v1/150x64/certified-head-qwen150x64-v2.receipt.json
```

## Explicit inputs

`scripts/benchmark_private_boundary_row_codec.py` requires two frozen **Experiment
JSON files**, not a recipe name. Control uses parallel dispatch and
`head_output_encoding="uniform_residues"`; candidate uses `"row_residues"`.
Everything else must be structurally identical: source/revision, deployment,
quantizer, nonlinear/linear body, cache, inventory, delivery, tokenizer, topology,
device, kernel, budget and experiment name. No prompt, stop policy or component
gets silently changed. Budget must already be 150+8 or 150+64, requests=1.
There is no command which silently retargets an 8-token experiment to 64.

The installed SDK exposes candidate boundary parameter
`head_certificate={"path": ABSOLUTE_PATH, "digest": SHA256, "size_bytes": INTEGER}`.
Only this additional field and `head_output_encoding` differ. Driver observes
that pin shape rather than generating/importing a certificate API. Integration
checks the SDK contract and this exact structural allowlist together.
Additional certificate parameters cause rejection, not permissive normalization.
The archive contains the generated-source certificate and two-arm experiments;
they cannot supply a real-checkpoint or different-workload pin.

`--certificate-accounting` requires a public JSON declaration:

```json
{
  "missing_owners": ["client", "boundary_a", "boundary_b"],
  "delivery_locations": {
    "client": "/absolute/local/certificate.bin",
    "boundary_a": "/absolute/local/certificate.bin",
    "boundary_b": "/absolute/local/certificate.bin"
  },
  "derivation": "Identify independent offline publisher, source locks, implementation and separate publication evidence here"
}
```

Locations are explicit operator declarations, not inferred proof of delivery.
Missing-owner list is a declared cache/setup scenario, never an observed remote
cache probe. Artifact size and SHA256 are read and verified against approved pin.
Even for a shared local path, fixed setup budget is artifact size times number
of declared missing logical owners, charged once each, **not once per selection**.
Local reads are not network measurements. Extra budget remains separate from
observed SDK TCP/body bytes: no double counting or fake measured values.
Checkpoint/tokenizer prepositioning is distinct from newly published certificate.

SDK response execution does not measure offline publication CPU or publisher
distribution. These remain `null`, with fixed byte budget explicitly labelled
scenario arithmetic. Full cold-network and SOTA claims are always forbidden.
An empty missing-owner list means already present certificate, not free global
distribution. Publication/source import costs require separate actual evidence.

## Invocation and replay

After freezing the admitted SDK/native build and source certificate, run each
workload in a **different invocation/output path**, with matching frozen budgets:

```sh
.venv/bin/python scripts/benchmark_private_boundary_row_codec.py \
  --control target/row-control-150x8.json \
  --candidate target/row-candidate-150x8.json \
  --certificate-accounting target/row-certificate-accounting.json \
  --native-head-commitments target/row-native-head-commitments.json \
  --prompt target/prepared-frontier-artifacts/prompt.txt \
  --max-output-tokens 8 --output target/row-codec-150x8.json --preflight-only
```

Use a **new output path** without `--preflight-only` for actual execution. Use
separate frozen 150+64 files, `--max-output-tokens 64`, and another fresh output
for long workload. Nothing pools or resumes independent invocations/private salts.
Token count is checked separately using ordinary SDK rendering/tokenizer and must
equal 150. Natural EOS remains unchanged; report actual emitted tokens, final EOS
selection, executed decode steps, material issued/claimed/burned/discarded.
Zero-output SDK telemetry is explicitly rejected as unsupported, not fabricated.

```sh
.venv/bin/python scripts/summarize_private_boundary_row_codec.py \
  target/row-codec-150x8.json --plan target/row-codec-150x8.plan.json \
  --receipt target/row-codec-150x8.receipt.json
```

Summary prints to stdout only. It revalidates raw run and frozen plan, never
loads providers/tensors, reads current checkpoint files, rewrites evidence or
merges cohorts. Missing extension/execution failure cannot produce a summary.

The accepted generated-source archive was replayed with:

```sh
target/frontier-sdk-env/bin/python scripts/summarize_private_boundary_row_codec.py \
  docs/evidence/private-boundary-row-codec-tiny-v1/attempts/certified-head-tiny-cohort-v6.json \
  --plan docs/evidence/private-boundary-row-codec-tiny-v1/attempts/certified-head-tiny-cohort-v6.plan.json \
  --receipt docs/evidence/private-boundary-row-codec-tiny-v1/attempts/certified-head-tiny-cohort-v6.receipt.json
```

This command needs the trusted verifier/SDK bytes pinned in that plan. Full native
dependencies and source caches remain external, explicitly hash-bound inputs.

Replay additionally requires a new terminal `.receipt.json`. Driver issues it
only after the persisted raw report matches the SDK return value, post-run frozen
source/input checks pass, and complete replay audit passes. Receipt binds cohort,
canonical JSON digests of the **entire plan and raw report**, and a separate digest
of the **entire public-prompt record** (path, SHA256 and size). An in-range size
change is therefore rejected too. Receipt schema/status/post-run-check fields are
strict; missing receipts cannot be manufactured from historical complete raw runs
by this driver. Existing receipt paths reserve an output, like other companions.

Interruption before issuance, failed post-run checks or audit rejection leaves no
receipt/summary, even if SDK already checkpointed six complete raw workers. Raw
attempts remain untouched. Receipt certifies the trusted driver's recorded checks,
not cryptographic authentication: coordinated forgery of plan/report/receipt can
recompute public hashes. JSON formatting is not part of canonical content digests.

Replay requires the **current trusted verifier checkout** to match the frozen
`verifier_sources` inventory exactly. This includes the driver, summarizer,
repository-local transitive helper imports and package initializers; installed
publisher, codec, transport, protocol, loader/model binding and response-audit/
target Python sources are additionally frozen under `sdk/` keys and checked
against each raw `BuildRef` SDK source manifest. Remaining SDK/native/dependency
provenance remains separately committed by `BuildRef`.
Missing/extra verifier entries, malformed SHA256 values, oversized manifests and
changed current verifier bytes reject replay. The verifier inventory must also
agree with `frozen_sources`; both manifests survive in the summary. Historical
plans lacking this inventory are not upgraded or silently accepted. To replay an
older verifier, independently inspect and select its trusted matching checkout;
this tool never imports or executes source paths supplied by an evidence file.

Before hashing or admission, each Experiment JSON is bounded to 1 MiB, delivery
declaration to 64 KiB and prompt to 4096 bytes. Certificate reads are bounded to
approved size plus one sentinel byte, with a 1 MiB ceiling. Other file hashing
streams bytes rather than allocating whole files. Replay CLI bounds plan/run
reads to 8/256 MiB respectively. Workload digests are rebuilt independently from
authoritative raw tokens, scenario, backend, WAN identity and sampling; matching
duplicated digest strings alone are insufficient.

### Versioned public/private workload binding (new state-audit measurements)

The six-client tiny attempt `target/certified-head-tiny-cohort-v2.json` completed
execution but failed the old auditor's `row_codec_raw_workload_digest_mismatch`.
The target produced a public workload digest, then the worker HMACed it together
with private inputs. Comparing that final opaque HMAC to the pre-HMAC public
digest was invalid. Historical raw reports/progress are **not rewritten**; that
attempt remains rejected. The repaired SDK was rebuilt before fresh v3 and later
attempts; the eventual accepted v6 cohort has its own source/build commitments.

The subsequent tiny v3 run passed workload binding but rejected on
`row_codec_head_layout_mismatch`: the actual authenticated SDK importer additionally
returns `artifact_digest` and `artifact_size_bytes` in certificate metadata. The
auditor now independently reconstructs these two fields from the approved pin;
it does not discard unexpected metadata. Actual SDK codec/importer regression
checks cover both layouts, missing fields and altered pins. The original v3
attempt remains rejected; subsequent invocations retain separate frozen identities.

Tiny v4 then passed the complete SDK/raw accounting audit but exposed the same
hash-domain mistake at the outer plan check: the plan's public prompt-file SHA256
cannot equal the SDK's privately salted prompt commitment. Fresh plans now record
the bounded public file path, size and SHA256 against their frozen input manifest,
plus independently rendered input-token counts. Replay validates those declarations
and keeps the opaque cohort commitment separate. The frozen driver checks the file
before and after execution and supplies its actual text to the ordinary private
input channel. This is trusted-execution provenance, not public verification of a
private hash preimage. Original v4 remains rejected; no evidence is rewritten.

Tiny v5 completed six workers but rejected `row_codec_planned_input_count_mismatch`:
predictor counted 105 input tokens while actual requests counted 104. The public
prompt file ends in a newline. Ordinary SDK dashboard `/api/run` strips string
prompts before validating and rendering them; the predictor previously rendered
the raw newline. Prediction now applies that same `strip()` plus shared SDK
`validate_context`, then uses the existing shared SDK renderer/tokenizer. Exact
predicted/observed count equality remains mandatory. Raw file bytes, SHA256 and
size remain frozen **without trimming**; those original bytes still enter private
benchmark stdin and salted input identity. No rewritten fixture or historical
v1–v5 evidence is accepted. The new v6 invocation and receipt pass this gate.

SDK currently inlines string stripping in its dashboard route rather than exposing
a shared preparation helper. This scoped repair mirrors that operation locally,
freezes dashboard/rendering source owners in verifier inventory, and tests the
actual SDK route against prediction. A future SDK extraction of shared benchmark
prompt preparation would remove this duplication; no global SDK/helper edit is
needed for this repair. Regression invokes route without lifespan/providers and
real tiny byte tokenizer/renderer, including generated-source template's exact
105-to-104 newline case, CRLF, whitespace and no-newline variants.

For newly completed response targets with `state_audit=True`, worker details now
include a strict `workload_binding` object:

```text
schema = "pllm.response_state_audit.workload_binding.v1"
public_workload_digest = target's original digest(tokens, scenario, backend, WAN, sampling)
private_inputs_commitment = HMAC-SHA256(private cohort salt,
    canonical({"schema": "pllm.benchmark.private_inputs.v1", "inputs": private stdin inputs}))
prompt_commitment = existing SHA256("pllm.benchmark.prompt.v1\0" || salt || context_bytes(prompt))
fixture_digest = checkpoint/source-lock fixture digest
final workload_digest = SHA256(canonical(workload_binding))
```

The raw target's public digest is retained rather than replaced by an unreplayable
mixed private HMAC. No private input, salt, or unsalted private-input digest is
published. The same salted input commitment must appear in all six clients.
Changing private inputs or cohort salt changes this commitment even if public
token counts stay identical. JSON canonicalization and input-domain version are
explicit; this is not concatenation of ambiguous private strings.

`fixture.digest` is **source identity**, not the private prompt hash. The latter
is `report.configuration.prompt_digest`. Before publishing, actual `Measurement`
independently recomputes that existing prompt hash from private stdin inputs and
salt, and checks fixture versus raw source lock. The auditor checks binding
cross-references to both retained hashes; it cannot verify their secret preimages.

`workload_measurement` separately records schema
`pllm.response_state_audit.workload_measurement.v1`, the workload-binding schema,
and the target's original measurement digest. Final `measurement_digest` is its
canonical digest. The auditor independently rebuilds that original identity from
raw report schema, scenario, cache convention, body-counter coverage and state-
audit version, then checks both the wrapper and final digest. New and legacy
measurement identities are not silently treated as equivalent.

The auditor does not call the worker's binding builder/finalizer. It recomputes
public workload from authoritative raw token usage, target scenario, raw backend,
WAN identity and configuration sampling (also checked against request sampling).
It rejects missing/extra/version-changed witnesses, public or final hash mismatch,
prompt/source cross-reference mismatch, measurement-version mismatch and mixed
private commitments, including individually resealed changed cells. New SDK
worker/helper/contract/context/coverage sources join the frozen verifier inventory
and must match each raw `BuildRef`.

These are **public consistency checks, not cryptographic authentication,
attestation or proof that a claimed private input was executed**. A public reader
cannot recompute an unrevealed salt/input HMAC. A coordinated forged report can
forge its own hashes; the normal trusted build/execution/evidence assumptions
remain. Opaque commitments establish equality of recorded values within the
cohort, not independently authenticated private preimages.

Ordinary non-state-audit targets retain the exact old final HMAC over
`{"target_workload": public_digest, "inputs": inputs}`. Failed state-audit targets
do not invent a completed witness. Worker wire protocol and typed result schemas
remain compatible; old typed reports remain readable, but the strict new row-codec
auditor rejects reports without the new witness. No salt recovery, fabricated
witness, repaired historical digest or bypass flag is offered.

Bounded tests (no model/provider/benchmark execution or install/build):

```sh
target/frontier-sdk-env/bin/python -m pytest -q -o pythonpath=. --noconftest tests/test_benchmark_workload_binding.py tests/test_private_boundary_row_codec_benchmark.py
target/frontier-sdk-env/bin/python -m ruff check python/pllm/_benchmark_worker.py python/pllm/_benchmark_workload.py benchmarks/research/private_boundary_row_codec.py tests/test_benchmark_workload_binding.py tests/test_private_boundary_row_codec_benchmark.py
```

Tests load owned uninstalled worker/helper sources through an explicit test-only
module overlay, not a wheel build or installed-file edit. Synthetic cohort tests
now invoke **actual `Measurement.finalize_workload`**, not a fake unsalted final
identity. Independent HMAC/canonical-JSON oracles test real commitment semantics,
Unicode/message contexts, changed inputs/salts, ordinary-target compatibility,
raw/final/version tampering, resealed cohort mismatch, and historical read-only
rejection. Copied timing/state observations remain synthetic fixtures, not evidence.

### Explicit tiny generated smoke

Parent first creates an isolated checkpoint with
`pllm.runtime.tiny_llama.create_tiny_llama_checkpoint(..., model_type="qwen2", with_qkv_bias=True)`
in a **fresh** directory. Parent then publishes/trusts its actual tiny certificate
using final SDK API and writes two explicit frozen Experiments referencing that
same generated source. Use tiny reference components from
`private_boundary_decode.recipes(tiny=True, generated_source=...)`, control arm
`coupled-parallel-uniform_residues`, and candidate differing only by codec/pin.
Freeze budget 128+2, requests=1, and a bounded public prompt file. Invoke driver
with `--tiny --max-output-tokens 8`; actual scenario cap is 2. This separate,
explicit fixture preparation is not hidden inside metadata-only preflight.
Tiny observations cannot support 150-token, long-output or SOTA claims.

## Execution, admission and process costs

Ordinary `BenchmarkTarget.response(..., backend="native", state_audit=True)` and
`benchmark_suite` execute six complete fresh clients, three per arm, serially,
one new private salt, no warmups, one repetition, one thread, timeout 3600 seconds
per worker, conservative 8 GiB admission budget. Native shared100/40 Mbps caps
remain enabled. Source/build/artifact hashes are checked around execution; SDK
also inspects build before and after cohort. Shared frontier lock prevents other
cooperating heavy runs. Do not edit SDK/build/source while a cohort runs.

Installed SDK `order="balanced"` produces accepted AB/BA/BA for two arms and
three replicates. Driver records and audits this actual schedule, never labels
it strict AB/BA/AB. No scheduling integration change is required. Splitting six
invocations would violate single-salt cohort and is not an acceptable workaround.

Admission inspects metadata before providers or tensor loading, including full
state-audit estimate. Exact byte admission estimates are emitted in plan rather
than invented here; observed RSS tiers never relax conservative admission.
Five CPU owners are client, inference, preparation, boundary_a and boundary_b.
Client peak covers complete fresh worker lifetime including SDK, supervision,
dashboard, relay and audit. Launcher excluded; final worker high-water gates
tiers using **maximum replicate**, not median. There are six SDK process
lifecycles plus role subprocesses, relay/supervision and auditor overhead; no
claim that only codec buffers are timed. Aggregate CPU sums all five owners
through first response; it does not measure post-response cleanup CPU. Complete
request and complete target lifecycle wall times are separate metrics.

All seven priority directions remain (client RSS/CPU, preparation RSS/CPU,
TTFT, decode throughput, network); TCP and covered bodies, complete request,
lifecycle, final worker RSS and aggregate CPU are also reported. Decode rate
uses N-1 generated-token denominator; one-output rate stays unavailable.

## Independent evidence checks and remaining integration

Auditor rebuilds explicit metric contracts/values; raw directed body edges;
startup/request TCP stream windows and native pacer counters; stage ledgers;
and sealed material conservation. Cached totals alone never suffice. It checks
all complete logit/active-KV, position/history, selected-token and RNG commitments
across all six clients, source/output identity, per-arm numeric identity across
replicates, build/device/environment/cache lifecycle, actual executed rows,
owner coverage and zero observed swap growth.

Numeric body fingerprints may differ only between arms; originals and complete
comparison provenance remain in per-replicate identities. Full-state commitments
must still match exactly. Request fingerprint must equal raw audit fingerprint.
Head reply, lifecycle control, prepared reply metadata and remaining startup
residual are attributed separately. MessagePack controls/timestamps may vary;
no assertion that total network delta equals a theoretical bit-width saving.
No requirement that candidate wins: valid slower/larger results remain evidence.

Certification now requires the latest raw snapshot at
`execution.details.private_boundary_head`, schema
`pllm.private_boundary_head_accounting.v1`. It must not remain in state-audit
bindings: public codec telemetry must not change numeric/output commitments.
Missing snapshots, archived public certificate bytes or independent native pins
reject the cohort; historical reports are never fabricated into certification.

The approved certificate bytes are retained as bounded `artifact_base64` in the
plan. Offline replay authenticates their exact size/SHA256 before bounded JSON
and zlib decoding. It checks complete source/checkpoint/config/head-weight/scales/
lookup identity, W8A8 numeric profile, geometry, width/bounds digests, publisher
implementation hashes, width histogram and packed bits against the snapshot.
Uniform width is independently rebuilt from `hidden * 128 * 127`; row widths
come only from the approved authenticated certificate. Bounds/source correctness
is a trusted publisher assertion, not a cryptographic proof or a new tensor scan.
Workers must have validated actual owned rows before recording successful samples.

`--native-head-commitments` supplies a bounded (64 KiB) independently approved
JSON object keyed by `parallel-uniform-residues` and `parallel-row-residues`.
Each value has exactly five SHA256 strings: `pipeline_digest`,
`composition_digest`, `schedule_digest`, `descriptor_digest`, and
`numeric_body_fingerprint`. Obtain these from independently trusted native
lowering/binding metadata for the frozen Experiments, **not by approving the
response snapshot currently under audit**. The snapshot alone supplies no separate
native-body-to-descriptor/schedule binding, so the auditor does not guess one.
Pins must match the frozen Pipeline, complete descriptor and raw numeric body.
They are required for execution; metadata-only preflight may omit them, but its
plan is then explicitly not execution/certification-ready. No native lowering or
checkpoint loading is hidden inside the driver.

Both owners must report exactly one reply per actual selection, including final
EOS. Payload/header/subframe totals are rebuilt exactly; outer response-body
totals must equal original head-download counters and fit each owner's online
download window. Existing directed body/TCP reconciliation still applies. Envelope
residual bytes remain explicit, not assumed zero or classified as payload savings.
Paired head deltas equal payload + codec-header + envelope deltas; lifecycle
controls and prepared-reply metadata remain separately charged. Only a complete
passing audit sets `codec_source_certification_verified=true`. Publication CPU/
distribution remain unknown and no full cold-network/SOTA claim follows. Fixed
certificate setup budget remains artifact bytes times declared missing owners,
once each, outside measured network totals.

Exclusive progress reservation prevents overwriting earlier attempts. SDK writes
completed arms atomically; interruption preserves incomplete raw cohort and
rejected progress, with no fabricated summary. Interruption after final raw
checkpoint but before post-checks likewise yields no terminal receipt. New attempt requires fresh paths.
Historical cohorts and aggregate research documents remain untouched.

## Bounded validation

```sh
.venv/bin/python -m pytest -q tests/test_private_boundary_row_codec_benchmark.py
.venv/bin/python -m ruff check benchmarks/research/private_boundary_row_codec.py \
  scripts/benchmark_private_boundary_row_codec.py \
  scripts/summarize_private_boundary_row_codec.py \
  tests/test_private_boundary_row_codec_benchmark.py
```

Tests exercise synthetic v2 whole-cohort validation and independent tampering of
state/RNG/history/token/output, numeric/source/build/environment bindings, raw
body/TCP/stage/material evidence, CPU ownership/sums, RSS/swap, cache lifetime,
metric windows/durations, cohort/order/completeness, frozen structural allowlist,
artifact byte pin and fixed setup budgets, plus exclusive-output refusal.
Additional regression tests reject fabricated workload digests, missing/changed
verifier sources, malformed/oversized manifests and oversized operator inputs
before hashing/admission; streaming hash and bounded certificate paths forbid
`Path.read_bytes()`.
