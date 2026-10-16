# Paper 1 implementation and conformance evidence

Audit date: 16 October 2026. This document supplies the implementation side of
P1-B/P1-C. It does not close the publication-build/release gate, the separate
primary-source literature review, or the matched performance/quality campaign.
No manuscript, bibliography, or shared gate status is changed by this audit.

**Accepted fresh execution:**
[`2026-10-16-clean-local-v1`](artifacts/conformance/2026-10-16-clean-local-v1/manifest.json):
all ten cases passed, 29 focused tests and 31 complete child identities.
[`2026-10-16-families-clean-v1`](artifacts/conformance/2026-10-16-families-clean-v1/manifest.json)
separately passes 46 tests and twelve operator/algebra/framing cases, with 26 child
identities. Both execute the same clean-local-v5 wheel, bound below.
[Replay instructions and retained failed attempts](artifacts/conformance/README.md).
Historical installed-v4 remains a distinct accepted functional record: its seven
cases passed, but its provider-child identities remain **unknown**.

## 1. Executed implementation identity

### Fresh clean-target build and provider-verified execution

The current execution is linked to
[`clean-build/2026-10-16-local-v5/receipt.json`](artifacts/clean-build/2026-10-16-local-v5/receipt.json)
through the separate, read-only
[`current-build link`](artifacts/conformance/2026-10-16-clean-build-links-v1/receipt.json).
The build receipt itself retains its original `built_and_installed_execution_pending`
status; subsequent execution evidence is supplied by the two new manifests, not by
rewriting that build-only record.

| Fresh identity | SHA-256 |
| --- | --- |
| Clean-build receipt | `4d274918781c78e3f0b4ebfdaef1501e2d19e741ee09e6847ba4e3ba665b243e` |
| Wheel | `1811d2d9d5c6cf1360091122a62fd9678031159acab1ab2a278b13b7872d2e8c` |
| Executed native binary | `826c2f41612912c38a2cd902a1988b50a069b4e278f1f05ffdb6980744e42c64` |
| Runtime `BuildRef` | `8a8803783aa2b61f84ac461c209925f911acbc447dc86bdc3120ee384d4c83d9` |

Independent read-only verification checked the build's **127 artifacts and 26,231
installed distribution files**, the actual wheel archive and its 355 package files,
and both runtime `BuildRef` canonical hashes. The Python SDK matches the inspected
checkout; the native binary differs from both historical v4 and the editable
checkout extension. The compiler used a newly empty Cargo target with incremental
reuse disabled, while registry sources, toolchain, OS SDK and Python wheel cache
were reused. This is local build evidence, not a hermetic or third-party/CI attestation.

Fresh main execution records **31** child launch/start/end/observed-exit lifecycles,
including **12 inference + 12 preparation** children. The family run independently
records **26**, including **4 inference + 4 preparation**. Each child's actual
package/native paths, 355 file hashes, import origins, isolation flags and exit
semantics are reconciled with the supplied wheel. These counts describe separate
executions; they are not repetitions of a measured performance cohort.

The fresh family manifest's `local_build_receipt_sha256` still equals historical
`b7d332fd599f7166f2b28d87012956b785b928cafa0b42e1b3bba60bd58a42fb`.
Its source lock correctly labels this `prior_build_receipt_sha256`. This is a
prior-evidence reference, **not the builder of the new native binary**. The explicit
current-build link verifies the actual `4d2749…` receipt, wheel, native binary and
runtime `BuildRef`; all `identity_equals_original_v4` fields are false. The raw
historical field is preserved, and does not prevent accepting the separately
verified new functional execution.

Two minimal-release attempts remain failed and separate from clean-local-v5.
`fresh-release-v1` failed provenance because Torch distribution metadata was absent.
`fresh-release-v2` passed SDK/gateway, the ordinary generated request, four bounded
operators and **11** child identities, but fixture generation and pytest collection
lacked Torch: zero tests ran and one collection error was recorded. Its wheel
`3b63ae30631f0b54773d708c0921b75dbf951142f14dfb80dd9eedac145043b6`
and `BuildRef` `e82b14e63bfa576a8144dbe00730c0610af707906b7e677cce9363028a2139c4`
are distinct. Making SDK Torch provenance optional did not make the generated
fixture or independent Torch/Transformers reference optional.

### Historical installed-v4 identity (unchanged)

The inspected checkout is `99e10e2c1e0e3dbeddc5f56dcc830e549cb1f2a5`.
The controller uses an **existing installed wheel** under Python `-I`.
Its byte identity is verified; this process flag does not establish provider-child
isolation. The original execution does not rebuild or install the native extension.

| Identity | SHA-256 / value |
| --- | --- |
| Distribution | `pllm.run==0.1.0a2` |
| Wheel | `55bc8e2b1f760fc4f96c9e88ea322e01408e24fe25c9154850c40e3748c21a50` |
| Installed native binary | `536b0b444a15c6a7d4bbf55bf47f10c93c95db269b64bd333a52edf3d8c11921` |
| Observed `BuildRef` digest | `db5b5374935fa41416a869383495ed68ac391ecb3535537d262110504e4a5f10` |
| Checkout's different native binary | `6b9370fba12ddd3df25979631d2ae9d37097d60bf0ae9bba1cd1bc2e8389a41e` |

The installed interpreter is `target/frontier-sdk-env/bin/python`; its wheel is
`target/certified-head-sdk-build-v3/wheels/pllm_run-0.1.0a2-cp311-abi3-macosx_11_0_arm64.whl`.
Every package file in that wheel is checked against the installation. All
installed `.py`/`.json` SDK files match the checkout at audit time. **The native
binary does not match the checkout's editable extension.** These are current
test sources run against that controller installation, with provider-child identity
qualified below. The original v4 snapshot
did not contain a build-to-source receipt; the later local supplement below
closes part of that provenance gap without rewriting the execution record.
This local supplement itself supplies no clean-build or third-party/CI attestation.

`provenance.json` retains the wheel/package hashes, installation `direct_url`,
dependency versions, Torch code hashes, native hashes, source differences, and
dirty checkout status. `installed-build.json` retains the complete SDK/dependency,
Python executable, numerical backend and hardware lock. The independent reference
process separately hashes its Torch and Transformers implementation; it never
imports PLLM. The wheel file and dependencies still need an immutable publication
archive and independent clean replay for P1-E.

### Provider-child identity qualification and bounded follow-up

[`artifacts/conformance/PROVIDER-CHILD-PROVENANCE.md`](artifacts/conformance/PROVIDER-CHILD-PROVENANCE.md)
records the launch-boundary audit and exact enforcement matrix. Installed
`runtime/servers.py` launches role children without `-I` and retains `PYTHONPATH`.
The frozen generic subprocess helpers removed `PYTHONPATH`, whereas the direct
generated SDK/gateway call did not scrub the parent's import environment. The
ordinary benchmark launcher independently removes all `PYTHON*`. None captured
provider-child import paths/native hashes. **Historical provider-child identity
is unknown**, including prepared tiling, Slalom/equalization SDK/gateway checks.
Their observed outcomes remain frozen; they do not establish all-provider
execution of the inspected wheel. No claim is made that those children actually
loaded the checkout or any other specific installation.

Both current research drivers now scrub Python/loader variables, enforce `-I`
on same-interpreter children, and recursively capture actual package/native
start/end identities before original entry points. They validate all 355 package
files against the caller-supplied wheel and require provider identity receipts
for provider-path selections. Runtime/package sources are unchanged.

`2026-10-16-child-isolation-v2/receipt.json` reports **seven passing focused
checks**, three captured Python processes and **zero provider/model/operator
executions**. It tests the installed role-spawn boundary with import-only code,
nested launch under replaced environment, native-mismatch rejection, missing
receipt rejection, and precise fixed-message accounting. The first diagnostic
attempt's parser error remains frozen in `2026-10-16-child-isolation-v1`.
These checks validate instrumentation, not a fresh SDK/gateway execution.
The later v3 instrumentation audit adds thirteen shutdown/isolation checks, and
the fresh runs above now supply actual SDK/gateway provider identities. Neither
follow-up retroactively establishes historical v4/family child identity.

### Local source/build supplement

[`artifacts/conformance/2026-10-16-local-build-v1/receipt.json`](artifacts/conformance/2026-10-16-local-build-v1/receipt.json)
archives the previously existing `target/certified-head-sdk-build-v3` source
manifest, successful build log and wheel lock. All **556** historical source
hashes match the current tree. A new build then executes successfully with:

```sh
CARGO_BUILD_JOBS=2 CARGO_NET_OFFLINE=true \
  target/frontier-sdk-env/bin/python -m maturin build --release --locked --offline \
  --out "$PWD/target/paper-publication-native-build-v1/wheels"
```

The receipt captures before/after hashes for **632 repository source/config
files** and **4,262 files across 81 resolved Cargo dependency source trees**,
including untracked files within the declared build-input roots. All remain
unchanged. Generated caches/binaries are explicitly excluded from source
inventories; output binaries are separately hashed. Compiler/tool identities
include Rust/Cargo 1.85.1, Rust LLVM 19.1.7, Maturin 1.15.0, Apple Clang 21.0.0,
macOS SDK 26.5, Python and compiler/tool binary hashes, Cargo metadata and relevant
environment/configuration. Exact absolute argv, timestamps, exit status and full
log are retained.

This is a **local cache-reusing incremental build**, not a clean or hermetic
rebuild: Cargo reports the release target finished in 0.06 seconds, with no crate
compilation reported. Its existing target/registry caches are reused;
`CARGO_INCREMENTAL` itself was unset. The receipt is local evidence of Cargo's
source/configuration checks and reproduced outputs, not an independent
source-to-binary attestation.

All **355 `pllm/` package files**, including the native binary, are byte-identical
to the historical conformance wheel and installed package. The complete new
wheel has SHA-256
`293578aea896ac7d5b4197f0bf7ef79949f70d8161c2d0b4d87566a642f789f4`.
Of its 361 payload files, only two differ from the old wheel:

- `pllm_run-0.1.0a2.dist-info/sboms/pllm-python.cyclonedx.json`: creation timestamp
  and `serialNumber` UUID differ; all other JSON values are equal.
- `pllm_run-0.1.0a2.dist-info/RECORD`: only the corresponding SBOM hash differs.

`payload-comparison.json` and the old/new SBOM/RECORD files retain these exact
deltas. Thus runtime-package/native output identity is established locally,
while **whole-wheel payload identity is false**. v4 retains its original wheel
hash; no tests are relabeled as execution of the new wheel. The existing
environment was not reinstalled. This cached-build complete wheel was not replayed;
the subsequently executed clean-local-v5 wheel above has a distinct identity.
Independent replay and immutable publication archival remain separate gates.
All four earlier execution directories remain
unchanged, and the previous outer archive index is preserved in the supplement.

## 2. Reusable extension: MLP prefill tiling

`MlpPrefillTiling(tile_rows=7)` is a registered immutable scheduler component
(`pllm/mlp-prefill-tiling/v1`). It uses the existing `Pipeline`, `Experiment`,
semantic stage bindings, linear executor, SDK, and gateway interfaces.

Source-to-runtime chain:

1. `python/pllm/schedulers/__init__.py` declares and validates the component.
2. `crates/pllm-compiler/src/mlp_tiling.rs` recognizes an exclusive, row-independent
   prefill region: fused gate/up linear projections, SiLU, multiplication, and
   down projection. It checks layer/shape/state constraints and rejects external
   uses of interior values. Public bounds include batch one, 1–256 tile rows,
   and at most 4,000,000 tile-row × intermediate-width elements.
3. The contract binds `source_plan_digest`, `source_schedule_digest`,
   `composition_digest`, tile size, and the exact regions. The runtime recomputes
   the source schedule digest before accepting the contract.
4. `python/pllm/runtime/mlp_tiling.py` executes each tile using the original
   compiled steps; tile-local scope retires temporary gate/up/product allocations
   before the next tile. Full attention/KV and decoder numerical policy retain
   their separate ownership. This is not a proof of whole-client memory usage.

The selected-component tests, rather than the research-only monkeypatch variant,
are executed. They exercise Qwen2 with QKV bias, Qwen3 with Q/K normalization,
and bias-free Llama; each uses a deterministic, **generated tiny checkpoint**.

| Check | Concrete scope |
| --- | --- |
| Serialization/selection | Component factory and pipeline roundtrip; configuration identity changes while declared numerical identity remains equal |
| Tiling parity | Tile sizes 1/7/16, 33-token batched prefill and three decode steps; byte-exact full logits and active KV; weak-reference checks for intermediate retirement |
| Separate roles/gateway | Per-family prepared SDK and gateway execution with real local provider children; full logits/KV equal the untiled control |
| Failure retirement | Exception after first tile; lease closes, active owner count reaches zero, reserved stage rows burn, subsequent `take` rejects |
| Incompatible composition | Verified prepared execution and Apple Metal scheduler combinations reject; invalid tile counts/types and mismatched selected tile size reject |
| Source/schedule checks | Wrong model, plan geometry, bundle tamper, post-bind mutation, stage drift and fused-weight reordering reject |
| Provider admission order | Missing/wrong plan and wrong body fingerprint return HTTP 409 before inventory availability is considered |

The focused suite contains **29 passing tests**: 13 selected tiling tests, seven
binding/admission tests, and nine complete-operator/evidence checks. Exact test
node IDs, commands, imported package location, logs, and JUnit records are retained
in the artifact bundle. Warnings are retained, including the disabled pytest
plugin's unknown asyncio configuration option.

Independent numerical conformance is a separate comparison: the driver generates
two-layer 32-hidden/64-intermediate, 258-vocabulary fixtures with seed 23; upstream
Transformers eager FP32 cached execution supplies logits and active KV. The
compiled graph emits all 258 vocabulary entries at the final prefill prediction
and each decode step; earlier prompt-position logits are not part of this exact
comparison. The check uses 33 input tokens and the fixed same-token sequence
`[5, 7, 9]`, comparing the exact same weights. W8A8 versus FP32 uses a predeclared
absolute tolerance of 0.1; it is **not** bit-exact numerical identity, representative
checkpoint quality, stochastic sampling validation, or an independent privacy proof.
Tiled versus untiled W8A8 is checked byte-for-byte separately. Tiled/reference arrays
and generated checkpoints are public fixtures retained for replay. Untiled arrays
were checked in process but were not separately serialized. The read-only archive
verifier recomputes all saved candidate/reference errors, without executing models.

Both historical v4 and the fresh main run separately check **29,592** bit-exact W8A8 values: per family, 1,032 logits
and 8,832 active KV values across four prediction positions. Measured maximum
absolute errors against upstream FP32 are:

| Generated family | Logits | Keys | Values |
| --- | ---: | ---: | ---: |
| Qwen2 | 0.042739 | 0.022892 | 0.023220 |
| Qwen3 | 0.051512 | 0.051037 | 0.021957 |
| Llama | 0.047203 | 0.021044 | 0.019906 |

The reference uses Transformers 5.16.1 and Torch 2.14.0. Complete unrounded errors,
bindings, inputs, generated checkpoint hashes and raw arrays are in
`generated-conformance.json`, `reference-*.npz` and `candidate-*.npz`.

### Published SDK/gateway path

The unchanged `examples/prepared_sdk.py` is executed with its default generated
`Model.tiny()` source and public fixture prompt. It performs two real prepared
requests. The same published `build_experiment` factory then completes an SDK
request and HTTP `/v1/responses` gateway request, with matching output text,
positive output-token counts, positive online steps, and no plaintext prompt/token
IDs in the provider audit. The gateway receives the explicit SDK-returned model
identifier. It is exercised via in-process ASGI HTTP; Inference/Preparation run as
actual local child processes communicating over TCP. This does not establish
independently operated services or Internet deployment.

**Paper 1 Listing 1 is a real, pinned Qwen model specification.** The generated
example replay does not execute that checkpoint and must not be relabeled as
Listing 1's real-checkpoint execution. Its parent-owned replay is separately
archived under `artifacts/listing1/qwen-v1/`; this worker does not modify or merge
that execution with the generated fixtures.

The ordinary `BenchmarkSuite`/`BenchmarkTarget.response` path also completes a
one-process, two-output-token generated request with `MlpPrefillTiling(7)`, an
explicit installed `BuildRef`, and one thread. Its ordinary runtime oracle passes;
this is a request-path check, not another independent full-logit oracle or a
performance comparison. `ordinary-benchmark.json` retains the full record.
The first 1 GiB whole-topology admission budget rejected a conservative
1,440,316,095-byte estimate before execution (`topology_memory_limit`, v3).
The accepted v4 replay uses a 2 GiB admission budget. Neither budget nor this
single request qualifies a client-memory tier; the benchmark controller is
explicitly excluded from that target's measurement ownership.

## 3. Six representative implementation families

The representative set now covers the **six families** selected in
`PRIOR-ART-REVIEW.md` §5. Paper-specific variants remain nested in their family:

| Family | Cards | Executed classification in this audit |
| --- | --- | --- |
| 1. Slalom-derived verified delegation | F1 | Generated end-to-end SDK/gateway adaptation |
| 2. SmoothQuant public equalization | F2 | Generated end-to-end adaptation; repaired accounting passes with captured frames; original failure retained |
| 3. Complete nonlinear representations | C1–C3: Curl/Haar, LogRow, Compact | Bounded complete gated-operator adaptations |
| 4. SIGMA/FuseFSS helpers | F4 | Bounded complete masked-wire operator, two layouts |
| 5. FastQuery/BumbleBee lookup/repacking | F5 | Bounded encrypted embedding and signed-conversion adaptations |
| 6. NEXUS/THOR nonlinear controls | F6 | Bounded complete encrypted softmax; decoder composition rejects |

None is a reproduction of an entire external system. Historical installed-wheel
supplements are `2026-10-16-six-families-v1` (**26 tests pass, one fails**) and
`2026-10-16-family-followup-v1` (four diagnostic cases pass). They retain the
original v4 wheel/native/BuildRef identities and refer to the separate local
source/build receipt. They neither replace v4 nor turn historical measurements
into current-source runs. Their exact accounting appears in §4.1 below.
The fresh clean-wheel family run passes **46** tests, including nineteen new
accounting controls, plus twelve operator/algebra/framing cases (§4.2). It supplies
fresh provider identities for F1/F2; bounded F4–F6 operator checks retain their
existing composition/security limits. C1–C3 are separately replayed in the fresh
main run. Counts and identities from these executions are not pooled with history.

The equations below state the exact PLLM integer functions, not purported
verbatim equations from the original papers. Original sections/figures and PDF
hashes come from the repository's source records; local PDFs are hash-checked.
Primary locators use the completed review, supplemented by direct reading of
the existing hash-pinned PDFs. `2026-10-16-family-source-review-v1` freezes the
locator audit. This is implementation mapping, not a new novelty survey.

### F1. Slalom-derived verified linear delegation

**Original source.** Slalom, ICLR 2019, arXiv:1806.03287v2; §3/Figure 1,
§3.2/Lemma 3.1 (precomputed verification), §3.3/Theorem 3.2 (privacy/integrity),
PDF pp.4–6. `papers/r07-slalom.pdf`, SHA-256
`ffdccc42057482eada2ca836e93dafbc35832fb3670e0bce0524b3412f7ef536`.
The original attested SGX/enclave boundary and its `2^24-3` field are material
parts of the original system, not inherited by this adaptation.

**Function and exact adaptation.** With implementation-oriented public integer
weights `W` of shape output × input, input row `x`, and proposed accumulator `y`,
Preparation derives secret field vectors `s_j` and projections `z_j=W^T s_j`.
Before dequantization, the client checks

```text
<y,s_j> = <x,z_j> mod p,       p = 4,294,967,291
k = ceil((b + ceil(log2(A))) / 31)
```

Here `b` is the requested failure-bit target and `A` the admitted session-wide
attempt bound. The code uses a conservative 31 bits/check and rejects unsupported
resource/soundness bounds. The public component accepts `b=1..80`; core admission
also caps total required strength. Signed input/output ranges, the matrix's
maximum row L1 norm, and `B_out < floor(p/2)` are checked, so modular equality is
not silently accepted as an unrestricted integer identity. The body separately
uses its admitted masked-residue ring (`u16/u24/u32` or prime); it does not change
the verifier field. Corrected integer outputs are verified before scales/bias
and nonlinear work are released to the decoder.

**Code and ownership.** `crates/pllm-core/src/freivalds.rs` owns projection
preparation, authenticated import, challenge derivation, constant-time equality,
attempt bounds and one-use claims; `python/pllm/verification/__init__.py`,
`runtime/transformer_engine.py`, `runtime/transformer_client.py`, `runtime/client.py`
and the compiled binding select the verified executor. Software trusted client
owns tokens, scales, nonlinear operations, logits and KV. Trusted Preparation
owns masks/projections; Inference receives masked linear inputs and public
weights. An allowed untrusted Inference must not learn secret challenges or
collude with Preparation. Malicious/colluding Preparation, TEE attestation,
protected key release, and original private-model protection are not supplied.
Public model/shape/stage inventory and traffic/timing leak as declared.

**Admission, lifecycle and complete owners.** Native policy bounds matrix,
projection, challenge and output elements, multiply-accumulates, batch rows and
attempts before allocation. For `R` inventory rows, `I/O` input/output widths and
`k` checks, projection payload alone is `4*R*k*I` bytes; a claimed `B`-row batch
also needs `4*B*k*O` challenge bytes, signed input/output buffers, authentication
and metadata. Preparation work, projection delivery, ordinary mask/correction
delivery, model/bundle delivery, online stage frames and client verification
remain separate charged costs. The projection subtotal is neither a whole-client
bound nor a traffic total. Material binds model/body/stage/weight/shape/numerics,
inventory/session and rows. Verification, malformed input, cancellation, failed
checks and reimport/replay consume or reject the associated material.

**Oracle and evidence.** Six current native-boundary tests check independent
small integer products, corruption, range/binding failures, cancellation,
reimport, malformed-type retirement and attempt composition. One current
generated-Qwen2 test completes both verified SDK and gateway requests through
actual local Preparation/Inference children: **seven tests pass**. The literal
integer oracle is independent of native verification; completed text is not a
separate upstream-FP32 model oracle. Historical records
`research-slalom-qwen25.json` and
`slalom-prepared-topologies-cold-cpu-2026-09-26.json` retain their original measured
workloads/builds. `research-slalom-admission.json` remains a blocked 10.04 GiB
estimate versus 7.74 GiB headroom, with no providers launched. The historical
cold CPU disadvantage and blocked geometry are not superseded by tiny success.

### F2. SmoothQuant-derived public per-channel equalization

**Original source.** SmoothQuant, ICML 2023/PMLR 202, PDF pp.3–4, Equations
(3)–(4), Figures 4–5. `papers/smoothquant.pdf`, SHA-256
`551636e4e0bd9dda9b01186473bc46c72c56e0c190a28fe6b7ed4fa80055372e`.
Equation (3) is the exact real-valued channel redistribution; Equation (4) chooses
scales from activation/weight maxima. PLLM's additional normalization/clipping
must not be attributed to those equations unchanged.

**Implemented function.** `runtime/public_equalization.py` fixes `alpha=3/4`:

```text
r_j = max(A_j,1e-6)^(3/4) / max(W_j,1e-6)^(1/4)
G   = exp(mean_j(log(r_j)))
s_j = float32(clip(r_j/G, 1/64, 64))
x'_j = x_j/s_j,       W'[:,j] = W[:,j]*s_j
```

The trusted client scales activations before ordinary per-row quantization;
weights are scaled before their quantization. Before quantization,
`x' W'^T = x W^T` over the reals. Separate rounding/saturation means equalized
W8A8 is a **different numerical identity**, not exact baseline-W8A8 or FP32
identity. Calibration uses caller-declared public offline tokens and the locked
source, covering every body stage; token lookup and LM head are excluded.

**Roles, leakage and deviations.** `PublicPerChannelEqualized` is a numerical
component, not a new privacy protocol. The fitted profile, calibration tokens,
source lock, per-stage scales and public profile digest are public. Online
token-dependent activations remain client-owned under the selected client-only
or prepared topology. Existing Preparation/Inference assumptions remain in force;
no private online range fitting or claim of protecting the calibration set is
made. Fixed alpha, `1e-6` floor, geometric normalization, bounded scales, selected
body stages and trusted-client division adapt the original recipe. No original
SmoothQuant training/model-quality result is reproduced.

**Admission and accounting.** Profiles admit at most eight sequences, 128 tokens
each, 1,024 stages and 16 MiB canonical MessagePack; scales are finite float32
within `[1/64,64]`. Hash/source binding, immutable scales, missing/wrong profile
and downgrade/tamper rejection occur before stage import. Count public calibration
work, retained/delivered profile, equalized weight preparation, local divisions,
ordinary W8A8 stage frames and bundle metadata separately. The historical Qwen
profile contains 96 stages/729,527 bytes and binds profile
`a1024e2b0376a5d8baa57b6c972d18f641a2817a57b204cc1e1c4182dc0f430b`.

**Oracles and historical result.** Independent Python scalar powers/logs match all
six adversarial/clipped scale examples; float64 matrix algebra checks 12 outputs
with maximum error `9.172864245954315e-9` (declared absolute tolerance `1e-6`).
Seven of eight original generated tests passed, covering profile integrity,
source/downgrade rejection, client/prepared compiled execution, both SDK/gateway
placements and pinned-source CLI handling. The eighth test fails: ordinary versus
equalized tuples are `(4,11580,16289)` and `(4,11580,16291)` for stage calls,
upload bytes and download bytes. **Exact total download-byte equality is not
established.** Current response framing includes variable-width MessagePack
`server_ns`; the separate deterministic follow-up changes only 65,535 to 65,536
and adds exactly two bytes. Original failing frames were not captured, so this
demonstrates an accounting mechanism, not retrospective field-level attribution.
That attempt's runtime/tests were not edited and its failure remains frozen.
The exact failing node is
`tests/test_public_equalization.py::test_prepared_numeric_choices_have_matched_online_stage_body_sizes`.
The [targeted follow-up](artifacts/conformance/PROVIDER-CHILD-PROVENANCE.md)
provides a principled test change: preserve call/upload equality; retain actual
response frames; compare static arithmetic geometry/payload and non-timing
framing exactly; account encoded `server_ns` widths separately and reconcile
every full download byte. Its negative controls reject non-timing differences.
**Fresh repaired result.** The source-test repair is now executed in
`2026-10-16-families-clean-v1`: all eight equalization tests and nineteen focused
accounting controls pass. Eight actual binary response frames and
`equalization-frames/test_prepared_numeric_choices_0/response-accounting.json`
are frozen. Each choice has four calls, 11,580 upload bytes, 16,128 arithmetic
payload bytes and 16,293 full download bytes. Non-timing normalized framing is
16,277 bytes per choice; four actual five-byte clock encodings replace four
one-byte normalization markers, reconciling the remaining sixteen bytes exactly.
Fresh masks/request identifiers are validated and retained; their contents may
differ, while their headers, geometry, payload sizes and other framing must match.
The original two-byte failure still has no retained frames and remains unresolved.

**Historical limits.** Calibration, controls and quality records are separately
locked. On the original 12-prompt upstream-reference diagnostic, ordinary W8A8
has 11/12 top-1 agreement versus equalized 10/12; maximum logit error falls from
3.62069 to 2.50925, with equal top-k recall. These mixed outcomes do not establish
a quality win, baseline-state identity or universal byte-for-byte traffic match.

### F3. Complete nonlinear representation alternatives

Let `R(n,d)` be signed round-to-nearest, ties-to-even division of integer `n` by
positive integer `d`. Every card accepts signed Q18 integers `g,u` satisfying
`|g| <= 48*2^18`, `|u| <= 128*2^18`. Values outside the public domain reject;
client-local float conversion and range enforcement are additional boundaries.

### C1. Curl/Haar whole SiLU × up

**Source record.** Curl, ePrint 2024/1127, Sections 4.1.1/4.2.1, Figures 4/6;
`papers/r17-curl.pdf`, SHA-256
`dc930fdb3219c989106f96eb9ae1f00f28aae16ec5d17eef0c5421eb4f241c1e`.
Exact rescaling uses the mixed-mode FSS Figure 14 universal interval construction,
whose recorded PDF digest is
`dc94c63a77a5cb52cfd12b186a327d045de4a1fc662060c1de6412b334f4fcf4`.

**Implemented function.** For `j=0,...,4095`, define
`T[j] = round_even(512*SiLU(j/32-64))`. For Haar level `h` in 0–8,
`T_h[j] = R(sum(T[2^h*j:2^h*(j+1)]), 2^h)`. The complete output is

```text
i = floor((g + 64*2^18) / 2^(13+h))
u9 = R(u, 512)
y9 = R(T_h[i] * u9, 512)
```

`crates/pllm-core/src/wavelet_gated_reference.rs` defines the numerical profile;
`crates/pllm-garble/src/wavelet_gated_reference.rs` composes protected index/up
rescaling, fresh masked-index one-hot lookup, a fresh ring32 Beaver product, and
protected output rescaling. `python/pllm/metrics/wavelet_gated.py::_profile/_oracle`
independently constructs the table and evaluates integer outputs. This oracle
does not call the native evaluator to obtain expected values.

**Bounds/ownership.** The complete table bounds the signed product below `2^31`;
`|u9| <= 65,536`. Admission covers 1–256 lanes and a conservative combined
allocation estimate no greater than 64 MiB, before issuance. Four peer rounds
charge both directions, including all three rescalings; full frames total
`1,248 + 176*lanes` bytes. Each party's lookup share alone is
`lanes*(2+4*(4096>>h))` bytes, but that is only one subtotal. Both-party keys,
public tables/profile, per-issuance metadata and client entry/exit estimates are
separate reported owners. Fresh material is bound to session/context/profile,
lane order and issuance; completion, malformed frames, replay and cancellation
retire it. These lifecycle properties are implemented in Rust; this audit's
Python subset does not replace a full native lifecycle-suite rerun.

**Adaptation.** Exact floor/ties-even helpers replace Curl's probabilistic
truncation. The mathematical SiLU table, Haar integer means, in-process dealer,
CPU execution and client float boundaries are explicit adaptations. There is no
Biorthogonal protocol, distributed issuance, authenticated provider transport,
or protected decoder binding in this operator.

### C2. LogRow whole SiLU × up

**Source record.** Heath, Kolesnikov and Ng, ePrint 2024/369;
`papers/r05-logrow.pdf`, SHA-256
`ac61a6a77b1f21656fa6a90a2f550e59d854542e137cc0eaebb29e718de98371`.
The cited mechanism is the older two-row masked one-hot setup, including all
masked table bits. Half-gates arithmetic follows ePrint 2014/756, Figure 2,
recorded PDF digest
`789ecaf8651b6dc8341d6033f951d76c0e6f4b04427cdde26cf91ecfb0af3083`.
**Resolved primary locator:** §3.4/Figure 1 (PDF p.10) supplies the one-hot
interface; §4.1 (p.12) defines `f'(x)=f(x XOR alpha) XOR r(x)`; §4.2 (pp.12–14)
constructs the recursive half-hidden random function; §5/Construction 1 and
Figures 3–5 (pp.15–18) specify the LUT, recursion and row-reduced masking gate.
`logrow.rs::prepare_with_encodings` maps to those operations: input color mask,
punctured one-hot tree, one ciphertext per output bit/depth for the half-hidden
mask, recursive folding, and every masked-table bit. It **does** implement the
Figure 5 row-reduced half-hidden-function step. Its one-hot tree retains
`2(n-1)+1 = 2n-1` blocks, rather than the `n-1` blocks charged in §6/Theorem 1
(pp.19–20). The pinned text itself lists `2n-1` in Figure 1 but `n-1` in §6;
this audit does not silently reconcile those two printed accounting statements.
For its LUT, the implementation charges `(2n-1+n*m)*128 + 2^n*m` bits, before
metadata/input labels/complete-product costs. The exact locator is resolved;
faithful reproduction of the paper's optimized headline cost is not claimed.

**Implemented function.** Exactly C1 with `h=0`; its numeric digest equals the
uncompressed Curl digest. Native `logrow_product.rs` selects the protected
12-bit index by dropping 13 low bits and toggling the sign-offset bit, evaluates
the 4,096-entry Q9 lookup, rounds `u`, multiplies and rounds the final product
inside one half-gates circuit. Intermediate output is not decoded to plaintext.
The independent Python expression is in
`python/pllm/metrics/logrow_product.py`; it uses the independently generated C1
table. `crates/pllm-garble/src/logrow.rs` supplies the lookup implementation.

**Bounds/ownership.** Input widths are 25/27 bits; output is signed 23 bits.
The complete-table product is bounded below `2^31`. A 32,768-wire ceiling and
16 MiB conservative serial-live allocation bound precede issuance. Each lane
gets fresh non-cloneable circuit/input/decoder handles and retires before the
next lane; one invocation admits 1–256 lanes and 1–16 samples. Garbler-client
encodings, evaluator lookup payload including masked table bits, arithmetic
ciphertexts, constant labels, public metadata and boundary delivery estimates
are all charged. **Zero evaluator-peer frames does not mean zero communication.**

**Adaptation.** Index reduction, complete signed product and exact shifts are
PLLM composition work; no distributed OT for independently supplied evaluator
inputs, authenticated transport, compiler tensor schedule or complete decoder
is implemented here. Client garbling differs from the two-worker Curl ownership.

### C3. Compact whole SiLU × up

**Source record.** Compact, arXiv 2309.04664, Sections 4.2.1–4.2.2 and 4.2.6;
`papers/r18-compact.pdf`, SHA-256
`e4497a81055e60cca1c462dac5ae04b4504c5f3fc76e392e170cc32bf8fee86c`.
The SDK citation uses the coarser Sections 4.1–4.2 locator. Half-gates has the
same source digest as C2.

**Implemented function.** Set `g12=R(g,64)`, `u12=R(u,64)`. If `g12 <= -8*4096`,
set `a12=0`; if `g12 >= 8*4096`, set `a12=g12`. Otherwise privately select a
dyadic interval `[L,U)` and its committed coefficient words `c_0,...,c_d`, with
degree two or three. Its normalized coordinate and Horner recurrence are

```text
t20 = (2*g12 - L - U) * 2^20 / (U-L)   # exact for admitted dyadic intervals
v = c_d
for k = d-1,...,0: v = R(v*t20, 2^20) + c_k
a12 = R(v, 2^20)
y12 = R(a12*u12, 4096)
```

`crates/pllm-core/src/compact_product.rs::Profile` fits and commits coefficients;
the garbled counterpart performs private piece/coefficient selection, Horner
arithmetic, tails and final product. The independent Python integer oracle is
`python/pllm/metrics/compact_product.py::_oracle`; it independently evaluates
committed coefficients, **not** an independent reimplementation of fitting.
The profile digest binds source, public calibration, requested/actual pieces,
degree and all coefficients.

**Bounds/ownership.** The fitter admits exactly 1,024 public counts, each no
greater than 1,000,000, total no greater than 100,000,000, and 1–16 pieces.
Coordinate magnitude is at most `2^20`; the coefficient-sum admission includes
rounding slack and a final half-unit before the `<2^38` check. Native circuits
retain signed 40-bit Horner words, 19-bit activation, 21-bit rounded up value,
and signed 26-bit result, preserving rounding carry before narrowing. The
131,072-wire ceiling preflights a conservative 64 MiB serial-live bound. Fresh
one-use client encodings, evaluator ciphertexts/constants, all selection/
rescaling/Horner/tail/product costs, public profile/metadata and boundary delivery
estimates remain separate. The operator admits 1–256 serial lanes and 1–16 samples.

**Adaptation.** Public histogram plus one pseudocount, fixed-bin integrated
absolute error and dyadic greedy refinement replace the paper's
batch-normalization distribution, threshold and annealing search. Q12, explicit
tails and the Boolean backend are additional changes. RMSNorm does not justify
assuming Compact's batch-normalization distribution. Client float conversion,
protected input-range proof, distributed role transport and tensor scheduling
remain absent.

### F4. SIGMA/FuseFSS masked-wire gated helpers

**Original sources.** SIGMA, final PoPETs 2024(4):61–79, §§2.5–2.6,
4.2.1–4.2.2 and 5.1, especially public gap arithmetic/right shift,
truncation/reduction and Figure 7's residual nonlinear
order; `papers/r13-sigma.pdf`, SHA-256
`f911b715b0e0523fa4d77f21c28c6a0440c5be781e67b5b30419f20e505a6ebf`.
FuseFSS, arXiv:2606.09551v1, §§3–5, §4.5/Theorem 4.6 and §4.6/Theorem 4.7;
`papers/r14-fusefss.pdf`, SHA-256
`6109399761ab0cc1ca6eb617fec1a811cbe59d474455ce3549467e5a5b77cbf0`.
Grotto suffix-parity comparison and Pika lookup/sign correction are additional
helper ancestry, retained in SDK citation records. No upstream implementation
is linked by the reference.

**Exact complete function.** `MaskedWireGatedMultiplyQ12(layout=...)` accepts
integer Q24 gate/up, `|g24|<=48*2^24`, `|u24|<=128*2^24`, in a signed 40-bit ring:

```text
T[i] = floor(4096*(i/64)/(1+exp(i/64))),  i=0..1023
g = floor(g24/4096); u = floor(u24/4096)
j = min(abs(floor(g/64)),1023)
y12 = floor((max(g,0)-T[j])*u/4096)
```

The negative-input asymmetry from floor-before-absolute-value is intentional.
Three 40/12 public-gap shifts, a 19/6 truncate/reduce, sign/interval predicates,
clipped-index selection, full-table lookup, ReLU-minus-residual composition,
masked multiplication and final shift all belong to the operator. Public
range analysis admits the input shifts and final product into
`[-2^(n-2),2^(n-2))`; an unrestricted ring shift is not substituted. Functions
live in `pllm-core/src/residual_silu_reference.rs`,
`pllm-garble/src/compact_dpf.rs` and `pllm-garble/src/masked_wire_reference.rs`.
The independent Python table/integer oracle is in `metrics/masked_wire.py` and
the SDK benchmark support; no expected integer comes from native evaluation.

**Roles, lifecycle and changes.** A trusted local dealer/client knows masks and
encodes/decodes bounded values. Two evaluator handles hold only their own
non-cloneable material, correction shares and masked intermediate wires. Public
geometry, profile/context, ordered frames and lengths are visible; input/output
masks and the other party's keys must remain hidden. Dealer/evaluator collusion,
malicious-result security and authenticated distributed transport are outside
this operator. Fresh issuance binds context/profile/party/phase/lane count;
malformed input, replay, cancellation and completion retire material. The CPU
DPF uses 128 seed bits plus flags, two AES calls per child, and 128-leaf early
termination; SIGMA's packed 127-bit seed/flag, half-PRG and GPU implementation
are not reproduced. Client float↔Q24 conversion and bounded residual SiLU replace
a complete original fixed-point decoder. FuseFSS's multi-view idea becomes a
specific eight-phase CPU layout; its entire compiler/proof is not reproduced.

**Admission and complete accounting.** At most 256 lanes and 16 samples; native
combined material/scratch/framing estimate must fit 64 MiB before random issuance.
The ordinary layout uses 11 ordered peer phases; fused uses eight, reusing a
single masked gate wire's properly corrected views within one invocation, not
keys across invocations. The fresh eight-lane reports charge respectively
`(5162,5162)` versus `(9124,9124)` raw key bytes, 2,254 versus 1,786 complete
both-direction peer-frame bytes, 8,192 public-table bytes **per party**, 120 client
mask bytes and 200 client entry/exit payload-estimate bytes. Estimated allocations
are 791,808 and 1,298,944 bytes. Fewer phases reduce frame overhead here while
material grows; keys, public table, offline delivery/framing and client conversion
must not disappear from a claimed total. Full physical wire/peak memory remain
unknown.

**Current/historical evidence.** Both layouts check eight exact integer outputs
plus eight fresh replay outputs, with identical numerical identity/quality and
stable per-layout resources. Three selected current tests pass, including fresh
sessions, oracle-corruption rejection and replay-contract rejection. Correct
integer execution still has nonzero float-SiLU approximation error. The historical
`research-masked-wire-reference-qwen25.json` retains the failed numeric screen
(10/12 prefill and 34/36 same-token decode selections, no exact full logits/KV)
and excessive projected full-decoder material. `research-masked-wire-fusion-sdk.json`
is a separate older cohort, not counted as fresh tests or new performance here.

### F5. FastQuery/BumbleBee private embedding and repacking

**Original sources and version separation.** FastQuery, DAC 2024 /
arXiv:2405.16241v1, §§2.2, 4.1–4.3, Figures 4–6;
`papers/fastquery.pdf`, SHA-256
`5a242b1a65ceaaaa256a3f02fd0d3cc6eb40c249ca39c0e645502176520711be`.
The implementation binds the **cached ePrint** BumbleBee, §IV/Figure 6,
`papers/r22-bumblebee.pdf`, SHA-256
`15583a86593998079d42b8b134dc4f4f5be9ca72a7ced8f31bba05f68893516a`.
The completed review separately checks the NDSS 2025 final §III/Figure 6,
whose bytes/section numbering differ; the final PDF is not silently substituted
for the code's source digest.

**Exact operator.** `PrivateEmbeddingLookup` operates in
`Z_8192[X]/(X^4096+1)`. If `B` is the public padded row count and secret token is
`t`, the query is an encryption of `X^(B-1-t)`. Public packed group `j` places row
`i` at coefficient `j*B+i`; useful product coefficient `(j+1)*B-1` selects row `t`.
Every response coefficient receives an independent uniform mask before delivery.
Mixed `(4,3,3)` signed fields occupy offsets 0/5/9 with guard bits, fitting 13 bits;
Int8 and paired Int4 are other admitted profiles. The garbled conversion computes
the sum of the two 13-bit shares modulo 8,192 **before** field extraction,
sign-extension and fresh ring32 output sharing. Extracting each share separately
would mishandle carries/borrows.

`repacking="pack_lwes"` extracts encrypted coefficients as LWE `(b,a)` with
decryption `b+<a,s>`, embeds the signed/reversed vector in RLWE, then uses complete
trace/FFT-style packing. `interleave` consolidates strided RLWE coefficients
directly using ciphertext-RNS division and trace; division in the plaintext ring
is not equivalent. Implementations are `pllm-core/src/packed_embedding.rs`,
`pllm-core/src/coefficient_repacking.rs`, garbled conversion support,
`runtime/coefficient_repacking.py`, and
`metrics/{packed_embedding,repacked_embedding}.py`.

**Roles, security scope and deviations.** Client alone owns the BFV secret key
and decrypts masked coefficients. Provider owns the table, uniform mask and public
evaluation keys. `conversion="party_local"` uses distinct native handles and
local-dealer random-OT correlations; each party receives only its conversion
inputs/output shares. The independent benchmark oracle alone reconstructs selected
signed integers. Public geometry, precision, frame lengths and number of groups
leak. Honest one-hot queries and non-colluding dealer/evaluators are assumptions;
no malicious-query proof, circuit-privacy review or authenticated provider service
is supplied. Local random-OT/half-gates replace SiRNN/distributed setup. No
FastQuery frequency-weighted STE fine-tuning, trained checkpoint, complete
BumbleBee decoder or private tied-head placement is reproduced.

**Admission/lifecycle.** Vocabulary ≤4,096, width ≤256, samples ≤16; dimensions
alone are insufficient: combined context/key/RNS/packing/serial-conversion
allocation must fit 128 MiB, and each conversion its 4 MiB bound. Ordinary BFV
uses one 50-bit prime; automorphism paths use 54+54-bit key context and one
54-bit data prime at degree 4,096. SEAL TC128 parameter validation and strictly
positive decoded noise budget are checked, not inferred. Fresh HE keys, complete
response masks and conversion issuance are used per sample. Plan/session/index
bindings, foreign labels/frames, malformed input, replay and cancellation retire
one-use conversion state. This does not certify multi-block full-vocabulary noise.

**Oracle, exercised geometry and complete costs.** Independent signed row lookup
checks all seven outputs for each initial call and fresh replay. The first
17×7 three-layout check is retained: its interleave branch has only one response
and **zero automorphisms**, so it alone would not exercise consolidation.
The separate follow-up uses a still-bounded 4,096×7 public table (28,672 scalar
codes), three response polynomials, and all three layouts with party-local
conversion. Both repacking modes execute **43 automorphisms**, retain nonzero
Galois-key payloads, preserve one common numeric digest, and pass positive-noise
checks. The initial minimum noise budgets are 10/20/19 bits for none/interleave/
PackLWEs; fresh randomized replay has its own recorded observations.

For that follow-up, complete online HE-plus-conversion frames total 284,168 bytes
without repacking and 144,904 bytes with either packing mode. Fresh setup delivery
is separately 70,144 versus 1,761,792 padded bytes. Raw client/provider key payload
is `(152352,79432)` versus `(1876768,1771080)` bytes. Public provider table/codes
occupy 53,248 reported bytes; metadata, client masks and setup overlap remain
separate owners. Allocation estimates are 76,398,592 / 89,374,720 / 90,161,152
bytes, not RSS. Eight selected current admission/conversion/framing/lifecycle and
decoder-rejection tests pass. These are functionality/resource-conservation
observations, not a performance cohort.

**Historical limits.** The locked packed/repacked cost records and separate
`research-packed-embedding-quality-qwen25.json` retain their original builds.
The clear requantization diagnostic fails exact logit/KV fidelity and does not
execute HE. A client-resident tied head still retains the whole vocabulary matrix:
embedding-only relocation releases **zero** tied-weight bytes. Full-vocabulary
projection, transport, quality and client-memory claims remain unestablished.

### F6. NEXUS/THOR complete bounded encrypted softmax

**Original sources and corrected locator.** NEXUS, *Secure Transformer Inference
Made Non-interactive*, NDSS 2025, §IV softmax, PDF p.8, Equation (4), the
**following unnumbered repeated-squaring formula**, and Algorithm 4;
`papers/r21-nexus.pdf`, SHA-256
`c809182345f27f76e21d983d97dffa0d086dbcd17dabc265f0df9fb6ca9b70f2`.
The SDK citation's shorthand “Equation (4) repeated-squaring exponential” is
imprecise: numbered (4) is normalized exponentiation. The following text supplies
`(1+x/2^r)^(2^r)` with `r=8`, already with a constant-shift optimization and
Goldschmidt division. THOR, ePrint 2024/1881 (the hash-pinned text reviewed for
CCS 2025), §5.1/Algorithm 3 and Appendix C's degree-15 coefficients;
`papers/thor.pdf`, SHA-256
`ddb57004b56ee8e00beffc4f563fd0921242753fda6c704065d4900b792439cd`.

**Exact complete profiles.** Let `I_k(u)=(1-(1-u)^(2^k))/u`, implemented by the
unrelaxed Goldschmidt product, valid for the publicly bounded `0<u<2`:

```text
NEXUS-derived: v_i = (1+(x_i-1/8)/256)^256
               a_i = v_i/n; y_i = a_i * I_3(sum(a))
THOR-derived:  v_i = p15(x_i/2)
               a_i = v_i/n; q_i = a_i * I_2(sum(a))
               b_i = n*q_i^2; y_i = b_i * I_2(sum(b))
```

`p15` is the exact 16-float coefficient vector in
`runtime/ckks_softmax.py::THOR_EXP`. The encrypted schedule uses balanced powers;
the independent Decimal oracle uses Horner/direct powers. THOR's selected
`delta1=1, delta2=2`, two unit-factor reciprocal steps per normalization and the
public narrow range replace its general relaxation/parameter schedule. The
specific NEXUS shift `1/8`, narrow range and three reciprocal iterations are
adaptations. Neither profile is mathematically exact softmax or numerically
interchangeable with the other.

**Bounded admission and backend.** Finite scores must lie in `[-1/8,1/8]`, widths
are 1/2/4/8/16/32, and 1–64 serial rows are allowed. Invalid values reject before
key generation, without clipping/private range fitting. Both complete schedules
consume 14 rescale levels, not just exponential depth: SEAL TC128 checks degree
32,768, `[50]+[40]*14+[50]` coefficient bits and scale `2^40`. Rust's
`ckks_layout.rs`/Python binding preflight the combined contexts, keys/copies,
delivery overlap, scratch and 32 live ciphertexts under 2 GiB. For width two,
the reported estimate is 1,334,325,248 bytes. The audit additionally admits at
least 4 GiB headroom beyond the existing host reserve and monitors the live host.
This is not a client-tier or measured peak-memory result.

**Roles, lifecycle and omitted functionality.** `runtime/ckks_reference.py`,
`runtime/ckks_softmax.py`, `runtime/seal_frames.py` and
`metrics/encrypted_softmax.py` orchestrate distinct client-secret/public-provider
contexts. Provider performs encrypted arithmetic; client decrypts only final
normalized output. Each sample generates fresh keys, rows reuse public evaluation
keys with fresh ciphertexts, and bounded sequence/binding/padding checks retire
malformed/replayed/cancelled sessions. Public width/rows/profile/key geometry
and traffic leak. The intended passive-provider encrypted-input boundary is
not a proof against malicious evaluation, circuit leakage or collusion. No secure
maximum, bootstrap (unsupported by this SEAL backend), QK/value product, causal
mask, KV, token feedback or complete model executor is supplied. Decoder
Experiments explicitly reject both components.

**Oracles and complete accounting.** A 60-digit Decimal profile oracle measures
ciphertext arithmetic error separately from a 60-digit `exp/sum` true-softmax
oracle and probability-mass error. Width-two, one-row calls check **two outputs
per profile**, then independently replay fresh keys/ciphertexts. All eight values
across the four calls stay within `0.0002`; exact-integer output count is zero.
Resource identity, complete depth and frame conservation pass; approximate
quality numbers are not required to be bit-identical on randomized replay.
One current invalid-range/geometry-before-key test passes; both decoder-rejection
outcomes are separately frozen.

Each width-two sample charges 260,190,720 setup-frame bytes (parameter, public,
Galois and relinearization delivery) plus 8,396,962 online encrypted request/reply
bytes. Reported raw client/provider payloads are 12,587,008 / 260,190,208 bytes;
these overlap setup owners and are not added into a fictional memory total. The
THOR coefficient table adds 128 public bytes. All rotations, products,
relinearizations, modulus switches/rescales are recorded; physical wire and
measured whole-client peak remain unknown.

**Retained rejection evidence.** `research-encrypted-softmax-cost.json` remains
its original synthetic cohort. The separate clear-checkpoint
`research-encrypted-attention-range-qwen25.json` finds all 48 inspected positions
outside this narrow profile, with 3,521,405 of 3,559,248 valid scores outside and
maximum absolute score 1,701.7544. It is a range rejection, not an encrypted
candidate-quality run or evidence that full NEXUS/THOR is infeasible.

## 4. Complete costs, oracles, and frozen historical evidence

Original v4 installed-wheel checks run four complete operators (C1 at levels 0/4,
C2, C3 at four pieces/degree two), each with eight lanes and one sample, then
replay fresh material. All 32 initial and 32 replay outputs pass the independent
integer oracles, and resource/numerical-quality reports agree on replay. These
same-process functionality observations are **not** a new matched cost campaign.
`operator-replay.json` retains each complete configuration, numeric/native/SDK
digest, citations, role costs and timing scopes. None marks a whole-model executor,
independent role transport, measured peak RSS, or full wire as available.

Historical source/evidence files are byte-locked in `source-evidence-lock.json`.
They retain their own builds and prompt cohorts:

| Record | Executed scope and oracle | Complete-cost interpretation |
| --- | --- | --- |
| `research-wavelet-gated-cost.json` | Nine fresh processes; 2,889 protected integer outputs; levels 0/2/4 and 1/64/256 lanes | Both-party keys, all rescalings, multiplication, four rounds, public setup and boundary estimates |
| `research-logrow-product-cost.json` | Six fresh processes; 1,926 protected outputs; matched uncompressed Curl/LogRow numeric identity | Client/evaluator owners differ; includes lookup and complete multiplication; zero peer frames excludes material delivery |
| `research-compact-product-cost.json` | Nine fresh processes; 2,889 protected outputs; three fixed profiles | Selection, input rescaling, full Horner/tails/product rounding and constants are included |
| `research-mixed-gated-cost.json` | Separate 36-process, 2,916-output cohort; six profiles and both backends | Optional arithmetic products still charge conversions/private selection; not pooled with older default-backend cohorts |

These paths are under `docs/evidence/`. Full-MLP projections use the explicitly
hypothetical geometry `24*4864*(39+8-1)=5,369,856` gated elements. For example,
the historical uncompressed Curl projection charges 213.08 GB both-party keys
and 971.27 MB peer frames; LogRow charges 12.89 GB client encodings and 326.40 GB
evaluator material, including 242.29 GB multiplication; four-piece Compact charges
1.971 TB evaluator material plus 13.40 GB client encodings. Decimal units,
boundary delivery, public metadata and model-independent source/cohort identities
remain as documented in their original records. These are **projections of the
implemented bounded layouts**, not protected checkpoint executions, actual WAN
traffic, proof that the original papers are infeasible, or client RSS bounds.

The separate clear-checkpoint numeric diagnostics do not preserve complete logits
or active KV for any of these changed profiles. The retained Curl/uncompressed
LogRow diagnostic has 10/12 prefill, 35/36 same-token decode selections and 10/12
eight-token trajectories, while all candidate full-logit/KV exactness checks fail.
Its unchanged control passes 48/48 logit/KV positions. The separate Compact
diagnostic likewise passes the fresh-runtime control but has no candidate exact
logit/KV positions. The newer mixed cohort uses different held-out prompts and
must not replace those historical results silently.

One explicit source drift remains in the historical Curl quality record:
`python/pllm/metrics/wavelet_gated.py` was recorded as
`30d2d08550a9f20f64ead0addf00354d54574859457aea1164feadc2ef29d8fd`, while the
audited checkout/installed SDK file is
`495a3b6251c4589d7e5e58429497ba7c432a06c2781818813ce0110ccf835ab6`.
The record and test intentionally retain their historical scope; a passing
evidence-conservation test does not turn this into a current-source quality run.

### 4.1 Historical five-family supplement: outcomes and conservation

The installed interpreter, all 355 wheel-package file hashes, wheel/native hashes
and `BuildRef` digest match original v4; no reinstall occurs. The 632 repository
build inputs still match the local source/build receipt. Additional test,
primary-PDF and historical-record locks are captured before/after each supplement.
TenSEAL 0.3.17 and the exact `_sealapi_cpp` binary/code files are separately
hashed in `crypto-dependency.json`. Native resource admission remains active.
Every subprocess is serial, requires 4 GiB available beyond the existing host
reserve, and runs the existing 250 ms live memory watchdog. Both new execution
directories record **zero observed host swap growth**, no watchdog trip, and
the original pre-existing host swap separately; swap is not counted as capacity.

| Current test source subset | Passed | Failed | Scope |
| --- | ---: | ---: | --- |
| Native Freivalds + generated verified SDK/gateway | 7 | 0 | Independent small integer products, rejection/retirement, actual request paths |
| Public equalization | 7 | 1 | Functional/source-bound tests pass; strict aggregate download-body equality fails |
| Selected masked-wire SDK tests | 3 | 0 | Complete fused accounting, fresh issuance/oracle corruption, replay contract |
| Selected embedding/conversion tests | 8 | 0 | Carry/sign conversion, admission, malformed codec/foreign state, decoder rejection |
| Encrypted softmax pre-key rejection | 1 | 0 | Invalid range/geometry cannot reach setup |
| **Total distinct pytest nodes in this supplement** | **26** | **1** | **No skipped tests; no native Rust suite rerun** |

`2026-10-16-six-families-v1/manifest.json` is intentionally **failed** because of
that one test. The seven successful operator/replay pairs in that directory are
not relabeled as a passing entire run. Their initial/replay counts are 16/16
masked-wire integer outputs, 21/21 embedding integer outputs, and 4/4 approximate
softmax outputs. The follow-up adds 21/21 embedding integer outputs at the
three-polynomial geometry, plus a deterministic framing diagnostic. Its passing
manifest does not rerun or supersede the failed test. Equalization's six-scale/
12-output real-valued algebra check is another separate result, not 18 pytest
cases. v4's 29 tests and its whole-gated operator outputs retain their own counts.

Raw reports preserve every complete configuration, numerical identity, per-party
owner, setup/online/conversion frame subtotal, observed error, diagnostic timing
and limitations. `2026-10-16-family-source-review-v1/check-ledger.json` derives
the exact node outcomes from JUnit and retains both initial and replay qualities.
Timings are incidental single-call diagnostics; no fresh performance ranking,
cohort, full external-system reproduction or complete-client memory conclusion
is drawn. The source review records exact `pdftotext -f/-l` commands and excerpts
from nine hash-checked local PDFs, resolving LogRow and correcting the NEXUS
equation locator without changing source/runtime citation objects.

Replay commands (output directories must be new):

```sh
target/frontier-sdk-env/bin/python -I paper/artifacts/conformance/audit-family-cards.py \
  --output paper/artifacts/conformance/families-replay-new

target/frontier-sdk-env/bin/python -I paper/artifacts/conformance/audit-family-cards.py \
  --output paper/artifacts/conformance/families-followup-new \
  --only equalization-framing embedding-active-none embedding-active-interleave embedding-active-pack_lwes
```

The current driver also includes the follow-up cases in its default selection;
each archive keeps the exact driver snapshot/argv that produced its own result.
That original traffic-equality assertion could pass or fail as variable-width
timing metadata changed; the repaired fresh execution does not erase its failure.

### 4.2 Fresh clean-wheel family evidence

`2026-10-16-families-clean-v1` passes all nineteen manifest cases, using the exact
clean-local-v5 identities in §1. JUnit nodes independently establish:

| Card/check | Passed tests | Fresh bounded execution |
| --- | ---: | --- |
| F1 Slalom-derived verification | 7 | Six native-boundary checks; generated verified SDK/gateway |
| F2 public equalization | 8 | Profile/source binding, numerical and prepared SDK/gateway checks; repaired framing assertion |
| F2 accounting controls | 19 | Actual encoded clock lengths; reject geometry, payload, other metadata, frame and accounting changes |
| F4 SIGMA/FuseFSS helpers | 3 | Two layouts, each eight exact outputs plus fresh replay |
| F5 FastQuery/BumbleBee helpers | 8 | Three small-table and three active-packing cases, seven exact outputs each plus fresh replay |
| F6 NEXUS/THOR softmax | 1 | Two width-two CKKS profiles plus fresh replay, both within declared tolerance |
| **This run only** | **46** | **Zero failed/skipped tests; no external-system reproduction** |

The twelve non-pytest functional cases comprise ten operator/fresh-replay pairs,
the independent six-scale/twelve-output equalization algebra and the separate
timing-framing mechanism. Operator pairs check **116 exact integer outputs and
eight approximate CKKS outputs** in total, with stable per-variant numeric
contracts/resources. Initial and fresh-replay reports are both retained. The
remaining nineteenth case verifies 26 child lifecycles. All **671** source/evidence
hashes match before/after. Existing bounded admission and serial host-guard records
remain part of each command record; these are functional checks, not new cost cohorts.

The fresh main run separately passes C1–C3's nine admission/evidence tests and four
complete-operator cases (Curl levels 0/2, LogRow, Compact), checking 32 initial plus
32 replay outputs. Only initial complete-gated-operator reports are serialized;
the second report's equality is an execution assertion. Historical cost/quality
records and the old failed equalization cohort remain unchanged.

Both exact invocations, complete child records and raw outcomes are in the new
manifests. The read-only archive verifier checks hashes, JUnit counts, actual
child records/roles/exits, saved FP32 errors, and the separate current-build link.

## 5. Remaining gates

- P1-B's generated-fixture extension, independent conformance, rejection,
  retirement, ordinary benchmark and published-example SDK/gateway functional
   outcomes now have **fresh provider-verified clean-wheel execution**. Historical
   provider-child identity remains unknown. Listing 1's parent-owned real-Qwen
  artifact remains separate; neither resolution nor a generated fixture replaces it.
- P1-C's six-family implementation/source/deviation/oracle/accounting cards are
  now present, including LogRow's precise original locator. Current evidence
   supports the explicitly bounded adaptations. Repaired exact non-timing
   equalization accounting passes; universal total-byte equality and retrospective
   attribution of the original failure remain unestablished. Optimized headline LogRow costs,
  protected full-model FSS/embedding/CKKS execution and external-system reproduction
  are not implied. The primary review remains its own record rather than an
  upstream implementation or independent cryptographic proof.
- A new clean-target local build and fresh-environment execution now supplement
   the old incremental receipt. Neither supplies third-party/CI attestation or
   hermetic/independently operated replay. No whole-client tier qualification,
  independently operated deployment, composable privacy proof, contemporary
  competitor defeat, or fresh matched real-model campaign is supplied.
- P1-E still needs a public immutable wheel/source/dependency archive and clean
  independent replay. Local `target/` paths alone are not publication artifacts.
