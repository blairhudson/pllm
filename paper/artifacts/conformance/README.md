# Paper 1 generated conformance archive

Accepted fresh run: **`2026-10-16-clean-local-v1`**, all ten audit cases passed,
29 focused tests and 31 complete child identities. Fresh family supplement:
**`2026-10-16-families-clean-v1`**, 46 tests and twelve operator/algebra/framing
cases passed, with 26 child identities. Both bind the clean-local-v5 wheel below.
Historical `2026-10-16-installed-v4` remains a separate seven-case passing
functional record; its provider-child import/native identities remain **unknown**.
The [historical qualification](PROVIDER-CHILD-PROVENANCE.md) is not retroactively changed.
Interpretation and implementation cards: [`../../IMPLEMENTATION-EVIDENCE.md`](../../IMPLEMENTATION-EVIDENCE.md).
This archive contains generated fixtures and bounded component checks, not real
Qwen-checkpoint performance, client-tier eligibility, or a publication release.

## Replay

From the repository root, with the already installed interpreters and wheel used
in this audit:

```sh
paper/artifacts/clean-build/2026-10-16-local-v5/work/venv/bin/python -I scripts/audit_paper_conformance.py \
  --output paper/artifacts/conformance/replay-new \
  --wheel paper/artifacts/clean-build/2026-10-16-local-v5/wheels/pllm_run-0.1.0a2-cp311-abi3-macosx_11_0_arm64.whl \
  --reference-python "$PWD/.venv/bin/python"
```

Output must not exist. No installation, download, native build, heavy checkpoint,
or external service is performed. Existing wheel environment needs PLLM, pytest,
Torch and normal runtime dependencies. Existing independent-reference environment
needs Torch and Transformers. The **current** drivers additionally enforce child
`-I`, scrub import/loader variables and capture package/native identities before
and after nested Python execution. Original snapshots lack these child receipts;
controller `-I` and pytest's `-o pythonpath=` alone did not establish child identity.
Low-memory generated
checks and local provider children run serially. The ordinary benchmark admits
under a 2 GiB conservative topology budget, not a measured client-memory claim.

On another machine, use an existing installed wheel's interpreter, its matching
wheel archive, and an existing reference interpreter. The script verifies every
installed wheel package file. Different platform/dependencies produce a different
`BuildRef` identity; do not label that execution the same build. The version
string alone is insufficient. Timings, random material, temporary paths and
request IDs vary; replay asserts numerical/resource contracts rather than
byte-identical new report files.

Exact accepted invocation and every subprocess command appear in
`2026-10-16-clean-local-v1/manifest.json` and `*.command.json`. The unchanged
published command executed by the driver is:

```sh
paper/artifacts/clean-build/2026-10-16-local-v5/work/venv/bin/python -I examples/prepared_sdk.py \
  --root NEW_TEMPORARY_DIRECTORY --prompt 'Public generated fixture'
```

The driver separately runs that example's `build_experiment` through SDK and
HTTP `/v1/responses` gateway with the explicit model identifier. HTTP gateway
transport is in-process ASGI; provider child processes use real local TCP.

## What is frozen

- `installed-build.json`, `provenance.json`: wheel/package/native/dependency and
  environment hashes; installed SDK versus checkout comparison. Native binary
  differs from checkout's editable extension. Later local build evidence is a
  separate supplement, not a rewrite of this original execution provenance.
- `driver.py`: exact driver snapshot for that attempt. Archival snapshots use the
  original repository-relative layout; run the accepted matching script under
  `scripts/`, rather than moving a snapshot and assuming its root is unchanged.
- `source-evidence-lock.json`: current implementation/test/source-record hashes,
  local primary-PDF verification and historical cost/quality record hashes.
- `fixtures/`: deterministic generated Qwen2/Qwen3/Llama safetensors and configs.
  These are public synthetic weights, not downloaded checkpoints.
- `reference-*.npz`, associated environment records, `candidate-*.npz`,
  `generated-conformance.json`: upstream FP32 and tiled W8A8 logits/KV,
  bindings, numeric errors and bit-exact comparison counts.
- `tiling.*`, `binding.*`, `operators.*`: 13 + 7 + 9 current test cases run against
  the installed implementation; logs and JUnit results. Not a complete test suite.
- `prepared-sdk.log`, `published-requests.json`,
  `published-generated-experiment.json`: unchanged example execution and
  generated SDK/gateway replay. Paper 1's real-Qwen Listing 1 is not executed here.
- `ordinary-benchmark.json`: one generated ordinary benchmark response. Its
  runtime oracle is not the independent-reference oracle.
- `operator-replay.json`: four bounded whole gated-operator reports, eight lanes
  each plus fresh replay. Local timing is diagnostic, not matched system cost.

`index.json` locks every file in its explicitly catalogued completed archives except
itself; it does not discover or index unrelated/in-progress directories. Generated
`__pycache__`/bytecode caches are excluded. Its run entries preserve the original
disposition. Byte integrity can be checked without importing PLLM:

```sh
python3 paper/artifacts/conformance/verify.py
```

The fresh wheel/build receipt are archived locally under `../clean-build/`;
installed dependencies and caches are local. Public immutable distribution/build
attestation and independent clean-machine replay remain publication gate P1-E.

## Fresh clean-build execution link

`2026-10-16-clean-build-links-v1/receipt.json` binds both passing fresh executions
to the actual clean-build receipt and verifies the actual wheel/package payload:

| Identity | SHA-256 |
| --- | --- |
| `../clean-build/2026-10-16-local-v5/receipt.json` | `4d274918781c78e3f0b4ebfdaef1501e2d19e741ee09e6847ba4e3ba665b243e` |
| Wheel | `1811d2d9d5c6cf1360091122a62fd9678031159acab1ab2a278b13b7872d2e8c` |
| Native | `826c2f41612912c38a2cd902a1988b50a069b4e278f1f05ffdb6980744e42c64` |
| Runtime `BuildRef` | `8a8803783aa2b61f84ac461c209925f911acbc447dc86bdc3120ee384d4c83d9` |

Independent read-only build verification passed **127 artifacts / 26,231 installed
distribution files**. The actual 355-file package matches both executions and
their child start/end identities; both runtime BuildRefs are canonically rehashed.
Fresh main records **31 children: 12 inference + 12 preparation + 7 others**.
Fresh families records **26: 4 inference + 4 preparation + 18 others**. All retain
launch, start, verified end and observed-exit records. Provider SIGTERM exits are
−15. The clean target starts empty and disables incremental reuse; registry sources,
toolchain, OS SDK and Python wheel cache are reused. This remains local, non-hermetic
evidence without CI/third-party attestation.

The fresh family raw field `local_build_receipt_sha256` names the **prior** cached
receipt `b7d332…`; its source lock explicitly calls this `prior_build_receipt_sha256`.
It does not identify the builder of this binary. The new link explicitly verifies
`4d2749…` instead; `identity_equals_original_v4` remains false for every identity
field. The original manifest and build receipt's execution-pending status remain
unchanged. The naming ambiguity does not block acceptance of separately bound
fresh functional evidence.

Fresh main independently repeats **29,592** generated tiled/untiled value checks,
29 focused tests, the generated 44+8-token SDK/gateway request, one ordinary
response, and four complete gated-operator checks (32 initial + 32 replay outputs).
Saved candidate/reference arrays reproduce all recorded FP32 errors read-only;
untiled arrays and second complete-gated-operator reports were not separately saved.
Their equality/replay assertions are witnessed by the frozen execution driver.
These checks do not rerun Listing 1's separately archived real checkpoint.

`accepted_run` and `installed_build_digest` in the index intentionally preserve the
historical v4 anchor used by old supplements. `accepted_fresh_run` and
`accepted_fresh_family_supplement` select the separately bound new executions.

## Local build/source supplement

`2026-10-16-local-build-v1/receipt.json` records a successful **locked, offline,
incremental local** Maturin build using existing Cargo caches and two build jobs.
It preserves the older 556-file source lock/build log/wheel lock, verifies that
lock against the current tree, and hashes 632 repository build-input files plus
4,262 resolved dependency-source files before and after. No inputs changed.
Compiler/tool versions, tool binary hashes, Cargo metadata, environment, exact
commands, full build log and complete wheel/package inventories are retained.
Cargo reports a 0.06-second release check; this is not compilation from a clean
cache, a hermetic build, or third-party/CI attestation.

All 355 `pllm/` package files, including native binary, match v4's wheel and
installed package byte-for-byte. The new full wheel hash is
`293578aea896ac7d5b4197f0bf7ef79949f70d8161c2d0b4d87566a642f789f4`.
Full wheel payload equality is **false**: SBOM timestamp/UUID and its `RECORD`
hash differ. `payload-comparison.json` and archived old/new SBOM/RECORD files
show the exact metadata-only delta. The existing environment was not reinstalled
and v4 is not relabeled as execution of this new complete wheel. This cached wheel
was not subsequently replayed; the fresh clean-v5 wheel above is a distinct output.
Independent replay and public immutable archival remain open.

To repeat this source/build audit, use **new** target and receipt directories:

```sh
target/frontier-sdk-env/bin/python -I paper/artifacts/conformance/build-source-receipt.py \
  --build-root target/paper-publication-native-build-replay \
  --receipt paper/artifacts/conformance/local-build-replay \
  --historical-build target/certified-head-sdk-build-v3
```

The recorded native command was:

```sh
CARGO_BUILD_JOBS=2 CARGO_NET_OFFLINE=true \
  target/frontier-sdk-env/bin/python -m maturin build --release --locked --offline \
  --out "$PWD/target/paper-publication-native-build-v1/wheels"
```

Receipt scope is local source/output identity; original execution directories
remain immutable. `prior-archive-index.json` preserves the index before this
supplement. The current verifier checks supplemental hash manifests as well as
the four original and three newly completed conformance execution manifests.

## Six-family implementation supplement

`2026-10-16-six-families-v1/` adds installed-wheel functional checks for Slalom
verification, public SmoothQuant-style equalization, masked-wire SIGMA/FuseFSS,
BFV embedding/repacking and CKKS softmax. Existing v4 and local-build files are
unchanged. The new provenance verifies the same wheel, native binary, BuildRef,
all 355 package files, and the 632-file build input lock; SEAL/TenSEAL code and
binary hashes are additionally archived. No dependencies were installed.

**Observed suite: 26 passed, one failed, zero skipped.** The overall manifest is
therefore **failed**, even though all seven initial operator/fresh-replay pairs
and the independent equalization algebra passed. The failing equalization test
expects equal total stage response bytes; it observed 16,289 versus 16,291, with
identical four stage calls and 11,580 upload bytes. No test/runtime was changed.
JUnit/logs retain the exact assertion and every selected node; counts are not
added to original v4's 29 tests or historical cohorts.

`2026-10-16-family-followup-v1/` **passed** four diagnostic cases. A deterministic
response-framing control demonstrates that changing only MessagePack `server_ns`
from 65,535 to 65,536 adds two bytes. It does not retroactively identify fields in
the original failing frames, which were not captured. Three 4,096×7 public-table
embedding controls then exercise three response polynomials: interleave and
PackLWEs each execute 43 automorphisms, pass exact independent row oracles and
fresh replay, and preserve the unrepacked control's numerical identity. This
closes the first 17×7 interleave check's zero-automorphism geometry limitation.
It is 28,672 scalar public codes, not a model checkpoint.

Every functional subprocess runs serially with the existing native allocation
admission plus at least 4 GiB host headroom beyond the existing reserve and a
250 ms live watchdog. Maximum observed swap growth is zero in both directories;
pre-existing swap is recorded and does not enlarge admission. CKKS uses its
existing 2 GiB operator ceiling, BFV 128 MiB, and masked-wire 64 MiB. These are
bounded functionality/resource-conservation checks, not performance or peak-RSS
measurements. Incidental operator timings remain labeled diagnostic.

Exact initial command (exit 1; retained failure):

```sh
target/frontier-sdk-env/bin/python -I paper/artifacts/conformance/audit-family-cards.py \
  --output paper/artifacts/conformance/2026-10-16-six-families-v1
```

Exact follow-up command (exit 0):

```sh
target/frontier-sdk-env/bin/python -I paper/artifacts/conformance/audit-family-cards.py \
  --output paper/artifacts/conformance/2026-10-16-family-followup-v1 \
  --only equalization-framing embedding-active-none embedding-active-interleave embedding-active-pack_lwes
```

For replay substitute **new** output directories. The current driver includes
the follow-up cases in its default selection; each existing directory freezes
its own exact driver snapshot and invocation. A future pass of the variable-size
traffic assertion must not erase the observed failed attempt.
New family runs may select `--wheel PATH` with that wheel's installed interpreter;
their new identity and explicit comparison to v4 are recorded without relabeling v4.

`2026-10-16-family-source-review-v1/` contains nine local primary-PDF source
records, page-selected extracts, extraction commands/tool hash and a JUnit-derived
outcome ledger. LogRow is mapped precisely to §3.4/Figure 1, §§4.1–4.2,
§5/Construction 1/Figures 3–5 and §6/Theorem 1. Its retained two-row one-hot setup
and the paper's inconsistent `2n-1` versus `n-1` setup accounting are explicit;
the implemented half-hidden-function step does use Figure 5 row reduction.
NEXUS Equation (4) is normalized exponentiation; repeated squaring is the
following unnumbered formula, followed by Algorithm 4. Source/runtime citation
objects remain unchanged. Cached versus final BumbleBee versions stay distinct.

```sh
.venv/bin/python paper/artifacts/conformance/review-family-sources.py \
  --output paper/artifacts/conformance/NEW-family-source-review
```

The primary extraction script reads existing local PDFs only; no novelty search,
upstream code execution, external-system reproduction or new model-quality result
is claimed. See the six implementation cards and exact limitations in
`../../IMPLEMENTATION-EVIDENCE.md`.

## Provider-child provenance follow-up

[`PROVIDER-CHILD-PROVENANCE.md`](PROVIDER-CHILD-PROVENANCE.md) supplies the exact
historical enforcement matrix, new research-only launch policy, focused command,
and minimal principled equalization accounting fix. `2026-10-16-child-isolation-v2`
passes seven import/framing checks with three captured Python processes and zero
provider/model/operator runs. The first parser-error attempt is retained as
failed in `2026-10-16-child-isolation-v1`.

Original v4/family outcomes stay frozen, with provider-child identity explicitly
unknown. Those probes validate instrumentation only. The separately archived fresh
SDK/gateway and family runs above now provide actual provider start/end receipts. The
original equalization failure remains unresolved because its frames are absent.

`2026-10-16-child-isolation-v3` adds **13 passing focused shutdown/isolation
checks**. Seven positive children retain verified package/native start/end and
observed-exit receipts. Four no-model Uvicorn fixture processes preserve raw
control behavior: SIGTERM exits −15 and SIGINT exits −2, with lifespan shutdown
and complete process-group cleanup. A nested custom-handler control retains exit
17. Forced SIGKILL, snapshot timeout and changed-native controls keep separate
failed identity summaries; no missing receipt is waived.

The current research helper uses a two-second final-snapshot budget, exclusive
atomic publication and original default-signal re-raise. Historical failures stay
frozen; no PLLM model/provider or whole-conformance replay was run for this fix.
The exact accepted command and signal/exit semantics are documented in
`PROVIDER-CHILD-PROVENANCE.md`. New replay still needs a new output directory and
an interpreter whose installation matches the supplied wheel.

## Fresh family supplement

`2026-10-16-families-clean-v1` passes **46 tests in six JUnit XML files**:
Slalom 7, equalization 8, accounting controls 19, masked-wire 3, embedding 8,
encrypted-softmax admission 1. Twelve additional cases comprise ten operator/replay
pairs plus independent equalization algebra and deterministic timing framing.
Pairs check **116 exact integer outputs and eight approximate CKKS outputs**;
initial and fresh-replay reports retain their separate results/resources. The
nineteenth manifest case checks all 26 child identities. All 671 source/evidence
hashes are stable before/after.

Eight actual equalization response frames and their accounting record are retained
under `equalization-frames/`. Each numerical choice records 4 calls, 11,580 upload
bytes and 16,293 download bytes. Exact arithmetic payload/geometry and non-timing
framing are compared after validating fresh IDs and preserving masked contents;
actual encoded `server_ns` lengths reconcile every download byte. Nineteen controls
reject unrelated byte/geometry/accounting changes. The original historical failure
remains unresolved: its frames were never retained. This new repaired pass does not
retroactively attribute its two-byte difference.

Replay, into a new output directory, using the already installed matching wheel:

```sh
paper/artifacts/clean-build/2026-10-16-local-v5/work/venv/bin/python -I paper/artifacts/conformance/audit-family-cards.py \
  --output paper/artifacts/conformance/family-replay-new \
  --wheel paper/artifacts/clean-build/2026-10-16-local-v5/wheels/pllm_run-0.1.0a2-cp311-abi3-macosx_11_0_arm64.whl
```

## Retained attempts and dispositions

| Attempt | Disposition |
| --- | --- |
| `2026-10-16-installed-v1` | Existing older `mlp-tiling-venv` wheel lacked public `MlpPrefillTiling`; no tiling tests collected. Driver also mishandled a historical scalar source hash and omitted required gateway model ID. Unchanged SDK example and bounded operator replay succeeded. This wheel has 100 SDK file differences from checkout. |
| `2026-10-16-installed-v2` | Correct installed wheel; all 29 tests, published SDK/gateway and operator checks passed. Independent-reference driver used tokenwise `forward_ids`, so asserted batch tile count failed. Driver error, not a numerical mismatch. |
| `2026-10-16-installed-v3` | Batched prefill exercised; driver incorrectly expected every prompt-position logit from a graph that emits final prediction logits. Ordinary request benchmark rejected 1 GiB topology budget before execution (required estimate 1,440,316,095 bytes). Other checks passed. |
| `2026-10-16-installed-v4` | Uses batched prefill, compares complete vocabulary at each emitted prediction, retains complete active KV, and uses 2 GiB benchmark admission. All checks passed. |
| `2026-10-16-fresh-release-v1` | Failed provenance before functional checks: minimal SDK environment lacked Torch distribution metadata. |
| `2026-10-16-fresh-release-v2` | Two cases failed: Torch missing for generated fixture and pytest collection (zero tests run, one collection error). SDK/gateway, ordinary response, four bounded operators and 11 child identities passed. Distinct wheel `3b63ae30…`, native `cb607aa0…`, BuildRef `e82b14e6…`; not the clean-v5 binary. |
| `2026-10-16-clean-local-v1` | All ten cases passed on clean-v5's full environment; 29 focused tests and 31 complete child identities. |

The driver makes Torch optional only for SDK provenance metadata. Fixture generation
and the independent Torch/Transformers reference still require it; neither failed
minimal-environment attempt is relabeled as a passing conformance run.

Earlier results are not silently overwritten, pooled as replicates, or used to
hide implementation/source mismatches. Failed driver attempts have their exact
source snapshots and tracebacks. v1's `driver.py` snapshot was archived after its
manifest was written, so the outer index also binds that additional file.
