# Checkpoint-specific tiny helper: executable first pilot

## Decision / present status

**Parent completed one real-checkpoint-guided helper fit; heldout screen failed.**
The 264-parameter extractive helper is trained but not demonstrated useful:
heldout QA is only 1/4, and fallback-inclusive projected arithmetic is 1.323916×
full input. No standalone replacement, serving improvement or memory-tier
qualification. Changing the method requires newly frozen held-out data; these
scores cannot be used to retune thresholds.

Owned implementation:

- `benchmarks/research/checkpoint_tiny_model.py`: corpus/source locks, sentence
  candidates, features, pairwise training, readable bound checkpoint API, costs.
- `scripts/train_checkpoint_tiny_model.py`: prepare-before-fit manifest, offline
  source preflight, real checkpoint teacher, fresh causal evaluation and retry.
- `tests/test_checkpoint_tiny_model.py`: independent behavioral gates.
- `docs/evidence/checkpoint-tiny-model-mechanics-r5.json`: current synthetic
  evidence only; initial mechanics artifact retained separately.

This is **not a standalone distilled replacement or universal learned encoder**.
The local learned component selects source-exact sentence subsets for one pinned
target. Full target supplies language generation. Its weights stay provider-owned
in the proposed serving placement; the local-clear research teacher temporarily
owns the whole target and reports its process peak separately. A target above 1B
does not invalidate a tiny helper, but moving that target into the serving client
would need a new complete local-ownership and parameter contract.

## Why this construction first

| Choice | Cold artifact / local compute | Target work / fallback | Decision |
| --- | --- | --- | --- |
| Learned extraction | 264 float64 parameters = 2,112 raw weight bytes; actual JSON artifact bytes measured, feature/tokenizer/target client boundary retained | Ordinary target receives fewer **changed** input tokens; full target decode still charged. Failed compressed attempt plus fresh full retry charged | Completed pilot: **heldout no-go**, fallback costs exceed full input |
| Latent compressor | Previous ~5.4M-parameter design could fit weight budget; still requires encoder work, target vocabulary/head, latent buffer, calibration and compatible decoder | New latent positions/KV and differentiation through frozen target; all failed attempts and full-prefill retry remain | Blocked new trained boundary/quality, not implemented here |
| Existing checkpoint drafter | SmolLM135M weights, tokenizer and state in addition to target boundary; independent import/draft/rollback cost | Prior exact greedy cohort reduced calls but increased rows/bytes and roughly doubled client execution work | Reject as first pilot, not rediscovered as solution |
| Posthoc rank/bottleneck cut | Small basis does not remove original MLP evaluation or residual/KV owners | Prior full-depth cuts lost held-out prefill/decode decisions | Not a trained useful-model result |

Prior evidence: [latent prompt](cold-latent-prompt-reference.md),
[cold request](cold-request-frontier-reference.md),
[wire objective](cold-wire-objective-reference.md),
[drafter](checkpoint-drafter-reference.md),
[pretrained bottleneck](../evidence/pretrained-bottleneck-screen-2026-09-29.json),
[Qwen rank oracle](../evidence/rank-interface-output-oracle-qwen25-2026-09-29.json).
Unlike the pooled histogram toy, retained sentences preserve internal token order,
punctuation and rare strings exactly. Dropped sentences can still destroy answers
or cross-sentence relations; separate copy/order/rare gates must expose that.

## Learned artifact and complete parameter inventory

264-parameter linear ranker, including bias, on eight scalar positional/coverage
features plus 256 signed hashed query-match / **ordered bigram** features. Three
candidate masks retain ceil(number_of_sentences/2) sentences: head, tail and
lexical-query-overlap. Each mask is sorted into source order before concatenation.
All Unicode source spans remain verbatim; tokenizer boundaries can nevertheless
change at joins. No equivalence to original token IDs is asserted.

The helper has **zero frozen backbone, vocabulary embedding or vocabulary head**.
Hash feature generation is fixed code, not hidden learned parameters. Runtime
parameter assertion sums *all* supplied owners and rejects >=1,000,000,000;
tests include adapters whose frozen backbone/head makes the total fail. Current
helper total is 264, not an adapter-only count. Pilot preflight separately counts
every source Safetensors tensor by public shape metadata, including target token
table, head if untied, norms and biases, without loading values. Target-plus-helper
local pilot inventory is retained; target is not secretly a helper backbone.

`Ranker.save(path)` writes readable JSON containing source binding, train-split
digest, full weight vector, complete parameter inventory and integrity digest.
`Ranker.load(path, source_binding)` rejects source substitution, corrupt inventory,
shape changes and nonfinite weights. Different checkpoint => independent teacher
scores, fit and artifact; no cross-checkpoint learned reuse is credited. Shared
algorithm uses no family-named execution branch.

Training: three frozen-target answer-continuation NLLs per public train example;
pairwise logistic rank loss, fixed 80 full-batch updates, learning rate .05 and L2
.001. Ties add no fabricated labels. This is checkpoint-guided supervised
compression, not adapter fine-tuning or a new language-model architecture.

## Public data and frozen evaluation contract

Public source: [SQuAD v1.1 dev data](https://raw.githubusercontent.com/rajpurkar/SQuAD-explorer/240e165ab706d95bd4323653bb92421f446208cf/dataset/dev-v1.1.json),
repository revision `240e165ab706d95bd4323653bb92421f446208cf`, Git blob
`e9a3f913ad1468ebe105b891334ca7b0bc0e2510`, 4,854,279 bytes.
Parent downloaded the locked public file; this worker did not download it. Prepare verifies blob
identity/size and records independent SHA256. Public QA excerpts retain the
SQuAD dataset's attribution to Rajpurkar et al. and CC BY-SA 4.0 license.

This deliberately repurposes official dev data into **custom** train/dev/heldout
cohorts; it is not a standard SQuAD leaderboard result. Teacher pretraining
exposure is unknown, so do not call it contamination-free validation.

Before any fit, assign entire article titles by SHA256 prefix modulo ten:
0–5 train, 6–7 dev, 8–9 heldout. Filter contexts to 120–850 characters and 2–16
sentences, answers <=8 words; select deterministic SHA256-QA-ID order with one QA
per context. Counts: 8 train, 2 dev, 4 heldout public QA. Add two disjoint public
procedural controls per split: unseen parcel-code copying and ordered-route
lookup. Train total 10, dev 4, heldout 6. Explicit cross-split context-duplication
check; no private or online prompt ingestion. Procedural controls alone establish
only mechanics. Tiny public QA sample is a pilot, not representative usefulness.

`--prepare` persists all rows, source hashes, complete policy, support snapshot
and training-code hashes in a digest-bound manifest. `--manifest` refuses changed
code/source. No training hyperparameter or candidate selection uses dev/heldout.
Main process and child both re-derive all rows from the blob-locked public file;
editing a manifest and recalculating its digest cannot inject private examples.
Heldout runs after fitting and checkpoint reload; each generation starts empty KV.
No prompt truncation: >512 native rendered tokens or >16 answer tokens fails.
Gold-answer BPE concatenation must equal actual causal continuation or fails.

Controls: unchanged full prompt, head, tail, lexical and trained selection.
Zero-weight ranker ties to head, giving an initialization control. Each uses
greedy causal generation, up to 24 selected tokens with EOS/truncation visible;
gold answers never force evaluation output. Reports normalized QA exact match,
literal copy/order/rare exact match, answer retention, EOS, actual input counts,
executed rows/calls, elapsed/CPU and independent heldout metrics with sample counts.

Fallback is deployable *only for this declared extractive task*: retry once when
EOS is absent, output is empty, or output is not a literal span of retained text.
No gold-answer/oracle routing. A wrong but present span **will not trigger retry**;
tests explicitly enforce that limitation. Retry executes again, rather than
borrowing the already measured full-control response. Both attempts count.

Predeclared pilot gates, independently on dev and heldout:

- QA EM >=75% and no loss versus full target's QA EM.
- Both order/rare controls literal-exact, all final responses terminate with EOS.
- Learned+fallback projected three-direction arithmetic <=80% of full.
- Separately report whether learned selection beats lexical QA EM or, at equal
  EM, uses fewer projected bytes. Tied lexical performance does not justify
  learning/extra delivery. Four heldout QA cases cannot establish a narrow CI.

Passing these gates would still **not** qualify full cold bytes, usefulness across
general generation, arbitrary sampling, private serving or Low/Medium memory.

## Completed parent pilot: strong heldout no-go

Parent ran the fixed v1 manifest once, serially, with exit status 0 and zero
observed swap growth. Public teacher fitting and causal evaluation completed; thresholds, split,
candidate policy and generation cap were not retuned. Dev passed its tiny screen:
2/2 QA, helper copy/order 2/2, fallback-inclusive ratio 0.794961. **Heldout failed**:

| Heldout choice | QA EM | Copy/order literal exact | Projected three-direction bytes |
| --- | ---: | ---: | ---: |
| Full context | 1/4 | 2/2 | 2,662,871,040 |
| Lexical compression | 1/4 | 2/2 | 2,105,607,168 |
| Learned before fallback | 1/4 | 2/2 | 2,039,872,512 |
| Learned including fresh full fallbacks | 1/4 | 2/2 | 3,525,417,984 |

Fallback-inclusive ratio is **1.3239161533**, a **32.39% increase**, versus the
predeclared <=0.8 gate. QA 25% misses the >=75% quality gate even though it ties
the weak full-target baseline. All heldout outputs ended with EOS; failure cannot
be excused as unfinished decoding. Copy/order success does not establish useful
language behavior. Four QA cases and two procedural controls are tiny denominators:
they support this bounded no-go, not a broad model/compression ranking. The raw
equal-EM byte improvement versus lexical is not a qualified learned-model benefit
once fallback and quality are charged. These are arithmetic projections, not
complete network measurements. Worker lifetime peak RSS was 1,172,668,416 bytes,
including the local-clear target teacher; this does not qualify client serving.

Archived raw artifacts preserve the original bytes:

- `docs/evidence/checkpoint-tiny-model-qwen25-pilot-v1.json`, SHA256
  `f431bf5b83c06ca65ee446ec7e709f020b0e457aeac69993b4b5c58587bc3bcd`.
- `docs/evidence/checkpoint-tiny-model-qwen25-helper-v1.json`, SHA256
  `29392cbbf430f5a072586cdd68cf414ea915c0fd4055a81e396231037c6a051e`.
- `docs/evidence/checkpoint-tiny-model-pilot-manifest-v1.json`, SHA256
  `740ca7839ba611223c7f9a5f784628a7f8f4bd765b8901df321ceb7006cfae0b`.
- Parent status is `docs/evidence/checkpoint-tiny-model-attempt-status-v1.json`;
  the unchanged launcher log is
  `docs/evidence/checkpoint-tiny-model-qwen25-pilot-v1.log.txt`.

Original manifest omitted `scripts/decoder_probe_support.py`. Parent's **external
after-launch verification** recorded SHA256
`6b102bd674403a431e898ef1d6d411e74a663f063555e021ac56a1159a8a0d3e`,
matching HEAD during the run and unchanged afterward. Do not describe this as an
original manifest commitment. Pre-edit source snapshots live in
`docs/evidence/sota-breakthrough-source-snapshots/` for the driver, research module
and decoder helper. New code below is a future hardening change, not the code
used for v1. Literal completed-result facts in the support inventory bind raw
report/helper/manifest hashes, never hashes of this module or its own evidence;
there is no circular source/evidence hash.

## Future driver hardening (not a new training experiment)

New manifests lock **four** sources: driver, research module,
`scripts/decoder_probe_support.py` and `scripts/frontier_probe_support.py` (shared
exclusive lock/host-reserve/swap/timeout runner). Exact dependency set and hashes
are checked before admission, in the child before work and again after it exits.
Old two-file manifests are rejected rather than silently upgraded.

Before validation/admission, `run_pilot()` exclusively reserves the requested output
as a complete structured **incomplete** JSON document. Snapshots use same-directory
temporary files, file fsync and atomic replacement; creation uses exclusive hard
link so existing output is never overwritten. Size is capped at 1MiB. Completed
train rows, partial current candidate scores, completed eval rows, partial mode
results, phase and helper path/existence/unqualified status survive failures.
The child updates each target pass; the parent preserves its last snapshot if the
guard aborts or 1,800s timeout fires. Catchable failure/interrupt/timeout is explicit,
with bounded error text. SIGKILL/power loss cannot be caught: last committed
snapshot remains visibly incomplete, never successful; incomplete `running` means
completion is unknown, not permission to promote a helper.

Helper creation does not confer qualification. If later evaluation fails, output
still names the existing helper with `serving_qualified=false`. Success replaces
the reserved output atomically using the previous result schema/status/metrics,
adding only `run_lifecycle`; even a completed screen is not serving qualification.
Injected tests cover train-candidate failure, partial eval, a completed row followed
by failure, timeout, cancellation, atomic-write failure, exclusive collision,
bounded snapshots, missing dependencies and success-schema compatibility. No new
tensor fitting or pilot rerun is needed to check this hardening.

## Checkpoint support and source admission

Adapter-level runtime summary: [`model-compatibility.json`](../data/model-compatibility.json).
That summary is **not a full public-checkpoint inventory**. `support_table()`
combines its six real-checkpoint rows with explicit checkpoint-level evidence for
Qwen3-4B; it does not infer every source using an adapter is supported. The seven
recorded real-checkpoint rows below are evidence-bound, not an exhaustive catalog.
Output snapshots summary SHA256, configs/config references, capability requirements,
source records, trained/admission status and explicit extension evidence hashes.

| Exact public checkpoint | Source revision / lock evidence | Tiny status |
| --- | --- | --- |
| Qwen/Qwen2.5-0.5B-Instruct | `7ae557604adf67be50417f59c2c2f167def9a775`; config+weights independently locked in [source lock](../evidence/orthogonal-frontier-qwen25-source-lock.json); tokenizer hashes frozen at prepare | **Trained helper, failed heldout screen**; not standalone or serving-qualified |
| Qwen/Qwen3-0.6B | `c1899de289a04d12100db370d81485cdf75e47ca`; pinned config fixture; runtime numeric gaps remain | Untrained; training tensor/tokenizer lock incomplete |
| Qwen/Qwen3-4B | `1cfa9a7208912126459214e8b04321603b3df60c`; [client memory](qwen3-client-memory.md), [SDK artifacts](qwen3-sdk-artifacts.md), publisher source-lock `308249a17bc5540401ce8ee373e24232036de08601b13e215fbb312943b8fdf1`, checkpoint digest `75a83dc4d370f01ae8e62e060c02d2a16316ef7638f9dfbc3b920be9882bc586` | **Untrained**; helper training remains unmeasured. Original memory-rejected quality attempt and later streamed-reference quality evidence retain their separate scopes |
| HuggingFaceTB/SmolLM2-135M-Instruct | `12fd25f77366fa6b3b4b768ec3050bf629380bac`; config+weights in [bottleneck evidence](../evidence/pretrained-bottleneck-screen-2026-09-29.json) | Untrained; tokenizer lock/pilot not admitted |
| Qwen/Qwen3.5-4B, text only | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`; pinned config fixture, real text runtime | Untrained; complete training source lock absent here |
| microsoft/Phi-4-mini-instruct | `cfbefacb99257ffa30c83adab238a50856ac3083`; pinned config and [runtime evidence](../evidence/phi4mini-reference-quality-2026-09-24.json) | Untrained; complete training source lock absent here |
| google/gemma-4-E2B-it, text only | retained fixture/source prefix `3e22461f`; narrow real runtime evidence | Untrained; full revision and tensor/tokenizer locks required before fit |
| google/gemma-4-E4B-it | config prefix `ee0ef602`, semantic schedule only | **Blocked**, no real checkpoint execution |
| Llama 3.1 8B / generic bias-free fixtures | config-only public mirror and generated tiny runtime, not official checkpoint import | **Blocked**, not real checkpoint support |

Config-only candidate architectures/MoE/multimodal capabilities are not supported
trained models. No “all Qwen/Llama” claim. Larger provider-owned target sizes do
not relax strict helper parameter inventory or client RAM admission. This driver
intentionally implements only one source-admitted pilot, not seven frameworks.

Qwen3-4B source config is independently read as public cached metadata only:
[revision-locked config](https://huggingface.co/Qwen/Qwen3-4B/raw/1cfa9a7208912126459214e8b04321603b3df60c/config.json),
SHA256 `8ba006f74fecfaaeb392872a60f4a480e7ec9860153d2e1b769ec81f9a147f8a`.
Its full config is recorded (hidden2560, 36 layers, Q32/KV8, head128, tied vocab151936).
Extension provenance includes hashes of native SDK evidence, memory/publisher
reports, configuration manifest and the [memory-rejected quality probe](../evidence/research-qwen3-client-memory-quality.json).
Source-lock and checkpoint digests must agree with those reports; unknown complete
training file locks remain blocked. No public Qwen3-4B tensors loaded for this audit.

## Cold cost, identity and privacy

For each attempt, actual consumed sequence rows are prompt_tokens+selected_tokens−1.
Three **projected arithmetic** directions price each stage's actual public wire
width: masked inputs C→I, outputs I→C and reserved corrections P→I. Reservations
charge output-cap rows even when EOS ends early. Full retry adds both attempts,
not only successful full work. Teacher training passes, both target import and all
baseline evaluation work remain in lifetime CPU/RSS; they are not serving savings.

Cold helper addition is actual JSON artifact bytes, plus fixed feature code/import
and bounded local scoring work. Existing target client tokenizer, vocabulary
embedding/head, logits, norms, KV, original prompt retained for fallback and all
runtime buffers **remain**. Local helper training+feature workspace allowance is
4MiB; helper parameter storage is 2,112 raw bytes. These are bounds, not measured
whole-client RSS. No extra local decoder is present. Target provider artifact
delivery must be counted at every owner if provider-cold; provider-cached is a
different denominator. Do not subtract raw target table bytes from compressed
historical network counters or amortize helper delivery over requests in H=1.

Unpriced/blocked serving obligations are explicit in output: common cold target
bundle/tokenizer delivery, optional provider-cold weights, seed requests,
tickets/layout/control/ACK/cancel/TLS, private transport and Preparation CPU/RSS,
correlation renewal, padding, serialization/import buffers. They are **unknown,
not zero**. No complete cost/performance claim until parent measures these with
unchanged placement/empty-cache/output policy. Latent alternative must add latent
and adapter delivery/calibration and compatible KV; drafter must add all rejected
target/draft work and both model owners. No synthetic estimate fills these cells.

Compression changes input IDs, positions and attention history: **no original
logit/KV/sampler identity**. Fresh KV derives from selected sequence only. It is
not safe to splice original-prefix KV or claim arbitrary-sampling equivalence.
Helper stays client-local; selected text, answers, activations and KV must remain
under existing private execution mechanisms. This local-clear teacher issues no
private masks and adds no SDK protocol. Never send raw selected private context
or latents to provider. Production bindings must include source, helper artifact,
positions, numeric plan and fresh one-use material; failure retires material.

## Archived execution contract / future replay reference

V1 is complete and negative. Its manifest requires its archived original sources;
the hardened driver rejects its incomplete dependency set. A future experiment
needs a new manifest and fresh paths. Method changes additionally require a new
frozen held-out cohort; reproducing this pilot does not supply independent quality
evidence.

No install/build/large model download needed. Existing pinned Qwen source and
tokenizer must already be cached; missing artifacts fail closed. Dataset must be
explicitly pre-positioned by parent (4.85MB; no automatic fetch in driver):

```sh
curl --fail --location 'https://raw.githubusercontent.com/rajpurkar/SQuAD-explorer/240e165ab706d95bd4323653bb92421f446208cf/dataset/dev-v1.1.json' --output target/checkpoint-tiny-model-squad-dev-v1.1.json
target/frontier-sdk-env/bin/python scripts/train_checkpoint_tiny_model.py --prepare target/checkpoint-tiny-model-pilot-manifest.json --data target/checkpoint-tiny-model-squad-dev-v1.1.json
target/frontier-sdk-env/bin/python scripts/train_checkpoint_tiny_model.py --manifest target/checkpoint-tiny-model-pilot-manifest.json --artifact target/checkpoint-tiny-model-qwen25-helper.json --output target/checkpoint-tiny-model-qwen25-pilot.json
```

Use fresh paths; no overwrite. Metadata preparation hashes checkpoint streams but
does not load tensor values. Admission requires **4GiB available after existing
host reserve** (`max(1GiB,total_RAM/8)`), 1GiB free disk and valid host/swap
telemetry. One serial fresh child owns existing frontier lock. Existing watchdog
monitors swap/host reserve and aborts under pressure; child timeout **1,800s**,
four CPU threads, offline environment. No GPU. Teacher is one target, never two
simultaneous checkpoint owners. Exact maximum: 30 teacher-forced training passes,
50 causal baseline/candidate responses, 10 possible fresh fallbacks: **90 target
passes**, <=47,910 sequence rows before model stage expansion. Pilot estimate is
not promised runtime; timeout retains no successful quality claim. An artifact
left by failed evaluation is not a completed pilot and must not be promoted.

Bounded checks (safe without source/data values):

```sh
target/frontier-sdk-env/bin/python -m pytest -q -o pythonpath=. --noconftest tests/test_checkpoint_tiny_model.py
target/frontier-sdk-env/bin/python scripts/train_checkpoint_tiny_model.py --mechanics --output target/checkpoint-tiny-model-mechanics-replay.json
target/frontier-sdk-env/bin/python -m ruff check scripts/train_checkpoint_tiny_model.py benchmarks/research/checkpoint_tiny_model.py tests/test_checkpoint_tiny_model.py
```

Generated integration regression exercises **actual `Teacher`**, not FakeTeacher:
one seeded local 16-hidden/one-layer checkpoint, real engine load, compiler/runtime
`prefix_f32`, two-token forced answer NLL independently recomputed from real logits,
then two identical capped greedy responses with separate sessions. Each session's
final position and KV cache lengths must be zero after return. Only source
resolution/manifest/file hashes are substituted; a transparent session observer
records actual runtimes without replacing numeric execution. No public checkpoint
values, installs or builds. This catches API/identity mismatch but establishes no
language usefulness. Teacher implementation required no API repair.

## Primary sources / what they establish

- [SQuAD paper](https://arxiv.org/abs/1606.05250), public extractive QA source and
  evaluation concept, not proof prompt compression preserves generation.
- [SQuAD source repository](https://github.com/rajpurkar/SQuAD-explorer/tree/240e165ab706d95bd4323653bb92421f446208cf),
  immutable data identity above; metadata source inspection only in this worker.
- [LLMLingua-2](https://arxiv.org/abs/2403.12968), task-agnostic token-classification
  distillation supports extractive compression direction. This independently
  implemented sentence-candidate ranker is **not** LLMLingua-2 or its accuracy.
- [ICAE](https://arxiv.org/abs/2307.06945), [AutoCompressors](https://arxiv.org/abs/2305.14788),
  [Gist tokens](https://arxiv.org/abs/2304.08467): compatible learned context paths,
  not small-independent-encoder guarantees for unchanged target. Their frozen
  backbone still counts when local. See prior latent reference for limits.
- [Pinned Qwen model card](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/tree/7ae557604adf67be50417f59c2c2f167def9a775),
  public target source; new helper uses no provider-private training prompts.

Nothing here establishes literature novelty or a useful real tiny model. The
single admitted checkpoint-guided fit is complete and negative; remaining
checkpoint-specific helpers are untrained. Retain this no-go rather than retuning
the gate, omitting fallback or promoting the helper as a standalone language model.
