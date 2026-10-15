# Four attempts to move the serving frontier

This investigation follows the [Low/Medium metric analysis](metric-relationships-reference.md).
Its four tracks target mechanisms behind the measured tradeoffs: full-vocabulary
reply delivery, serial cross-stage execution, fresh linear corrections, and the
number of prompt rows that require private computation. The
[investigation contract](../plans/sota-breakthrough-four.md) records worker
ownership, source constraints and the checkpoint-specific training requirement.

## Comparison rules

The existing matched cohorts are controls for subsequent serving experiments,
not samples in this new research screen. In particular, adding a boundary worker,
changing the input representation, or training a helper does not preserve the
original topology, numeric identity or quality evidence automatically.

| Track | First independently checkable gate | Subsequent serving gate |
| --- | --- | --- |
| Smaller exact head replies | Actual public row certificates, complete integer reconstruction, layout delivery and two-worker bodies | Full logits/KV/sampler parity; matched parallel/residue control; complete client and role costs |
| Cross-stage MLP transport | Real bounded channel, exact arithmetic, fixed ordering, cancellation and row conservation | Compiler-bound generated decoder parity, then full/tiled/conveyor comparisons at 150 and 288 input rows |
| Fixed-W correlations | Explicit party views and setup/query construction for arbitrary small W; full renewal/expansion bill | Independent security review, source-conditioned material and live one-use transport before decoder measurements |
| Checkpoint-specific tiny helper | Complete runtime parameter count, frozen train/evaluation split, useful held-out behavior and complete artifact costs | Supported source-specific model interface, changed-contract quality, cold private serving and fallback accounting |

A smaller reply or owner is a mechanism result. Lower whole-client RSS requires
fresh complete clients; faster full responses require matched lifecycle timing.
All seven priority metrics remain visible even when one was the primary target.
Low/Medium eligibility uses the **largest complete-client lifetime peak across
all replicates**, with RSS windows preserved. The helper, failed attempt,
fallback and overlapped work remain in that owner.

Before measurements, each arm freezes source/numeric identity, workload, build,
topology, lifecycle and the per-party checkpoint/tokenizer/artifact cache
inventory. Historical source-present results cannot supply unknown source-cold
costs. A new nondominated tradeoff can extend the frontier without dominating
its control. Domination requires no worse values on every comparable objective
and a strict improvement on at least one; missing costs prevent that claim.
Changed topology retains its distinct trust contract and topology-local leaders.
An exploratory serving winner still needs independent confirmation.

The read-only prepared-incumbent reference is
[`private-boundary-tiers-qwen25-2026-10-06.json`](../evidence/private-boundary-tiers-qwen25-2026-10-06.json),
SHA-256 `0bc6061e24f41fdd12ae60c67c9e313ad8551fbefdb36e92bceadfd88e12e1ca`.
Its seven prepared configurations qualify at Low under their original
confirmation cohort. Replay retains all nine available serving metrics:

```sh
node docs/scripts/research-relationships.mjs \
  --cohort qwen25-private-boundary-tiers-150-8 --memory low --topology Prepared
```

The command was replayed for this investigation; these historical samples do
not replace fresh matched controls for a new implementation.

## Checkpoint-specific training

The learned model may differ for every supported public checkpoint. Its complete
runtime model must contain fewer than one billion parameters; a frozen large
encoder with a small trainable adapter does not meet that requirement. A small
helper beside a provider-owned target and a standalone replacement are distinct
designs. The target's original generation contract cannot be attributed to the
replacement or to a compressed prompt without held-out evidence.

Shared tooling should discover source dimensions, tokenizer and supported
interfaces from locked metadata. Learned weights, training data provenance and
evaluation remain checkpoint-specific. Semantic lowering alone is insufficient
to mark a checkpoint trainable or deployable. A bounded pilot is evidence for
that checkpoint and task, not completion of the remaining support matrix.

Cold evaluation charges helper/tokenizer delivery, local loading and compute,
protected ingress, remaining target computation, state, sampling and any failed
compressed attempt plus fallback. Public fitting and teacher computation are
additional creation costs. Copying, rare entities, ordering and causal continuation
need explicit disjoint checks alongside useful task quality.

The primary task metric and numeric acceptance margin must be recorded before
fitting. Baselines include full original input and zero-training compression.
If the target is adapted, compare **full original, full adapted and compressed
adapted** execution, so adaptation error cannot disappear into a compression
claim. A helper acting only on original input does not silently activate an
adapted model.

## How to prioritize the next experiment

The mechanism has to match the objective's window. Reducing one decode exchange
can be valuable for stream speed yet have little leverage over a short cold
request dominated by public delivery and prefill. Conversely, reducing prompt
rows can improve cold traffic and first text while leaving per-token head costs
unchanged. The original cohorts cannot be pooled to predict those effects.

Use these decision rules when reviewing each track:

- **Exact head codecs:** compare net bytes after layout delivery and both worker
  replies. A reply-size floor belongs to the stated reconstruction design; it
  does not prove all secure selection protocols require full logits on the wire.
- **Transport overlap:** require an observed overlap interval and conserved
  work/bytes, then inspect the additional simultaneous owners. Elapsed time saved
  in a fixture is not multiplied by decoder layers to manufacture TTFT.
- **Correlation generators:** first check which party can expand which values.
  If an online worker learns its own mask, low seed traffic does not preserve
  input privacy. Correct ideal correlations alone do not implement their setup.
- **Trained helpers:** freeze the task and held-out gate before fitting. A
  compressed input that is cheaper but less useful is a different quality/cost
  point, not an exact optimization. Include a zero-training heuristic so learned
  selection must justify its extra creation and runtime costs.

Surviving tracks need their own matched experiments before combining them.
Composition can change admission, source identities, state and available overlap;
percentage improvements from separate cohorts are never multiplied.

## Evidence levels

This screen keeps four levels distinct: construction/counterexample, synthetic
correctness, measured operator or transport fixture, and complete SDK serving.
No lower level supplies missing measurements at a higher level. Per-worker peaks
and logical allocation bounds do not qualify Low or Medium whole-client memory.

## Completed four-track screen

Four Sol 6.1 workers built bounded references; parent ran checkpoint scanning,
training and transport measurements serially under shared admission. Independent
review corrected kernel-failure retirement, preserved incomplete transport
evidence and reconciled the correlation PRG ledger. These remain separate
construction/operator/quality cells, with **no new serving SOTA or Low/Medium
qualification**.

| Track | Concrete result | Decision |
| --- | --- | --- |
| [Compact full head](compact-private-head-reference.md) | All 151,936 pinned Qwen rows scanned twice; 759,680 native reconstruction checks. Mostly 23-bit instead of 25-bit replies; 8.31% less payload, about 0.598 MB net over eight heads under trusted-layout delivery | Small exact-codec candidate. The 7.88 ms/head upload-floor difference is analytical; trusted thin-client certificates and SDK parity/cost remain open |
| [MLP conveyor](mlp-conveyor-transport-reference.md) | Six fresh 150/288-row v2 workers preserve outputs and 86.14 MB framed bodies, with zero swap growth. Conveyor takes 2.511/4.780 s versus serial 2.653/5.043 s and full 2.550/4.830 s | Fixed-order single samples; no causal speedup. Narrow adjacent-gate write-overlap metric is zero; fixture memory rises. Improve the schedule/measurement gate before SDK integration |
| [Fixed-W correlations](fixed-weight-correlation-gate.md) | 84 exact output checks and exhaustive tiny leakage controls. Zero corrections can preserve input privacy, but require delivering or locally computing the masked image; subset tables move work into expansion | No cheap-client generator. Preserve arbitrary full-ring masks and price masked-image production; do not reinterpret removed corrections as free work |
| [Checkpoint-specific helper](checkpoint-tiny-model-reference.md) | Real Qwen-guided 264-parameter ranker trained from eight public QA items plus two stresses. On four held-out QA items, full/lexical/learned each score 1/4; learned copy/order scores 2/2. Fallback-inclusive projected bodies rise 32.39% | Held-out gate fails. This helper is not a useful standalone LLM or a private-serving result; remaining six recorded checkpoint cells stay untrained |

### Training result and next gate

The source-locked local-clear pilot freezes article-group splits and thresholds
before fitting. Development passes, but held-out evaluation fails its 75% minimum
QA accuracy and at-most-80% projected-body ratio. Learned compression alone uses
2,039,872,512 projected arithmetic bytes versus full input's 2,662,871,040
(23.39% less); failed attempts plus fresh fallback require **3,525,417,984**.
The same 1/4 task score does not establish useful quality. The lexical advantage
flag reflects a smaller pre-fallback estimate, not a successful quality/cost gate.

The 5,251-byte helper artifact has no hidden backbone. Local teacher fitting and
evaluation cost about 695.45 process CPU seconds; 1,172,668,416-byte peak RSS includes
the target checkpoint and is not a serving-client peak. Checkpoint/tokenizer source
was pre-positioned, and all observed swap growth was zero. Private transport,
full cold delivery and broad language quality remain unmeasured. Original source
snapshots and the after-launch numeric-dependency check preserve the pilot's
provenance limitation; future manifests include that dependency from the start.

A next learning experiment should first establish a useful **full-input** task
control on fresh public articles, then compare a richer checkpoint-specific
compressor against lexical selection and its complete fallback bill. Keep the
entire runtime model below one billion parameters, preserve the per-checkpoint
requirement, and freeze new held-out data before changing the method. This failed
four-QA pilot neither validates all supported checkpoints nor rules out trained
model/protocol co-design generally.

### Priority after these gates

1. Exact source-certified head coding is the closest bounded implementation, with
   modest expected benefit and a thin-client verification gate.
2. Conveyor work needs a demonstrably useful schedule before promotion; current
   transport mechanics alone do not justify a decoder-wide speed claim.
3. Fixed-W work still needs a genuinely asymmetric generator with cheap client
   expansion, not a relocated tape or dense client multiply.
4. Tiny-model research needs a useful quality baseline and a stronger learned
   representation. No thresholds are relaxed after this pilot's failure.

Raw evidence and frozen implementation snapshots live under `docs/evidence`;
the linked track references contain exact commands and retained earlier attempts.
Historical serving cohorts are untouched and are not pooled with these screens.

### Read-only evidence replay

`scripts/audit_sota_breakthrough.py` checks the frozen pilot split, original
helper refit, fallback-inclusive arithmetic, head certificate and native codec,
six conveyor cases and corrected correlation ledger. Executable snapshots have
fixed repository SHA-256 pins; an alternate evidence directory supplies data
only. The pilot report, manifest, helper, launcher status and public log are all
archived. Its numeric-helper hash was first observed **after launch**; subsequent
driver hardening does not retroactively create a pre-launch lock.

```sh
PYTHONDONTWRITEBYTECODE=1 target/frontier-sdk-env/bin/python \
  scripts/audit_sota_breakthrough.py \
  --output target/sota-breakthrough-audit-replay-v2.json
```

Use a fresh output path. Full replay requires the original installed native and
NumPy dependencies plus the pinned local PDF/text copies named by the correlation
reference. `--skip-native --skip-primary-sources` explicitly omits installed-codec
and full-text revalidation while retaining the bounded archive checks; its report
does not claim those omitted checks passed. Portable helper refitting declares
an absolute float64 tolerance of `1e-12`; full replay requires exact weights and
the original numeric dependencies. No mode reloads target checkpoints,
retrains a target, reruns timing cohorts or qualifies serving performance/RAM.
The retained `docs/evidence/sota-breakthrough-audit-v1.json` and
`docs/evidence/sota-breakthrough-audit-v2.json` preserve their own source hashes
and coverage; v2 also reconciles the archived public launcher log.
