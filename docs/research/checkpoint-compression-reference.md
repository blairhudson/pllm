# Fresh checkpoint-specific compression: useful-baseline-first gate

## Status and scope

**Real local-clear Qwen2.5 baseline rejected by frozen optimistic gate.**
Thirty valid admission QA cases: **13 correct, 17 errors (43.33% observed EM)**.
Even perfect remaining QA yields **47/64 = 73.4375%**, below required 75%.
Teacher labels, fitting and heldout evaluation remain **unrun and blocked**.
No useful-learning, compression benefit, traffic reduction or serving qualification.
The prior 264-parameter pilot remains a failure: original and learned QA 1/4;
fallback-inclusive arithmetic 32.39% worse. This experiment does not retune that
holdout or count mechanics as useful learning. Parent alone operates real stages.

Owned files: `benchmarks/research/checkpoint_compression.py`,
`scripts/benchmark_checkpoint_compression.py`,
`tests/test_checkpoint_compression.py`, `tests/test_checkpoint_compression_audit.py`,
and synthetic-only `docs/evidence/checkpoint-compression-mechanics-v4.json`.
Historical mechanics remain untouched. V4 adds pre-freeze audit repairs;
its synthetic result is not the real baseline below or a useful-quality claim.

V4 analytic fixture digest (canonical train controls, analytic labels, policy):
`db5eef7686b14f96546f96b905033559fec3558368857002c6a5a4e82b5e736d`.
Evidence-file SHA256:
`7902709645bfb314613de7a4e72ec0960a6241ea4dab9d32394fb8b7d32ff1f9`.
Both execution-code hashes matched that mechanics evidence. Historical focused
tests: **62 passed**; Ruff clean. Those mechanics checks loaded no public checkpoint
tensor values; subsequent parent-operated baseline did load the pinned public target.

## Archived negative baseline: Qwen2.5 v1

Bounded evidence slice:
[`docs/evidence/checkpoint-compression-baseline-qwen25-v1/`](../evidence/checkpoint-compression-baseline-qwen25-v1/).
`originals/target/` preserves original v2 manifest, entire run directory (including
failed admission), prepare/preflight artifacts, baseline 01–05 logs, available
incremental reports and final report-05 byte-for-byte. `audit.json` is explicitly
derived, not a fabricated execution result. `checksums.json` inventories all other
archive files, byte lengths, original paths and frozen code-hash bindings; it
excludes itself to avoid a recursive checksum. Historical mechanics are separate.
Archive contains **46 files, 1,646,439 bytes**: 23 original artifacts, seven
launcher dependency snapshots, thirteen installed-SDK source slices, two derived
JSON documents and checksum inventory. Exact paths are listed in `checksums.json`.
Inventory file SHA256:
`a12dca40d4d2994fb09687624243d2fd7cd6a2d19d69e647edeb9f257c625da0`.
Documented stdlib snippet was executed successfully: **45 archive hashes verified;
30 valid / 13 correct / 17 errors; maximum 47/64**, including failed-attempt retention
and reconciliation of recorded pass costs. No new model runs/builds/installs/tests.

| Parent tranche | Valid QA | Correct | Quality errors | Outcome |
| --- | ---: | ---: | ---: | --- |
| 01 | 8 | 3 | 5 | Complete |
| 02 | 8 | 3 | 5 | Complete |
| 03 | 0 | 0 | 0 | Host admission rejected before child |
| 04 | 8 | 5 | 3 | Complete retry |
| 05 | 6 | 2 | 4 | Frozen stop before next case |
| **Total** | **30** | **13** | **17** | **Baseline rejected** |

All valid cases are distinct admission/public-QA cases with completed full-context
passes. Quality errors include one generation without EOS, not an infrastructure
failure. Error rate is 56.67%. Remaining 34 QA and all 32 admission controls were
not executed. Maximum possible QA score is 47/64; optimistic Wilson lower bound
0.615169 still meets 0.60, but EM cannot meet 0.75. Even 79/96 aggregate successes
would not override this per-kind gate. No labels/fit/evaluate attempts or learned
artifact exist in this lineage. This is a negative result for this frozen target,
numeric mode, prompt and cohort, not proof that compression generally cannot work.

Original report's `baseline.reason` remains `missing/incompatible baseline cohort`:
it describes missing full-cohort rows, **not an observed source incompatibility**.
Terminal quality rejection is recorded in `baseline.status` and `baseline.optimistic`.
Original reason is not rewritten. Reusing this inspected cohort to retune gates,
prompts or helper architecture would not be a fresh experiment.

Five attempts are retained/charged; four workers completed. Tranche 03's
`baseline-85fd76bca785443c9da81932ab0cd4dd.json` records
`frontier probe exceeds post-reserve headroom`, zero items/passes, 3.946172 parent
CPU seconds and 4.404301 wall seconds. It consumes attempt/retry budget but no
quality error. Zero provenance-invalid or observed pressure-invalid attempts.
Invalid/failed attempts are not silently deleted; eligible quality and all-attempt
costs remain separate.

Manifest digest:
`14402f3e35997316366edb9f93126b2cabc9d0f167d69ae4230b281b9683b6e7`.
All four workers agree on source/checkpoint locks and build digest
`db5b5374935fa41416a869383495ed68ac391ecb3535537d262110504e4a5f10`.
Audit rehashed all 1,009 listed environment/code/native/interpreter files
(60,437,548 bytes), with no changed/missing entries, and verified public dataset
identity. Seven launcher dependencies and thirteen essential installed SDK source
slices were copied only after checking frozen manifest hashes. Installed SDK
snapshots, not potentially changed checkout files, explain actual executed helpers.

Target: Qwen/Qwen2.5-0.5B-Instruct revision
`7ae557604adf67be50417f59c2c2f167def9a775`, 494,032,768 parameters,
`W8A8-prefix_f32-local-clear`. `external-artifacts.json` references dataset,
model snapshot, resolved interpreter/native paths and SHA256 identities; original
manifest retains all eight model-file hashes and full environment hash map.
Large weights were **not rehashed by this lightweight audit**; their identity is
recorded by original execution-time source validation. No checkpoint/native binary,
full dataset, entire interpreter or SDK is duplicated in this bounded archive.
Public data attribution remains Rajpurkar et al., SQuAD v1.1, CC BY-SA 4.0.

Observed child lifetime peak RSS: **1,192,771,584 bytes (1.111 GiB)**, not whole
client/private-serving RSS. Four valid workers report **zero observed swap growth**
against **2,417,623,040 bytes existing host swap**; this is not swap-free execution
or proof against between-sample transients. Lowest recorded available memory:
7,353,942,016 bytes, above full 4 GiB host reserve. Host rejection has no child RSS.

Thirty pass totals: 423.072344 CPU seconds / 253.694187 wall seconds; 6,734 input
tokens, 252 selected tokens, 6,956 executed rows, 667,776 stage rows, 24,192 calls.
Worker partial-window known subtotals: 466.769205 CPU / 299.281192 wall seconds,
one missing record; totals remain null. Parent window: 21.255618 CPU / 336.079404
wall seconds, **including child wait: do not add overlapping wall windows**.
Full child-lifetime CPU/wall, prior downloads/setup, interrupted tails, private cold
traffic, transport and whole-topology Low/Medium RSS remain unknown, not zero.
Projected arithmetic bytes are not measured traffic savings.

Declared task: source-grounded short extractive QA on public encyclopedia
paragraphs, plus separate procedural literal-copy, rare Unicode/code, ordered
route and approximately 1,000-character context controls. This is not general
generation, realistic 32K-context evaluation, a standalone replacement, or a
standard SQuAD leaderboard run. Public teacher pretraining exposure is unknown.

## Frozen data before target work

Preparation re-derives the original pilot from the pinned SQuAD v1.1 dev source,
verifies it against the SHA256-locked archived original manifest,
then excludes **all twelve articles** touched by its fourteen public QA cases
(eight train, two dev, four heldout). It does not only remove those question IDs.
Remaining entire article titles are deterministically assigned with a new
domain-separated SHA256 split: 60% train, 20% admission, 20% final heldout.
Contexts are globally unique, with one selected QA per paragraph. IDs, articles
and context split integrity are checked. No online/private prompts enter fitting.

| Split | Fresh public QA | Copy | Rare | Order | Long-context | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 192 | 16 | 16 | 16 | 16 | 256 |
| Baseline admission | 64 | 8 | 8 | 8 | 8 | 96 |
| Final heldout | 128 | 16 | 16 | 16 | 16 | 192 |

Source: Rajpurkar et al., SQuAD v1.1, CC BY-SA 4.0;
[immutable public JSON](https://raw.githubusercontent.com/rajpurkar/SQuAD-explorer/240e165ab706d95bd4323653bb92421f446208cf/dataset/dev-v1.1.json),
Git blob `e9a3f913ad1468ebe105b891334ca7b0bc0e2510`, 4,854,279 bytes.
No downloads in this driver. The prepositioned public-text unit test confirmed
544 rows and split digest
`00073e7715d43277d140ebc085219daf21bd61854d45d2ea383ec0cc9d6ae1a2`.
This text-only check did not inspect public model tensor values.

`prepare` persists all canonical rows, exclusions, source and independent data
hashes, complete policy/hyperparameters, environment and code locks in an
exclusive digest-bound manifest. Every stage re-derives the rows from the locked
public file; editing/re-hashing a manifest cannot inject different data. Do not
change prompts, thresholds, data filters, architecture or hyperparameters after
observing admission/heldout answers. A changed method after inspection requires
new disjoint data, not another manifest over this same cohort. The software
cannot prevent a human creating another directory; that is not an authorized
second experiment or a way around attempt budgets.

## First gate: original target must be useful

`baseline` runs **only full original inputs**, greedy, fresh empty-KV sessions,
serial chunks of at most eight examples. No teacher candidate scoring, neural
fitting, compressed target runs or target adaptation before admission.

Admission requires all 96 fixed examples, public QA EM >=75% with 95% Wilson
lower bound >=60%, and >=87.5% literal-exact/EOS success independently for each
control kind. QA uses the existing normalized extractive EM; controls are literal.
Missing rows, weak full target, unfinished responses or incompatible provenance
keep training blocked. Baseline quality is never rescued by a compressed arm.
At 64 QA, 48/64 passes the frozen QA screen; 47/64 fails. This is a bounded
screen, not a statistically established population noninferiority result.

**Frozen quality-only early stop:** assume every missing/unexecuted example will
succeed, at the original fixed denominators and thresholds (including Wilson).
If even that optimistic completion cannot pass, write `baseline_rejected` and
stop before the next target call, including mid-tranche. Seventeen provenance-
valid completed QA errors among 64, or two valid errors in any eight-item control
kind, suffice. Sixteen QA errors/one control error do not suffice. Missing,
in-flight, source/code/environment-invalid and pressure-invalid items are not
quality errors. Missing results can never admit training. This rule is frozen in
`POLICY` before any real baseline; it does not use heldout data or change quality
thresholds. Repeated baseline commands after rejection launch no child.

The manifest defaults to independently pinned Qwen2.5-0.5B-Instruct source from
the previous infrastructure, but does **not** reuse its results or learned
artifact. A complete `--source-spec` JSON (`model`, immutable 40-hex `revision`,
`files` mapping filenames to SHA256) admits another prepositioned source through
the same model-neutral Teacher API. Each source gets its own manifest, baseline,
labels and helper. No family-specific neural execution branch. Missing tokenizer,
tensor/config locks or unsupported Teacher execution fails closed. This is not
a claim that every model family/checkpoint is supported. Sources whose weight
bytes ×3 +256MiB exceed this pilot's 4GiB resource estimate are blocked.

Declared file hashes are anchors, **not the inventory boundary**. Preparation
discovers and locks the entire snapshot, every safetensors index and referenced
shard (including nested shards), all discovered/unreferenced safetensors files,
tokenizers/templates/config and resolver metadata. Missing/escaping references
fail closed. Parameter/header and byte accounting includes all discovered shards.
Preparation records SDK checkpoint digest, source-lock digest and complete SDK
source-lock document as well as the larger execution inventory. Immediately
after actual source resolution, a child-local resolver guard compares all of
these against the actual resolved source and its manifest **before the shared
Teacher constructs or loads the engine**. Extra/replaced shards or incompatible
actual resolved locks invalidate provenance, not merely posthoc results.

## Stronger small encoder, complete ownership

The helper is a fully trainable **19,776-parameter** contextual sentence extractor:

| Owner | Parameters |
| --- | ---: |
| UTF-8 byte embedding, including padding | 8,224 |
| Ordered width-three byte convolution | 3,072 |
| Convolution bias | 32 |
| Query-conditioned contextual sentence hidden matrix | 8,320 |
| Hidden bias | 64 |
| Sentence output vector | 64 |
| Frozen backbone / frozen embedding / target vocabulary head | 0 / 0 / 0 |

All parameters count, not an adapter subset. Runtime weights are float64:
158,208 raw bytes; **actual complete JSON artifact bytes** are measured separately.
The evaluator temporarily owns a second initialized encoder for its untrained
control, so up to 39,552 helper parameters, not 19,776, coexist in that process.
No target weights, embedding or head are an undeclared helper backbone.

The **<1B limit applies to the entire tiny helper**, including any frozen
backbone/embedding/head, not to the remote checkpoint teacher plus helper.
Current constructor asserts exactly 19,776 helper parameters and zero hidden
backbone/embedding/target head. Metadata names `entire_helper_parameters` and
`teacher_parameters` separately; research teacher-plus-helper totals are charged
inventory, **not a combined parameter admission gate**. A >1B teacher is not
rejected merely for its parameter count. This bounded pilot's separate teacher
weight-byte/RAM admission may still block it. No teacher backbone moves secretly
into client ownership, and teacher source/import/RAM/compute remain charged.

Trainable CNN encodes ordered byte triples, averages contextual byte states,
then conditions sentence scores on query state, sentence×query interactions,
previous/next sentence states and source position through a tanh hidden layer.
This is nonlinear learned local representation, not more linear hashed features.
It is deliberately small and may still fail natural QA. It is not LLMLingua-2,
an implementation of upstream weights, or a novelty claim.

Training labels use frozen-target gold-continuation NLL for each contiguous
half-sentence window plus a lexical half mask (deduplicated; at most fourteen
candidates). Temperature-one soft labels supervise averaged selected sentence
scores. Inference selects the highest-scoring ceil(n/2) sentences and restores
source order, without gold answers. Every retained source span remains verbatim;
input tokenization, positions, logits and KV nevertheless change. There is no
original-logit/KV or arbitrary-sampling equivalence claim.

All hyperparameters are committed in `POLICY`: seed 610162, byte dimension 32,
hidden 64, sixteen deterministic epochs, Adam .003/.9/.999/1e-8, norm clip 1,
L2 .0001. All embeddings, convolution and head train; no pretrained/frozen
feature extractor. NumPy supplies math; no PyTorch or new dependency required.
The target is never adapted, so adapted-target controls are inapplicable.

## Stages, failure retention and controls

1. `prepare`: canonical public manifest, metadata/header reads and stream hashes;
   no checkpoint values. Exclusive output, no overwrite.
2. `preflight`: revalidate metadata/code/native/environment and write manifest-bound
   host snapshot; resource headroom is advisory until rechecked at every launch.
3. `baseline`: bounded serial full-original admission chunks. `report` exposes
   the useful-baseline gate; all other target stages enforce it themselves.
4. `labels`: resumable train-only candidate-NLL chunks, only after admission.
5. `fit`: one bounded local neural fit from **all** completed train labels;
   source-bound artifact and train-split digest. Exactly one fit attempt per root.
6. `evaluate`: final heldout only, exact completed fit artifact reloaded; fresh
   full original, lexical compressed, initialized-neural compressed, and learned
   compressed original-target sessions for every case.
7. `report`: same-cohort quality and full retained cost ledger, never serving
   qualification. Reports from different manifests are not pooled.

Each compressed arm can invoke a **fresh** full-input fallback when EOS is absent,
output empty, or output not a literal span of its retained context. A wrong
present span does not trigger retry. Gold retention is diagnostic only, never a
fallback router. Controls do not borrow the already measured full response.
At most two fallbacks total per chunk; exhausted routes are explicit, and any
learned exhausted route prevents its quality gate passing. Learned fallback
fraction must be <=10% across the final cohort. Evaluation order is frozen:
full, lexical, untrained, learned, with each arm's necessary fallback immediately.

Useful final quality and learned value are **separate**: full heldout must itself
pass the useful baseline screen; learned QA >=75%, controls each >=87.5%, no
per-kind drop >2 percentage points from full, and fallback gates pass. Learned
value additionally requires >=2 percentage-point QA EM gain over lexical both
before and after fallback, so extra full-target retries cannot masquerade as
learned value. Both raw and fallback-inclusive quality tables are reported.
Tying lexical is not evidence that learning helps. None of these gates admits
traffic/memory claims.

Run directories bind one manifest. Every attempt gets a new UUID status file;
intent is fsynced before each target pass, result after each pass, and item
completion after all modes. Completed items survive a later item failure and
are resumable; partial current items are rerun from scratch in a **new child**.
Failed teacher/provider sessions never continue. Original partial/failure/timeout
attempts stay and are charged. Pressure-invalidated attempts remain visible but
their items cannot admit a baseline or be reused. Source/code/environment or
build mismatches, even after the child completed, set `provenance_invalid=true`
and exclude every item/artifact in that attempt from reuse and all gating.
Ordinary timeouts retain independently completed, still-valid items; incomplete
items never count as quality errors. No automatic retries; at most
two attempts per item, 138 total child attempts per root. SIGKILL/power loss leaves
the last explicitly incomplete snapshot, never an inferred success. Snapshot
writes and helper creation use bounded same-directory temporary files, file
fsync, atomic publication and containing-directory fsync (including cleanup).
Existing artifacts are never overwritten by helper creation. Failure may leave
a fully published but unqualified helper; its path/status never imply a fit pass.

## Resource/work and accounting bounds

All real target/local fitting work uses imported shared `fresh_workers`: one
exclusive frontier lock, fresh process group, four CPU threads, offline mode,
900-second timeout, host-pressure watchdog and group termination/kill cleanup.
Admission requires **4GiB available after** max(1GiB,total_RAM/8) reserve and 3GiB
free scratch disk. This driver tightens the shared watchdog: **any positive new swap
growth** or loss of the full reserve aborts. Source/native/SDK code and environment
must match before child work and after completion. Locks include both new files,
actual imported SDK Python sources and native binary, shared Teacher/ranker,
decoder helper, frontier runner, installed package versions, interpreter binary,
and critical NumPy/BLAS/tokenizer/safetensors/guard source and native files.

The frozen public cohort has exact default-eight-item work ceilings:

| Work | Upper bound without failed attempts |
| --- | ---: |
| Original-only baseline | 96 target responses; 12 children |
| Teacher candidate labels | 1,049 forced-answer passes; 32 children |
| Local fitting | 4,096 updates; one child; no target passes |
| Full final heldout, all arms/fallback cap | 816 target responses; 24 children |
| Total | 1,961 target passes; <=1,049,135 sequence rows; 69 children |

Rendered prompt <=512 native tokens; selected output <=24, forced answer <=16.
No truncation or BPE-boundary relaxation: out-of-bounds/boundary mismatch fails
the attempt. Teacher inherits these caps through the locked shared module.
Default complete-run timeout ceiling is 69×900s =17.25h, **not a runtime promise**;
138-attempt overall cap is 34.5h of child deadlines. A baseline tranche is <=8
passes/4,280 sequence rows/900s, one live child. Labels tranche <=112 forced
passes; heldout tranche <=34 causal passes. Smaller chunks alter process/fallback
ceilings; manifest work-budget assumption explicitly says default eight.

Each helper has <=6,144 context UTF-8 bytes, <=2,048 bytes/sentence and <=512
query bytes; no silent byte truncation. Helper tensor/cache workspace is a small
bounded addition (conservative 64MiB estimate, not measured whole-client RAM).
Local-clear teacher plus SDK stays inside the separate 4GiB admission estimate,
not Low/Medium serving certification. Stage status/manifest/complete helper
artifacts are capped at 16MiB each; <=138 attempts imply <2.2GiB absolute JSON
ceiling, covered by 3GiB initial scratch admission absent unrelated disk use.
Normal expected research JSON is tens of MB, plus already cached ~1GB target,
4.85MB source corpus and <1MiB helper. Parent should reserve 3GiB scratch for the
absolute ledger ceiling. Source validation hashes existing target streams repeatedly;
that CPU/I/O and every source import are charged, not treated as free preparation.

Worker clocks are explicitly **partial measured windows after validation/import**,
not whole child lifetime. Full child lifetime CPU/wall remain null. Parent
launch/window timing is separately labeled and overlaps child wall time.
Missing killed-child measurements yield null totals and missing-record counts;
known snapshot subtotals are retained, never substituted for complete costs.
Incomplete attempts have explicitly unknown post-snapshot tails. Observed OS
lifetime peak RSS may miss a killed child's later peak, so is labeled observed.
Ledger reports these windows and missing counts, plus
preparation/source-prepositioned bytes, every completed pass's input/output tokens,
sequence/stage rows, calls and three-direction **projected arithmetic**, local
selection CPU, full artifact size, all failed attempts and unknown interrupted
work. Per-arm raw and fallback-inclusive counts, CPU/wall and directed arithmetic
totals include every retained attempt, including invalid/failed work, separately
from eligible quality cohorts. Incomplete passes make aggregate totals null,
with known subtotals and missing counts retained. Parent wall includes child wall
and must not be summed with it. Post-last-
snapshot interrupted CPU/work and prior downloads remain unknown, **not zero**.

**>=20% net first-request measured traffic reduction under Low=256MiB or
Medium=2GiB is a future serving target, not arithmetic extrapolation.** It must
include helper delivery, source/setup at every owner, original retained fallback
prompt, tokenizer/table/head/logits/KV/runtime, failed/fallback work, preparation,
control traffic, masks/correlations, padding and transport. No such measured
serving cohort exists here. Compression remains client-local in future serving;
target stays masked through existing SDK. This experiment is explicitly
**local-clear**, issues no one-use provider material and adds no private protocol.

## Parent commands and replay boundary (no new target runs)

Parent-operated v2 command forms below correspond to retained prepare/preflight,
five baseline logs and final report. Logs preserve outcomes, not full shell command
history: redirections, shell wrappers and spelling of default `--limit 8` are not
independently recoverable. These are documented CLI arguments, not fabricated
verbatim shell transcripts. **Historical commands only; do not execute them again.**

```sh
target/frontier-sdk-env/bin/python scripts/benchmark_checkpoint_compression.py prepare --data target/checkpoint-tiny-model-squad-dev-v1.1.json --manifest target/checkpoint-compression-v2-manifest.json
target/frontier-sdk-env/bin/python scripts/benchmark_checkpoint_compression.py preflight --manifest target/checkpoint-compression-v2-manifest.json --output target/checkpoint-compression-v2-preflight.json
# Five parent baseline invocations: complete, complete, host rejection, complete, baseline_rejected.
target/frontier-sdk-env/bin/python scripts/benchmark_checkpoint_compression.py baseline --manifest target/checkpoint-compression-v2-manifest.json --run-dir target/checkpoint-compression-v2-runs --limit 8
target/frontier-sdk-env/bin/python scripts/benchmark_checkpoint_compression.py report --manifest target/checkpoint-compression-v2-manifest.json --run-dir target/checkpoint-compression-v2-runs --output target/checkpoint-compression-v2-baseline-report-05.json
```

Prepare produced `prepared_unadmitted`; preflight reported metadata-only resource
admission true. Tranche 03 later failed fresh host admission despite preflight.
Tranche 05 and final report record terminal `baseline_rejected`. Labels/fit/heldout
commands were **not run** and remain blocked; no follow-on target work is authorized.

### Portable read-only count/hash audit

From repository root, run this separate local **stdlib-only** snippet. It imports
no archived executable code or SDK, loads no model, acquires no benchmark lock,
writes nothing and needs no external artifact. It verifies slice checksums,
manifest/source-snapshot binding, valid item uniqueness and independently recomputes
EOS-aware normalized QA counts, optimistic score and pass-cost totals.

```sh
python3 - <<'PY'
import hashlib, json, math, re
from pathlib import Path

root = Path("docs/evidence/checkpoint-compression-baseline-qwen25-v1")
load = lambda p: json.loads(p.read_text())
sha = lambda b: hashlib.sha256(b).hexdigest()
inventory = load(root / "checksums.json")
files = inventory["files"]
assert {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()} == set(files) | {"checksums.json"}
for name, record in files.items():
    raw = (root / name).read_bytes()
    assert len(raw) == record["bytes"] and sha(raw) == record["sha256"], name
originals = root / "originals/target"
m = load(originals / "checkpoint-compression-v2-manifest.json")
digest = sha(json.dumps({k: v for k, v in m.items() if k != "digest"},
                       sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
assert digest == m["digest"]
for name, record in files.items():
    if "manifest_code_path" in record:
        assert record["sha256"] == m["environment"]["code_native"][record["manifest_code_path"]], name
rows = {r["id"]: r for r in m["data"]["rows"]}
norm = lambda s: " ".join(re.sub(r"\b(a|an|the)\b", " ", re.sub(r"[^\w\s]", "", s.lower())).split())
runs = originals / "checkpoint-compression-v2-runs"
attempts = sorted((load(p) for p in runs.glob("baseline-*.json")), key=lambda a: a["started"])
assert len(attempts) == 5
assert not any(runs.glob("labels-*.json")) and not any(runs.glob("fit-*.json")) and not any(runs.glob("evaluate-*.json"))
seen, correct, errors, passes = set(), 0, 0, []
for a in attempts:
    assert a["manifest_digest"] == digest
    guard = a.get("guard", {})
    eligible = not a.get("provenance_invalid") and not guard.get("error") and not guard.get("maximum_swap_growth_bytes", 0)
    if a["items"]:
        assert a["build"]["expected_digest"] == "db5b5374935fa41416a869383495ed68ac391ecb3535537d262110504e4a5f10"
        assert all(a["teacher_lock"][k] == m["source"][k] for k in ("checkpoint_digest", "source_lock_digest", "files", "pilot"))
    wins = losses = 0
    for key in a["ids"]:
        if key not in a["items"]:
            continue
        item, row = a["items"][key], rows[key]
        assert eligible and item["status"] == "complete" and key not in seen
        assert row["split"] == "admission" and row["kind"] == "public_qa"
        assert len(item["passes"]) == 1 and item["passes"][0]["status"] == "complete"
        value = item["full"]
        assert item["passes"][0]["result"] == value
        seen.add(key)
        passes.append(value)
        win = bool(value["eos"] and any(norm(value["text"]) == norm(answer) for answer in row["answers"]))
        wins += win
        losses += not win
    correct += wins
    errors += losses
    print(a["outcome"], len(a["items"]), wins, losses)
assert (len(seen), correct, errors) == (30, 13, 17)
assert sum(not v["eos"] for v in passes) == 1
policy = m["policy"]
assert policy["baseline_early_stop"] == "quality_only_optimistic_all_remaining_success_v1"
assert (policy["counts"]["admission"], policy["qa_min_em"], policy["qa_min_wilson_lower"]) == (64, .75, .60)
maximum = 64 - errors
p, n, z = maximum / 64, 64, 1.96
lower = (p + z*z/(2*n) - z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))) / (1 + z*z/n)
assert maximum == 47 and maximum / 64 < policy["qa_min_em"] and lower >= .60
report = load(originals / "checkpoint-compression-v2-baseline-report-05.json")
assert report["baseline"]["status"] == "baseline_rejected"
assert report["baseline"]["optimistic"]["valid_completed_quality_errors"]["public_qa"] == errors
assert not report["heldout_complete"] and not report["costs"]["helper_artifacts"]
for key, metric in report["costs"]["all_passes_including_invalid_and_failed_attempts"]["counts"].items():
    assert math.isclose(sum(v[key] for v in passes), metric["total"], rel_tol=1e-12, abs_tol=1e-8), key
print("PASS:", len(files), "archive hashes; 30 valid / 13 correct / 17 errors; maximum 47/64; Wilson", lower)
PY
```

This verifies archived source slices against recorded hashes, not availability of
all external dependencies. Optional external hash verification may read paths in
`manifest.environment.code_native` and `manifest.data_path`; model weights remain
external. Archive checksum inventory is an integrity record, not a signature or
independent proof of historical execution.

### Original environment-locked full report route

Original report command is shown above. To re-render only that report, preserve
original checkout paths, frozen code, interpreter, full installed SDK/native and
dependencies, external dataset and exact source snapshot. Choose a **new unused**
output path; exclusive publication must not overwrite report-05. For example,
only with original locked environment restored and separate permission:

```sh
target/frontier-sdk-env/bin/python scripts/benchmark_checkpoint_compression.py report --manifest target/checkpoint-compression-v2-manifest.json --run-dir target/checkpoint-compression-v2-runs --output target/checkpoint-compression-v2-baseline-report-audit.json
```

Not executed during archival. Report validates/hashes external artifacts but does
not rerun inference; it can fail closed after environment changes. Copied source
slices are **not** a standalone executable SDK. Full numeric rerun additionally
requires original pinned external SDK/native/source artifacts and authorization;
this archive makes **no portable complete replay claim**. Do not recreate a numeric
run or new cohort merely to repair missing provenance.

Dependencies already present: project SDK/native, NumPy, pytest, Ruff,
huggingface_hub, tokenizers, filelock, psutil, Jinja2 and shared runner/Teacher
dependencies. Missing dependencies fail; no install/build/fetch fallback.

Actually executed bounded checks:

```sh
target/frontier-sdk-env/bin/python -m pytest -q -o pythonpath=. --noconftest tests/test_checkpoint_compression.py tests/test_checkpoint_compression_audit.py
target/frontier-sdk-env/bin/python -m ruff check scripts/benchmark_checkpoint_compression.py benchmarks/research/checkpoint_compression.py tests/test_checkpoint_compression.py tests/test_checkpoint_compression_audit.py
target/frontier-sdk-env/bin/python scripts/benchmark_checkpoint_compression.py mechanics --output docs/evidence/checkpoint-compression-mechanics-v4.json
```

Tests independently check scalar-loop CNN inference, finite-difference gradients
for every learned owner, loss reduction using an independent oracle, real tiny
generated-checkpoint NLL from raw logits, fresh session/KV reset, public split
integrity, baseline admission/blocking, fallback budgets/fresh work, timeout and
partial-row retention, atomic failure/exclusive output, code/native/environment
locks, complete source discovery/index coverage, pre-engine rejection, >1B
teacher versus helper-cap separation, late provenance invalidation, optimistic
early-stop boundaries/mid-chunk behavior, durable helper publication and per-arm
cost/null accounting, and synthetic dispatch for every parent stage. Generated checkpoints
establish API/numeric mechanics only, never language usefulness.

Prior negative gates remain relevant, not retested claims:
[tiny ranker](checkpoint-tiny-model-reference.md),
[latent prompt](cold-latent-prompt-reference.md),
[drafter](checkpoint-drafter-reference.md),
[four-track contract](../plans/sota-breakthrough-four.md).
