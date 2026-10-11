# Small decode-inventory epochs

**Status: exact decoder/prepared-stage reference, not an executable SDK choice.**
Two fresh pinned Qwen2.5 workers complete **15 paired comparisons**, preserving
complete final logits/KV, output and pending state with zero observed swap growth.
Stopping at 17 of 128 allowed outputs reduces covered serialized stage bodies
**55.22%** with four-output epochs. Full-cap use increases bytes **1.00%** and
adds 31 idle boundaries. These are in-process stage measurements; SDK admissions,
public delivery, all-link streams and pause times remain unmeasured. They do not
establish the earlier illustrative 36% whole-request estimate.

## Pinned checkpoint result

The source is `Qwen/Qwen2.5-0.5B-Instruct`, revision
`7ae557604adf67be50417f59c2c2f167def9a775`, explicit W8A8. Both independent
workers use 16 public synthetic input IDs and a 256-token continuation bound.
The first executes three full-demand controls and six candidates at a 32-output
cap; the second executes three controls and nine candidates at a 128-output cap.
Each source is loaded once per worker. They are byte/parity screens, not repeated
timing cohorts or generation-quality measurements.

| 128-output allowance | Epoch | Control stage bodies, MB | Epoch stage bodies, MB | Body reduction | Added idle boundaries |
| --- | ---: | ---: | ---: | ---: | ---: |
| Explicit completion after 17 outputs | 4 | 170.754 | 76.467 | 55.22% | 4 |
| Explicit completion after 17 outputs | 16 | 170.754 | 86.670 | 49.24% | 1 |
| Explicit completion after 17 outputs | 32 | 170.754 | 86.563 | 49.31% | 0 |
| Full 128 outputs | 4 | 332.987 | 336.304 | −1.00% | 31 |
| Full 128 outputs | 16 | 332.987 | 333.736 | −0.22% | 7 |
| Full 128 outputs | 32 | 332.987 | 333.308 | −0.10% | 3 |

Controlled EOS after 17 emitted outputs also preserves exact final state; E=4
reduces covered bodies 54.75%. At the smaller cap, explicit completion after five
outputs saves 30.34% with E=4, while its full-cap control grows 0.69%. Full-cap
correction demand is unchanged; repeated seeds, layouts and frames cause the loss.
Every execution reconciles reserved rows with claimed plus burned rows. For
17/128 explicit completion, E=4 burns 288 unused stage rows versus 10,656.

`docs/evidence/research-decode-epochs.json` retains both complete source-bound
reports and hashes, plus the historical tiny fixture. A separate attempted
64-input case rejected the probe's explicit 16-input limit before checkpoint
execution; it contributes no result. Epoch sizes 16/32 deserve a live comparison
alongside four: their smaller byte win also incurs fewer preparation pauses.

## What executes

`scripts/probe_decode_epochs.py` uses the installed compiler, native seeded-ring
Preparation, `PreparedStageRows`, demand `PreparedInventory`, reservation leases,
`PreparedRemoteLinear`, and ordinary `prefix_f32` semantic decoder. Correction
pushes and online requests/replies are actually serialized and decoded. Inference
adds each prepared correction to its native masked matrix result. The reference
holds the three algebraic owners in one interpreter; it is neither a deployment
nor a proof of network authorization/privacy isolation.

An epoch is an output allowance, selected from public sizes 1/4/8/16/32/64.
The first execution pays the complete first prefill, compiler terminal pruning,
and up to E−1 decode rows. Later executions consume the exact pending sampled
token on one freshly prepared continuation row, then up to E−1 further decode
rows. Preparation cannot execute while a lease is active. There is no online
material fallback. Finishing an execution closes its lease, burns unused reserved
ranges, cancels its inventory, and discards correction arrays. A fresh random
session, request seed and tickets are issued next time. Failed preparation retires
already published material and counts its partial request/correction bodies.

The decoder remains local. Resumption uses the installed native continuation
admission and actual response-owned snapshots. Bare incremental KV is not a legal
continuation basis. Snapshot ownership is renewed at each boundary; token IDs are
never reconstructed from text. The final emitted token remains pending on explicit
completion/full cap. An EOS stop after N emitted outputs instead executes the Nth
token before the next selection; a stop exactly on an epoch boundary therefore
pays one extra fresh execution. Full-cap controls execute identical work, not a
shortened generation passed off as savings.

The research workload uses public synthetic token IDs and greedy sampling. EOS
tests use an explicitly injected public termination predicate, not a claim that
this checkpoint naturally emits EOS there. Sampled values are otherwise unchanged
between matched controls/candidates; a full-cap harness treats sampled EOS as an
ordinary output in both. Natural model EOS and stochastic sampler-state continuity
remain serving-integration gates.

## Strong control and accounting

Each candidate is compared with the same exact decoder and compiler-derived demand
allocation for the full output cap, **not the obsolete 64-row eager inventory**.
`compiled.prepared_stage_rows` prices each actual stage, terminal pruning and the
all-stage minimum-one-row authorization rule. Reserved = claimed + unused burned
in every completed or injected-failure execution. Preparation failures separately
charge issued/discarded rows before admission to execution.

Measured directed bodies are:

* Client → Preparation: packed per-stage seeded requests.
* Preparation → Inference: complete packed correction pushes, including frames.
* Client → Inference and Inference → Client: complete packed online stage bodies.

Bodies are counted once, never once at both endpoints. This uses current
`stage_packed` inputs, `row_residues` outputs/corrections, terminal pruning and
demand allocation, with compiler-checked committed layouts. `CorrectionPush`
uses the native packed correction codec; `pack_row_response` and
`PreparedRemoteLinear` use matching native output codecs/dequantization.
SDK authorization/admission, response/session
control, correction acknowledgements, public artifacts, TCP framing, checkpoint
distribution and tokenizer distribution are **unknown**, not zero. Historical
control is read through `scripts/probe_all_link_frontier.py:conserved_control`,
not copied from prose: **493,575,247** total body bytes, **139,060,689** correction
bytes, **90,443,518** ingress bytes, **124,905,091** public-bundle bytes,
139,123,150 result bytes and 42,799 control/seed bytes. These values are context
only; reference bytes cannot be subtracted from them.

The worker replaces equal-per-row arithmetic with
a separate **compiler geometry forecast** for 150 input / 512 cap / 80 used. It
counts projected reserved stage rows and their committed per-column output widths. This
forecast excludes frames, admissions and timings and has no whole-request saving
fraction. It is **not an admitted execution**: this reference caps continuation
at 256 tokens and output at 128; 150+511 exceeds that bound. Packed codec value
limits also prevent silently compiling the old 1024-row prefill bound for the
widest Qwen stages. Full-cap forecasts save zero packed correction residues; additional
epochs only add framing/control and pause costs.

## Tiny functional evidence

The collection's `historical_tiny_functional` record retains the original
functional scope and frozen identities. That fixture loaded no real checkpoint.
A generated two-layer 32-hidden Qwen2
test fixture uses five input tokens, nine-output allowance, three early outputs.
All six matched cells preserve complete final logits/KV/output/pending state.

| Workload | Epoch | Control covered stage bytes | Epoch bytes | Executions |
| --- | ---: | ---: | ---: | ---: |
| Full cap, nine outputs | 1 | 53,646 | 90,401 | 9 |
| Full cap, nine outputs | 4 | 53,646 | 62,839 | 3 |
| Explicit completion, three outputs | 1 | 33,822 | 35,681 | 3 |
| Explicit completion, three outputs | 4 | 33,822 | 27,713 | 1 |
| Controlled EOS after three outputs | 1 | 37,126 | 44,801 | 4 |
| Controlled EOS after three outputs | 4 | 37,126 | 31,017 | 1 |

Both small-epoch early cases are negative controls: fewer reserved correction
rows are insufficient to beat repeated framing/layout metadata. Early explicit E=4 burns eight
unused stage rows in its last block, versus 48 under the full-cap control. Its
final active KV/logit logical payload is 2,824 bytes in both cells. These numbers
are fixture results, not Qwen2.5-0.5B predictions.

57 focused tests pass against the installed wheel. Checks cover six public epoch
sizes, full cap, zero/early outputs, both termination lifecycles, exact final and
every executed-row logits/all active KV against an independent clear native
target, pending ownership, terminal pruning, over-bound rejection before issuance,
forbidden online preparation/fallback, fresh tickets, idempotent close, partial
Preparation failure, and injected execution failure/cancellation retirement.
The cancellation injection verifies material lifecycle, not async worker drainage.
Unexpected exceptions explicitly reset the runtime as well as retiring material.
The historical packed-codec evidence is a tiny direct functional check with
before/after frozen identities, not a timed/RSS cohort. Its host/swap maxima
remain null. The later checkpoint workers report host/swap admission separately;
neither measures separated-role CPU/RSS.
Worker cells assert reserved = claimed + burned before returning.

## Resource owners and integration gates

The proposal keeps **100% of dense body matrix MACs remote**; client embedding/head
and nonlinear attention/KV remain ordinary owners. No body matrix moves to client.
The reference itself is in-process and cannot report provider/client RSS separately.
It records active KV/logits logical bytes, not RSS. Source/bundle import, resident
matrices, tokenizer, prefix cache, masks, correction buffers and scratch are extra
owners. The implemented snapshot/restore boundary adds full active-state copies:
running state plus old/new snapshots can coexist, with additional restore scratch.
An optimized runtime could retain the live state but must preserve the same owner
basis contract rather than silently bypass admission.

Added pauses equal completed material blocks minus one. Pause duration, first-text
delay, N−1 throughput, client CPU, provider birth CPU/RSS, lifetime client peak RSS,
source/hash/import CPU and all-link stream bytes remain null until measured. The
byte reference does not support a throughput or client-memory improvement claim.

### Calculated allocation admission

Config/header-only `memory_admission` reuses installed `estimate_memory`
components. It prices **one** resident engine, not two provider copies. Loading,
whole-bundle import and execution are mutually exclusive temporary phases; their
maximum is added to retained owners, then 25% slack. Source mappings remain
charged. Bundle import retains the helper's conservative six-copy allowance.
Execution includes decoder/snapshot overlap, mask/correction owners, prefix-cache
allowance and 64 MiB native/FFI scratch. Explicit ceilings only tighten admission.

Pinned Qwen config/header calculation, **not measured RSS**:

| Owner | Bytes |
| --- | ---: |
| Resident engine i8/native upper bound | 630,095,872 |
| Resident bundle weights/metadata | 182,521,344 |
| Worker import baseline | 268,435,456 |
| Source checkpoint mapping | 988,097,824 |
| Launcher imports/hash buffers | 268,435,456 |
| Largest temporary phase: bundle import | 1,095,128,064 |
| Total with 25% slack | **4,290,892,520** |

For 16 input / 32 cap / 256 bound, execution temporaries total 571,287,976 bytes.
For default 128 cap they total 863,653,288 bytes; bundle import still wins. Both
fit **4 GiB (4,294,967,296 bytes)**. This is not an OOM guarantee or permission
to reduce shared host reserve. Reject unless input ≤16, cap ≤128 and
input+cap−1 ≤ bound ≤256. `fresh_workers` admits the calculated total and retains
all shared reserve/swap guards. Reports include initial host/swap values, minimum
available memory, maximum host-used memory, maximum swap growth and a conservative
absolute swap maximum bound. Host telemetry is not independent process RSS.

Required minimal runtime extension, **not implemented here**:

1. Explicit same-generation yield/resume that preserves output/history, full exact
   state, pending sampled ID, sampling/RNG state, request-wide output cap and fixed
   source/composition/continuation admission. `Conversation.send("")` renders a new
   turn and is not equivalent.
2. End the provider execution/session completely before fresh Preparation; authenticate
   each new authorization, bind its fresh inventory, reject stale/over-bound resumes,
   price all-stage reservations/acks/unused final rows and response controls.
3. Drain failed/cancelled online work before issuing material or publishing/reusing
   state. Retire unseen committed epochs. Preserve remote operator separation and
   privacy audits with zero Preparation requests during active execution.
4. Installed SDK parity on Qwen2/Qwen3/Llama and the pinned public checkpoint, natural
   EOS and stochastic sampling; then matched all-link bodies/streams, CPU, RSS, pause
   latency and complete time against current demand allocation.

Existing `BenchmarkTarget.preparation`/`stage` isolate installed stage kernels but
have no same-generation epoch orchestration contract. Existing Conversation provides
admission/response-owned state mechanisms, not this public API. No new SDK option is
asserted by this note.

## Serial replay, parent-owned

Only existing local checkpoints are accepted; no downloads or installs. The launcher
freezes actual installed PLLM and NumPy/msgpack trees, native binary, dependency
RECORDs, driver/tests/helper, checkpoint contents and historical control digests
before and after the worker. Imported code is installed wheel code, not a repository
`PYTHONPATH` replacement. Checkpoint/source hashing and imports remain charged owners
outside any future isolated timing windows. Private keys/seeds/tokens/snapshots are
never written to evidence; output/state use a fresh private cohort salt commitment.

```sh
target/frontier-sdk-env/bin/python -m pytest -o pythonpath= -q tests/test_decode_epochs_probe.py

target/frontier-sdk-env/bin/python scripts/probe_decode_epochs.py \
  --checkpoint "$HOME/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775" \
  --input-tokens 16 --cap 32 --used 5 --epochs 4 16 --bound 256 \
  --memory-bytes 4294967296 --output target/decode-epochs-qwen-minimal.json

target/frontier-sdk-env/bin/python scripts/probe_decode_epochs.py \
  --checkpoint "$HOME/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775" \
  --input-tokens 16 --cap 128 --used 17 --epochs 4 16 32 --bound 256 \
  --memory-bytes 4294967296 --output target/decode-epochs-qwen-cap128-replay.json
```

This minimal single serial worker loads the checkpoint once and executes nine
matched cells (three controls, six candidates), including full-cap controls.
The lock is `~/.cache/pllm/frontier-probe.lock`; host reserve/swap admission and
`MemoryWatchdog` come from `scripts/frontier_probe_support.py:fresh_workers`.
An existing output path rejects rather than overwriting another cohort.
The second command replays the separately completed 16 input / 128 cap / 17 used
screen with epochs 4/16/32. The 150/512/80 case remains geometry-only until explicitly budgeted.
CPU/GPU timing and independent-role RSS need the serving extension above; this
entry point currently measures only actual serialized reference bodies and parity.
