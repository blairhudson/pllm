# Existing constant-state checkpoint screen

## Scope and present result

This slice selects existing open checkpoints, never trains a replacement model.
Pure recurrence removes attention KV growth; it does **not** establish equivalent
quality, cheaper protected nonlinear execution, or a smaller complete client.
Qwen3.5 is a hybrid and must not be presented as constant-state end to end.

Current executable result: independent local-clear stable weighted-exponential
state contract (RWKV-v4 projected WKV), and an owned gated-delta decode contract
reusing existing `BoundedGatedDeltaRecurrence` geometry and reference. The latter
keeps one state slab and fixed outer-product scratch instead of exporting a fresh
state copy at every token. It is a research script, not a new production runtime
or compiler operator. No Rust additions or build required. No checkpoint values
were loaded. Thirteen focused NumPy-only regression checks and focused Ruff lint
passed after serial measurement.

## Serial measured result

`docs/evidence/research-constant-state-measured.json` contains **18 fresh timing
workers**: three repeats for each of 64, 256 and 1024 one-token transitions, for
both unchanged existing `copy_reference` and owned candidate. Total executed
timing transitions: **8,064**. All **nine matched pairs** have identical final
public-synthetic output and state SHA-256 commitments (dtype/shape framed), with
finite outputs/state. Control and candidate match exact F32 bits, not merely
tolerances. The measured paths are both the original gated-delta semantics;
neither drops work, changes model family, truncates state or compresses history.

| Steps | Median control CPU s | Median owned CPU s | CPU reduction | Median control peak RSS bytes | Median owned peak RSS bytes | RSS reduction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 0.071839 | 0.038852 | 45.92% | 49,725,440 | 45,383,680 | 8.73% |
| 256 | 0.281882 | 0.210810 | 25.21% | 51,740,672 | 46,055,424 | 10.99% |
| 1024 | 1.231705 | 1.007533 | 18.20% | 53,952,512 | 45,875,200 | 14.97% |

Median wall seconds, control/owned: 64 steps **0.072671/0.038913**; 256 steps
**0.288129/0.214529**; 1024 steps **1.297826/1.036537**. Longest cohort yields
**1.2225× median CPU speedup**, not 100×. All nine paired CPU ratios favor owned
state, ranging 1.2060–2.4470×; short runs have substantial relative variance and
are not a basis for a whole-response claim. No independent repeat cohort beyond
these three repeats was run.

Both paths retain exactly **2,097,152 state bytes**, independent of elapsed
sequence length. Candidate owns **4,259,840 persistent bytes** including bounded
state-sized outer-product scratch and q/k/delta/output buffers. Benefit is
avoided per-token state export/allocation churn, not smaller recurrent state or
better retention capacity. Process peak RSS includes NumPy/runtime imports and
setup allocations; CPU and wall timers cover the recurrence loop only. Source
freeze/hash verification, child startup, imports, setup, oracle execution and
commitment hashing are outside those CPU/wall values and their complete costs
remain unmeasured, not zero. No weights, GPU transfer, crypto keys, protected
protocol or peer link is exercised. Seven serving metrics remain null.

One additional fresh oracle worker executed **44 upstream-oracle rows**:
17 tiny gated-delta rows, four rows at pinned Qwen state geometry, and 23
eight-channel stable weighted-exp rows including strongly scaled keys. Gated
delta is also exact-bit equal to existing reference on its 21 oracle-cohort
rows. Independent upstream Torch F32 agreement uses separate `rtol=2e-5`,
`atol=2e-6` contracts: maximum absolute output/state error **5.96046448e-8**
(tiny gated delta), **4.47034836e-8** (checkpoint-shaped gated delta), and
**1.19209290e-7 output / 0 state** (weighted-exp). This is numeric operator
evidence only, not source checkpoint values or full-model fidelity/quality.

Runtime identities: NumPy **2.5.3**, Torch **2.14.0**; installed build digest
`d4c5bc12436c5354c1974b2731c5ccd45ef6e68f35c22023c5dd057e78b0a04e`.
Measured script SHA-256:
`bfd45e61b3e90f44e6362ee6b5aa26a074a172a20940044b68afa6497ce7cd01`.
Actual executed `semantic_hybrid.py` bytes:
`55bc7d78060cc382acd6cba96e648d25beddfd24ffea3717a0c2b2248ab82903`.
Evidence SHA-256:
`8ce9f4facbb135a2908f7573ec39b453273305dd936bd0847f265aaa8e2e1d6d`.
Script, tests, helper, audit, all three configs/oracle source files and actual
reference source were frozen **before** checks/workers and verified unchanged
afterward. The complete manifest is retained in evidence. Python executes the
hashed reference bytes directly rather than accepting stale bytecode caches.
The exclusive shared lock, offline children, 768 MiB post-reserve admission,
build identity check and swap-pressure watchdog completed without a blocker.
Actual Torch-oracle peak RSS was not recorded and is unknown.

**Narrow admission:** retain this as an exact reusable recurrence ownership
optimization research candidate. Integration into the existing semantic decoder,
whole-response benefit, protected state operators/role ownership, full source
import, task-quality equivalence and long-context checkpoint behavior all remain
open. Pure RWKV and Falcon state/config estimates are separate from this measured
Qwen-shaped projected-operator cohort; no pure-recurrent full model was executed.

## Immutable source identities and architecture audit

`docs/evidence/research-constant-state-audit.json` records config and source URLs,
revisions, SHA-256 identities and observed local cache inventory. Config/source
bytes are retained under `docs/evidence/research-constant-state-sources/` solely
as specification/oracle evidence; upstream Apache-2.0 headers remain intact.
Transformers source revision: `536ecc007387a50e77603bb5d92100e9b07514cc`.
Upstream classes are never imported as runtime implementations. Serial numeric
oracle execution AST-selects only pure Torch fallback functions, strips hub
decorators and validates whole-file hashes first.

| Existing checkpoint | Pinned revision | Architecture and retained F32 state |
| --- | --- | --- |
| `RWKV/rwkv-4-169m-pile` | `46bdc280eb97b6141d5d51a935e0c4870ecaefcc` | 12 layers, width 768; two token-shift vectors plus three log-scaled WKV vectors per layer: 184,320 bytes (180 KiB), no KV |
| `tiiuae/falcon-mamba-7b` | `080ad94b3619e2c2d0afa59bafdc6113465b7006` | 64 SSM layers, intermediate width 8,192, state width 16, conv history 4: 41,943,040 bytes (40 MiB), no KV |
| `Qwen/Qwen3.5-4B` | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` | 24 recurrent + 8 full-attention layers; recurrence/convolution 53,477,376 bytes (51 MiB), **plus 65,536 F32 KV bytes per token** |

Falcon config includes `expand=16`, but pinned implementation uses explicit
`intermediate_size=8192`, not `hidden_size * expand`. State estimate is a uniform
F32 accounting profile, not a measured checkpoint/runtime allocation. Mixed
F32 SSM/BF16 convolution would instead occupy 36 MiB. Qwen's recurrent state is
`[1,32,128,128]` per recurrent layer; convolution is `[1,8192,4]`.
Eight attention layers retain two `[1,4,T,256]` F32 tensors each. At configured
262,144 positions, KV alone is 16 GiB (8 GiB if genuinely BF16). Neither changing
state dtype nor dropping attention layers is authorized by this probe.

## Context and quality gate

- RWKV config `context_length=1024`: this is the pinned training/kernel profile,
  not evidence of useful 100K-token task behavior. CPU recurrence can carry state
  past that length mathematically. This probe supports **1–4096 one-token state
  transitions**, not validated model contexts past 1024.
- Falcon config declares no maximum context position: report **unknown**, not
  unlimited supported context. Sequential SSM execution has no attention KV;
  actual long-context retention/task success remains untested.
- Qwen config declares 262,144 positions. Existing PLLM gated-delta/conv reference
  accepts **1–256 tokens per prefill call**, exactly one per decode call. A
  compiled decoder separately fixes input/output and attention state capacity;
  source's 262K bound does not mean this runtime has tested 262K execution.
  Prior real-checkpoint evidence covers two short prefixes and four next-token
  comparisons (W8A8 2/4 top-1; float32-body diagnostic 3/4), not long-context
  equivalence. This probe's owned decode horizon is explicitly 4096 steps.

Admit a new pure-recurrent checkpoint only after separately evaluating the exact
checkpoint/tokenizer on held-out user-relevant instruction following, completion,
long-document retrieval, multi-hop facts, and state distraction at 1K, 4K, 16K,
64K and 128K where the source genuinely permits testing. Compare task success
and calibrated quality with the current target under equal task budgets, not
equal state shape. Pre-register acceptance thresholds with the user. Numeric
operator agreement is necessary but cannot satisfy this task-quality gate.
RWKV-169M is an old small base model, not an assumed instruction-quality peer.
Falcon-Mamba-7B likewise cannot replace the current task target without this gate.

## Checkpoint availability, boundary and full-import floors

Pinned public repository metadata, inspected without downloading weights:

- RWKV public `pytorch_model.bin`: 677,440,778 bytes; absent from observed standard
  Hugging Face local cache. A pickle checkpoint needs trusted/source-bound safe
  loading or a validated safetensors conversion, then a complete value map.
- Falcon public three safetensors shards: 4,956,184,024 + 4,987,536,920 +
  4,601,680,888 = 14,545,401,832 bytes; absent from observed standard local cache.
- Pinned Qwen snapshot present: two safetensors files, 9,319,828,096 bytes including
  outer multimodal checkpoint data. No tensors loaded in this stage. Existing
  text adapter is available; new owned state is not integrated into that decoder.
  Cache inventory is not proof of complete shard/content validation.

Boundary-only original-dtype weight payload, excluding layer weights, scales,
packing and runtime owners:

| Checkpoint | Embedding/output boundary floor | Notes |
| --- | --- | --- |
| RWKV | 308,901,888 bytes at source FP32 | Two untied 50,277 × 768 tables; an int8 hypothetical is 77,225,472 bytes before scales |
| Falcon | 1,065,353,216 bytes at BF16 | Two untied 65,024 × 4,096 tables; int8 hypothetical 532,676,608 bytes |
| Qwen | 1,271,398,400 bytes at BF16 | One tied 248,320 × 2,560 table; F32 materialization alone is 2,542,796,800 bytes; int8 635,699,200 bytes |

These are payload floors under stated representations, not physical RSS claims
or authorized quantization changes. Tied weights may acquire separate embedding,
quantized head, packing or device owners. Full source import must count mapped
weights plus resident/conversion copies, all linear layers, logits, projections,
state/workspace, Python/native/Torch/crypto libraries and private protocol material.
Actual private-client runtime floor is **unknown**, not zero. Even tiny recurrent
state does not remove a large local embedding/head requirement.

Full import feasibility: Qwen uses an existing reusable hybrid text adapter and
has real short-prefix evidence, with documented numeric/quality gates still open.
RWKV needs token-shift, mixed-key/value/receptance projections, stable weighted-exp
state, LayerNorm, squared-ReLU channel mixer, per-six-layer rescale semantics,
embedding/head and tokenizer. This probe only supplies the WKV numeric contract;
there is no full-model lowering or protected WKV protocol. Falcon needs selective
SSM discretization/scan/state, causal depthwise conv, input-dependent B/C/dt RMS
normalization, SiLU gating, residual normalization and weight binding. Existing
conv/linear primitives help but do not provide the complete SSM adapter. Existing
Falcon source also supplies newly evolved weightless RMS defaults: complete import
must bind config defaults/version semantics, not read dimensions alone.

## Reusable contracts and resources

Gated delta, after grouped-head expansion and L2 normalization:

1. `S <- exp(g) S`
2. `delta <- beta * (v - k^T S)`
3. `S <- S + k delta^T`
4. `y <- q^T S / sqrt(key_dim)`

This retains F32 state `[1,Hv,K,V]`. Work is O(Hv K V) per token, including four
state-sized multiplication traversals plus reductions/updates; projections,
convolution, nonlinear coefficient production and complete decoder are excluded.
For Qwen geometry, state is 2,097,152 bytes per layer. Candidate also owns an
equally large outer-product scratch, q/k buffers and delta/output buffers:
**4,259,840 counted persistent bytes**. Temporary normalization/finite-check
arrays and NumPy/runtime allocation are additional. Constant-state does not mean
scratch-free or lower peak RSS. Current control copies the prior state and
allocates an outer product; the serial cohort above measures a scoped CPU/RSS
advantage, with unchanged retained state bytes.
Control and candidate own one evolving state; caller consumes each output and
discards it before the next token. Candidate output is borrowed and overwritten;
callers retaining history must charge explicit copies. Both execute identical
public synthetic inputs, same F32 geometry and one-token lifetime.

Weighted-exp keeps three vectors: `N,D,M`, representing `N exp(M), D exp(M)`.
For output, use `P=max(M,k+first)` then rescale numerator/denominator to P.
For next state, use `P=max(M-exp(log_decay),k)` and add the rescaled new item.
This independently implemented contract is O(channels) per token and three
F32 vectors retained, plus fixed-size coefficients and bounded temporary vectors.
It is deliberately separate from gated-delta and selective-SSM semantics.

Owned gated-delta input/geometry checks fail before mutation. Numeric failure
zeros state and burns the owner; rollback is not promised. Research owner is
single-thread/single-party local-clear. No mask/key material, network traffic or
independent production-party privacy is implemented or implied.

## Reproduction and serial-stage admission

Read-only interpreter:
`/Users/blairhudson/workspace/pllm/target/frontier-wheel-env/bin/python`.

```sh
# Initial stage: small pinned config/source retrieval only; output must be new.
python scripts/probe_constant_state.py --audit --output docs/evidence/research-constant-state-audit.json
python -m pytest --noconftest -q tests/test_constant_state_probe.py

# ONLY AFTER coordinator authorizes exclusive serial measurement:
python scripts/probe_constant_state.py --measure --output docs/evidence/research-constant-state-measured.json
```

Tests bypass repository-wide conftest to avoid native/server imports in this
low-memory stage. Ordinary pytest initially failed because the unbuilt isolated
worktree's `pllm._native` is absent; no build attempted. Serial script uses the
installed wheel through `fresh_workers`, not isolated unbuilt package imports.
No Cargo/maturin build needed. Shared fail-fast cohort lock, build identity,
four-thread caps, offline workers, host reserve and swap watchdog remain active.
Admission requests 768 MiB; estimated peak dominated by Torch oracle import,
roughly 0.4–0.7 GiB. No GPU or model loads. Three repeats of fresh matched
control/candidate pairs use 64, 256, 1024 decode steps at pinned
Qwen state geometry; separate fresh oracle checks tiny/full-size gated delta
and RWKV projected WKV. Evidence records build/probe/reference hashes, local CPU,
wall time and process RSS. All seven serving metrics remain null: this is not a
whole-response or protected serving experiment. Serial execution is complete;
independent confirmation and serving integration remain required.
