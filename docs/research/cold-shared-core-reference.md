# Cold single-request shared-core alternative (#6)

## Result and scope

**Distinct-weight download can shrink; inference depth, quality and KV do not
shrink automatically. Fully local private execution is an alternative topology,
not a predominantly-remote-compute optimization.** No TEE, checkpoint weights,
hardware timings, heavy training, builds or installs used here.

`scripts/probe_cold_shared_core.py` independently implements a small causal
single-head decoder: shared attention/MLP matrices, separate per-depth RMSNorm
scales and diagonal adapters, separate embedding and vocabulary head. This is
an executable architectural fixture, **not** a reproduction of Huginn, Qwen's
gated recurrence or an evaluated language model. NumPy float64 random weights
prove only accounting and computation contracts.

Independent oracle computes entire causal sequence with matrix attention;
decoder computes one token at a time with depth-indexed KV. Tests also cover
greedy autoregressive continuation. Exact materialized copies of a model
**defined with tied weights** produce bit-identical oracle outputs. Incremental
versus full-sequence oracle error is below 1e-12 (floating-point association
differs). This does not prove training, quality or bitwise equivalence of all
kernels.

Counterexample: replacing independently initialized untied blocks by their mean
changes logits by max absolute **1.1844**; replacing all blocks by first block
changes logits by **2.2920**, on fixed eight-token fixture. Posthoc tying is a
model change, not lossless checkpoint compression. Paper's checkpoint EMA is
averaging *training checkpoints*, not averaging untied depths.

## Ownership and cold artifact

For width `w`, depth `d`, vocabulary `v`, fixture parameter counts are:

- shared: `6w² + 3dw + 2wv`;
- untied/materialized: `6dw² + 3dw + 2wv`.

Six shared matrices remain one owner. Two norms and one diagonal adapter per
depth remain distinct. Embedding and head remain distinct in this fixture.
Explicit object identity, not byte equality, defines sharing. At `w=8,d=4,v=19`:

| Ownership | Distinct parameters | Float64 weight payload |
| --- | ---: | ---: |
| Shared core | 784 | 6,272 B |
| Untied or copied unrolling | 1,936 | 15,488 B |

Complete synthetic downloadable bundle also charges manifest, core-owner map,
integer-ID tokenizer specification, framing and **entire execution source**.
Actual deterministic bundle bytes/digests are in
`docs/evidence/cold-shared-core-synthetic.json`. Python/NumPy are explicitly
preinstalled; provisioning them is additional if absent. Manifest models a
synthetic integer vocabulary, not an omitted real text tokenizer. This file
format is accounting fixture, not production signed package/distribution API.
Downloaded arrays are not assumed free or already warm.

Depth-specific KV is unchanged: `2*d*t*w*8` live bytes. Eight fixture tokens
need **4,096 B KV**, **32 block calls**, **27,008 dense matmul FLOPs** and
**4,608 attention matmul FLOPs**, tied or untied. Different depths yield
different K/V even though projections share weights. Blindly sharing cache
would change computation. No constant-state claim follows.

Resident memory lower bound is unique weight payload plus live KV. Paged model
keeps one block plus embedding/head/all depth extras resident; untied blocks
can be loaded successively but still need download/storage and repeated page
reads. Fixture reports both window and matrix-touch estimates. Tying can allow
resident core reuse, but does not imply core fits CPU/GPU cache or removes
memory-bandwidth costs. Page cache, BLAS, Python, activations, allocator and
runtime are excluded: **payload/window estimates are not RSS**. No empirical
RSS/timing claims. Default arrays are tiny; explicit geometry/horizon caps keep
individual execution under 128 MiB added-array budget. Avoid accumulating
unbounded fixture instances.

## Primary specifications and pinned consumer path

1. Dehghani et al., [Universal Transformers, v3](https://arxiv.org/abs/1807.03819v3)
   specifies self-attentive recurrence across depth and optional adaptive
   halting. This is architectural ancestry, not a consumer decoder checkpoint
   benchmark or proof that arbitrary pretrained Transformer layers can be tied.
2. Geiping et al., [recurrent-depth paper, v2](https://arxiv.org/html/2502.05171v2),
   §§3–4: prelude, injected embedding/latent-state adapter, four-layer shared
   core and coda; variable-depth pretraining and truncated backpropagation.
   Huginn shape `(2,4,2)`, width 5,280, intermediate 17,920, 55 KV heads × 96,
   vocabulary 65,536; depth at 32 loops is **132 executed layers**, not eight.
   Recurrent core norms/adapter are shared across loops in this real design;
   fixture's per-depth extras are deliberately conservative separate owners.
3. [Huginn checkpoint](https://huggingface.co/tomg-group-umd/huginn-0125/tree/bb6621b65e90b6a4b9b29ef88dc83866d450470c),
   pinned revision `bb6621b65e90b6a4b9b29ef88dc83866d450470c`.
   Read config, model card, safetensors index and model source only as provenance;
   no upstream import/execution. SHA-256s and revision-locked file inventory URL
   are in evidence. `raven_modeling_minimal.py` lines 542–580 construct shared
   core; `iterate_forward` around 727 loops it. `HuginnDynamicCache.update`
   around 147–200 indexes state by unfolded step. Card explicitly warns normal
   KV caching must not overwrite earlier recurrent steps.

**Actual uploaded artifact matters:** checkpoint index reports 3,911,400,096
stored FP32 parameters / **15,645,600,384 B** tensor payload. Model card notes
duplicate tied embedding in uploaded checkpoint. Four safetensors files
including headers total **15,645,609,096 B**. Pinned required runtime files
(weights, index, config, generation config, two Python files, tokenizer and
special-token metadata) total **15,648,442,384 B**. Media, README and git
attributes excluded because not execution dependencies; including every repo
file would be **15,651,519,639 B**. Downloading FP32 then loading BF16 still pays
FP32 download. HTTP envelopes, runtime dependencies and cache/security checks
are additional; inventory fetched without downloading weights.

Removing one duplicated 346,030,080-parameter embedding gives estimated
3,565,370,016 unique parameters, ~**7.131 GB BF16 payload**. Hypothetical 4-bit
payload lower bound ~**1.783 GB**, before scale/packing metadata. Neither is a
verified distributable compact checkpoint here. No quality equivalence asserted
for quantization, altered initialization, reduced loop count or cache sharing.

At 4,096 context and BF16, full depth-indexed KV estimates are **11.419 GB** at
32 loops or **3.114 GB** at eight loops. BF16 unique weights plus 32-loop KV
already exceed **18.55 GB** before scratch/runtime. Consumer CPU RAM may hold
this, but no cheap CPU speed follows. 8–12 GB consumer GPU cannot fit that
32-loop/context resident case; 24 GB GPU might fit only after measured admission.
Paper §6.2 explores bounded cache slots/overwriting and reports benchmark
results. That is an explicitly different recurrence/cache contract, not an exact
consequence of tying. Paper's per-token/latest-depth lookup likewise needs its
own quality and execution tests.

Credible next experiment requires an **independently implemented, model-neutral
PLLM recurrence adapter** for a pinned checkpoint, with loop budgets 8/16/32,
fixed random-state seed, short prompt/output and explicit full-cache mode on
consumer CPU or GPU. Upstream implementations are specification/provenance/oracle
only: do not import or run upstream model code as a runtime dependency. Derive
parameter ownership, embedding injection, normalization, initialization and
depth-indexed cache contracts from pinned specifications; validate independent
execution against permitted oracle evidence before quality/resource experiments.
CPU requires an independently supported attention path; published optimized
GPU/training paths do not establish CPU execution. No adapter or checkpoint
experiment is implemented here; no family-specific module is prescribed.

## Cold break-even, not warm-model accounting

For single cold request, compare complete local download `D`, local import/
conversion/preload `A`, local execution `C_local(p,n,r)` against prepared graph's
critical-path setup/issuance `S`, complete per-link bytes `G`, RTT/round trips
and remote/client execution `C_prepared`. Serial approximation:

`T_local = D/B_download + A + C_local`

`T_prepared = G/B_effective + S + C_prepared + RTT_cost`.

Real directional links and overlap need dependency/critical-path accounting,
not sum of all-link bytes divided by client downlink. If total `x=p+n` and
illustrative prepared bytes `G=G0+g*x`, then local wins when

`(g/B + 1/s_prepared - 1/s_local)*x > (D-G0)/B + A-S-RTT_cost`.

Evidence runs sensitivity scenarios, **not calibrated prepared-graph prediction**:
uploaded Huginn complete runtime files `D=15.648 GB`; `G0=64 MB`, `g=8 MB` per
input/output token; effective local 2 or 20 tokens/s, prepared 100 tokens/s;
100 or 1,000 Mbps. These rates include neither real prefill/decode asymmetry
nor attention's context growth. Setup/import/RTT assumed zero for both;
adjust terms before deployment. Same byte assumptions across workloads allow
inspection, not matched-quality comparison.

| Prompt+output | 100 Mbps, local 2 t/s | 100 Mbps, local 20 t/s | Prepared scenario |
| --- | ---: | ---: | ---: |
| 32+8 | 1,271.88 s | 1,253.88 s | 31.12 s |
| 512+128 | 1,571.88 s | 1,283.88 s | 421.12 s |
| 4,096+512 | 3,555.88 s | 1,482.28 s | 3,000.32 s |

100 Mbps crossing: ~8,312 total tokens at 2 t/s, ~2,078 at 20 t/s.
1,000 Mbps crossing: ~5,195 at 20 t/s; at 2 t/s local is slower for all
nonnegative token counts under these assumptions (negative algebraic crossing
is not an attainable workload). Short first request loses strongly on actual
uploaded artifact. Quantized/repacked **complete** artifact and quality-validated
lower loop budget could change this; neither can be assumed.

Existing prepared-graph context, not a matched competitor:
`docs/research/prepared-delivery-reference.md` records Qwen2.5-0.5B 150+8-token
cold control **476.953 MB covered all-link application bodies**, **127.451 s**
complete request at native 100/40 Mbps caps. Its counters exclude checkpoint
distribution and physical wire; checkpoint/quality differ from Huginn. Do not
equate warm graph bytes or covered application counters with fully priced cold
first install. Replace scenario `G,S,C_prepared` with exact target graph's fresh
artifact inventories, preparation/upload, tokenization and client work before
making any ranking. No remote-predominance result follows from local execution.

## Compact boundary and privacy

Fixture eight IDs need 16 B at uint16; width-eight FP32 hidden boundary 256 B;
per-message int8 plus float32 scale 68 B. Reconstruction error ~0.00297:
**not lossless**, not quality validated, and quantization is not encryption.
Real vocabulary 65,536 can use uint16 IDs but special tokens/contracts need
validation. A trained latent bottleneck or discrete codebook requires its own
training, quality and leakage gate. Remote latent/token transmission leaks
information unless protected by a proven protocol. Compressing clear boundary
does not preserve remote privacy requirements.

Fully local inference has **zero inference network boundary bytes**, not zero
cold download. Download service can observe model request/IP; execution must
disable telemetry, remote tools and external callbacks. Model confidentiality
from client is also lost when full weights are delivered. Token/boundary byte
compression does not excuse complete weights, tokenizer, runtime or state.

## Training/distillation/no-training routes and next gate

- **No training:** use checkpoint already trained/defined with shared core;
  repack true tied duplicate tensors without changing semantics. Separately
  validate serialization loader, dtype and numerical identity. Existing shared
  checkpoint is viable starting specification, not same-quality replacement for
  current target model. Reduced depth/adaptive exit/cache overwrite are quality
  changes; enforce workload-specific acceptance.
- **Distillation:** train shared recurrent student against untied teacher, with
  fixed/variable loop budgets and optionally learned compact boundary. Teacher
  tokens/activations, student training cost and consumer inference must be priced.
  No exact conversion guarantee; model and task quality can deteriorate.
- **New training:** pretrain shared architecture with variable-depth objective,
  normalization/initialization designed for stability. Paper reports failed
  recurrence setups and large-scale pretraining; cannot call this cheap consumer
  adaptation. Small task-specialized shared models could be separately trained,
  but random tiny fixture supplies no evidence of useful quality.
- **Not valid no-training route:** average arbitrary untied blocks, reuse first
  block, or claim same model because dimensions match.

**Next actionable gate:** make local-topology change, target task/quality floor
and consumer memory/download budgets explicit; architecture changes are already
permitted, not pending approval. Independently implement a model-neutral PLLM
recurrence adapter for a pinned checkpoint, without upstream model-code runtime
dependencies. Before loading, require complete distributable artifact inventory/
digests, independent numeric/cache-contract checks and compatible consumer CPU/
GPU execution with resource admission. Then run one isolated checkpoint
experiment (no concurrent heavy jobs): measure cold download/import/conversion,
TTFT, complete request, p95 decode, peak RSS/VRAM/swap and quality at loop budgets
8/16/32 for short/medium/long workloads against equally cold prepared graph.
Stop if full artifact exceeds download/memory budget or quality fails. Published
Huginn's ~15.65 GB cold payload loses for short requests in these scenarios;
tying alone does not reduce depth-indexed KV. Synthetic results pass accounting/
correctness gate only; no production promotion yet.

## Replay

```sh
target/frontier-sdk-env/bin/python -m pytest -q -o pythonpath= tests/test_cold_shared_core.py
target/frontier-sdk-env/bin/python scripts/probe_cold_shared_core.py \
  --output docs/evidence/cold-shared-core-synthetic.json
# Optional: verify four small source/config/card/index hashes, never weights:
target/frontier-sdk-env/bin/python scripts/probe_cold_shared_core.py --verify-sources
```

Evidence embeds script/test SHA-256s; source revisions/hashes are stable and
default replay needs no network. Float64 tolerance/oracle and scalar estimates
are portable; low-order BLAS differences may alter recorded numerical digits
and artifact digest changes whenever execution source changes. Replay is
deterministic in same installed environment. No clock/RSS samples mixed into
JSON. Only owned four files changed; no commits or SDK edits.
