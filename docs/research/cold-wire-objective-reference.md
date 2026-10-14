# Cold single-request training against protocol bytes

## Verdict and scope

Promising **changed-model** research objective, not an admitted Qwen optimization.
`scripts/probe_cold_wire_objective.py` independently implements bounded public
synthetic integer distillation, exact residue codec and specified MessagePack
body-size oracle. It checks installed native/layout/request/correction/response
serializers. No real checkpoint loads, provider runs, timing cohorts, installs,
builds, or shared SDK changes. No existing exact Qwen identity changes.

Evidence: `docs/evidence/cold-wire-objective-synthetic.json`. Real task quality,
peak RSS, transport, latency and real weight-width distributions remain **unknown**.
Declared extra-array design bound is 8 MiB, not a measured RSS result. Largest
training arrays are 128×32 int64; teacher has 256 integers. Hypothetical geometry
uses sizes rather than allocating full weight/payload arrays. No GPU or TEE used.

## Exact compiler/protocol target

Let public integer weight row be `q_r`, public supported activation maximum be
`A` (127 for symmetric A8, 7 for A4), and `L_r=sum(abs(q_r))`. Then

```
B_r = A L_r
b_r = max(1, bit_length(B_r) + 1)
L_r <= floor((2^(b_r-1)-1)/A)
```

Signed reconstruction requires **strict** `B_r < 2^(b_r-1)`: positive endpoint
cannot be included. Zero row still costs one bit. For A8, width 16 requires
`L<=258`; 259 needs 17. Width 20 permits 4,128; 4,129 needs 21. Width 21 permits
8,256; 8,257 needs 22. Width 24 permits 66,052; 66,053 needs 25. Full threshold
table includes A4 in evidence. Negative i8 -128 contributes 128 to L1; ordinary
symmetric quantizer emits [-127,127]. No observed activation maximum substitutes
for public worst-case bound. Lower A must be enforced by clipping/quantization
for **every** private input under separately declared numeric identity.

Specifications read directly:

* `crates/pllm-core/src/offset_transport.rs`: `row_residue_bits`, `pack_row_residues`,
  `reconstruct_rows`, `pack_difference`, `add_packed`.
* `python/pllm/runtime/preparation_protocol.py`: `seeded_ring_profile`,
  `PreparationRequest.pack`, `CorrectionPush.pack`.
* `python/pllm/runtime/stage_protocol.py`: `PreparedStageBatchRequest.pack_encoded`.
* `python/pllm/runtime/residue_codec.py`: `encode_layout`, `pack_row_response`.
* `python/pllm/runtime/prepared_stage.py`: `stage_layout`; input bits are
  `max(output_residue_bits)` for `stage_packed`.

Arithmetic execution ring still rounds to 16/24/32 bits according to stage maximum
bound; row-residue body widths need not be byte multiples. For `R` rows and `N`
inputs, stage-packed input costs `ceil(R N max_r(b_r)/8)`, whereas each correction
and response tensor costs `ceil(R sum_r(b_r)/8)`. Thus one wide blocker row can
prevent **all input savings**. Per-body terminal padding and MessagePack bin
headers at 256/65,536 bytes matter. Row width reductions without a packed-byte
boundary crossing save nothing. Tensor widths also change compressed public layout
bytes; public delivery and lifecycle metadata cannot be silently dropped.

Concrete original traffic:

1. Client→Preparation: compact request with public session, 16-byte attempt,
   row count and fresh 32-byte seed. Public model/stage/weight/ring context was
   already registered; compact request does not eliminate registration costs.
2. Preparation→Inference: fresh one-use correction `W*r-s`, full identity/ring
   metadata, duration and inline zlib/base64 public widths. Correction is not a
   reusable checkpoint or cache object.
3. Client→Inference: `PLLMPSB3`, uint32 big-endian row count, MessagePack array
   `[batch128, concatenated row tickets128, columns, wire_bits, input_bits, data]`.
   Data contains masked integer `x-r`, not original float activations. Each row
   carries a distinct ticket; batch and row ticket fields are charged separately.
4. Inference→Client: `PLLMPR01`, MessagePack
   `[ticket_string32, stage, rows, wire_bits, width_digest32, server_ns, data]`.
   Data contains `W*(x-r)+(W*r-s)`. Client adds private `s`, reconstructs centered
   integer result, applies client-held activation/weight scales and bias.

All values are packed LSB-first in row-major order with fixed public widths.
Discarded input high bits vanish modulo every output ring because
`input_bits>=b_r`. Probe verifies this with actual integer matmul, native packing
and native reconstruction at 1/3/16 rows. Deterministic test masks are **oracle
fixtures only**, never valid production randomness. Scales of private activations
remain client-side. No source activations, residual sparsity, private maximum,
training-derived private threshold or prompt-dependent width is published.
Public weights/layouts are not a model-weight secrecy mechanism. Usual separate
Inference/Preparation non-collusion and authenticated one-use ownership assumptions
remain necessary; consumer software without TEE does not enforce non-collusion.

## Training objective, not nominal W4

Useful objective is task/distillation loss plus weighted **all-directed-body**
cold request bytes, subject to public row-bound and privacy constraints:

```
min loss_public_train(Q, scales, A) + lambda * bytes_cold_public_prep_online
subject to Q in signed i8, public A enforced, fixed public widths and source identity
```

Enumerating public width budgets and minimizing loss within each budget is a
constrained alternative to differentiating discontinuous byte cost. This probe
uses exact integer SSE marginal costs: reduce coordinates toward zero until row
L1 budget holds, then two bounded coordinate-descent passes. It learns integers
from public train inputs/teacher outputs rather than merely assigning W4 or
shrinking floats. Fixed per-row scales are explicit. Whole-transformer QAT could
learn weights and scales with STE and a row-budget surrogate, but final selection
must recompile and score **actual serializers**, not surrogate L1 or precision.
This script does not implement LSQ or claim transformer fine-tuning.

Primary literature used as specifications, independently implemented; no external
implementation code imported:

* [Learned Step Size Quantization](https://arxiv.org/html/1902.08153), §2,
  equations 1–2 define clipped round-and-rescale; §§2.1–2.3 specify step-size/STE
  training, §3.7 teacher distillation. Task-loss learning can differ from minimizing
  quantization error (§3.6). Its ImageNet quality results are **not** Qwen evidence.
  Our discrete constrained SSE optimizer is an alternative, not its gradient rule.
* [SmoothQuant](https://arxiv.org/html/2211.10438), §2 equations 1–2 describe
  symmetric per-token/per-output-channel quantization; §4 equations 3–4 specify
  `X D^-1 * D W = X W` and public calibration-based diagonal equalization.
  Float equivalence does not imply integer/numeric/wire equivalence; inner-axis
  scales are not free and transformer residual/nonlinear boundaries constrain fusion.

Earlier `network-io-10x.md` / `prepared-precision-reference.md` already reject
quality-free precision/equalization promotion: W4A8 had 1/12 prefill selections,
A4 had 0/12, and equalized W8A8 lost three decode selections versus W8A8.
Those frozen PTQ/equalization choices did not optimize trained integer row budgets,
worst-row input cliffs and complete cold bytes jointly. Training offers extra
degrees of freedom and teacher/task feedback—not exemption from those failures.

## Frozen synthetic experiment

Seed 6103; train 128 public rows, held-out 96 disjoint public rows; eight output
rows ×32 input channels. Train/held-out samples uniform integers [-127,127].
Teacher rows have fixed 127 outlier coordinate; other coordinates [-48,48].
Synthetic quality uses exact teacher integer products, **not** natural-language
task performance. Width budgets/candidate list freeze before held-out scoring;
no held-out candidate fitting/selection. Eight additional teacher-sign-aligned
activation rows probe adverse worst-case bounds. These are stress cases, not
exhaustive quality evaluation.

| Frozen choice | Stage max input bits | Four directed stage bodies, 16 rows | Held-out relative MSE | Held-out argmax |
|---|---:|---:|---:|---:|
| unchanged W8A8 teacher | 19 | 2,695 B | 0 | 96/96 |
| trained row budget, -1 bit | 18 | 2,623 B | 0.00004252 | 96/96 |
| naive shrink, -1 bit | 18 | 2,623 B | 0.00034975 | 94/96 |
| trained row budget, -2 bits | 17 | 2,525 B | 0.14621 | 74/96 |
| coarser fixed scale ×2 | 18 | 2,599 B | 0.00040809 | 94/96 |
| same scale ×2, integer trained | 18 | 2,599 B | 0.00042620 | 96/96 |
| coarser fixed scale ×4 | 17 | 2,501 B | 0.00116449 | 91/96 |
| same scale ×4, integer trained | 17 | 2,501 B | 0.00131642 | 93/96 |
| public A4 identity | 14 | 2,235 B | 0.00503497 | 89/96 |

Mild training modifies only enough of blocker row to cross cliff: 2.67% covered
stage saving, not exact parity; worst held-out teacher error 746, adverse 2,032.
With two raw provider weight copies and client fp32 row scales, **known cold
payload plus stage bodies** is 3,239→3,167 B (2.22%). Manifest/control/full wire
unknown. More severe budget loses quality; sign-aligned relative MSE is 0.16437
and absolute error 67,564. Coarser-scale training improves argmax but worsens MSE;
not a universally better frontier. A4 sign-aligned error happens to be zero because
these stress values are exactly endpoints: zero adverse error alone is misleading.
Naive -2-bit shrink leaves one row at width 18 due to rounding, despite intended
budget. This prices failed threshold crossing rather than assuming it succeeds.

### Quantizer rescaling and exact equivalence screen

Positive scalar shrink followed by existing per-row max-normalized W8 quantization
does **not** lower integers: both float weights and quantizer step shrink together.
Probe verifies scalar factor 1/4 recovers identical integers and widths. Thus an
L1 penalty on float weights alone can be ineffective. Fixed/coarser learned scales
must survive export; re-normalizing each row to peak127 can destroy byte savings.

Constructed even-valued integer rows permit exact `Q'=Q/2`, `scale'=2*scale`.
All integer products times scales are identical for every input; one bit per row
is saved, no extra axis scales, existing eight fp32 row scales still 32 B.
Constructed stage bodies fall 2,811→2,715 B. Power-of-two scaling avoids additional
rounding at dequantization in ordinary finite non-overflow ranges, but full decoder
bit parity still needs an end-to-end gate. General rows need common divisor>1;
odd coordinates block this construction. No real Qwen divisibility observation.
Compiler must preserve authoritative integer values/scales; today's ordinary
float-source max-normalizing import is not silently equivalent to such export.

## Separate whole-request hypothetical geometry and all cost axes

Evidence explicitly labels **hypothetical** 24-layer body, hidden896/intermediate4864,
Q896/K128/V128/O896, gate/up/down; 357,826,560 body parameters and 304,128 fp32
output scales. Client head/embeddings/nonlinear artifacts omitted and unknown.
One 39-input/8-output terminal-pruned request executes 39+7 rows. Prefill frames
16/16/7 and seven single-row decode frames, request-sized Preparation issues 46
rows once per stage. Real on-demand windows, inventory maximums and source shapes
must be checked before matching an SDK cohort. Widths uniform 24→23 are assumptions,
not fitted or scanned Qwen distributions.

| Direction/cost | Uniform24 | Uniform23 |
|---|---:|---:|
| Publisher→Inference raw signed-i8 weights | 357,826,560 B | same |
| Publisher→Preparation raw signed-i8 weights | 357,826,560 B | same |
| Publisher→client fp32 weight row scales | 1,216,512 B | same |
| Client→Preparation + Preparation→Inference framed bodies | 42,058,354 B | 40,309,618 B |
| Client→Inference + Inference→client framed bodies | 76,243,060 B | 73,081,204 B |
| Known hypothetical subtotal | 835,171,046 B | 830,260,454 B |

One-bit uniform gain saves 4,910,592 B (~0.588%) when two provider raw deliveries
are cold; no tenfold claim. If provider weights already resident, cold **client**
request needs a different cache/delivery denominator. Stored i8 W4 values are
still one byte unless actual nibble/compressed delivery serializer is selected;
no automatic half-weight-delivery assumption. Entropy/compression bytes and CPU
must be measured from new artifacts separately.

No extra axes in implemented candidates (0 B). A diagonal input equalization
variant must add `4*N` bytes per unfused input-axis scale vector, price each
destination/copy, recompute public layouts and client arithmetic; it is not
implemented here. Existing row scales are `4*O` bytes. Model checkpoint is replaced,
not added to inference delivery alongside baseline: synthetic publisher fp32
checkpoint is 1,024 B; hypothetical body publisher checkpoint 1,431,306,240 B.
If a system sends float checkpoints, delivers public layouts separately or includes
provider scale files, add those directed bytes. They are not included in known
raw-weight payload subtotal. Master weights+gradients+two Adam slots alone require
5,725,224,960 B for hypothetical body; training activations, teacher, embeddings,
head, optimizer metadata and copies are additional. This is analytic resource
estimate, not observed training/RSS. Publishing privately trained checkpoints
can leak training data; private-data training privacy is not solved by this protocol.

Full request remains unknown: public manifest/artifact wrappers, tokenizer,
embedding/head/nonlinear delivery, registration/reservation, tickets/inventory
issue and ACKs, refill/cancel/lifecycle, authentication, HTTP/TLS headers, padding,
retries and transport compression must all enter final directed-body ledger.
Zero placeholders are not substituted for these unknowns. Fresh masks/material
cannot be amortized across unrelated requests. Evidence prices all four stage
directions and explicitly lists unpriced obligations; no incomplete subtotal is
called complete cold network cost.

## Concrete next Qwen quality/resource gate

1. Parent's **serial** source-locked real integer scan must identify actual rows
   within a small L1 distance of public bit cliffs and global-max blocker rows;
   check common divisors separately. Keep source/tokenizer/config/license hashes.
   Do not infer distribution from this synthetic teacher.
2. Publish a separate changed-checkpoint/quantizer Experiment with integer/scales,
   activation bound, width/layout, calibration/training cohort and artifact digests.
   Freeze candidate widths and loss/task budgets before held-out work. Exact Qwen
   comparator remains untouched. Use public train-only tokens; no private prompts.
3. First Qwen2.5-0.5B local-clear numeric screen: replay existing 12 held-out prefills
   and 36 same-token decode selections against upstream float reference **and**
   unchanged W8A8, full-logit error and per-row bound verification. Proposed narrow
   rejection gate: no lost unchanged-W8A8 selections and no larger worst error in
   either phase. Passing is only numeric screening, not representative task quality.
   Freeze a second independent public set (e.g. WikiText-2 test perplexity plus
   PIQA/HellaSwag task scores); train/calibrate on disjoint public data. Proposed
   task gate: relative perplexity rise≤1%, task-score drop≤1 percentage point with
   uncertainty reported. Policy budgets are proposals, not obtained measurements.
4. Budget publisher training before execution. Full body Adam state alone ~5.73 GB;
   full model/teacher/activations likely exceed modest 8-GB consumer headroom.
   Start streaming stage/block training or small trainable adapter and integer
   export, minibatch/gradient accumulation, no full `4864²` float64 covariance
   (~189 MB). Example 32×4864 row block fp32 master/grad/two slots is 2.49 MB,
   before input/output minibatches/teacher/SDK. No runtime estimate without measurement.
   Qwen3-4B full-parameter Adam minimum ~64 GB at 16 B/parameter is not an 8-GB
   consumer plan. Parent owns admission and serial real-weight scans.
5. Only after quality: serial matched cold one-request SDK cohort, same privacy,
   placement/prompt/output/cache/issued-row contracts. Count **all** cold public,
   Preparation and online directed bodies including acknowledgements/control;
   separately measure actual compressed artifact bytes, CPU/elapsed, peak client
   and provider RSS/swap, source privacy, output quality and full transport. Retain
   lower byte count only with declared quality/resource tradeoff. No current admission.

## Replay and validation

```
target/frontier-sdk-env/bin/python scripts/probe_cold_wire_objective.py \
  --output docs/evidence/cold-wire-objective-synthetic.json --check
target/frontier-sdk-env/bin/python -m pytest -q -o pythonpath=. \
  tests/test_cold_wire_objective.py
```

To generate a new report, omit `--check` and choose a new output path. Exact replay
includes source/script/native/installed serializer hashes and dependency versions;
different installed binary or zlib/serializer implementation can legitimately
require a new evidence identity. Current checks: 22 tests, native threshold oracle
both sides at A7/A127, zero/power-of-two/signed32 boundaries, LSB codec/padding,
MessagePack bin cliffs, global-max blocker, public constrained training,
1/3/16-row installed serializers and exact masked matmul/unmask, deterministic
evidence and hypothetical directed-body arithmetic. Pytest duration is a unit-test
runner report, not an inference latency or training benchmark.
