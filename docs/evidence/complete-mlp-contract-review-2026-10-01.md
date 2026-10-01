# Complete bounded protected-MLP contract and cost gate

Date: 2026-10-01. **Reject this specified conventional construction.** Its
conversion/material payload alone exceeds either saving gate by hundreds of
times. The complete mathematical function and two-worker interface are specified
below; a complete executable circuit/protocol is **not** implemented. Missing
coverage independently rejects admission, even with unlimited traffic budget.

This is a complete **MLP placement contract**, including its client boundary,
not another scalar SiLU lookup and not an admitted resident whole decoder.
No training, checkpoint modification, HE, TEE, or provider plaintext fallback.

## 1. Fixed source, actual semantics, and bounded execution

Source: `Qwen/Qwen2.5-0.5B-Instruct` at
`7ae557604adf67be50417f59c2c2f167def9a775`, W8A8 `SymmetricPerRow`.
The numeric target is the existing quantized semantic runtime, **not** the
unquantized Torch/BF16 checkpoint forward. Imported public weights and their
original float32 per-output-row scales are retained exactly.

The probe reads only cached config and six native-loader quantized MLP stages,
at layers 0, 12, 23. It checks compiler plan/schedule digests, exact operation-ID
coverage, semantic SiLU shapes, 48 MLP stage specs, source geometry and absence
of MLP biases. No operation-ID suffix parsing supplies semantic roles.

| Quantity | 39+8 | 39+32 |
|---|---:|---:|
| Body rows, `39 + generated - 1` | 46 | 70 |
| Layer-row MLP invocations | 1,104 | 1,680 |
| Gate/up pair elements | 5,369,856 | 8,171,520 |
| Native prefill / decode SiLU shape | `[1,39,4864]` / `[1,1,4864]` | same |
| MLP public integer MACs **per worker** | 14,434,172,928 | 21,965,045,760 |

24 layers; hidden H=896; intermediate M=4,864. Native gate/up is a fused
`9728 × 896` stage; down is `896 × 4864`. All prefill rows execute, even though
only physical-last logits choose the first generated token. Decode repeats
7/31 times; the last selected token requires no subsequent body execution.

### Actual widths from cached W8 weights

`B = 127 max_j sum_i abs(W[j,i])` is a whole-domain accumulator bound, not a
prompt-dependent measurement. `seeded_ring_profile(B)` gives actual prepared
width. Native quantization is enabled in this run (NumPy 2.5.3, arm64).

| Layer | Gate/up B / prepared width | Down B / prepared width |
|---|---:|---:|
| 0 | 4,453,890 / 24 | 18,993,866 / 32 |
| 12 | 4,339,590 / 24 | 17,455,007 / 32 |
| 23 | 4,507,357 / 24 | 18,888,456 / 32 |

All sampled weight and float32-scale hashes are in the JSON. These are sample
widths, not a claim that every unsampled stage has the same width. The chosen
construction uses **R32 throughout**. Public W8A8 geometry certifies all layers:
gate/up `896*127² = 14,451,584`, down `4864*127² = 78,451,456`, each below
`2^31`. Thus signed decode of the reconstructed R32 word is exact. No secret
u24 share is independently widened as if that preserved additive reconstruction.
Down bounds exceed `2^24`; integer-to-float32 rounding is genuinely relevant.

## 2. Complete numeric function

One invocation has public layer/row identity and public fixed Wg, Wu, Wd, their
float32 row scales wg, wu, wd, and post-attention RMSNorm coefficient gamma.
The client owns finite float32 post-attention residual row `h[H]`. The existing
attention/KV and client normalization placement is retained. Normalization is
part of the specified MLP boundary but **not secretly moved to the workers**:

```text
z = existing SemanticDecoderRuntime RMSNorm(h, gamma, epsilon=1e-6)
  = h / sqrt(mean(h*h, axis=-1) + epsilon) * gamma
(q0, s0) = existing dynamic row W8A8 quantizer(z)

ag = Wg @ signextend(q0); au = Wu @ signextend(q0)     # exact integer dots
g_j = f32(f32(f32(ag_j) * s0) * wg_j)
u_j = f32(f32(f32(au_j) * s0) * wu_j)

if g_j >= 0:
    e_j = EXP_NP32(-g_j)
    a_j = f32(g_j / f32(1 + e_j))
else:
    e_j = EXP_NP32(g_j)
    a_j = f32(f32(g_j * e_j) / f32(1 + e_j))
p_j = f32(a_j * u_j)

maxp = max_j abs(p_j)                               # whole M=4864 row
s1 = f32(maxp / 127) if maxp > 0 else f32(1)
q1_j = i8(clip[-127,127](RNE_f32(f32(p_j / s1))))

ad = Wd @ signextend(q1)
d_i = f32(f32(f32(ad_i) * s1) * wd_i)
y_i = f32(d_i + h_i)
```

`f32` means the actual rounded operation boundary; no reassociation, fused
multiply-add, rational SiLU, common-scale cancellation, public guessed scale,
or intermediate requantization. Signed zeros/subnormals, comparisons, negative
branch multiplication **before division**, and the `maxp=0 -> s1=1` rule matter.
The client RMSNorm uses the existing NumPy reduction order, epsilon handling,
square root and float32 behavior, not a replacement real-number formula.

`EXP_NP32` is the installed NumPy float32 exponential path used by
`semantic_executor.py:697-716`. It is **not** automatically a correctly rounded
IEEE exp or a portable scalar libm exp. Its vector implementation, dispatch,
special cases, libm dependencies and machine/compiler behavior would have to be
locked and lowered exactly. This probe specifies the required function but has
**no executable exp circuit or binary lock proving that lowering**.

`quantization.py` dispatches to the installed native extension. Actual native
`crates/pllm-core/src/codec.rs:99-131` checks finite inputs, reduces absolute max,
computes an f32 scale, rejects nonpositive/nonfinite scale, divides in f32,
uses `round_ties_even`, clamps, then converts to i8. Python fallback uses
`np.rint` with the same required rounded boundaries. Division must retain f32
rounding before clipping; it cannot be replaced by exact rational rounding. Full underflow,
overflow, NaN/Inf rejection and intermediate exception behavior remain lowering
obligations. MLP projection biases are absent in this pinned source.

For total-function specification, invalid inputs/scale failure produce a private
failure word to the client and dummy masked outputs. Success/failure is evaluated
without a worker-visible secret-dependent branch or short transcript. The client
may terminate; termination leakage is part of the public service failure model.
This error path is unimplemented and unpriced, as are valid finite edge cases.

## 3. Chosen construction: additive linear shares + two Yao bridges

Participants: trusted client C, remote **non-colluding semi-honest** A and B.
A is garbler, B evaluator, in both bridges. Public weights/program/shapes/row
positions and response lengths are visible. Neither worker gets plaintext token
IDs, residuals, activations, row scales, the other party's pads, or an output
decoding that reconstructs private values. Malicious correctness is not claimed.

Arithmetic type: `R = Z/(2^32)`. Float32 words are *bit strings*, **XOR-shared**;
their shares are never interpreted as floating-point numbers or multiplied
locally. An arithmetic share and a Boolean/float encoding are distinct types.

### C -> A/B: entrance

1. C runs the existing post-attention RMSNorm and dynamic input quantization.
2. For each q0 coordinate, sign-extend its i8 value to R **before sharing**.
   Draw uniform `r0_i in R`; A receives `r0_i`, B receives `q0_i-r0_i mod R`.
3. C draws independent uniform 32-bit XOR pads for each residual word h_i and
   the exact s0 word. Send pad to A, `word XOR pad` to B.
4. Each worker computes `ag_party=Wg*q0_party`, `au_party=Wu*q0_party mod R`
   locally. No `Wr-s` correction or linear inter-worker transcript is required
   for this public map. This admitted algebra does not implement nonlinear work.

### GC1: accumulators -> whole-row product quantization -> down input shares

A's private circuit inputs:

- `ag_A[M], au_A[M]`: 2M R32 words;
- s0_A: one XOR-shared word;
- fresh `rho_q[M]` R32 output pads and `rho_s` 32-bit scale pad.

B's private inputs: `ag_B[M], au_B[M]`, s0_B. Each B input bit uses a
1-out-of-2 OT to obtain only its active wire label. A's active labels are sent
directly, one **full label** per A input bit; no seeded compressed-label encoding
is specified or credited. No selected-label count is substituted for an OT
transcript.

Inside the circuit:

1. **A2B:** modularly add each accumulator share pair, discard final carry,
   signed-decode the resulting 32-bit word; s0 = s0_A XOR s0_B.
2. Evaluate every rounded g/u rescale, stable SiLU, multiplication, private
   4,864-way maximum, zero/scale checks, divide/RNE/clip as specified above.
   Neither maximum index/value nor scale is released to either worker.
3. **B2A:** sign-extend q1_j to 32 bits, then compute
   `bq_j = signextend(q1_j) - rho_q_j mod R`. B privately decodes bq_j.
   A retains rho_q_j; these are legal R32 down-projection input shares.
4. B privately decodes `bs = bits(s1) XOR rho_s`. A retains rho_s. s1 remains
   XOR-shared, not public, and not arithmetic-scaled per share.

The GC output-decoding metadata is only for **masked words** bq/bs. It is never
the plaintext product, q1, maximum or s1. B can know bq without A learning it;
A knowing rho_q alone does not reveal q1.

### Local down projection; GC2; A/B -> C

Each worker computes `ad_party = Wd*q1_party mod R`. GC2 gets:

- A: ad_A[H], s1_A, h_A[H], fresh output XOR pads rho_y[H];
- B: ad_B[H], s1_B, h_B[H], via new input OTs/labels.

GC2 A2B-adds down accumulators in R32, XOR-reconstructs s1 and h **inside the
circuit**, performs rounded int->f32, the two scale multiplications and residual
add. B decodes only `by_i = bits(y_i) XOR rho_y_i`. A sends rho_y[H] to C;
B sends by[H] to C. C XORs them and reinterprets the exact y float32 words.
No 4,864-wide gate/up/product is returned to C for local nonlinear evaluation.

This deliberately reconstructs **one H-wide final MLP residual row at the client
per layer-row**, matching retained client attention ownership. It is not an
unimplemented claim of secret-share-resident attention/KV. C owns next-layer
input norm, attention, rotary and KV append/reuse exactly as before, then enters
the next MLP using fresh shares. At the final layer C applies retained final
norm/head and client token selection. Each of 7/31 next-token decode inputs is
chosen only after the preceding output and receives fresh shares/material.
No private remote argmax, private remote embedding lookup, or batching of future
unknown token inputs is credited.

### Freshness, state, setup and messages

Public circuit structure/weights and authenticated endpoint setup may be cached.
For **each** layer-row and each GC, A draws new label pairs/global free-XOR delta
(selection bit set), fresh tables and output pads; OT randomness/extension rows
are one-use, with base-OT state following the selected secure extension's rules.
The construction is parameterized half-gates with **256-bit labels**, under its
required free-XOR/correlation-robust hash assumptions. No reviewed 256-bit hash
backend or OT extension is supplied by this probe; this is a conventional
construction specification with honest implementation gaps, not a crypto audit.

Tables have two full ciphertext labels per AND. A -> B sends tables plus masked
output translation bits, potentially offline. Online A -> B sends A active
input labels, then OT request/response flows occur in both directions for B's
inputs. B decodes masked outputs locally; there is no separate output-label
return charged on top. C ingress and both output-share links are charged below.
There is no third-party dealer in this chosen topology. Introducing one would
add dealer -> A/B material links; nothing here gives that placement free bytes.

Every message binds version, checkpoint/body/quantization digest, plan/circuit
digest, session, layer, token-row position, phase (GC1/GC2), attempt ID, ring,
dimensions and exact byte lengths. Party state retains public weights, local
input/accumulator/output shares, OT state, one-use material indices and burn
ledger; C retains h, original scales, attention/KV and generation state. No
common worker-expandable seed exposes both shares or a garbling delta to B.

Reserve before first send; durable monotonic state commits consumed/burned IDs.
Partial failure/cancel/expiry consumes or burns all reserved rows, labels,
tables/pads and OTs; retry gets a new attempt/material, never rolls back. After
GC2/client acknowledgement erase residual/scale/accumulator and garbling state.
Streaming row-at-a-time can bound working state but does not shrink recurring
payload. Prefill traffic is batched only across its known 39 rows. Actual
transport, hash instantiation, OT transcript, durable ledger, replay protection,
peak RAM, setup/cold weight delivery and backend CPU remain **unknown**.

Dense MLP compute is remote at both workers; client normalization/attention/head
are retained and have real costs. The protocol does not make C generate
garbling/linear corrections. MAC placement demonstrates remote dense work, not
a measured total compute-cap pass.

## 4. Exact removal ledger: two different scopes, no double credit

All baseline amounts below are reconciled **application-body bytes**, counted
once per directed sending edge, warm with recurring preparation. Raw matched
`client-attention` archives at +8/+32 supply directed edges; the independent
prepared stage-attribution document cross-checks MLP role sums. Prompt/source
keys and full canonical compact ledger are locked in the probe. +8 and +32
are separate cohorts, not latency-matched prompts.

### Requested narrow gate: only gate/up returned outputs and corrections

| Body / budget | 39+8 | 39+32 |
|---|---:|---:|
| Attention-placement covered warm control | 97,070,658 | 148,297,986 |
| Remove Prep -> inference gate/up correction | 32,227,142 | 49,037,126 |
| Remove inference -> C gate/up output | 32,246,354 | 49,145,954 |
| Total removed | 64,473,496 | **98,183,080** |
| Hold all other bodies fixed | 32,597,162 | **50,114,906** |
| New replacement allowance for 25% saving | 40,205,831 | **61,108,583** |
| New replacement allowance for 50% saving | 15,938,167 | **24,034,087** |

Allowance is `floor(control * remaining_fraction) - fixed_other`. Entire
gate/up **stage** is 67,540,904 / 103,070,408 bytes; its input/control bodies
are not part of the 64.47/98.18 MB removal. Down stage is not removed in this
narrow gate. New inter-worker bytes still count even if the client link shrinks.

### Specified full protocol: retire both prepared MLP stages

Our chosen full protocol also replaces entrance and down projection transport,
so its proper scope retires **all five directed bodies of both MLP stages**,
and adds the new C -> A/B, A -> B, A/B -> C links. It does **not** add those
new links while pretending the old down protocol is still executing.

| Full-region body / budget | 39+8 | 39+32 |
|---|---:|---:|
| Retired complete MLP stage bodies | 97,069,168 | 148,296,496 |
| Retained covered authorization remainder | 1,490 | 1,490 |
| Replacement allowance for 25% saving | 72,801,503 | 111,221,999 |
| Replacement allowance for 50% saving | 48,533,839 | 74,147,503 |

This extra full-region credit is 32,595,672 / 50,113,416 bytes, separately
justified by transport retirement. It is not an extra deduction from the narrow
scope. Both scopes independently conserve `removed + fixed = control`. This is
a conditional retirement if the proposed protocol existed; demonstrated runtime
crossings eliminated remain zero.

## 5. Analytically checked construction costs

The explicit ripple netlist uses `2w-3` ANDs for addition or subtraction modulo
`2^w`, with no top carry. For w=32 this is **61**. The first carry uses one AND;
each of the remaining w-2 carry positions uses two; final sum uses no carry
generation. XOR/NOT are free **table bytes**, not free CPU/storage.

Per MLP row, GC1 reconstructs 2M accumulators, B2A subtracts M target-ring pads,
GC2 reconstructs H down accumulators. Known conversion ANDs:

```text
A2B = (2M + H) * 61 = 648,064
B2A = M * 61         = 296,704
total               = 944,768 per layer-row
```

The private scale and residual XOR reconstructions are truly XOR gates and do
not generate half-gate ciphertexts. Their **input labels/OTs** still cost work.
No ANDs from rounded rescale, exp, SiLU, multiply, abs/max/scale, RNE/clip,
range checks or failure handling have been implemented or counted as zero.

For n=24*executed_rows and chosen label size ell=32 bytes:

```text
garbler direct input bits = [3M*32 + 3H*32 + 96] * n = 553,056*n
evaluator input OTs       = [2M*32 + 2H*32 + 64] * n = 368,704*n
masked output bits       = [(M+H)*32 + 32] * n       = 184,352*n
C -> each worker         = [H*4 + H*4 + 4] * n      = 7,172*n bytes
each worker -> C         = H*4*n                    = 3,584*n bytes
table bytes              = conversion_ANDs * 2*ell
```

Garbler bit accounting includes fresh q pads/scale pad (GC1), scale share,
residual share and final output pads (GC2). These are not forgotten because
they are random. Output translation is one bit per masked output wire, tightly
packed. The A -> B table payloads and direct labels are disjoint; **OT bytes are
not** silently identified with B's selected label payload.

| Known payload, bytes | 39+8 | 39+32 |
|---|---:|---:|
| A2B conversion tables, ell=32 | 45,789,609,984 | 69,679,841,280 |
| B2A conversion tables, ell=32 | 20,963,917,824 | 31,901,614,080 |
| A-owned direct active labels | 19,538,362,368 | 29,732,290,560 |
| Masked output translation bits | 25,440,576 | 38,713,920 |
| C ingress to both workers | 15,835,776 | 24,097,920 |
| Final H-wide shares from both workers | 7,913,472 | 12,042,240 |
| **Known subtotal, chosen R32/32-byte labels** | **86,341,080,000** | **131,388,600,000** |
| **Optimistic R32/16-byte-label subtotal** | **43,195,134,912** | **65,731,727,040** |
| **Invalid/unadmitted uniform R24/16-byte comparator** | **32,244,585,408** | **49,067,847,360** |
| **Complete recurring protocol / full wire** | **unknown** | **unknown** |

Uniform R24 is an optimistic comparator only: it fails the sampled down bounds
and whole-geometry certificate. It is not an admitted exact protocol. Neither
comparator gains coverage. Chosen half-gate labels have 255 hidden bits after
exposed selection bit; conventional serialized 128-bit labels have only 127
hidden bits. The cheaper 16-byte comparator is therefore deliberately generous,
not a claim to the requested 128 hidden-bit security.

An even smaller **direct-label-only construction screen**, granting away every
pad label, conversion table, OT, nonlinear operation and other link, still sends
one 16-byte label per A-owned gate/up accumulator bit: **5,498,732,544 /
8,367,636,480 bytes**. This is a real label-width floor for our chosen **explicit
uncompressed direct-label delivery**, not an information-theoretic OT cost or a
lower bound on seeded/structured/compressed FSS and other protocols. Even one
fresh half-gate AND per gate/up pair would be 171,835,392 / 261,488,640 bytes at
32 B/AND; this is a conditional screen, not a claim that every MLP needs that
many ANDs. The actual specified carry netlist is much more expensive.

Moving tables offline changes latency placement, not fresh per-response
all-link accounting. The online A-input labels alone exceed both entire matched
warm controls. Public circuit structure can be reused; private ciphertexts,
labels and pads cannot. Half-gates conversion hash calls alone are
4,172,095,488 / 6,348,840,960 at A and 2,086,047,744 / 3,174,420,480 at B.
CPU seconds and peak RAM remain unmeasured.

## 6. Prior references and construction-specific vetoes

- **Half-gates:** Zahur, Rosulek, Evans,
  [Two Halves Make a Whole](https://eprint.iacr.org/2014/756), Table 1 / Fig. 2,
  gives two ciphertexts per AND, free XOR, four garbler/two evaluator hashes.
  Primary landing page fetched this turn; full-PDF review already retained in
  `quantizer-circuit-review-2026-10-01.md`. Parameterized construction is used;
  no paper microbenchmark is imported as Qwen latency or assurance.
- **Earlier arithmetic garbling / scalar synthesis:** R01–R06 in
  `docs/data/research/papers.json` and the quantizer circuit review are reference
  families, not this executable full float32 row. Earlier 7,657-AND public-scale
  8-bit scalar costs are **not charged** to actual R32 Qwen accumulators. LogRow
  logarithmic ciphertext count does not erase masked table bits or conversions;
  Duty-Free Bits needs its own fully specified reviewed conversion backend.
- **Explicit truth tables:** after accumulator reconstruction, GC1's function
  has `2*4864*32 + 32 = 311,328` semantic input bits before output pads. An
  explicit complete truth table for that encoding enumerates `2^311328` inputs;
  no such allocation is attempted. Public fixed scales do not describe this
  function. This vetoes **enumeration**, not succinct circuits or compressed FSS.
- **SIGMA / FuseFSS:** R13/R14 registry and
  `adjacent-residue-review-2026-10-01.md:176-193` distinguish supported offset,
  comparison, truncation and DPF families from arbitrary multivariate rounded
  functions. Neither complete exact NumPy-exp row keys nor exact private scale/
  B2A/down continuation are supplied here. Fresh keys, both worker deliveries,
  setup/expansion and all bridge work would need a new ledger. Today's prepared
  path has **no FSS keys or garbled gates to remove**; no reported fused-helper
  speedup is applied to its output/correction bodies. This is **not** rejection
  of every possible compressed FSS or a universal communication lower bound.

## 7. Executable coverage, meaningful controls, and evidence

Implemented only: cached native stage import/quantization/whole-domain widths,
compiler/shape provenance, bounded **clear** XOR/AND carry evaluator, independent
integer-to-float32 control, exact scope removal/accounting and construction-specific
analytical payload projections. No protected full MLP or full checkpoint forward.

**Explicitly unimplemented:** exact float32/libm-or-NumPy exp lowering; all rounded
rescale/multiply/divide edges; whole-row max/private scale; RNE and clipping;
private valid/error checks; secure modulus lift/conversion and OT/garbling
backend; down float restoration; client/next-layer continuation parity; two
successive decode steps; full transport/freshness/durable state; CPU/memory/cold
setup admission. We avoid one modulus-lift obligation by sharing initially in
R32 and sign-extending inside B2A; we do not claim the general lift is free.

Tests independently exhaust 10,912 modular add/sub circuit assignments at
widths 2..6, verify the 61-AND real-width netlist on negative/minimum/wrap cases,
exhaust 65,536 i8/target-ring share reconstructions, check public linear share
composition against integer dots, and check 121 signed int->f32 threshold cases
against an integer-only ties-even oracle. Concrete wrong lift: R8 shares (5,124)
sum to signed -127, but separate signed lifts sum to +129. These tests validate
conversion arithmetic, **not exp or complete float execution**.

Accounting tests verify directed sums, disjoint removal, no duplicate/negative/
unknown/excess scope credits, correct narrow/full budgets, exact per-direction
conversion/label totals, saved evidence reproduction and fail-closed missing
coverage even with a generous `10^15`-byte budget. **24 focused cases passed**.
Focused Ruff lint and formatting checks passed.

Owned files:

- `scripts/probe_complete_mlp_contract.py`
- `tests/test_complete_mlp_contract.py`
- `docs/evidence/complete-mlp-contract-screen-2026-10-01.json`
- `docs/evidence/complete-mlp-contract-review-2026-10-01.md`

Executed/reproduction commands (existing environment, no dependency sync):

```sh
rtk uv run --no-sync python scripts/probe_complete_mlp_contract.py \
  --archive-dir /var/folders/3d/16hqtng12ng_bfz1n99jg04w0000gn/T/opencode
rtk uv run --no-sync python scripts/probe_complete_mlp_contract.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  VECLIB_MAXIMUM_THREADS=1 rtk uv run --no-sync pytest -q tests/test_complete_mlp_contract.py
rtk uv run --no-sync ruff check scripts/probe_complete_mlp_contract.py tests/test_complete_mlp_contract.py
rtk uv run --no-sync ruff format --check scripts/probe_complete_mlp_contract.py tests/test_complete_mlp_contract.py
```

The script sets offline flags and all relevant native thread limits to one
before importing NumPy/native code. Default replay reads only the compact saved
matched-control ledger, checks its pinned canonical hash, then recomputes native
weight samples and costs. Optional archive extraction checks the original raw
SHA-256 locks through that same canonical ledger. Full regenerated JSON equals
saved JSON structurally; formatting differs. No large circuit/table/key arrays
are allocated. Evidence contains config/checkpoint, individual sampled weight/
scale, source-code and archived ledger hashes. Unknown costs are JSON `null`.

**Feasibility result:** no 25% or 50% saving for this specified construction,
even under the more generous full-region retirement. Complete executable
coverage is absent, so no plan is admitted. Reopening requires a concrete
different encoding/protocol with exact rounded whole-row function, legal private
scale/output continuation and a complete fresh-material/transport ledger—not
another scalar lookup or a discount copied from an incompatible placement.
