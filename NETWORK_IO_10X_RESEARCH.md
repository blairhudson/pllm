# Tenfold network-I/O research: existing-model privacy

## Current direction: measured incremental gains

The tenfold target below is the original research screen, not a requirement for
progressing useful improvements. Preserve numeric quality, privacy and remote-heavy
execution; benchmark incremental savings with their actual cold/warm costs.
Removing the tenfold requirement does not rescue a method that increases traffic
or fails its numeric-quality gate.

### Priorities from existing evidence

1. **Fresh prompts, fixed Qwen: role-selective client attention projections.**
   Implemented as `ClientLinearRoles`, selecting
   semantic `qkv_projection` and `attention_output` stages across all layers.
   Keep ordinary prepared gate/up/down stages and existing client nonlinear work;
   do not introduce a resident protected MLP or another correlation generator.
   Pinned W8A8 prefill/decode logits are bit-identical on three checked prompts.
   A matched 39+32 warm run measures **178.97 → 148.30 MB** covered bodies
   (**17.14%**) and **113.55 → 93.22 MB** online. Attention weights add
   **44.04 MB** of signed-i8 values and **0.20 MB** of row scales to client
   delivery, plus **44.04 MB** of native weight snapshots. These are declared
   storage counts, not peak-memory samples. About **87.69% of body linear MACs
   stay remote**; this is not a full-response CPU percentage. Remote body stage
   calls fall from 96 to 48 per executed row. Raw cold bundle delivery grows;
   compression, aggregate CPU and representative quality remain separate gates.
2. **Shared-prefix workloads: progress `ClientPrefixReuse`.**
   The existing 168-of-175-token overlap cohort measured **279.29 → 11.58 MB**
   online. This is the largest demonstrated workload-specific win, not a
   fresh-prompt gain. Expand realistic multi-turn tests and cache-budget/eviction
   accounting; preserve explicit prefix-equality/access-pattern visibility.
3. **Cold short requests: progress request-sized inventory.**
   A matched 39+1 cohort measured **268.05 → 244.70 MB** covered cold bodies
   (**8.71%**), from issuing 39 rather than 64 rows per stage. Online bytes did
   not change. Spare rows are not inherently wasteful if later requests consume
   them; compare repeated sessions before changing policy defaults.
4. **Bandwidth-limited cold delivery: keep compression opt-in.**
   With request-sized inventory already selected, framed zlib measured
   **244.70 → 223.62 MB** covered cold bodies (**8.61%**). Aggregate cold CPU rose
   **29.32 → 34.99 s** in one pair. Warm transfer is unchanged, cached bundles
   remain raw, and the option has not passed a matched compute-cap gate.
5. **Existing exact client-prefix placement: useful comparator.**
   Two local layers measured **63.84 → 58.52 MB online** (**8.33%**) but
   **247.23 → 268.71 MB cold covered bodies**. Keep one/two-layer candidates in
   the matched placement benchmark; accept client-weight delivery only when
   amortized over enough requests. Counts of body linear work remain remote-heavy.

If changing the checkpoint is allowed, pinned SmolLM2-135M measured **88.40 MB**
warm covered 39+32 bodies versus Qwen's **178.97 MB**, and **55.68 versus 113.55 MB**
online. This is a separate model/task-quality decision: its 11/12 W8A8 prefill
agreement is with its own float32 reference, not evidence it answers as well as
Qwen. Public task-quality comparison must precede a cross-model winner claim.

Evidence: [incremental SDK/benchmark cohorts](docs/evidence/incremental-network-qwen25-2026-10-01.json),
[stage attribution](docs/evidence/prepared-stage-attribution-qwen25-2026-09-29.json),
[prefix reuse](docs/evidence/client-prefix-reuse-qwen25-2026-09-28.json),
[inventory sizing](docs/evidence/prepared-inventory-cold-sizing-2026-09-27.json),
[compression](docs/evidence/client-bundle-compression-qwen25-2026-09-27.json),
[prefix placement](docs/evidence/client-owned-prefix-layers-qwen25-2026-09-28.json),
[model/cold controls](docs/evidence/prepared-cold-compare-qwen-smol-2026-09-29.json).
All network figures are scoped serialized bodies, not complete wire measurements.
Do not add percentages from different cohorts or assume optional components compose
without a matched run.

**Combined workload evidence:** attention ownership, prefix reuse and framed zlib
on three fixed growing contexts measure **647.60 → 240.68 MB** online bodies
(**62.8%**) and **1,171.96 → 547.41 MB** setup-inclusive covered bodies
(**53.3%**). Request-sized/prewarm inventory and delivery choices are immutable
SDK components; the ordinary benchmark ranks exact ordered context cohorts and
includes startup/prewarm once. This is a shared-prefix result, not a saving of
that magnitude on unrelated fresh prompts. Compression increases cold CPU in
the checked pairs. Counts exclude wire framing and upstream model distribution.

**Next build order:** separate first-request delivery/preparation, new executed
rows, reused rows and discarded inventory in residual budgets. Improve exact
continuation-cache storage and suffix execution only where measured misses or
overhead leave room; compare placement frontiers under explicit client budgets.
Retain ordinary prepared controls and existing one/two-layer placements. Do not
promote the failed W8A6 override,
tested speculative pairing, public-domain certificate or replicated three-party
layouts simply because a smaller gain is now acceptable.

## Original tenfold objective and admission

Reduce **all-link** network bodies by at least 10× for a *fresh* response from
an existing pretrained decoder, without training, moving most transformer-body
compute to the client, or exposing plaintext prompts/activations to providers.
Keep the current Client / honest-but-curious non-colluding Preparation and
Inference privacy contract, or explicitly declare a distinct two-online-worker
non-collusion contract. Count every Client↔Preparation, Preparation→Inference,
Client↔Inference and worker↔worker body, offline material, setup, and cold model
distribution separately. One-use material must burn on cancellation or replay.
No secure-runtime or quality claim follows from the bounded references below.

Pinned control: Qwen2.5-0.5B-Instruct W8A8, **39 input + 32 generated tokens**,
same checkpoint/body fingerprint and 96 remote stages: **178,970,558 covered
all-link** and **113,545,024 covered online** serialized-body bytes. Tenfold
budgets are **17,897,055** all-link and **11,354,502** online bytes (strictly
less than one tenth). MLP accounts for 148.30 MB, and *even removing all MLP
stage bodies for free leaves 30.67 MB*. Any proposed win must address attention
and MLP, pass held-out prefill/decode parity and keep most MACs remote. Full
wire, cold source distribution and independent operation are **unmeasured**.
See [compiled attribution](docs/evidence/prepared-stage-attribution-qwen25-2026-09-29.json),
[pretrained screen](docs/evidence/pretrained-bottleneck-screen-2026-09-29.json)
and [cold comparison](docs/evidence/prepared-cold-compare-qwen-smol-2026-09-29.json).

These are independently testable PLLM design hypotheses, **not** claims of
literature novelty or executed tenfold improvements.

## 1. Predictive modular execution: exact result from a narrow residue

Client constructs a **cheap** public-weight predictor `p ≈ W·x` for a stage.
Inference evaluates `W·x mod q` using fresh masks and returns only a residue;
Client selects the integer congruent with the residue nearest `p`. If the true
integer result is `y`, recovery is unique when `|y - p| < q/2` in accumulator
units. **Observing a small residual on public prompts cannot establish that
condition for a private input**: Client cannot tell which lift is correct from
`p` and `y mod q` alone. Before material issuance, require a public,
stage/row-bound *worst-case certificate* `B < q/2` for every admitted input.
For an integer linear predictor `A·x` under unrestricted W8A8 activations,
`B_j = 127 × Σ_i |W_ji − A_ji|` is a sound per-output bound. At `q=4`, any
nonzero integer weight residual already gives `B_j ≥ 127`: a linear predictor
must contain the **entire** weight matrix. An input-specific certificate needs
its own cost and privacy contract; an uncertified nearest lift may silently
change model output. Equality at `q/2` is ambiguous and **must fail closed**.
Biases, scales, numeric-domain bounds and every predicted stage must bind to
the compiled plan. Predictor must use far fewer client MACs than remote body
execution, and its once-per-model weight snapshot must be counted for cold
distribution and disk. Stage input/output masking still needs fresh uniform
one-use ring material; neither provider may learn the predictor or full input.

At the *existing schedule's* optimistic 110.53 MB uniformly-u16 stage-body
estimate, replacing each 16-bit residue by a **2-bit** residue would project
13.82 MB all-link and roughly 8.49 MB online *arithmetic bodies*, before
tickets, metadata, framing, model transfer or predictor compute. This is a
conditional width calculation, **not** permission to narrow current rings:
`q=4` demands sub-two-unit prediction errors for *every* output element.
Weight-only low-rank cuts previously failed token-decision fidelity; unlike
those lossy cuts, successful modular recovery would restore exact integers.

**First gate:** implement bounded one-use masking with certified exact/ambiguous
recovery; capture real compiled-stage integer inputs on pinned public prompts,
fit only public predictors, and report *both* worst observed residual and the
distinct worst-case certificate per stage, exact-recovery fraction as an
offline diagnostic, client/remote MACs, stored predictor weights and projected
all-link bytes. No private-error-dependent message length or fallback (which
would leak information) without a separately admitted fixed-size protocol.
**Veto:** any stage needs wide residues, prediction carries most remote compute,
or combined bytes exceed the whole-response budget. A single-stage success
does not admit a decoder.

**First gate result (2026-09-30): veto the tested sparse predictors.** The
bounded one-use reference proves exact signed-integer lifts only when the
whole-input certificate passes. All 96 stages of the pinned real Qwen 39+8
and 39+32 W8A8 compiled decoder were then cost-gated using public predictors
retaining 10%, 50% and 90% of weight columns. *None* certifies even 8-bit
residues; certified per-stage widths remain **19–26 bits**. At 39+32, the
best tested arithmetic-only projection is **141.51 MB all-link / 87.67 MB
online**, before control, versus 17.90 / 11.35 MB tenfold budgets; it also
ships **322.24 MB** of client predictor weights once per model and duplicates
**90%** of the tracked body integer MACs. The observed-prompt residual widths
are not an exactness certificate. A hypothetical uniform 2-bit schedule would
cost 13.82 MB arithmetic bodies, but is **not certifiable** by these
predictors. See [locked evidence](docs/evidence/predictive-modular-qwen25-2026-09-30.json).
Reproduce after locally caching the pinned checkpoint:

```sh
uv run --no-sync python scripts/probe_predictive_modular.py --max-output-tokens 32 --summary
```

Further predictor work needs a *different* succinct certified representation,
not a private-error-dependent fallback or another unverified low-rank cut.

## 2. Correlation-compiled complete layers

Compiler specializes joint one-use correlations to public weights and the
complete attention + MLP + exact-rescaling layer rather than issuing unrelated
per-element triples, Boolean carries or independent gate coefficients. Two
online workers keep one share each; correlated material is party-local and
non-reusable, never a provider-expandable common seed. Attention/KV, scale
conversion and nonlinear arithmetic must remain protected. The trusted client
may select each token at the final boundary, but cannot reconstruct intermediate
hidden states to mask the actual layer cost.

For Qwen's 24 layers, 70 executed rows and 896-wide hidden state, an **assumed**
pair of 12-bit hidden-source openings to two parties costs ~9.03 MB online.
This leaves at most ~2.32 MB online and ~8.87 MB all-link for everything else.
Twelve-bit arithmetic is **not validated** for Qwen: real bounds, exact
truncation and output shares may defeat this design. The earlier correlated
quadratic MLP reference needed 141.6 MB dealer material across both parties
for 39+1, and a second hidden source pushed optimistic online openings above
that cohort's tenfold target. Ring PCG toy OLE does not generate the required
joint material; FuseFSS savings cannot be applied to today's prepared path.

**First gate:** compile one *complete* semantic layer into a symbolic link,
material and bounded-numeric schedule; construct one joint correlation block
with measured setup/expansion bytes, CPU, memory, one-use and replay semantics.
**Veto:** unbounded per-element correction, expanded material or private
attention/rescaling exceeds the per-layer whole-decoder budget. No issuer or
benchmarkable topology until full-layer arithmetic and fidelity pass.

## 3. Token-boundary encrypted execution

Client encrypts embeddings; Inference keeps decoder activations and KV
encrypted across **all** body layers, returning only final hidden ciphertext
for client-side head and token choice. Client encrypts each next embedding.
Preparation may distribute public evaluation keys, but no server gets the
decryption key. This retains client plaintext ownership and remote body compute
under a different HE assumption, without requiring server-side private argmax.
It avoids the failed *per-gate* ciphertext boundary; it does **not** make
encrypted attention, depth or nonlinear evaluation cheap.

For 4,096 slots and width 896, an idealized 39+32 response has 9 initial
embedding ciphertexts, 31 later embeddings and 32 returned hidden ciphertexts:
**72 boundary ciphertexts**. At a measured *shallow* TenSEAL CKKS size of
~235 kB, one-way-plus-return boundaries already cost ~16.93 MB before keys,
headers and controls, and miss the online target. At an *unachieved* 128 kB
average, ~9.22 MB boundary bodies would leave limited budget. The earlier
per-gate CKKS probe projected 615 MB **one-way** on this Qwen cohort; this
new schedule changes the boundary, not that measurement.

**First gate:** lower one complete real-checkpoint layer into encrypted
arithmetic, measure multiplicative depth/bootstraps, rotations, key/context
bytes, ciphertext size, CPU/memory and held-out numeric error. Project
response cost with tokens and complete wire separately. **Veto:** bounded
whole-decoder depth, CPU cap, memory or cold key distribution fails despite
possible traffic reduction. No plaintext fallback at the provider.

## Work order

1. Predictive modular gate: **tested sparse predictors vetoed**; only revisit
   with a genuinely cheaper whole-input certificate and bounded client compute.
2. Joint-correlation full-layer gate: direct two-worker MLP cut and 24-bit
   two-source construction vetoed on known bodies; 12-bit remains unadmitted.
   Reopen only with a concrete compact generator and exact shared arithmetic.
3. Token-boundary HE gate: **tested TenSEAL CKKS parameter sets vetoed** on
   depth and, for 39+32, sampled boundary bodies before whole-layer work.
   Other HE backends require their own complete-layer cost and fidelity gate.

Publish failed gates as evidence. Do not promote a scalar reference, an
unmeasured projection, an online-only saving or a different privacy contract
as a tenfold complete-response result.

## Direction review: charge complete regions before constructing correlations

Sparse predictive lifting failed its whole-input certificate and byte budget.
Gate 1 compared two **distinct placements** against the same
compiled schedule and 39+8/39+32 controls. A compiled operator grouping alone
does not remove cryptographic openings, protected rescaling or one-use dealer
material. A missing operation has **unknown cost**, never zero cost.

### Placement A: client attention, protected remote MLP

Client keeps attention, KV, normalization, public Q/K/V/output projections and
token boundary; both non-colluding workers receive separate shares of one
post-attention hidden MLP input and return separate shares of one complete MLP
output. Gate/up/down linear MACs remain remote; report actual client projection,
attention, norm and output-head work separately. This differs from the existing
PreparedProviderRoles graph: new peer nonlinear and fresh dealer correlations
are needed before it can execute. Client downloads attention-side public weights
once, with cold delivery and disk charged separately.

The older **10,536,960-byte** 39+32 narrow-boundary estimate counted *one*
input and *one* output at 24/32 bits per layer row. The direct two-worker
share interface sends that input to **each worker** and receives **each output
share**: **21,073,920 bytes** online at 39+32, versus a 17,897,055-byte
all-link budget. At 39+8 the same correction gives **13,848,576** versus
11,684,396 bytes. These are arithmetic-body floors *before* peer messages,
dealer issuance, framing and client model delivery. A different protocol must
justify a smaller interface rather than silently reuse the one-copy estimate.

### Placement B: two-worker resident complete layers

Workers keep one activation and KV share each through complete attention,
normalization, MLP and exact scale transitions. Trusted client reconstructs
only a final hidden/token boundary each generation step. Existing compilation
keeps selection and token feedback client-local; do not charge a fictitious
private full-vocabulary argmax or assume deferred token selection. Every
dealer→worker and worker↔worker message counts, including two *independent*
attention-query and MLP hidden sources established by semantic dependencies.

At assumed 24-bit source openings, this **specific two-source construction**
already costs **18,063,360 bytes** online at 39+32 and **11,870,208** at
39+8, each above its all-link target without attention arithmetic or any
material. Hypothetical 12-bit packed openings halve those figures, leaving
~8.87 MB / ~5.75 MB all-link for *every* other message. Twelve-bit Qwen
numeric range, exact shared rescaling and a compact joint correlation
generator have **not** been validated. This bounds this construction, not all
possible resident protocols.

### Ordered admission gates

1. **Compile a directed-link contract:** one immutable per-semantic-layer
   schedule with role, operation, tensor shape and phase. Charge startup and
   online separately for Client↔each worker, worker↔worker and dealer→each
   worker. Count MLP and attention projection MACs; mark nonlinear,
   normalization, rescaling, KV, share renewal, feedback and token-boundary
   costs unknown until specified. Veto any placement whose *known floor*
   already exceeds either matched budget; a sub-budget lower bound is not a
   pass. No model-family-name parsing or stage-name heuristics.
2. **Specify one complete joint correlation:** input/output function, both
   parties' views, seed independence, concrete correlated distribution,
   one-use/burn contract, serialized issuance, peer openings, expansion CPU
   and peak memory. Existing Ring PCG toy OLE does not generate arbitrary
   nonlinear coefficients. Reject any proposed generator unable to account
   for the whole layer, including attention and exact scale transitions.
3. **Numeric checkpoint gate:** first substitute the *entire* arithmetic
   path into local compiled execution for pinned real weights. Check public
   ranges, every rounded edge, held-out prefill, same-token decode and free
   generation. Approximate reference quality cannot be inferred from a Q7
   polynomial numerator; an exact candidate must match baseline integers.
4. **One protected region:** only after gates 1–3, execute one complete
   layer with party-local shares and material, cancellation/replay burn, no
   hidden client reconstruction, exact measured bodies across every link,
   CPU and memory. Then extrapolate to full decoder and benchmark via ordinary
   `Experiment` path. Keep TEE and HE under distinct trust/compute contracts.

Token-boundary HE was cost-gated ahead of 12-bit carry work; its sampled
backend result is below. Full-wire measurement remains open; macOS
per-process `nettop` snapshots failed even to reconcile
covered online application bodies and cannot close that gap.

### Gate 1 result: compiler-derived placement comparison (2026-09-30)

`pllm.runtime.region_contract_cost.compiler_region_contract_cost` derives
semantic roles, both independent hidden sources, per-layer tensor counts,
projection MACs, and directed optimistic serialized bodies from the native
runtime schedule. It rejects incomplete stage coverage and tracks missing
work as `body_bytes: null`. The pinned source/configuration, schedule, body
fingerprint and two workload budgets are locked in
[`docs/evidence/compiler-region-contract-qwen25-2026-09-30.json`](docs/evidence/compiler-region-contract-qwen25-2026-09-30.json).

| 39-input response | Prepared covered all-link / online | 10× all-link / online budgets | Two-worker MLP cut known online floor | Resident 24-bit known online floor | Resident **hypothetical** 12-bit known online floor |
| --- | ---: | ---: | ---: | ---: | ---: |
| +8 tokens | 116.84 / 73.83 MB | 11.68 / 7.38 MB | **13.85 MB** | **12.17 MB** | 6.24 MB |
| +32 tokens | 178.97 / 113.55 MB | 17.90 / 11.35 MB | **21.07 MB** | **18.67 MB** | 9.64 MB |

The two-worker MLP cut and the tested 24-bit resident two-source opening
**fail both budgets on known bodies alone**. Their unknown dealer, peer
nonlinear, attention and scale costs only increase traffic. This does not
veto other layouts or opening-free protected arithmetic.

At 12-bit resident source width, only **1.14 MB (+8)** and **1.72 MB (+32)**
of the online budgets remain for every omitted online peer operation and
control, including exact lifting from small shares to full accumulators.
All-link room before fresh material and complete wire is **5.44 / 8.26 MB**.
No Qwen-valid 12-bit share-lift or whole-layer correlation generator exists.
The existing quadratic-numerator dealer alone projects **172.97 / 263.21 MB**
for the two parties with the additional attention source; that is a
comparator for the existing method, **not** a lower bound on every new method.

A bounded exact-arithmetic regression shows why the 12-bit assumption cannot
simply widen each worker's additive share for a public 32-bit linear layer.
For shares `a,b` of signed-i8 `x` modulo 4096, correct wide arithmetic needs
`x = a + b − 4096k`, where `k` may be **0, 1 or 2**. Ignoring `k` changes even
a single public-weight integer product. Publishing `k` is not a valid repair:
`k=0` rules out negative `x`, and `k=2` rules out positive `x` for the checked
domain. A protected, secret-shared carry and signed scale conversion must be
specified and charged *before* claiming 12-bit peer openings can run Qwen.

Client-attention placement retains 87.69% of *tracked body projection MACs*
at each remote worker, but adds ~44.24 MB of once-per-model client attention
weights/scales. It also gives the client Q/K/V and output projection work,
token head work and unmeasured norms, attention and KV. This fraction is **not**
aggregate CPU, energy, or a proof that the full response remains remote-heavy.

**Reopening gate for resident sharing:** specify exact 12-bit source-share
lifting into a wide accumulator and public fixed-scale rounding, including
how each party privately corrects the modular carry **without revealing it**.
Prove exact parity on bounded inputs and price every carry/correlation across
the two source tensors; reject if
the 1.14/1.72 MB remaining online budgets are exhausted. Only then attempt
one concrete **whole-layer** joint-correlation distribution. Do not build
another gate-only polynomial simulator or activate a topology to test it.

Reproduce the matched cost gates with locally cached pinned config:

```sh
uv run --no-sync python scripts/probe_region_contract_cost.py --max-output-tokens 8 --summary
uv run --no-sync python scripts/probe_region_contract_cost.py --max-output-tokens 32 --summary
```

### Gate 3 result: token-boundary encrypted-layer feasibility (2026-09-30)

`pllm.runtime.he_layer_feasibility.token_boundary_he_layer_gate` traces the
complete compiler-declared Qwen2 layer in prefill and decode, including both
body RMSNorms, Q/K/V, rotary and private KV updates, attention scores, causal
mask, softmax, value accumulation, residuals, gated SiLU MLP and seven public
linear projections per layer. A *new* HE placement would return the final
pre-normalization hidden vector to the trusted client, which performs final
RMSNorm, head and token choice. The current baseline schedule does **not**
activate that placement. Missing protected operators remain unpriced.

Even optimistically charging **zero** ciphertext depth for both body RMSNorms,
causally masked softmax, public-weight projections and every scale conversion,
one forward pass needs **four causally ordered ciphertext products per layer**:
`Q·K`, attention probabilities times `V`, a non-affine polynomial SiLU gate,
then SiLU×up. Compiler dependencies give a **96-product serial depth floor**
across 24 layers. Keeping encrypted KV between feedback steps may raise the
whole-response depth further; this is not a complete arithmetic circuit or
a universal lower bound for different HE constructions.

A real, secret-free TenSEAL 0.3.17 provider context with 8,192-degree CKKS,
`[60,40,40,60]` coefficient bits and `2^40` scale executed **two** successive
896-value ciphertext products; a third failed `scale out of bounds`. The
16,384-degree `[60,40,40,40,40,60]` context executed **four**, then failed on
the fifth. Neither exposes a Python bootstrap API. The provider held no secret
key; both multiplication chains matched their synthetic plaintext oracle within
`1e-3`. They did **not** evaluate RMSNorm, softmax, a matrix or a full layer.
Public contexts with rotation and relinearization keys serialized to about
**35.29 / 179.50 MB per client key** respectively. Distribution may be
amortized over later requests but exceeds either response's 10× all-link target
if included in that first response; it is not a once-per-model global key.

At 39+32, the idealized boundary sends **40 client ciphertexts and receives
32** with 4,096 slots; the larger context uses 8,192 slots and sends **36**.
Using the observed synthetic input ciphertext and the **smallest last-level
response** ciphertext for every return gives illustrative online bodies of
**17.44 / 46.31 MB**, versus **11.35 MB** tenfold online. This assumes free
repacking, no evaluation refreshes, no controls and no wire framing. At 39+8
the smaller context projects **6.35 MB**, below its **7.38 MB** online target,
but fails even one layer's depth; the larger projects **14.74 MB**. These sizes
are one backend's sampled serialization, **not** ciphertext-size lower bounds
for all HE schemes. Full-layer latency, memory, numeric fidelity and cold
checkpoint distribution remain unmeasured because depth vetoes execution.

See [locked HE evidence](docs/evidence/he-token-boundary-feasibility-qwen25-2026-09-30.json).
With pinned config cached, reproduce both cohorts, key/context sizes and depth:

```sh
uv run --no-sync python scripts/probe_he_token_boundary.py --summary
```

**Direction after these gates:** none of the three *tested* constructions
achieves 10× on fixed Qwen2.5 under the current privacy/remote-compute
requirements. Do not implement a whole encrypted decoder in this TenSEAL
context or another scalar HE island. Reopen token-boundary HE only when a
specific backend supports the full private attention + nonlinear layer,
ciphertext refresh through at least 24 layers, and measured packing/key/CPU
costs below both matched cohort budgets. Reopen resident sharing only with a
concrete private scale-lift and joint correlation generator whose **complete**
layer fits the remaining online and one-use material budgets. Continue to
improve the executable prepared baseline and measure full wire separately;
neither an unmeasured transport gap nor another model's lower bytes licenses
a fresh-prompt 10× claim for Qwen.

## Five additional ideas: bounded screens (2026-09-30)

All five were screened against pinned Qwen2.5 dimensions and the same
39+8/39+32 compiler schedules. The arithmetic/table inputs are **synthetic**;
this is not real-checkpoint quality evidence or executable protected inference.
Results and environment are locked in
[`docs/evidence/interaction-reduction-screen-qwen25-2026-09-30.json`](docs/evidence/interaction-reduction-screen-qwen25-2026-09-30.json).

### 1. Specialized secure reductions

The bounded one-use two-party sum-of-squares correlation produces shares of
`sum(x*x)` in a 32-bit ring. Each party opens only its share of `x-r`; the
dealer gives each party its own vector-mask share and one scalar `sum(r*r)`
share. At width 896, both-direction online bodies are **7,168 bytes** versus
**14,336** for generic elementwise Beaver squaring; total dealer bodies are
**7,176** versus **21,504**. A vector-times-secret-scalar correlation also
avoids transmitting repeated copies of the same scalar opening and material.
Both primitives pass exact modular parity, binding, malformed-input and
one-use tests. This local dealer is not an independently distributed protocol.

However, these two pieces together cost **14,344 online / 21,520 dealer bytes
per norm row**, without privately computing the inverse square root, dividing,
rescaling or rounding. For 39+32, 3,360 body-norm rows project **48.20 MB online
and 72.31 MB dealer bodies**; 39+8 projects **31.67 / 47.52 MB**. This specific
32-bit construction already fails the tenfold budgets before attention or MLP.
It is not a lower bound on all possible norm protocols.

Summing before per-element rounding is not automatically exact: for squares
of `[1,1]` divided by two, ties-to-even per-element rounding sums to zero,
while rounding the total gives one. Float32 RMSNorm fidelity and a full private
inverse/quantization path remain unimplemented. **Decision:** useful incremental
primitive, but do not scale this construction into a whole decoder.

### 2. Private token-to-embedding lookup

An 18-bit additive DPF reference expands each key once over a public
151,936×896 signed-i8 synthetic table. Worker-local outputs reconstruct the
exact selected quantized row, including signed coefficients. Measured query
bodies total **872 bytes across both workers**, versus **7,168** for directly
sending 32-bit embedding shares. If outputs remain worker-local, no return
traffic is needed for the lookup itself. Returning both shares to the client
would cost another 7,168 bytes and undo that comparison.

Each worker stores/scans **136.13 MB** of table and performs **136.13 million
modular MACs per token**. Selector storage is 0.61 MB; bounded conversion
windows are 3.67 MB rather than a full u32 table copy. The sampled Python
key expansion, table hash and scan took **1.32 / 1.33 CPU seconds** per worker.
At 39+32 the conditional boundary upload saving is only **0.44 MB**, at the
cost of **19.06 billion** extra lookup MACs across both workers. Batched/native
lookup might change CPU, but this sample does not establish compute admission.

The table's per-token float scales, conversion into the decoder numeric
representation, HE output conversion and authenticated independent workers
remain unimplemented; the DPF construction has no security review. The current
prepared baseline does local token lookup with zero provider token-query bytes,
so these savings apply only to a proposed resident-share boundary, not prepared
traffic. **Decision:** keep as a boundary option for a future viable resident
decoder; not a remedy for today's large body traffic.

### 3. Exact public-linear fusion audit

Conservative compiler-graph discovery found **zero serial linear→linear
candidates** through identity reshapes in either phase. Norms, attention
values and gated multiplication separate the projections. The baseline already
has **48 parallel groups per phase** (Q/K/V and gate/up), avoiding 72 duplicate
input sends per layer sweep. These are existing savings, not new gains.

Even where serial maps exist in another graph, intermediate dynamic activation
quantization and float32 rounded edges prevent claiming exact W8A8 equivalence
from `B(Ax)=(BA)x`. Product weights, bias, layout and new numeric behavior need
their own admission. **Decision:** no new eliminated messages for this audited
Qwen graph; no speculative compiler optimization was activated.

### 4. Same-client HE SIMD batching

Real secret-free CKKS evaluation of one square used the existing degree-8,192
context and 4,096 slots. Four independent 896-value sequences fit in one
ciphertext; sixteen require four. Both-direction bodies per sequence were
**566.66 kB / 141.56 kB / 141.60 kB** for batches 1/4/16: approximately **4×**
better slot utilization, plateauing once those slots are full. Checked errors
were below `1e-3`; provider never received a secret key.

This is a real isolated packing gain, not whole-decoder acceleration. Attention
rotations must preserve separate sequences, encrypted KV layouts and depth
refresh still need implementations, and batching different client keys is not
supported. Waiting for concurrent requests also has latency costs not measured
here. **Decision:** most positive measured packing result, relevant only to
same-client concurrent workloads and a separately feasible HE computation.

### 5. Seeded ciphertext serialization

Three samples each of ordinary asymmetric and symmetric CKKS vector encryption
both serialized requests to approximately **331 kB**, with evaluated square
responses around **235 kB**. Symmetric encryption alone does not make these
Python vector bodies seed-compressed.

The low-level SEAL `encrypt_zero_symmetric()` overload returns
`seal::Serializable<seal::Ciphertext>`, which TenSEAL 0.3.17 cannot convert to a
Python type. `CKKSVector.serialize(self)` exposes no seeded option. The probe
records that actual `TypeError`, rather than pretending a supported transport
exists. A native binding could reopen supported seeded input serialization;
it would still require expansion, arithmetic and response-cost measurements.
**Decision:** backend-binding gap; no measured compression gain or activated path.

### Direction after all five

The only positive measured HE result is same-client SIMD utilization, which
does not repair the depth veto or meet a single-sequence decoder target.
Private lookup is compact but expensive and affects an already small boundary.
Specialized reductions improve generic arithmetic but the tested whole-response
norm floor remains over budget. Exact fusion and current seeded serialization
offer no new saving in the checked paths. These results do **not** justify
combining the five into a claimed 10× decoder.

Keep prepared inference as the executable control. Reopen share-resident work
only with a complete-layer protocol that eliminates substantially more vector
openings and fresh material; reopen HE batching/seeded input only after a real
private layer has passed depth, numeric and resource gates. No new Experiment
or provider topology is admitted by these screens.

Reproduce all five (cached pinned config and optional HE dependencies required):

```sh
uv run --no-sync python scripts/probe_interaction_reductions.py
```

## Adjacent MPC/HE representations: parallel review (2026-10-01)

Five independent tracks reviewed concrete constructions, their numeric and
privacy contracts, and the same locked 39+8/39+32 controls. Primary-source
results, synthetic arithmetic, compiled shapes and conditional cost projections
remain separate. No protected complete layer, checkpoint-quality improvement,
independent-worker deployment or tenfold result was established.

| Track | Result | Decision |
| --- | --- | --- |
| [Homomorphic secret sharing](docs/evidence/adjacent-hss-review-2026-10-01.md) | Direct HSS removes peer interaction for **restricted** multiplication programs. Computed register × computed register is not a native instruction. Optimistic BKS embedding encodings project 19.96/59.87 MB input bodies. DCR encrypted token bits instead project 2.54/3.87 MB, with lookup, complete program, terminal conversion and KV continuation unpriced. | Most structurally different candidate, but require a legal program and output interface before implementation. Small ingress is not a decoder pass. |
| [Projective/rational normalization](docs/evidence/adjacent-projective-review-2026-10-01.md) | Exact real identities exist, including epsilon-preserving squared-denominator RMSNorm. Unrestricted delayed division changes float32 and quantized outputs. Naive residual denominator alignment alone projects 31.69/48.22 MB online. | Reject unrestricted cancellation. Investigate bounded rounding-aware norm→quantization predicates instead of creating more secret denominators. |
| [Collective HE refresh](docs/evidence/adjacent-refresh-review-2026-10-01.md) | Concrete Lattigo/OpenFHE protocols exist, but refresh needs **reserved** modulus capacity. Conditional OpenFHE two-party FIXED layouts with sampled parameters project 390.07/1,560.28 MB uncompressed worker-link bodies even granting one live ciphertext per epoch. | Veto these parameter/layout transfers; other refresh designs need actual primes, serialized bodies, full layer and output-key ownership checks. |
| [Function-specific joint correlations](docs/evidence/adjacent-correlation-review-2026-10-01.md) | Reusing one masked source within a joint statistic/normalization block really removes an opening. Norms plus token boundary still project 16.15/24.72 MB online at 32 bits. An in-process BFV generator has exact modular parity but unsanitized ciphertext noise, so no circuit-privacy admission. | Keep algebra as incremental optimization; the tested layouts fail before private inverse, attention and SiLU. Material compression alone cannot repair online cost. |
| [Delayed residue reconstruction](docs/evidence/adjacent-residue-review-2026-10-01.md) | CRT preserves bounded polynomial islands, not signed comparisons or rounded float semantics. A 28-bit two-source layout costs 14.15/21.68 MB online. A hypothetical 14-bit tuple leaves only 154,246/211,846 online bytes for every omitted operation. | Veto tested wide layout. Narrow layouts need exact protected lifting/order/scale transitions; channel-local tables cannot supply those for free. |

All paired values refer to **39+8 / 39+32**. Communication projections describe
specified encodings/layouts, not universal cryptographic lower bounds. Refresh
figures use raw RNS coefficient storage, not measured serialized network bodies.
Public setup and cold checkpoint distribution remain separately charged; HE/HSS
correctness error needs whole-response accounting, not one-operation parameters.

### Corrections and cross-track lessons

1. **A stronger encoding does not mean unrestricted local nonlinear evaluation.**
   BGI/BKS/Roy–Singh-style HSS supports original-input × retained-register
   multiplication and branching programs. A transformer circuit's native gate
   count is not its branching-program length. Re-encoding computed activations,
   finite-machine rounding and terminal decoding need concrete costs. Recent
   SIMD HSS and one-bit-per-gate garbling still have construction-specific
   assumptions, preprocessing and offline FHE/material; no headline multiplier
   prices this decoder.
2. **Exhaustion depth is not a refresh interval.** At scale `2^40`, the inspected
   Lattigo helper's nominal two-party 128-bit mask condition needs about 169
   active-modulus bits. The small sampled context cannot meet it; the large
   context leaves roughly one product before refresh, not four. Stronger
   magnitude/session bounds may consume that remaining level. Collective keys
   held by workers also change who can decrypt and how final client output is
   delivered. CKKS refresh preserves prior approximation error; it does not
   restore W8A8 arithmetic exactly.
3. **Joint reuse can remove an interaction, but only for the same logical value.**
   A source mask can serve a fixed public-map statistic and a broadcast product
   in one admitted block. Reusing it across distinct values reveals their
   difference. Normalized hidden state, attention probabilities and later MLP
   input are new sources; treating them as the original source undercounts cost.
4. **Numeric representation is an algorithmic boundary.** Rational comparisons
   can implement bounded ties-to-even without a reciprocal, but that does not
   reproduce all intervening float32 operations. Residue or projective arithmetic
   can postpone a reconstruction only until an actual order/scale/rounding
   consumer. The residue inventory's 23 obligation sites per layer are semantic
   obligations, not 23 forced plaintext openings or network rounds.

### Direction and next admission gates

**Near-term numeric experiment:** screen a rounding-aware
**RMSNorm→dynamic-quantization predicate fusion**. In ideal arithmetic, codes
of a row with a common positive denominator depend only on its numerator;
its absolute activation scale still depends on that denominator. The new probe
already disproves unconditional float32 cancellation. A useful method must
certify code decisions against the actual rounded program, handle epsilon,
zero rows, signs and half ties, and retain the scale needed by QK, residuals
and SiLU. Refinement/fallback must not leak private ambiguity; oblivious or
padded execution costs count. Reject if certificate bounds are ineffective or
the whole protected region remains over budget. This is not an authorized
compiler rewrite or an assumption that preserving codes preserves logits.

**Longer-horizon representation experiment:** choose one concrete adaptive HSS
construction and produce a typed complete-layer program certificate with two
KV-backed decode steps. Every product must be legal in that construction, or
use an explicitly charged bridge. Price terminal shares as their actual
cryptographic objects, not unimplemented i32 outputs. Reject on program-size,
global correctness, local-compute or directed-body bounds before key generation.
The DCR token-bit ingress is a reason to investigate this certificate, not a
reason to deploy an HSS worker.

Park the tested refresh/residue layouts and explicit HE correlation generators
as tenfold routes. Reopen only with a different complete construction that
changes the failed floor. Continue using prepared inference as the executable
control; none of the parallel screens activates an Experiment or component.

### Reproduction and validation

These repository-root commands were executed by the tracks; pinned-config
options require the existing cached official configuration, not model weights.
The BFV option requires installed HE dependencies and measured about 735 MB
process peak RSS for its bounded synthetic generator; it is not circuit private.

```sh
uv run --no-sync python scripts/probe_hss_feasibility.py
uv run --no-sync python scripts/probe_projective_feasibility.py --with-pinned-config
uv run --no-sync python scripts/probe_collective_refresh_feasibility.py
uv run --no-sync python scripts/probe_structured_correlations.py --bfv
uv run --no-sync python scripts/probe_residue_feasibility.py --semantic
uv run --no-sync pytest -q tests/test_projective_feasibility.py tests/test_collective_refresh_feasibility.py tests/test_structured_correlations.py tests/test_residue_feasibility.py
```

The combined new regression gate passed **54 tests**. All nine new Python files
passed Ruff lint and formatting. HSS also passed exhaustive 256-point sign and
rounding interpolation checks and adaptive-input packing assertions. These are
bounded arithmetic/protocol-accounting checks, not executed secure HSS,
collective refresh, distributed dealer generation or a full decoder.

## Four follow-on Qwen screens — 2026-10-01

Evidence: [`conditional-execution-qwen25-2026-10-01.json`](docs/evidence/conditional-execution-qwen25-2026-10-01.json).
These are independently runnable, non-selectable research diagnostics. The pinned
Qwen checkpoint, baseline W8A8 body, semantic schedule and binding are recorded.
Numeric agreement is with that W8A8 execution, not broad float32 task quality.

### Communication-aware mixed precision

Candidates are chosen by actual quantized-weight row-L1 bounds and the resulting
u16/u24/u32 transition, not nominal bit count. Lowered weights come from checkpoint
artifacts, not requantized W8 snapshots. The clear callback carries an explicit
numeric override; it is not admitted as a mixed-stage Experiment. Seven candidates
cover whole semantic roles and sparse MLP-down placements. Four public tuning
prompts precede two separate six-prompt held-out cohorts, with distinct bounded
token sequences and recorded digests.

All-role W8A6 MLP-down loses a tuning prefill decision. The tuning-selected last
four MLP-down layers match all four tuning prompts, but the first held-out cohort
matches only **3/6 prefill and 4/6 same-token decode**. Confirmation matches 6/6
each; it does not erase the failed first cohort. At 39+8 the candidate matches all
eight generated selections; at 39+32 it matches only **8/32**.

At 39+32 the selected override projects **1,612,800 fewer online arithmetic bytes**
and **250,880 fewer correction bytes**, just **1.86 MB** combined (about 1.05% of
the arithmetic baseline). Integer MACs are unchanged. The arithmetic baseline is
176.98 MB, smaller than the historical 178.97 MB accounted application bodies;
these projections exclude tickets, framing, cold distribution and unused inventory.
The quality failure and small saving do not justify a live component.

### Greedy private-speculation prerequisites

A client-local controller verifies a pinned pretrained SmolLM2-135M draft against
the unchanged compiled W8A8 target. Different tokenizers are bridged through draft
text; correctness is determined entirely by target decisions. Exact KV snapshots
retain only accepted rows, while rejected target evaluations remain charged.
Four actual 39+8/39+32 runs (draft widths two/four) preserve every target selection.
Sampling is greedy, with fixed output caps and no EOS early stop.

At 39+32, widths two/four accept **17/29** and **18/54** offered tokens. Rejection
adds **12/36** consumed target rows: 43/67 decode rows versus 31, raising total
prefill+decode body work by **17.14%/51.43%**. The current compiler admits only
one decode query row; verification therefore makes sequential stage calls.
This is a tested controller and clear-kernel consumption trace, not a protected
inventory integration, batched query plan or WAN latency benchmark. Real batching
needs a new compiler-bound query/state contract and admission for all rejected
attempts. The tested pairing does not reduce fresh-prompt bytes. Other drafts or
latency-focused batching are separate investigations, not disproved here.

### Decision-directed certification

A bounded certificate uses pairwise coefficient differences, preserving shared
input uncertainty instead of independent logit intervals. It accounts for actual
float32 head rounding, uses strict positive margins and falls back to full
computation on ties or unresolved choices. Exhaustive small completions and a
correlated synthetic-tail example check useful certificates; a complete tiny
compiled prefill/decode checks parity with exact body and KV.

On six fresh public Qwen prompts, prefill and same-token decode yield **0/12 early
certificates**. All require all **896** head features. Body computation and KV
are already exact; **zero body work and zero provider bytes are avoided**. Bound
evaluation also scans unresolved public coefficients, so dot-product reduction
alone would not prove a CPU saving. This cheap gate rejects the tested public-domain
head bound before protected whole-decoder integration, not all possible tighter
trajectory-aware certificates.

### Honest-majority three-party screen

The screen evaluates explicit replicated arithmetic layouts, inspired by
[ABY3](https://eprint.iacr.org/2018/403), with one semi-honest corrupt party among
three independent operators. Each party holds two adjacent shares. Test-local
linear and multiplication algebra agrees with independent clear modular products;
distributed randomness, active security and exact Qwen numerics remain unimplemented.

For 39+32, six-copy ordinary public-linear execution projects **150.29 billion
MACs**, versus **50.10 billion** for two offset workers. Computing each unique
share once reduces this to **75.14 billion**, but retaining replicated outputs
adds **255.47 MB** of 32-bit peer bodies. Gated products alone project **98.06 MB**
online before attention, normalization, truncation, private quantization or
fresh zero-sharing costs, against an 11.35 MB online target. These are alternative
explicit layouts and arithmetic counts, not measured full-response CPU or a
universal three-party lower bound. Both tested layouts fail admission.

### Reproduce and decision

All commands below ran from repository root:

```sh
uv run --no-sync python scripts/probe_wire_precision.py
uv run --no-sync python scripts/probe_greedy_speculation.py
uv run --no-sync python scripts/probe_decision_refinement.py
uv run --no-sync python scripts/probe_three_party_cost.py
uv run --no-sync pytest -q tests/test_decision_refinement.py tests/test_greedy_verification.py tests/test_wire_precision_screen.py tests/test_three_party_cost_screen.py tests/test_quantization.py tests/test_runtime_model_binding.py
```

The focused gate passed 71 tests, including native compiled tiny prefill/decode,
rounding/subnormal/tie behavior and non-refundable rejected evaluations. The
eleven new Python files pass Ruff lint/format checks. CPU samples from concurrent
screens are diagnostic only; no matched latency or compute-cap ranking follows.

**Do not promote any of these tested methods as a fresh-prompt network winner.**
Keep the exact verification controller as a prerequisite if latency batching later
becomes a separate objective. Reopen certification only with demonstrably tighter
sound bounds, mixed precision only with new held-out parity, and three-party work
only with a different interaction/work schedule. Prepared inference remains the
working control.
