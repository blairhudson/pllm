# Paper-to-SDK capability plan

Status: SDK composition search, its matched finalist scorecard and the first
Qwen Slalom/SmoothQuant adaptation controls are implemented. Maverick and
SIGMA/FuseFSS have bounded native component references with explicit unresolved
resource/security gates. The SIGMA-derived masked-wire block and FuseFSS-inspired
fused/unfused signed-rescale helper also have installed SDK operator benchmarks.
The masked-wire component additionally binds complete dense float32 gated regions
in the CPU two-offset Experiment. Generated Qwen2/Qwen3 prefill/decode matches an
independent integer oracle, and separate worker processes complete SDK and gateway
requests. Client conversion and online issuance, per-party one-use material and
both client-relay legs are explicit. The fresh twelve-prompt Qwen2.5 numeric
diagnostic matches 10/12 prefill, 36/36 same-token decode and 9/12 eight-token
greedy trajectories against locked W8A8; logits/KV are not exact. Compact,
Curl and LogRow now have validated bounded component/operator bindings,
with complete material owners and separate compressed numeric identities.
The first protected response-cost sweep hit its 3,600-second request timeout
without a complete comparison report; that failed attempt remains explicit.
The bounded transport follow-up identifies fresh key generation/delivery as
dominant. Opt-in one-chunk issuance overlap and concurrent worker requests lower
capped synthetic row latency by 39.3%/40.6% for unfused/fused layouts, with unchanged
traffic and almost unchanged aggregate CPU. Client chunk/frame admission doubles
to 140 MiB before margin. A subsequent fresh-client 30+2-token Qwen2.5 pair
measures 39.61% lower full-response latency (2,705.809 to 1,634.109 seconds),
1.01% higher client peak RSS and identical 5.72 GB TCP-stream traffic; outputs
match and observed swap growth is zero. Aggregate CPU is almost unchanged.
The bounded native profile checks 1,926 outputs. An interleaved implementation
pair removes repeated DPF expansion and resource serialization: 11,556 outputs
remain exact, while 256-lane issuance falls 28.22%/33.22% for unfused/fused layouts.
The separate opt-in `material_encoding="seeded"` reconstructs only each worker's
own independent random key fields from fresh, role-bound seeds. Client masks and
correlation-dependent correction shares remain separate. Expanded-key memory and
one-use lifecycle bounds remain charged; both packet versions execute the same
Q12 numeric profile.
The matched 30+2 whole-response encoding pair lowers latency 14.79% and
TCP-stream traffic 17.17% (5.7166 to 4.7350 GB), with matching outputs and zero
observed swap growth. Aggregate CPU increases 2.73%; client peak RSS falls 1.36%.
Evidence and reproduction: `docs/research/nonlinear-decoders.md`.
The paper landing map is
`docs/research/paper-module-landing-map-2026-09-23.md`; the 85-entry source/citation
inventory is `docs/data/research/paper-library.json`. `papers/` is an ignored
local research cache; run `uv run python scripts/sync_paper_library.py --fetch`
to populate it, never during documentation builds. The existing 24 locked
research records in `docs/data/research/papers.json` are separate. A source in
the chronology is **not** an implementation or an eligible inference composition.
The packaged `python/pllm/components/planned_methods.json` binds every paper to one
importable Python symbol. The [generated research method roadmap](https://pllm.run/sdk/components/research-method-roadmap/)
lists those symbols and their next gates.

## Current round: shared prerequisites and complete consumers

The current priority is the working prepared memory/latency frontier and complete
SDK compositions, followed by shared research bottlenecks. Prepared comparisons
separate cold bundle import, preparation, online execution and second-request
reuse. The earlier 150+8 public-prefix variants spent almost the same time online;
their request-time difference alone does not isolate paging-kernel overhead.
`benchmarks/research/prepared_frontier.py` holds matched serial/windowed, explicit
reuse and resident-client controls. `PreparedInventory(stage_window_bytes=...)`
binds an optional larger work window while existing whole-topology admission still
prices every concurrent owner.
The matched ordered two-request cohort passes all checks with identical outputs
and zero observed new swap. Windowed paged preparation lowers cold preparation
42.04 to 38.18 seconds and complete cold/repeat requests 2.67%/3.67%, with equal
covered bodies and 2.02% more cold-first aggregate CPU. Against resident client
weights it halves client peak RSS, 564.77 to 281.85 MB. Explicit prefix reuse
reduces the exact-repeat request 103.45 to 11.67 seconds and two-request TCP-stream
bytes 830.67 to 495.07 MB; the first response remains costly. The ordered-sequence
factory, publisher costs, source hashes and retained rejected initial screen are
documented in `docs/research/prepared-frontier.md`. These single pairs do not
establish a general fresh-prompt or 10× memory improvement.

The subsequent matched Qwen3-4B 16+8-token pair uses fresh clients, indexed
tokenization and native 100/40 caps. Paged versus resident client storage lowers
client peak RSS 1,020.46 to 261.05 MB (3.91×), with identical output text and zero
observed new swap; request time rises 142.665 to 155.460 seconds. A separate
resident ownership fix halves retained matrix payload but leaves import peak
RSS unchanged. These results do not establish 10× memory savings or broad model
quality. Evidence and replay: `docs/research/qwen3-client-memory.md`.
The held-out FP32 quality attempt rejects before tensor loading: 16.09 GB of
reference weights alone exceeds 13.84 GB of post-reserve headroom. Its separate
replay and rejected admission are retained; a quality score remains pending.
A later compiler-bounded KV allocation removes the short-response 64-row floor,
preserving generated-model logits and active KV bits. The matched Qwen3 pair
reduces array payload from 18.87 to 6.78 MB but changes client peak RSS only
235.24 to 234.98 MB (0.11%). Further frontier work must target the measured
process peak; this payload reduction alone did not improve it.

Protected selection now exposes bounded independent phase batches, with complete
frame/header accounting and fresh material per operation. Its fresh-process
comparison lives in `benchmarks/research/batched_selection.py`; provider-state
integration still requires a complete cost justification. Thirty-six fresh
processes check 468,585 outputs. Full-width top-k reduces exchanges 19,200 to
1,200, but keeps 295.26 MB keys and nearly unchanged local CPU; its live-allocation
estimate rises 15.45 to 58.29 MB. Evidence:
`docs/evidence/research-batched-selection-cost.json`. Encrypted attention
range/bootstrap work remains a separate longer-horizon track.

1. **Mixed arithmetic — implemented bounded operator slice.** Curl, LogRow and
   Compact accept an explicit mixed backend without changing numeric profiles or
   default identities. Thirty-six fresh processes check 2,916 exact outputs.
   LogRow/Compact combined material improves 24.9%/12.2–29.6%; Curl regresses.
   Whole-MLP costs remain 203 GB–3.02 TB before remaining decoder work, and fresh
   twelve-prompt quality checks preserve no complete logits/KV. See
   `docs/research/mixed-gated-reference.md`. No tensor/provider promotion follows.
2. **FastQuery — implemented bounded repacking/conversion slice.** Extracted-LWE
   packing and direct interleaving compose with independently supplied party
   inputs and local-dealer OT conversion. Eighteen fresh processes check 3,456
   outputs. At 4,096×64 mixed precision, online frames fall 1,642,896 to 180,624
   bytes, with higher setup material and compute. Symbolic Qwen costs still need
   234.32 MB online plus 409.73 MB conversion keys and setup. Tied-head ownership
   releases zero weights. Full-width noise, distributed setup and transport remain
   open; see `docs/research/packed-embedding-reference.md`.
3. **MPCache / CipherPrune — implemented bounded mixed selector.** Garbled
   predicates and fresh arithmetic complete-record moves check 171,303 outputs
   across 33 fresh processes. Partial pruning improves its matched small control;
   empty/keep-all cases regress. Full 256-row/128-feature K/V selection fits a
   15.45 MB live-allocation estimate, but needs 295.26 MB keys, 78.59 MB peer frames
   and 19,200 serial rounds to retire 131,584 clear-state bytes. Protected scoring,
   authoritative owner binding and provider transport remain open. CipherPrune's
   mixed backend has ordinary SDK/replay; top-k is an internal measurement binding.
   Details: `docs/research/shared-selection-reference.md`.
4. **Encrypted linear foundation — implemented bounded SDK slice.** Public
   diagonal matrices and private CKKS vectors execute in separate client/provider
   contexts. Eighteen fresh processes check 1,224 outputs within `9.37e-7`
   float64 error. One sequential rotation key cuts 32×32 setup frames from 25.16
   to 1.20 MB with unchanged online bytes. Native admission, finite depth,
   resource accounting and replay are explicit. See
   `docs/research/encrypted-linear-reference.md`. MOAI attribution
   requires acquiring and reviewing its full source; the OpenReview browser
   challenge remains unresolved.
5. **NEXUS / THOR — implemented bounded encrypted softmax slice.** Source
   exponentials, encrypted reciprocals and final normalization execute without
   intermediate decryption. Eighteen fresh processes check 738 outputs; worst
   softmax error is below `3.21e-5`. Both profiles consume fourteen levels and
   charge 260.19 MB setup plus 25.19 MB for three width-32 rows. The fixed
   [-1/8,1/8] range rejects all 48 prefill/decode positions in a separate fresh
   twelve-prompt pinned W8A8 audit. No encrypted candidate is executed there.
   Secure maximum, QK/value products, masks, bootstrap, persistent decode state
   and private token feedback remain separate gates. See
   `docs/research/encrypted-softmax-reference.md`.

Each slice uses installed SDK replay, complete ownership/cost records, independent
numeric validation and a validated commit. Whole-response measurements require
an admitted complete consumer; bounded operator availability is a separate claim.

## Next delivery order: a PLLM comparative study

Prioritize recent papers with direct relevance to private LLM execution, established
published reference systems, and reusable prerequisites. This is an engineering
priority, not a citation-count or original-paper performance ranking; no current
bibliometric survey has been performed. Paper versions, authors and source links
come from `docs/data/research/paper-library.json`, while actual PLLM coverage comes
from `docs/data/research/implementations.json`.

| Priority | Papers | SDK work to deliver | First decisive gate |
| --- | --- | --- | --- |
| 0 — establish comparable controls | SmoothQuant (2023), Slalom (2019) | Put the existing channel-equalization and Freivalds adaptations through saved Qwen2.5 configurations, explicit quality checks and complete response accounting | SmoothQuant needs a separate numeric/quality cohort; Slalom's verified preparation must fit admitted memory. Neither is an original-hardware reproduction |
| 1 — newest direct linear-delegation target | Maverick (2026) | Implement reusable private matrix delegation and coding verification contracts in protocols/verification; extend beyond the current bounded coding reference | Full-width code/resource bounds, the actual masking/privacy construction, repeated queries and malicious-result rejection. The current Walsh toy lacks input privacy |
| 2 — shared nonlinear execution foundation | FuseFSS (2026), with SIGMA (2024) as the reference baseline | A party-local fixed-point/FSS backend, then independently selectable mask-aware fusion; save unfused and fused paper configurations | Exact/reviewed rescaling, signedness, preprocessing consumption and two-role transport before a whole decoder. Existing shared-runtime truncation remains inadmissible |
| 3 — state and long-context costs | MPCache (2025) | Complete the model-neutral retention transform, numeric-state executor and protected dynamic-selection contract | Held-out same-token prefill/decode quality and observable access-pattern tests; ordinary prefix reuse is not MPCache |
| 4 — compare nonlinear methods on the shared backend | Curl (2024), Compact (2024), LogRow (2024) | Promote the existing bounded references to explicit tensor schedules with transport and material admission | Include protected input conversion/truncation, all one-use material, offline issuance and whole-decoder quality; a lookup table or scalar approximation is insufficient |
| 5 — additional private transformer systems | CipherPrune (2025), NEXUS (2025), THOR (2025) | Source-specific methods expressed through generic semantic transforms and separately declared HE/hybrid protocols | Private pruning decisions and model-quality changes for CipherPrune; complete autoregressive state/token feedback and ciphertext costs for HE systems |

Bifrost (2026) follows availability of an actual attestation/key-provisioning
executor and suitable hardware. Euston's analyzed vulnerable transmission remains
a regression/repair review target. Neither receives a simulated successful SDK score.

### Current follow-on queue

After the SIGMA/FuseFSS material comparison, prioritize these ten paper
reimplementations by client-memory benefit, communication cost and readiness for
reusable SDK integration. This updates the delivery sequence above; completed
Slalom/SmoothQuant controls and historical failed gates retain their evidence.

| Order | Paper | Next bounded delivery |
| --- | --- | --- |
| 1 | MPCache | Physical retained-state ownership, compiler-bound causal execution, then protected top-k/gather and fresh held-out quality |
| 2 | Curl | Complete gated-block composition, conversion costs and compression/quality tradeoffs |
| 3 | LogRow | Tensor scheduling and party transport with full conversion/product accounting |
| 4 | Compact | Protected polynomial evaluation costs and checkpoint-range coverage |
| 5 | Compressing Correlations via Secret Replication | Establish exact arithmetic/party assumptions, then a bounded independently implemented PCF construction and complete correlation costs |
| 6 | FastQuery | Private embedding queries, including retained output-head ownership and measured client-memory impact |
| 7 | CipherPrune | Protected pruning/degree decisions and held-out quality under explicit new numeric identities |
| 8 | Maverick | Resolve code/field security assumptions and the 22.90 GB preprocessing gate |
| 9 | MOAI | Reusable packing/operators for an actual non-interactive encrypted baseline |
| 10 | NEXUS | Actual ciphertext execution, complete decode state and token feedback after the HE foundation |

The first MPCache slice implements bounded native retained-row storage and an
internal GIL-released Python binding. Original token positions survive physical
compaction, causal masking and cluster selection; discarded native storage is
erased and released. Independent controls cover source release, exact retained
bits, atomic append/compaction failures and dense-versus-retained attention.
Owned payload and transient costs stay explicit. The next internal slice binds
static client-local retention to the native semantic schedule, validates per-layer
causal state and bounded decode, and rejects full-KV snapshot/continuation reuse.
Generated Qwen2/Qwen3/Llama whole-decoder controls preserve every logit and KV bit
at full retention; sparse execution matches an independent selected-view oracle
and releases discarded backing arrays. Protected provider-held selection remains
open; the prior selected-view cohort retains its
original scope. A new fixed twelve-prompt Qwen cohort preserves all 48 checked
logit vectors and final KV at full retention, reducing median owned KV from 6.29
to 5.16 MB. Keeping 75%/50% uses 3.90/2.62 MB but each matches only 35/36
same-token decode selections. Diagnostic process peaks remain 1.21–1.23 GB and
CPU rises; no SDK-client memory or speed improvement follows. Static-retention
quality and protected provider-held selection remain distinct work. The static
contract is now selectable through `ClientLocalKv(retention=KvCacheEviction(...))`
in client-only and prepared Experiments. Generated SDK/gateway requests and
two prepared role children pass; response completion/cancellation retire native
state, while history follow-ups re-prefill. Benchmark matching binds the retention
numeric policy and placement admission prices its state/workspace owners. Two
fresh-client prepared SDK cohorts at 52/379 input tokens plus eight outputs under
native 100/40 caps now measure full-retention client peaks of 257.77/524.85 MB versus
dense 264.90/546.31 MB. Full-retention output text matches, while the longer 75%
response changes text. Covered bodies stay 257.79/989.08 MB for every policy and
latency is nearly unchanged. Sparse retention does not monotonically lower peak
RSS; the one-response cohorts do not establish representative quality or a 10×
client-memory benefit. Different numeric contracts retain empty rankings. The
original longer workload's residue-frame rejection is preserved separately from
the admitted 22-entry rerun; memory admission did not price that codec limit.
The separate bounded client-garbled top-k/gather substrate binds one-use private
owner revisions and moves complete KV records through fixed bitonic networks.
Twelve fresh native processes match 474 independently checked output words.
Selecting eight of sixteen two-feature rows spends 1.10 MB of label/ciphertext
material plus 2.06 MB estimated public metadata to retire 160 bytes of state.
The 64 MiB allocation gate rejects 16×16 records. A follow-up compact-index sort
plus complete oblivious gather checks 4,041 exact output words in 36 fresh
processes, including two/four-row Qwen-width state. It reduces public metadata
estimates but increases label/ciphertext bodies and native CPU in every matched
case. Half-retaining 256 rows of 128 features projects 8.71 GB per layer versus
1.77 GB for full-record sorting; all complete-state shapes fail preflight before
issuance. Provider-resident scores/state, compiler-validated context, transport
and a cheaper complete selection construction remain gates. A future method must
price oblivious routing, protected scores and all boundary owners before transport;
the current substrate stays Rust-only.
Details: `docs/research/mpcache-reference.md`.

The next Curl slice is now the installed `CurlWaveletGatedMultiplyQ9` operator:
private Haar lookup, protected compressed-index reduction, exact up/output
rescaling and fresh Beaver multiplication. Exact interval-key helpers replace
the paper's probabilistic truncation. Nine fresh processes check 2,889 outputs;
levels 0/2/4 project 213.08/81.58/49.06 GB combined keys and 971.27 MB peer frames
at historical 39+8 geometry. The four-level rescaling subtotal alone is 37.91 GB.
A fresh twelve-prompt W8A8 Qwen cohort fits the fixed ranges but matches only
10/7/8 prefill selections, 35/33/33 same-token decode selections and 10/5/4
eight-token trajectories; no logits/KV are exact. Each native clear profile
checks 130,043,904 independent integer outputs. SDK operator availability is
implemented; fidelity, complete cost, protected float conversion, review and
distributed role execution still gate decoder admission. Historical Q7 evidence
is unchanged. Details: `docs/research/wavelet-gated-reference.md`.

The installed `LogRowGatedMultiplyQ9` now composes its 4,096-row private lookup
with opaque half-gates rescaling and multiplication. Six fresh processes check
1,926 outputs on the same uncompressed numeric profile. Hypothetical 39+8 costs
are 12.89 GB of client encodings and 326.40 GB of evaluator material, including
242.29 GB for multiplication. Zero evaluator-peer frames do not include material
delivery or boundary labels. The unchanged profile retains its failed Qwen
fidelity screen. Serial 16 MiB allocation admission, one-use execution and
installed benchmark access do not authorize tensor/provider execution.
Details: `docs/research/logrow-product-reference.md`.

The installed `CompactGatedMultiplyQ12` now composes private polynomial selection,
exact input/intermediate/output rescaling and signed multiplication. Its dyadic
density-weighted fitter and half-gates backend are scoped adaptations, rather
than Compact's annealing or complete MPC system. Fresh Qwen diagnostics for three
frozen profiles have no range violations. The 8-piece cubic matches 9/12 prefill,
35/36 same-token decode and 9/12 generation trajectories; no complete logits/KV
are exact. Native and independent integer results agree over 131,444,736 elements
per profile. Nine fresh operator processes check 2,889 protected outputs; complete
39+8 MLP geometry projects 1.97–3.42 TB of evaluator material plus 13.40 GB of
client encodings. Polynomial arithmetic dominates. These are negative costs of
the Boolean adaptation, not Compact's paper system. Details:
`docs/research/compact-product-reference.md`.

Secret Replication now has the installed `SecretReplicationPcf` scalar/vector
operator: two independent line-PCF directions, expanded/punctured keys and a
one-use full-field consumer. Eighteen fresh processes check 5,778 exact outputs.
At field 257 and 64 lanes, combined keys fall 16,424 to 328 bytes while expanded
correlations remain 1,032 bytes and peer frames 676 bytes. Expansion still needs
65,664 combined field-output PRFs; compact keys do not remove that work.
Restricted-domain masks fail exact arbitrary-input privacy checks, and one shared
scalar mask leaks differences if reused for independent elementwise inputs.
The local dealer and prime-field consumer therefore do not replace ring32 triples
or FSS keys. Full-width expansion, distributed setup and a compatible decoder
consumer remain open. Details: `docs/research/secret-replication-reference.md`.

FastQuery now has the installed `PrivateEmbeddingLookup` bounded operator:
encrypted one-hot coefficient packing, fresh uniform full-polynomial masks and
one-use signed packed-share conversion. Nine fresh processes check 1,215 exact
outputs. That historical control uses full ciphertext replies and local half-gates
conversion, before the separate repacking/party-local slice above. Its mixed 4/3/3 full-vocabulary
39+32 projection still needs 1.643 GB of HE frames and 293.35 MB of conversion keys,
before boundary labels, setup and the rest of the decoder. A separate public
W8A8-source requantization diagnostic checks 1,007,104 elements per profile; mixed
4/3/3 matches 10/12 prefill, 34/36 decode and 6/12 eight-token trajectories, with no
exact logits/KV. Token lookup and the client head share 136.13 MB of integer
weights, so embedding-only relocation releases zero matrix bytes. This establishes
neither paper-system costs/quality nor full-response client-memory savings.
Details: `docs/research/packed-embedding-reference.md`.

CipherPrune now has the bounded `CipherPruneTokenPruning` operator: exact Q16
score-numerator thresholds, disclosed count, stable full-record compaction and
compacted degree selection. Fifteen fresh processes check 381 exact outputs.
Public-count and degree disclosures remain explicit; even normalized two-token
attention can produce distinguishable counts. A hypothetical 39-token/896-feature
invocation retaining 19 records projects 631 MB evaluator material before client
encodings and metadata to retire 143 kB of two-party payload. Worst-case 64 MiB
admission rejects full-width issuance. Learned thresholds, selected polynomial
execution and causal KV mapping remain absent, so checkpoint quality is unknown.
See `docs/research/token-pruning-reference.md` for source deviations and costs.

Maverick now exposes the existing RAA algebra through the installed
`MaverickDelegatedLinear` operator with explicit geometry and configuration replay.
Nine fresh processes check 50,688 exact signed outputs. Public preprocessing,
fresh masking, evaluator work and post-response checks have separate timings;
the admitted Qwen attention shapes retain 51.72/66.45 MB client public payload.
Both MLP shapes still reject; whole-body P+Q remains at least 22.90 GB. The source's
cited field-range theorem excludes BabyBear and no seeded-code distance or
dual-LPN certificate has appeared. Operator availability therefore adds no
decoder/privacy claim. See `docs/research/maverick-reference.md`.

The authorized LogRow → Compact → Secret Replication → FastQuery → CipherPrune →
Maverick sequence now has installed bounded operators, replay and cost evidence.
The individual source, numeric, storage and security failures above still prevent
new full-decoder promotion. Each slice retains paper deviations and explicit
unsupported coverage.

Each whole-model delivery ends with:

1. A paper-to-method contract stating **full reproduction**, **scoped adaptation**
   or **component reference**, with deviations from the cited source.
2. A reusable executable SDK capability, bounded native admission and real role
   transport for the claimed scope, including failure/cancellation behavior.
3. A repository Python `Experiment` factory, pinned public calibration where
   needed, and immutable source/configuration/evidence identities.
4. A Qwen2.5-0.5B matched cohort against both the relevant baseline and the best
   eligible SDK combination. Qwen3-4B follows after memory admission. Comparable
   model/source/workload/lifecycle and trust contracts are explicit.
5. Quality-measured cohorts for methods that change numerics or semantics. Such
   methods cannot enter today's exact-body/output scorecard by suppressing its
   identity checks. Record quality, bytes on every link, setup/material, CPU/GPU,
   memory and failures separately; unknown measurements remain unknown.

The meta-study should first establish useful head-to-head **methods**, then compose
compatible methods through SDK search. Do not relabel the fastest PLLM stack as an
individual paper's reproduction or mix original-paper speedups into PLLM scores.

### Experimental benchmark eligibility and recommendation are separate

User-driven benchmarking should expose correct, bounded paper methods even when
their measured cost or quality is poor. A failed optimization screen blocks a
recommendation or exact-control ranking; it does not by itself block an opt-in
SDK research benchmark. Preserve the negative result, source attribution and
replayable configuration so users can measure different supported workloads.

Use three explicit coverage levels:

1. **Operator benchmark:** installed SDK access to the bounded native reference,
   public numeric contract, resource admission, one-use lifecycle, citations and
   measured material/CPU/rounds. An in-process dealer stays an explicit local
    research assumption. The masked-wire block and fused/unfused rescale helper now
    have this installed SDK boundary and typed reports. Checked examples live in
    `examples/benchmarks/masked_wire.py` and `examples/benchmarks/shared_rescale.py`.
2. **Experimental whole-model benchmark:** actual admitted tensor schedule,
   conversions, complete operator/state coverage and declared role transport.
   Report quality changes and costs under a separate numeric identity. A partial
   block must not silently activate a decoder or be counted as completing a paper.
3. **Recommended serving composition:** independently satisfied privacy/security,
   quality, resource and deployment requirements for the claimed workload.

Correctness in the declared numeric domain, implemented execution coverage and
bounded resource/lifecycle behavior remain hard requirements at every level.
Unavailable functionality and unknown costs stay explicit. Source coverage and
SDK availability are recorded independently of whether a method wins a benchmark.
References currently callable only from repository scripts still need installed
SDK bindings, typed reports and checked examples before claiming SDK availability.

### First control delivery — 2026-10-05

Saved factories under `benchmarks/research/` now run Slalom-derived verification
and SmoothQuant-derived public equalization on pinned Qwen2.5-0.5B. The verified
native 100/40 pair completes a 50+8-token response with matching outputs;
composing exact transport/delivery choices reduces 374.00 to 334.21 MB of
TCP-stream traffic. Correct native Freivalds check counts and role-local buffers
replace the historical all-role memory overestimate. This is an accounting fix,
not a weaker verifier or a memory optimization.

Fixed equalization calibration uses eight public sequences and a separate
twelve-prompt prefill cohort. Top-1 regresses from 11/12 to 10/12 while worst
logit error improves from 3.6207 to 2.5092. It cannot be presented as a numeric
quality win or merged with the original W8A8 performance cohort. Quality,
offline calibration, native role costs and exact factories are retained in the
Research data rather than selectively promoting the better statistic.
Within the fixed equalized body, exact SDK composition reduces one native
100/40 response from 78.26 to 70.24 s and 289.41 to 251.25 MB TCP-stream traffic,
with client peak RSS 540.85 to 262.96 MB; aggregate CPU rises 34.47 to 37.28 s.
These pairs are baseline-versus-composition controls, not exhaustive searches
for the best verified or equalized stack. Whole-generation quality remains open.

Maverick now has a separate bounded full-width RAA masked-delegation reference,
with exact sparse sampling, one-use client ownership and explicit allocation
admission. Its source review identifies an unresolved BabyBear applicability gap
in the quoted distance-failure bound; fresh masks and successful corruption tests
do not resolve it. See `docs/research/maverick-reference.md` and its saved Python
reproduction. Concrete code-distance/dual-LPN review and the 22.90 GB Qwen2.5
preprocessing lower bound still block a recommended delegated-decoder method.
Bounded local benchmark exposure must retain the unresolved security contract.

SIGMA/FuseFSS now share a native, one-use signed-rescale/nonnegative helper
reference with fixed public key/frame shapes, an exact ties-to-even extension
and complete key-payload accounting. The initial cohort uses one quadratic
prefix-DPF backend for both layouts, not either paper's optimized implementation.
Twelve fresh-process cases check 3,840 outputs. For 64 16-bit/seven-bit-shift ties-even lanes,
fusion halves peer bodies to 668 bytes but increases local online elapsed time
from 29.98 to 49.32 ms and key payload from 5,888 to 9,428 bytes per party/lane.
A hypothetical 39+8-token SiLU-width placement projects 63.24/101.25 GB of keys.
The independently implemented Figure 1 compact DCF now passes the same numeric
and lifecycle checks. A separate 24-case matched cohort reduces fused 16/7
ties-even keys from 9,428 to 1,718 bytes per party/lane and local online time
from 96.32 to 11.13 ms. The hypothetical 39+8 key total remains 18.45 GB; review
and complete-operator cost reductions still precede tensor admission or live
roles. A separate multiple-interval follow-up now reuses one universal DCF per
width within each one-use lane/phase. Its 24-case cohort checks 7,680 outputs,
reduces fused 16/7 keys from 1,718 to 758 bytes and issuance 12.83 to 5.74 ms,
with nearly unchanged online work and 668-byte peer bodies. Hypothetical 39+8
fused payload is still 8.14 GB. Complete-operator numeric mapping and cost,
review and independent issuance/transport remain gates; this primitive does not
activate planned APIs.
See `docs/research/shared-rescale-reference.md` for source deviations, exact
equations, fresh-process measurements and the rerunnable Python configuration.
The compact follow-up is in `docs/research/compact-rescale-reference.md`.
Universal interval keys are in `docs/research/interval-rescale-reference.md`.

The complete shared Q7 gated-block follow-up composes four exact rescalings and
two one-use Beaver multiplications. All 11,540 checked outputs match the integer
oracle, but 36.07% of pinned Qwen gate/up pairs exceed its declared range. The
39+8-token fused layout projects 41.33 GB of keys and 409.12 MB peer traffic,
before other decoder work. This closes the current profile's Qwen recommendation
path; out-of-domain inputs still reject in research benchmarks. See
`docs/research/nonlinear-block-reference.md`.
A separate public Q8/Q9 affine-SiLU follow-up now covers its twelve fresh
held-out prompts without domain rejection, but the best profile preserves only
10/12 prefill and 35/36 same-token decode selections. Its complete vector
coefficient lookup reduces matched raw material 11.51%, while hypothetical
39+8 still needs 60.10 GB material and 465.16 MB peer bodies. Four rescalings
consume 86.42% of the material. This is another optimization no-go, with all profiles frozen
before evaluation; a new numeric hypothesis needs new held-out data and a
materially cheaper complete rescale/product protocol before a decoder recommendation.
See `docs/research/piecewise-gated-reference.md`.
A separate joint-output screen now uses fixed Q12/Q16 mappings and one final
rounding. Q12/2,048 pieces matches 12/12 fresh prefill and 36/36 same-token decode
selections, with non-exact logits/KV and 2.78 worst logit error. Other profiles
still miss selections. Its direct dense function-sharing representation fails
storage admission, and even one uncompressed half-gate per element exceeds the
whole-response byte comparator. Narrow-ring openings alone exceed a tenfold
target. This closes those concrete representations before protected issuance,
not all joint FSS or vector correlation designs. The next gate is a specified,
costed construction that changes complete material/opening representation;
promising selected-token agreement alone does not supply an executable protected method. These
twelve prompts are now spent as held-out evidence. See
`docs/research/joint-gated-reference.md` for the complete candidate, both-party
views, native carry/privacy counterexamples and all four numeric results.
The subsequent primary-paper audit confirms the 60.10 GB arithmetic but limits
the rejection to our reference: 51.94 GB is generic ties-even rescaling, while
SIGMA's DPF-based gap shifts, residual SiLU lookup, effective widths and
masked-wire composition were missing. The follow-up now independently implements
that bounded path with proven public gap bounds, one-use party ownership and
floor/Q12 numerics. Twelve fresh processes check 5,136 protected outputs; 39+8
MLP geometry projects **6.93 GB combined raw material and 397.12 MB peer frames**,
8.67 times less material than the old, numerically different layout. The fresh
Qwen screen matches 10/12 prefill and 34/36 decode selections; both optimization
gates still fail. The opt-in `pllm.nonlinear.MaskedWireGatedMultiplyQ12` now exposes
this operator with user-supplied bounded inputs, immutable source-bound reports,
independent integer parity and separate approximation errors. Paper implementation
records link its SDK example and measurements without claiming full-model coverage.
The installed `pllm.nonlinear.SharedRescale` additionally compares three backends, both
layouts and both rounding modes. Its twelve-configuration wheel cohort checks
4,608 exact outputs; fusion halves peer frames while increasing keys and local
runtime. Floor and ties-even are distinct numeric cohorts. Reproduction and
per-party costs are in `docs/research/interval-rescale-reference.md`.
Both follow existing `NonlinearProtocol`/`ComponentRef` conventions. The separate
`pllm.metrics.benchmark_operator` owns lanes, samples and reports; historical
probe-v1 evidence remains replayable with explicit migrated identities.
The complete masked-wire block now exposes `layout="fused"`: Q12 rescaling,
Q6 index reduction and sign/interval views share the original masked gate input.
Eighteen fresh-process SDK cases check 5,778 exact outputs. Rounds fall from eleven
to eight; 64-lane peer frames fall from 6,020 to 5,552 bytes, while both-party keys
rise from 82,592 to 145,984 bytes and local online time from 7.31 to 17.51 ms.
The historical 39+8 geometry projects 6.93/12.25 GB unfused/fused keys and
397.12/387.30 MB peer frames. Both layouts remain opt-in operator choices.
The two-offset Experiment binding now implements tensor execution, explicit
client conversion and authenticated party-local transport. A fresh twelve-prompt
clear replay checks 132,145,152 integer outputs, preserving 10/12 prefill,
36/36 same-token decode and 9/12 eight-token greedy trajectories; worst logit
error is 3.46. Its reference is the locked W8A8 baseline, not upstream float,
and token agreement is not a semantic task-quality score. See
`docs/research/nonlinear-decoders.md`. Validate improved numeric mappings on
new prompts. Poor measured
quality/cost stays visible rather than excluding the method from research use.
Protected conversions, other operators, reviewed transport and the paper's GPU
system remain open. See `docs/research/masked-wire-reference.md` and
`docs/research/fss-paper-cost-audit.md` for exact source/accounting boundaries.
MPCache now has a native selected-view oracle with exact full-cache control and
8,712 independent index/equation checks. Static retention matches 34/36 same-token
decode selections; combined selection and adjacent-layer sharing matches 31/36.
The runtime retains full KV. Protected selection, sparse-state execution and
numeric quality remain gates; see `docs/research/mpcache-reference.md`.

The common Compact/LogRow/Curl Q7 screen retains different material and output
contracts. Even four-level Curl projects 1.40 GB material plus 730.30 MB peer
bodies for activation alone at 39+8, before protected conversion and the gated
product; see `docs/research/nonlinear-frontier.md`. NEXUS/THOR plaintext softmax
references reject all eight Qwen prompts under the fixed public score range,
with no candidate quality or ciphertext result. CipherPrune source review makes
its count/compacted-mask leakage and trained thresholds explicit. See
`docs/research/he-pruning-gates.md`. Next implementation decisions need a new
public numeric policy or materially cheaper complete protocol before a serving
recommendation. Correct bounded operators remain candidates for explicit SDK
research benchmarks with their failures and unresolved contracts retained.

The promoted indexed tokenizer now has a separate ordinary Qwen3-4B SDK pair
with fresh clients and native 100/40 caps. At 16+8 tokens it lowers client peak
RSS from 360.89 to 262.73 MB (27.20%); outputs and covered application bodies
match, but sampled aggregate memory rises and latency remains about 166 seconds.
Its 9.29 MB public index and offline compiler CPU remain additional. This closes
the earlier probe-to-SDK measurement gap for this choice, not the resident-weight
control or generation-quality gates. See `docs/research/qwen3-sdk-artifacts.md`.

## Importable Python stubs are not executable components

Every pending class has a stable public import, paper URL, reserved identity,
and missing first gate. Constructing it raises `NotYetImplementedError`, and
its identity is rejected through component factory, serialized `Pipeline`,
provider, and search entry points. Pending classes are absent from executable
component discovery; the existing 24 locked records keep their identities.
The additional bibliographic entries are not silently converted into reviewed
source locks. Ripple remains an unverified *lead*, and MOAI/MoZZarella lack
verified local full texts. Later method proposals need explicit semantic
operator, numeric domain, state/freshness, trust, dependencies, and evidence
contracts before any pending class is promoted.

Paper pages have **Planned Python API** callouts to the generated SDK reference;
SDK reference entries link back to the exact Research paper, and the roadmap
groups the missing first gates. Scoped implemented-component citations
keep their separate reciprocal links. Do not expose pending identities in the
runtime component registry, `Pipeline` selectors, `pllm.components` discovery,
or runnable examples.

To promote a method, replace its placeholder with a real capability-family
class, give it a reviewed executable component identity, and set that paper's
manifest `status` to `implemented` with an accurate scope/evidence description
in `gate`. The paper link persists, while the `pllm/planned/...` reservation
remains unusable; generated status percentages and paper callouts follow the
manifest and the exported class. A partial primitive alone does not pass this
promotion gate.

## Land proposals in shared capability contracts

| SDK home | Mapped research examples | First reusable contract |
| --- | --- | --- |
| `pllm.correlation` and `pllm.preparation` | PCFs, ring/field generators, silent OT, VOLE, MoZZarella | Correlation kind, algebra, share holders, authentication, expansion cost, one-use recipient and consumer |
| `pllm.nonlinear` and compiler numerics | Compact, LLAMA, SIRNN, ReDASH, FLUTE, LogRow, FSS/DPF lookup | Exact signed domain, approximation, protected selection, conversion and per-lane material bound |
| `pllm.modeling`, `pllm.passes`, and `pllm.kernels` | CipherPrune, Ditto, MPCFormer, ATLAS, SHAFT | Model-neutral semantic transform, numeric/quality invariant, operator schedule and artifact binding |
| `pllm.protocols`, `pllm.roles`, and `pllm.schedulers` | Bifrost, NEXUS, SIGMA, FuseFSS, Euston, THOR, MOAI | Topology, online parties, privacy contract, required offline material, message and state lifecycle |
| `pllm.verification` and Rust `pllm-assurance` | Maverick, EMVP, LAMP, DeepProve, matrix coding | Exact checked claim, soundness/attempt budget, challenge secrecy, trusted verifier and failure burn |
| `pllm.state` and `pllm.passes` | MPCache, Cachemir, non-autoregressive GPT | Valid-prefix/state handoff, token feedback, protected selection and generation quality |
| `pllm.assurance` controls and `pllm.nonlinear` source leads | Breaking Euston, Game of Arrows, Carnival, SoK, Ripple | Negative controls or source provenance only; no executable candidate from a survey, attack, or unverified lead |

The homes are capability families, **not paper-named models, profiles, or
compiler switches**. Different trust topologies cannot share an executable
contract merely because they perform matrix multiplication. A new family is
created only when input/output, lifecycle, or trust semantics genuinely differ.
Static provider manifests may be discoverable, but native provider runtime
loading is not yet available; a provider proposal must not imply it can run.

## Promote one vertical slice at a time

1. **Source gate:** verify the primary paper, source URL, authors, citation,
   license/provenance, claimed protocol and role assumptions. A PDF fingerprint
   can help when needed, but is not a general publication gate. A
   bibliographic record is insufficient for a security claim.
2. **Reference gate:** implement one bounded method independently in the
   existing Rust/Python stack; test against exact/independent numeric and
   adversarial controls, account for setup and retained material. Upstream
   artifacts are oracles/provenance, never linked dependencies.
3. **Composition gate:** register a typed component only after its input,
    output, trust, state, numeric, one-use and resource contracts are enforced.
    Bind its serialized selection, lineage and implementation digest to a
    `Pipeline` and fail closed on missing operators or incompatible roles. A
    narrow native reference alone stays a pending public component, not a
    default or a search candidate.
4. **Execution gate:** demonstrate the complete required path, including
   preparation, transport, recurrent state and error/cancellation burns. A
   scalar primitive or semantic plan is not whole-decoder support.
5. **Evidence gate:** an inference-applicable method becomes an explicit,
    optional `Pipeline`/`Experiment` choice only with complete operator and
    runtime coverage. Check parity and generation quality on an exact locked
    model, then benchmark against an eligible baseline with the same body,
    tokens, output cap, warm state, environment and privacy/assurance contract.
    Measure latency, TTFT, throughput, CPU, memory, disk, protocol bytes,
    offline material, quality and failures; price/energy and external SOTA
    remain unmeasured until comparable cohorts exist. Search varies compatible
    options, never paper names; promotion to a default requires review and
    evidence. Attack and assurance controls stay independent research checks,
    not fake inference backends.

LogRow currently stops at a bounded one-element Rust/compiler reference: its
`pllm.protocols.LogRowGarbledLookup` symbol is *not* an `Experiment` option.
Its 2,144-byte material count alone cannot prove an inference speedup. Next
gate is a protected tensor/decoder execution contract before matched benchmarks
or candidate admission.

## Useful first slices

1. Prototype **one correlation source** in a domain actually consumed by a
   prepared stage, with seed expansion and one-use accounting; do not swap
   finite-field material into the integer-ring baseline.
2. Compare **one delegated linear/verification method** from EMVP or Maverick
   against the bounded native Freivalds stage, including no-wrap fidelity,
   challenge distribution, failure handling and full offline work.
3. Prototype **one bounded FuseFSS nonlinear operator** in its required
   multi-party topology. Test fresh keys, online interactions and numeric
   conversions before any whole-model benchmark.
4. Run **Breaking Euston** and **Game of Arrows** as attack-aware controls on
   proposed masks or model partitioning; they are not execution backends.

These slices create tested reusable contracts, then executable option pages and
examples under `/sdk/components/` and `/sdk/inference/`. The 84 paper-linked
symbols are already installed as importable, fail-closed declarations; their
presence does not mean that any corresponding method can run or join search.
