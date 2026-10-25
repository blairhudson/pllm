# CacheMir source → capability integration contract

**Specification only; user62077 C-NP approval.** Select base
`meta-llama/Meta-Llama-3-8B` and the WikiText-2 raw-v1 calibration protocol in
[numeric packet §3](cachemir-numeric-state-packet.md#3-numerical-attachment-proposed-freeze-with-blocking-empty-fields).
Preserve **standard Goldschmidt**. Approval does not select coefficients,
certified domains, numerical tolerances, backend, encrypted execution, disclosure,
or a decoder conversion. This is one integration contract, not another research
campaign or an adopted CacheMir profile. Original owner: Main; review handoff:
Astra `ses_e8aacf8bbffewvpwIOJbZIvWAk`. No commit before review.

Repository inspection basis: `2702f9810af29a2647bce2136aee063242087f6a`;
concurrent uncommitted implementation work is not evidence for this packet.
`MISSING` means required evidence not recovered within this assignment, never
that an artifact does not exist. Every missing join rejects before construction;
unknown numerical error, cost or owner capacity is never zero.

## 1. Immutable identities, conflicts and missing fields

Fresh retrieval on 26 October 2026 used one sequential worker, **5 public
text/metadata requests, 23,784 response bytes**, against limits 25 requests,
2,097,152 response bytes and 1,048,576 report bytes. Redirects and retries disabled;
no credentials, tensor/shard/index bodies, tokenizer vocabulary, dataset rows,
archives or executable source downloaded. Simple request/status/byte counts and
selected metadata are in ignored `target/cachemir-source-capability-join.log`.
Prior audits below are inherited evidence, not additional requests or fresh
verification. No source weights, trained indices, model or calibration read/run.

| Identity / evidence | Pin or observed conflict | Missing field / refusal |
| --- | --- | --- |
| Governing paper | arXiv `2602.11470v1`; PDF SHA-256 `67db77545b62a4ec0f6906706b947492e6d1bcd9a37b84e5afc32445b41c98e7`; Appendix E p.16 chooses standard Goldschmidt and newly profiled larger-model ranges | Evaluated all-node manifest and checkpoint linkage **MISSING**; Table 6 RMSE and Table 8 examples are not certificates. |
| Original Zenodo releases | `22844188` / `23189548` / `23377151`, separate immutable records; archive hashes and operator locations in [prior audit](../research/paper-module-landing-map-2026-09-23.md#cachemir-source-recovery-and-implementation-decision-sheet) and [numeric packet §1](cachemir-numeric-state-packet.md#1-governing-evidence-and-the-conflict-this-proposal-preserves) | All three use adaptive reciprocal, not standard Goldschmidt. Fixed THOR ranges, generated normalization/dummy weights and hardcoded refresh targets are not trained-model provenance. Do not substitute these numerics. |
| First archive Git identity | HEAD `0d5695869730427c1a5dc4320d3bd375e727d2b5`; tree `f646b8e4197d5babbc292b2320dcf4206f5b9388`; annotated tag target `dcbb3645b15f8ffde3d08ab5a1fa09dd53386779` differs | A tag does not identify archived HEAD. Latest archive commit **MISSING**; immutable archive hash remains its source identity. |
| Separate sandbox | `JerryFishY/rotation-sandbox@7c953a13e12e0646ef935eae22477c833f21d2d5` | Different tree, dummy `0.01` weights/KV, disabled operations; no transfer of Zenodo path/line identities or trained-driver claim. |
| Model repository metadata | [public API](https://huggingface.co/api/models/meta-llama/Meta-Llama-3-8B): revision `8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920`, manually gated, base model ID as above | Repository revision is a **candidate source pin**, not authors' evaluated run, tensor authentication, configuration verification or execution authority. |
| Model configuration | [pinned config descriptor](https://huggingface.co/meta-llama/Meta-Llama-3-8B/resolve/8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920/config.json), HTTP 401, 134 bytes | Configuration bytes/hash and exact epsilon, activation, RoPE, Q/KV geometry, precision, head tying and source-module mapping **MISSING**. No authenticated retry or mirror inference. |
| Tokenizer descriptors | Same model revision: `tokenizer_config.json` and `special_tokens_map.json`, each HTTP 401, 134 bytes | Tokenizer implementation/version and complete byte identity, BOS/EOS IDs, special insertion semantics and tokenization identity **MISSING**. No tokenizer vocabulary fetched. |
| Corpus repository metadata | [public API](https://huggingface.co/api/datasets/Salesforce/wikitext): `Salesforce/wikitext@b08601e04326c79dfdd32d625aee71d232d685c3`; `wikitext-2-raw-v1`; card declares `text:string`, train 36,718 rows, validation 3,760, test 4,358; repository card lists CC-BY-SA-3.0/GFDL | Candidate revision/config recovered, not split-content identity. Exact split bytes/order/hashes, loader/version and applicable data-use review **MISSING**. Card counts/licenses are metadata, not verification of corpus contents or permission inference. |
| Trained parameters and evaluated driver | No tensor/index body acquired or read | Tensor/shard identities, source tensor-to-module bijection, checkpoint precision, evaluated driver and calibration observations **MISSING**. Outside this acquisition grant. |

Successful API response identities: model metadata SHA-256
`b06980289916a84df24b4882fe2b693fec447c80457dc2f0b26c36f82f8c54c4`
(18,372 bytes); dataset metadata SHA-256
`09d581714bb298afc9b55909c45bd84f721a32722fb465e2e04b54fe139f806f`
(5,010 bytes). These bind observations only, not numeric or publisher authority.
Three denied descriptor responses account for remaining 402 bytes.

## 2. One join, with no new IR or issuer

The following are **document record requirements**, not new Rust types, registered
profiles or instantiated capabilities. A record may become selectable only after
its original evidence and every dependency are complete:

```text
SOURCE = (paper pin, release/archive-or-commit pin, model revision,
          verified config/tokenizer/tensor-map identities, original source owner)
OP = (SOURCE, module path, semantic operation, layer, phase, public position,
      ordered operand IDs, source shape/dtype, source semantics)
MAP = (OP, physical layout/version, Q→KV mapping, permutation/masks,
       active extent, slot addresses, conversion capabilities)
NUM = (OP, MAP, calibration identity, admitted domain/certificate,
       approximation/evaluation bytes, CKKS/error recurrence and units)
ALT = (OP, MAP, NUM, ordered PublicCiphertextState inputs, exact output,
       PublicCiphertextCapability, original kernel evidence, cost/owner quotes)
ADMIT = (original source borrow, original graph borrow, OwnedBudget,
         construction quote, cumulative metadata/control ledgers,
         finite DAG/trace limits, original backend/source preparation grant)
```

`OP` paths below are a **required inventory to verify** against pinned original
model source; no configuration or tensor metadata recovered here establishes
these paths. Include every layer instance, prefill/decode phase and active
geometry; an exemplar layer cannot stand in for remaining layers. Bind weights
to the original immutable tensor owner when a later acquisition is authorized;
neither an operator name nor a digest creates that owner.

Reuse [`original_source::OriginalSourcePlan`](../../crates/pllm-compiler/src/original_source.rs)
for actual registered config-factory provenance, retaining original input owners
and cumulative construction-work ledger. It does **not** authenticate checkpoint
weights, tokenization, publisher, runtime or feedback. Reuse
[`PublicCiphertextGraph`/`PublicCiphertextCapability`](../../crates/pllm-compiler/src/encrypted_level_schedule.rs),
[`construction` quotes and ledgers](../../crates/pllm-compiler/src/encrypted_trace_construction.rs),
and [`OwnedBudget`](../../crates/pllm-garble/src/shared_allocation.rs).
Do not mint authority by reparsing exported graph JSON, cloning records or
matching metadata identities. No CacheMir-specific parallel source/admission IR.

## 3. Physical slot and GQA join

Reuse [`interleaved_kv_layout`](../../crates/pllm-core/src/interleaved_kv_layout.rs)
exactly. For its **equal-head** geometry, let `H=heads`, `p=head_dim`,
`d=H*p`, `t=S/d`, `G=S/H`, token `n`, head `h`, feature `f`:

```text
embed(x,h) = (floor(x/t)*H+h)*t + (x mod t)
K: ct=floor(n/t), slot=embed(f*t+(n mod t),h)
v=n mod G; c=(floor(v/t)+p-f) mod p
V: ct=floor(n/G)*p+c, slot=embed(f*t+(v mod t),h)
live_K=ceil(active/t); live_V=p*ceil(active/G)
```

This is specification transcription, not a second implementation/oracle. Keep
distinct K/V validity masks and original backing; K tiles are not V groups.
`S`, `H`, `p` must satisfy existing power-of-two/divisibility/capacity guards.
The integer oracle supplies no CKKS parameters or real placement kernels.

| Source geometry | Legal integration requirement | Present refusal |
| --- | --- | --- |
| GPT-2's 12 attention heads | Existing power-of-two `H` excludes 12. A padded/non-power-of-two layout needs its own explicit source-preserving mapping and charged transforms under shared contracts. | Do not use GPT-2 toy geometry as evidence for Llama, round 12 to 16, or silently change source width. No such extension selected. |
| Llama-3-8B GQA | Expected family geometry is 32 Q heads, 8 KV heads, `p=128`, width4096; these values remain **unverified for retrieved revision**. If verified, ratio4 grouping requires an explicit source-proved head permutation and `q→kv` map, not equal-head relabeling. | Config/source mapping **MISSING**; current oracle has one `H`, not `Hq,Hkv`. Raw 8-head KV cannot directly join 32-head Q. |
| Candidate explicit expansion | Only if source proves contiguous grouping, candidate `kv(q)=floor(q/4)`. Expand each original K/V head into four immutable query-head lanes before using `H=32,p=128`; charge real copy/rotation/mask kernels, errors, expanded backing and original-KV liveness. This is a possible alternative, **not approved or proven**. | Expansion equivalence, physical transforms, repeated-head sharing/ownership and measured costs **MISSING**. Metadata alias is not physical replication; cache/output sharing cannot be assumed. |
| Candidate native GQA | Extend shared layout and kernels with separate query/KV axes and explicit broadcast/gather/reduction maps; retain original KV owners. | No selectable shared GQA mapping/kernel exists in this evidence. Do not add paper-local packing or select extension here. |
| Artifact BSGS V / rectangular projection / RoPE | Bind actual artifact/model address permutations, rectangular VMM axes and RoPE pairing/frequency/position to explicit transformations into/out of shared oracle. | Semantic equivalence, source pins and typed physical kernels **MISSING**; THOR diagonal frames or logical reshape cannot substitute. |

Even conditionally using `S=32768,H=32,p=128` gives `t=8,G=1024`, not
`t=S/p=256`. This arithmetic is not slot/parameter adoption or a capacity quote.
All active-group score reductions must preserve GQA head identity; V outputs
must return to source Q-head order before `o_proj`.

## 4. Concrete operator dependencies and exact missing evidence

Each row is an edge group in one original model block, not an independent
registration milestone. `L=model.layers.<layer>`; path existence and exact
semantics remain to be verified. Recovered artifact locations refer only to
`Cachemir_ae/` in immutable v0.1.2-ae, as recorded in prior audit.

| Edge / ordered source dependency | Required NUM and legal typed alternatives | Exact missing evidence before this edge can execute |
| --- | --- | --- |
| E0 original source/input → block input `x` | Actual admitted source construction and input ciphertext state; no private embedding invented | Verified config/operator/tensor map; input producer custody, original source owners, input domain, precision, scheme/key/level/scale/error and capacities. Private embedding/decoder conversion remains outside contract. |
| E1 `x` → `L.input_layernorm` → `z` | Source RMSNorm: square → reduction → source epsilon → rsqrt → multiply with source norm weight. Ordered fork retains `x`; reduction/broadcast must have explicit layouts. Alternative rsqrt schedules require separate certified records. | Source epsilon/axes/weight; reduction order; rsqrt coefficients/count/domain/error; real kernels and owners. Artifact `normThor:920–957,1068–1093` is centered normalization, not evidence for RMSNorm. |
| E2 `z` + `q_proj/k_proj/v_proj` weights → `q,k,v` | Source rectangular projections and dtype/accumulation; exact packed input/output states and conversion capabilities | Tensor identities/dimensions/bias semantics/precision; rectangular slot maps, actual projection/placement kernels, approximation/encoding/error bounds, cost and peak quotes. Dummy `util.go:521–541` is not weight provenance. |
| E3 `q,k` + position → RoPE `qr,kr`; `kr,v` → staged KV | Verified pairing, frequency, position and precision; then independently typed K/V placement, legal alignment, fresh paired outputs | Pinned RoPE semantics and physical rotations/masks; GQA map; source-prefill/new-token producer; retained level/scale/error targets and native staged owner/commit protocol. Artifact `linear.go:237–278` fixed-position K-first mutation is not transactional append. |
| E4 `qr` + all retained/new K → scores | Query/KV head map, active masks, score scaling `1/sqrt(p)` as source semantics; immutable aligned reads or charged copies; no parent mutation | Exact QK kernel/address/permutation proof and reduction order, score domain/error/cost/owners. Artifact `QK_T:289–392` alone does not establish trained source mapping. |
| E5 scores → fitted exponentials → one denominator per Q head | Complete exp stage coefficients/scaling/order and positive-domain certificate; mask inactive **exponentials** before all-active-group reduction | Per-node calibration and certified domain including upstream CKKS error; complete polynomial bytes/evaluation graph, positivity and reduction errors. Artifact `SoftmaxThor:1125–1206` includes `1/8` prescaling/fixed fits, not a selected paper-aligned record. |
| E6 denominator → reciprocal; reciprocal + exponentials → probabilities | Standard Goldschmidt fork `r_next=r*(1+e)`, `e_next=e*e`, old values, fixed public count; exact aligned/scalar-broadcast capabilities after explicit alignment | Certified `0<a<=denom<=b`, selected `m,tau`, encoded `1/b`, full per-step CKKS/refresh error proof and domain, states/owners/costs. Adaptive `Inv:620–806` cannot fill any standard-Goldschmidt field. |
| E7 probabilities + all retained/new V → context → `L.self_attn.o_proj` → attention residual `x+a` | All-active-group weighted V, source head-order restore, rectangular output projection, explicit state-aligning transforms before residual `Aligned` join | Distributed-V/BSGS/GQA equivalence and real weighted-V kernel (`AttnV:404–496` is only artifact lead); output weights, source order, propagated domain/error, residual join states and live-`x` owner overlap. |
| E8 residual → `L.post_attention_layernorm`; `gate_proj/up_proj` → SiLU(gate) × up | Source RMSNorm followed by separately certified SiLU and product; explicit shared-input fork and aligned joins | Source norm/MLP dimensions/weights; SiLU coefficients/stages/evaluation and domains including norm/projection error. Artifact `nonlinear_moai.go:13–35` single Chebyshev fit is not approved composite/per-layer calibration. |
| E9 product → `L.mlp.down_proj` → second residual → next block | Exact source projection plus charged level/scale alignment retaining original residual; output error certificate covers downstream reuse | Down-projection tensor/kernel/map, full branch error/cost/owner join, next-block admitted domain and repeated decode horizon. Hardcoded `DecoderThor:49–78` refresh 9/7/13/7 is not planner input or numerical authority. |
| E10 final blocks → final norm/head/token loop | Final norm needs its own record; stop before unresolved private head/output boundary | Actual private head, tie/error policy, selection/EOS, feedback/re-embedding, allowed disclosure and decoder conversion contracts **MISSING and not invented**. No client logits/decrypt/argmax fallback. |

Each edge additionally requires original backend preparation, real input/output
capacity/alignment, full temporary/recovery/serialization/retirement overlap,
public metadata disclosure review and finite cumulative work admission. Table
entries describe distinct missing evidence; satisfying one row never waives
these common prerequisites or another edge.

## 5. Numerical record → typed alternatives → original admission

Use [numeric packet §3](cachemir-numeric-state-packet.md#3-numerical-attachment-proposed-freeze-with-blocking-empty-fields)
for standard-Goldschmidt equations and certificate obligations. Every `NUM`
contains exact operator/MAP identity; calibrated observations/protocol digest;
independently certified admitted domain including preceding approximation and
CKKS error; target/scaling/shift; complete coefficient bytes and stage order;
reduction/rescale/refresh graph; selected tolerance; propagated errors and units;
and public disclosure authorization. Observed extrema are not certificates.
Normalization rsqrt, exponential and SiLU require independent records; a valid
reciprocal record completes none of them. Proposed Horner/binary64 convention in
older packet remains a proposal, not coefficient/evaluation adoption here.

For each candidate `ALT`, populate existing capability fields: exact ordered
input states and output state (`scheme,key_domain,layout,scale,level,shape,error`),
`kernel_identity`, primitive, join, execution, error assumption,
`scratch_bytes`, `recovery_overlap_bytes`; supply explicit cost identity/unit and
full owners. `Unary`, `Aligned` and `ScalarBroadcast` are only their existing
typed contracts, not arbitrary shape/GQA joins. Choose no `Hybrid` alternative
without an independently approved conversion/disclosure contract. A `View`
cannot change physical layout, level, scale, key or backing capacity. Any real
conversion/refresh needs its own legal kernel and new charged owner.

The existing [encrypted-level DAG](../../crates/pllm-compiler/src/encrypted_level_dag.rs)
searches **supplied fixed SSA states/capabilities**, preserving ordered parents,
fork/join liveness and stable ties. Different output levels require explicitly
supplied legal graph/state variants and joins; search does not synthesize them.
This is not CacheMir's paper hierarchical/residual optimizer. Do not flatten
Goldschmidt/residual/GQA forks through chain-only trace bridge. Missing-cost or
missing-error alternatives are rejected, not assigned zero; capacity/search
exhaustion rejects a claimed optimum, not merely a losing path.

### Existing append controls stay authoritative

Reuse [`interleaved_cache_append::analyze_append`](../../crates/pllm-compiler/src/interleaved_cache_append.rs)
at `22bbf91fe9ae0736a7be11d335c7a2bf7e91de59`, with cumulative-control boundary
coverage from `2ace0da2ecd17fe5f6c1eaba96871ef58645b4a4`. It borrows ORIGINAL
graph and complete ordered old K/V parent arrays; packed incoming replacements
must already be original graph inputs. Their packing is an external synthetic
assumption, not proof of E2/E3 or encrypted prefill.

Preserve actual sequence: predebit control visits; prepay full finite search
work on **same** `MetadataLedger`; admit final trace; bound metadata and preflight
geometry/parents/capabilities; charge complete original/clone/staged/search/trace
overlap on original `OwnedBudget` before emit/search; validate constructed graph
and final trace through borrowed construction quote and same `ControlLedger`.
No new ledger per layer/append/failed candidate, no reset/refund, detached graph
promotion, mutable parent or partial result publication. Original source
construction work, metadata/DAG work and control visits remain distinct
cumulative obligations; no scalar substitutes for all three.

Retain all old K/V owners through reads and staged replacements, including
untouched shared backing. Capacity/alignment come from originals, not visible
shape/caller byte counts. Cancellation burns pending work; committed parent
remains unchanged. Real retirement requires observed release; modeled logical
drop cannot refund uncertain native backing. Existing paired-commit proposal
does not authorize native state execution. Existing 19 append cases were **not
rerun**; these synthetic controls are not new method progress.

### Separate common Secure HE Source dependency

Consume only common Source owner's future interface: fallible, purpose-/epoch-
typed preparation result tied to original key/context owners, complete finite
sampling/work/metadata/capacity admission, typed exhaustion/burn and required
erasure/release evidence. Failure propagates before evaluation, without biased
fallback or replacing original owner with metadata. Do not implement RNG, inspect
private source material or create a CacheMir preparation owner here.
[D53](../../paper/methods/DECISIONS.md#d53-bounded-unbiased-rejection-sampler-design)
approves bounded unbiased-rejection **design only**, not implementation,
sampling/key generation, backend adoption or readiness. Consume interface when
its independent owner qualifies it; no sampler/version claim closes this gate.

## 6. Calibration attachment and earliest executable original region

Protocol selected by user62077, **not executed**: pinned WikiText-2 raw-v1 train
rows in split order, strict UTF-8 joined by two LF, no shuffle/template/Unicode
normalization; tokenization with special insertion disabled; first 128 disjoint
complete 511-ID segments, prepend verified model BOS to obtain 512 IDs each;
no inserted EOS, reject fewer than 128, discard incomplete final segment. Freeze
tokenizer/loader versions, row/segment offsets, token stream/input digests and
source operator inventory. Validation selects candidates after frozen fitting
records; test stays held out. Never report this independent completion as authors'
evaluated calibration. No rows/token stream/activations acquired here.

Observe every semantic-node input/output for all layers/phases/active geometries;
training windows alone do not certify arbitrary later decode/cache history.
Calibration records stay source-side; ranges, coefficients/errors/costs and raw
activations are not automatically public. Independently justify admitted domains
and upstream error propagation; do not inspect private denominators or clamp
inputs to make records fit. Explicit finite acquisition/calibration/metadata/work
quotes and source-data authorization are prerequisites to any later run, not
permission supplied by this protocol approval.

**Earliest candidate ORIGINAL connected region:** E1→E2→E3, block0
source `input_layernorm` → original Q/K/V projections → physical Q/K RoPE and
fresh paired K/V staging, from an independently authorized original encrypted
block input. Stop before E4 attention and before any output/head/feedback.
This is not an embedding bypass or selectable decoder conversion: obtaining
that input is independently blocked by E0. The region is **not executable now**:
E0 ownership, verified model configuration/tensor map, E1 numerical certificate,
E2 real projection/placement, E3 physical RoPE/GQA/staging, common Source and full
original owner/admission evidence are missing. If starting with E0 instead, even
the source-bound original config construction cannot yet be claimed for this
pin because config bytes were denied. An isolated supplied/synthetic append or
rational reciprocal is not an ORIGINAL trained-model region.

**Next bounded native/Python command: NONE.** No complete selectable
`SOURCE→OP→MAP→NUM→ALT→ADMIT` chain exists. Do not invent a command, coefficients,
backend fixture or an admission estimate to create one. Only after complete
contracts and separate implementation/execution approval may Main derive one
ordinary bounded command from selected region and real preconstruction quote;
no bespoke attestation driver or repeat of 19 append cases.

## 7. Review decisions remaining; freeze boundary

| User decision still required | Evidence still required regardless of approval |
| --- | --- |
| Whether to authorize later gated config/tokenizer and separately scoped tensor/index/data acquisition; accept/revise candidate model/corpus pins after review | Exact authorized bytes/hashes, source operator semantics and tensor mapping; metadata pin alone cannot supply them. Current grant does not permit these acquisitions. |
| Explicit GQA strategy and any shared layout extension; no implicit 32↔8 or GPT-2 padding | Source-proved head/permutation/slot equivalence, actual transforms and measured/certified owners/errors/costs. |
| Numerical coefficients/evaluation schedule, certified-domain/tolerance policy and retained-state targets/horizon, separately reviewed | Actual all-node records, independent domain/error certificates including refresh and decode reuse; approving numbers without proof does not qualify them. |
| Separate calibration, backend/native implementation and execution grants with finite original-owner/work/metadata/control quotes | Common Source readiness, real source-bound kernels, parameter/key/security manifests, capacity/alignment/cleanup and independent correctness evidence. |
| Any private head, selection/tie/error, feedback/EOS/re-embedding, disclosure or decoder conversion completion | Complete independent contract and evidence; this packet specifies none and grants none. |

Freeze only this new document and [numeric packet](cachemir-numeric-state-packet.md).
No DECISIONS/common library/runtime/source/index edits, commit, builds, encryption,
keys/RNG, installs, Orca/new trees, calibration or qualification run. Local link
existence/fragment and whitespace checks are document checks only, not method
qualification. No policy/adoption claim beyond user62077 specification approval.
