# NEXUS–Qwen generation extension: decoder binding contract

**Astra 67589; design only, not executable/adopted.** Batch 67562–67563 authorizes a separately labelled generation extension and source-supported parameter derivation, not new tolerances, disclosure, default/backend selection or method implementation. Prepared against HEAD `23d691c6f0d353d988bcb405305bf4a9a1674479`; repository and docs AGENTS inspected. Parent independent Astra review must precede any method implementation. This document activates no API, capability or runtime through metadata and proves neither encrypted execution nor whole-trained-model correctness.

## 1. Sources, scope and reusable owners

- **N**: final NDSS 2025 paper, DOI `10.14722/ndss.2025.230868`, [`papers/r21-nexus.pdf`](../../papers/r21-nexus.pdf), SHA-256 `c809182345f27f76e21d983d97dffa0d086dbcd17dabc265f0df9fb6ca9b70f2`; lock [`source-lock.v1.json:71–78`](../../paper/methods/source-lock.v1.json#L71). Citations below use printed pages. N §II.B p.3, §III pp.4–7, §IV pp.7–9, §V p.9, Table IV p.12, Appendices B–D pp.16–17 govern source facts.
- **A**: pre-publication artifact `zju-abclab/NEXUS@e15058c7f8f58fcf037c4709211109cb1be7d1e0`, not final-paper numeric identity. Recovered function/line anchors are recorded in [`nexus-numeric-contract-audit.md:49–150`](../research/nexus-numeric-contract-audit.md#L49); no network/source execution performed here. Artifact formulas below are explicitly recovered-artifact alternatives, not silently adopted N rules.
- **M**: current [`crates/pllm-models/src/lib.rs`](../../crates/pllm-models/src/lib.rs): `ModelOperator`/`ModelOperation`/`StateTensor`/`DecoderGraph`/`DecoderPlan`; `lower_dense_decoder`, `dense_rope_attributes`, `full_kv_cache_update`, `full_kv_cache_view`, `linear` and `validate_dense_qwen_graph_tail`. Current Q/K graph edges are at 1817–1937, RoPE attrs at 2133–2149 and cache helpers at 2152–2245. These are graph semantics and structural checks, not authenticated checkpoint tensors or HE kernels. The historical preparation HEAD above is not a claim that these current anchors were present at that HEAD.
- **O**: [`original_source.rs:1–17,81–225`](../../crates/pllm-compiler/src/original_source.rs#L1), `OriginalSourcePlan`: immutable actual factory result, original root custody, cumulative nonrefundable construction work and sticky failure; no adoption of an arbitrary plan/hash.
- **R**: [`row_fold_reference.rs:85–189`](../../crates/pllm-compiler/src/row_fold_reference.rs#L85), `CENTERED_ENERGY_REFERENCE_POLICY`; **C**: [`commutable_projection_reference.rs:1–20,158–233`](../../crates/pllm-core/src/commutable_projection_reference.rs#L158), `add_client_layer`; **D**: [`encrypted_level_dag.rs:1–24,54–124`](../../crates/pllm-compiler/src/encrypted_level_dag.rs#L54), `PublicEncryptedDagSelection` and checked DAG.
- Existing governing [`NEXUS contract §§4.1–4.5`](../../paper/methods/encrypted-computation-contracts.md#4-nexus-contract), [source audit](../research/nexus-numeric-contract-audit.md), [N-G proposal](nexus-gelu-acceptance.md) and [original normalization track](nexus-centered-energy-prefix.md) remain unchanged. Preserve original GELU/LayerNorm BERT track and reported one-word Llama extension; table labels do not establish Qwen RMSNorm/SiLU/RoPE/GQA fidelity. N-G candidates are **not adopted**, and none implements Qwen SiLU.

Implementation shape: consume validated existing operations and their ordered input edges; reuse shared model-/method-neutral row reduction, ring/compression, source ownership, transport/admission and typed SSA scheduling. Do not build a second hard-coded Qwen-family graph. Unsupported operators/attrs/state/profile combinations refuse; do not rewrite them to available primitives with different semantics.

## 2. Exact functional contract, separate from representation

Notation: batch `B`, query length `T`, hidden width `H`, intermediate width `I`, query heads `Nq`, KV heads `Nk`, head width `d`, capacity `C`, visible key-domain bound `K`, vocabulary `V`, `g=Nq/Nk`. Read dimensions from actual graph/config, not BERT constants; require divisibility and exact output shapes. Logical weights below use source tensor orientation (`x W^T` for stored output-by-input weights), with declared bias only. Arithmetic formulas specify ideal functions; model-native evaluation/rounding and HE approximation require separate contracts (§3).

Every binding retains `id`, `operator`, `layer`, `state_kind`, ordered `inputs`, exact `attributes`, `output_shape`, source/config and weight owners. Attribute strings/digests are not tensor authenticity or permission to load weights. Whole model includes both graphs, `state_inputs`, `state_outputs`, `graph.output` and `plan.token_feedback`; copying an operation list alone is insufficient.

Retain each graph's `mode`, `batch`, `query_sequence` and `maximum_key_sequence`; each input/output `StateTensor` retains `id`, `layer`, `kind`, `shape` and `maximum_sequence`. Retain complete `DecoderPlan` identity: `schema_version`, `model_family`, `adapter`, `config_digest`, ordered `transformations`, `prefill`, `decode` and `token_feedback`. Each `AppliedMethod` retains `component`, `implementation`, `method_id`, `input_digest` and `configuration_digest`. These fields must agree with the original plan and bound graph/state semantics; family labels or equal digests alone confer no authority.

| Actual M operation / binding | Required exact input → output semantics | N source support / explicit extension gap |
| --- | --- | --- |
| `TokenLookup`, `token_lookup`; `input.tokens`; `weight=model.embed_tokens.weight` | Token IDs `[B,T]` in `[0,V)` select embedding rows → `[B,T,H]`. Generated selections must re-enter same table/token namespace. | N §II.B starts with embeddings; private ID lookup and model-private embedding custody are not specified. No public-table substitution. |
| `RmsNorm`, `layer.l.input_norm`, `post_norm`, `final_norm`; optional `q_norm`,`k_norm` | For each last-axis row width `n`, `v=(sum x_j²)/n + epsilon`, `y_j=(weight_j+weight_offset)*x_j/sqrt(v)`; dense attrs require exact source `epsilon` string and `weight_offset=0`. Preserve weight names: input/post norm, `model.norm.weight`, optional `self_attn.q_norm.weight`/`k_norm.weight`. Q/K norms occur after head reshape, before RoPE, width `d`, not `H`. | **Extension.** N Eq.(5) is centered LayerNorm, not RMSNorm. No centering, beta, removed epsilon or implicit unit weights allowed here. Inverse-square-root qualification missing (§3). |
| `Linear`, `q_linear/k_linear/v_linear` from `input_norm` | Declared `self_attn.{q,k,v}_proj.weight` and optional `.bias`: Q width `Nq*d`, K/V width `Nk*d`; output sequence preserved. | N §III.B–C gives packing/linear algebra, not complete encrypted-hidden-layer preprocessing composition. Bind bias null versus actual tensor; do not add/remove bias. |
| `Reshape`, `q_heads/k_heads/v_heads` | Logical index map `[B,T,heads*d] → [B,heads,T,d]` under `layout=batch_heads_sequence_feature`; later `attention_hidden` inverts head concatenation to `[B,T,Nq*d]` (`batch_sequence_hidden`). | Logical reshape is not proof of free HE view: require explicit slot/coefficient permutation or checked alias. |
| `RotaryEmbedding`, `rope_q/rope_k`; second input `input.positions` | Apply declared theta, rotary dimensions, split-half pairing and sequential absolute positions to Q/K only; tail unchanged. For unscaled dimensions `r`, `omega_j=theta^(-2j/r)`, `(x_j,x_(j+r/2)) → (x_j cos(p omega_j)-x_(j+r/2) sin(p omega_j), x_(j+r/2) cos(p omega_j)+x_j sin(p omega_j))`. Preserve actual `coefficient_profile` and any `frequency_scaling`, not formula substitution. | **Extension.** N has no pinned RoPE implementation. M default `pllm.numeric.rope.q30.libm.v1` is coefficient identity, not automatically adopted CKKS encoding. Scaled profiles need their own coefficient semantics. No position reset at decode. |
| `KvCacheAppend`, `key_append/value_append` | K input is `rope_k`, V input `v_heads`. Prefill inputs `[current,positions,attention_mask,sequence_lengths]`; decode prepends original named state. Initialize/append at absolute valid write positions, capacity `C`, layout `batch_kv_heads_sequence_feature`; output `[B,Nk,C,d]` with matching Key/Value kind. Preserve prior valid cells; no cross-request alias. | **Extension.** N Table IV establishes no persistent cache. Explicit valid-cell/length invariant and state ownership required (§4); append is not unrestricted overwrite. |
| `CacheSuffix`, `key_view/value_view` | Inputs `[append,positions,attention_mask,sequence_lengths]`; `axis=2`, `semantics=visible_valid_prefix`, bound `K`; output `[B,Nk,K,d]`, includes current valid entries. Padding/unwritten cells cannot become visible keys. | **Extension.** Name does not authorize suffix eviction or cache acceleration. Resolve validity/visibility through actual inputs, not physical buffer contents. |
| `AttentionScores` → `AttentionScale` | `[rope_q,key_view]`: `s[b,h,t,k]=sum_j q[b,h,t,j]*key[b,floor(h/g),k,j]`; declared `group_size=g`. Scale by `1/sqrt(head_dim)` → `[B,Nq,T,K]`. | N §II.B and Appendix C/Alg.5 support attention dot products; rectangular decode/GQA head sharing and scale precision need explicit mapping. Never expand/reindex KV heads incorrectly. |
| `CausalMask` → `Softmax(axis=-1)` | Inputs `[scaled,input.positions]`; combine causal `key_absolute_position <= query_absolute_position` with valid-prefix/padding invariants from cache inputs. For visible set `J`, `p_k=exp(s_k)/sum_(j in J) exp(s_j)` for `k in J`, otherwise exactly zero in ideal semantics. Empty visible rows require explicit refusal/semantic policy. | **Extension** for causal/padded decode. N §IV.B Eq.(4) supplies softmax target/approximation, not these masks. Never feed symbolic `-infinity` to finite polynomial; finite sentinel is a new error/domain decision, not equivalent masking. |
| `AttentionValues`; `[probabilities,value_view]` | `o[b,h,t,j]=sum_(k in J) p[b,h,t,k]*value[b,floor(h/g),k,j]` → `[B,Nq,T,d]`; same `group_size` as scores. | N attention formula; complete probability-diagonal/column/value-layout conversion remains missing. Attention-value multiplication is not omitted. |
| `Linear o_proj` → `ResidualAdd attention_residual` | Reshaped attention → declared `self_attn.o_proj.weight`, null bias → `[B,T,H]`; add **pre-input-norm residual**, not normalized input. | N linear/add algebra; maintain both live branches and scale/key compatibility. |
| `post_norm` → `gate_proj/up_proj` → `Silu` → `Multiply gated_multiply` → `down_proj` | `a=post_norm(attention_residual)`; `u=a W_up^T`, `z=a W_gate^T`; `silu(z)=z/(1+exp(-z))`; `m=silu(z)*u` elementwise `[B,T,I]`; `down=m W_down^T` → `[B,T,H]`. M `.mlp.{gate,up,down}_proj.weight`, null biases. | **Extension.** N Eq.(3) GELU is not SiLU; no pinned SiLU polynomial, range, error or rescale schedule. Gate/up branches cannot be merged or switched. |
| `ResidualAdd mlp_residual` → next layer / `final_norm` | Add `down` to **attention_residual**, not post-norm or original earlier residual; all layers preserved; final RMSNorm uses `model.norm.weight`. | Add algebra supported, complete pre-norm Qwen transformer remains extension. |
| `LastToken last_hidden` → `OutputHead output_head` | Prefill `[final_norm,input.sequence_lengths]`, `axis=1`, `selection=last_valid`; one-query decode `[final_norm]`. `[B,H] → [B,V]` over **entire source vocabulary**, tied `model.embed_tokens.weight` iff `tied=true`, else `lm_head.weight`; preserve `model_native` compute/output dtype attrs. | N output selection supports restricted output leakage, not exposing full logits. Full head must exist internally; no truncated vocabulary to satisfy packing. No claim ideal-real equality implies model-native bit identity. |
| `GreedyTokenSelection` → `TokenFeedback` | `[output_head] → [B]` token IDs, then `[token_selection] → [B,1]`; both `policy=pllm.greedy.v1`, `source=execution_policy`; graph output is feedback. Bind actual execution policy's deterministic tie rule; M dense tail does not encode a tie-break attr. | N Alg.2 outputs encrypted selection vector, nominally multi-hot on ties, not ID or feedback. Exact source-vector-to-policy-ID and next lookup mapping are extension obligations; approximate vector is not assumed Boolean. |

Checker obligations compare complete attributes and ordered input roots, not only the table's abbreviated formulas. For `KvCacheAppend`, compare `state`, `mode` (`initialize`/`append`), `state_capacity`, `state_layout=batch_kv_heads_sequence_feature`, `absolute_write_positions_input=input.positions`, `padding_mask_input=input.attention_mask`, `valid_lengths_input=input.sequence_lengths`, and the complete `attention_domain` object: `layout=batch_kv_heads_sequence_feature`, `maximum_sequence`, `includes_current=true`. For `CacheSuffix`, compare `axis=2`, `maximum_sequence`, `semantics=visible_valid_prefix` and those same three root-name attributes, alongside matching Key/Value `state_kind`. For prefill `LastToken`, compare `axis=1`, `selection=last_valid` and `valid_lengths_input=input.sequence_lengths` with its length input edge; one-query decode instead has `[final_norm]` and only `axis=1`. Missing, changed or extra attributes must not pass through formula equivalence; these checks do not resolve the numeric, state or security profiles below.

## 3. What parameters can actually be derived

Do not conflate ideal function, clear numerical implementation and encrypted realization. Each admitted per-operation realization needs bound input/output domains, approximation target/statistic, coefficient conversion, evaluation order, scale/level and rescale-prime schedule, key domain, layout, rounding/refresh error, and resource capability. Missing fields remain missing, not inferred tolerances. Parameters below are source facts or conditional algebra, not adopted profile defaults.

| Unit | Supplied source / derivable quantity | Exact missing values; permissible next choices (none selected) |
| --- | --- | --- |
| RMS inverse square root | N Eq.(5)/Alg.4 says Newton with suitable seed but gives neither seed nor count/domain. A `src/layer_norm.cpp:6–32` uses mean-square scale `1/768`, `invert_sqrt(...,4,2)`, no centering/affine/epsilon. A `src/ckks_evaluator.cpp:366–484,517–531` recovered seed `y=-1.29054537e-04*v+1.29054537e-01`; four `y←1.5y-0.5v y³`; set `x=v*y,h=y/2`; twice `r=0.5-x*h; x←x+x*r; h←h+h*r`; return `2h`. | Missing certified operand interval, seed contraction/error, coefficient conversion and node rescale/error. Options: recover final-paper author profile; or separately qualify recovered-artifact schedule on **RMS operand including actual epsilon**, with documented scaling. Never transplant `1/768` to arbitrary width. For normalized operand `u=lambda*v`, reconstruct `v^-1/2=sqrt(lambda)*u^-1/2`; lambda/domain are not supplied choices. |
| Reciprocal | A `src/ckks_evaluator.h:100`, `.cpp:487–508`: `e=1-u,r=1+e`, four additional `e←e²;r←r*(1+e)` gives `u*r=1-(1-u)^32`. Thus for admitted `u∈[a,b]⊂(0,2)`, `rho=max(abs(1-a),abs(1-b))<1`, exact-real relative residual ≤ `rho^32`, absolute inverse error ≤ `rho^32/a`. | Missing actual positive `[a,b]`, conversion/rescale/error and final-paper seed/count. Options: source recovery or explicit artifact-derived adaptation with certified domain. This conditional error formula is implementable algebra, not supplied Qwen error tolerance. To invert `D` via `u=lambda*D`, use `lambda*r(u)`, not `r(u)/lambda`. |
| Softmax | N p.8 Eq.(4), unnumbered exponential formula and Alg.4 p.9: `z=s-c`, `e=(1+z/256)^256` by **eight** squarings; sum visible exponents and divide. A `.cpp:534–547` instead `/128`, seven squares; `src/softmax.cpp:23–39` uses `u=0.01*sum(e)` and final factor `0.01`. | Missing public `c`, lower `z` bound, positive denominator interval, seed/count for paper, underflow and masking error. Options: recover paper values; separately qualify artifact variant, never mix identities. Given independently certified visible scores `[L,U]`, choosing `c>=U` implies `z∈[L-c,U-c]`; it does **not** supply c or certify exponential accuracy. Conditional exponent bounds `0<e_min<=e<=e_max`, visible count `m`, yield `m*e_min<=D<=m*e_max`; qualify scaled reciprocal on that interval. Polynomial shift is not exactly shift-invariant softmax. |
| SiLU | M supplies exact target `z/(1+exp(-z))` via `Silu`, then separate `Multiply`. N/A audited sources provide no SiLU schedule/domain or certificate. | Missing `[L,U]`, approximation coefficients/degree, exponential treatment for positive arguments, reciprocal scaling/count, evaluation order and total error. Options: obtain pinned SiLU source profile; or separately specify/certify exp–reciprocal composition or direct approximation. N's nonpositive exponential formula cannot be reused on all `-z` without a domain/sign derivation. Do not fabricate degree, clipping, tolerance or reuse N-G. |
| Selection | N §II.D p.4: public difference normalization, degree-nine f/g with `(dg,df)=(2,2)`, `alpha=20`; Alg.2–3 pp.7–8: `max(a,b)=(a+b+(a-b)*SGN(a-b))/2`, `b_i=SGN(a_i-max)+1`, packing `2m<S`. For logits `[L,U]`, pair differences lie in `[L-U,U-L]`, so magnitude bound `U-L` (nondegenerate case) follows. | Missing actual logit/intermediate-max bounds, near-tie separation, thresholds, tie-to-ID rule, padding and multi-ciphertext tournament layout. Normalized `2^-20` means original-unit width `Delta*2^-20`, not absolute `2^-20`. Options: recover author whole-vocabulary layout; separately derive a checked shard/tournament with original policy and total error. Exact ties need distinct semantics, not assumed separation. |
| HE realization | N §VI.A pp.9–10/p.12: ring degree `65536`, slots `32768`, modulus 1763 bits, depth `35`, bootstrap depth `14`, nominal usable `21`, `q0≈2^60`, others `≈2^50`, scale `≈2^50`, sigma `3.2`, secret weight `192`. N §V favors refresh after width reduction. | Missing exact Q/P/bootstrap chains, encoding, evaluation keys/security treatment, per-node rescale/refresh/error and layout ledger. Source rounded values are not executable parameters. Options: source recovery or separately approved qualification; no Poulpy/new backend/parameter adoption here. Table IV `20→21` matmul is unresolved, not a refresh rule. |

**Conditional softmax branch feasibility** ([source audit](../research/nexus-numeric-contract-audit.md)): for paper `E(z)=(1+z/256)^256`, requiring the positive-base, nonpositive-input branch and a uniform `e_min>0` on certified `[L-c,U-c]` requires `-256<L-c<=U-c<=0`.
Equivalently, `U<=c<L+256`, feasible iff `U-L<256`; width `256` cannot supply a uniform strictly positive lower bound under this sufficient branch assumption, since `E(-256)=0`.
Here `E'(z)=(1+z/256)^255` is negative for `z<-256`; exact wrong-branch control is `E(-768)=2^256 > E(-512)=1` despite the lower input.
This is conditional branch/domain derivation, not approximation accuracy, HE underflow, denominator-reciprocal qualification or full numeric-profile acceptance; it chooses no public shift/default and does not claim all wider intervals impossible for every different profile.
The table's missing lower bound remains unresolved; existing `ckks_softmax.evaluate` fixed `BOUND` path is unchanged, and artifact `/128`/seven-squaring identity is neither adopted nor mixed with the paper formula. Native HOLD remains unchanged.

Selection caveats: printed f/g literals recovered from Cheon–Kim–Kim are in [audit:196–224](../research/nexus-numeric-contract-audit.md#L196); A `.h:98`, `.cpp:181–211` instead inserts intermediate `0.5` factors. Four degree-nine compositions have formal degree `6561`, not `36`; N p.12 error discussion cannot certify them. Existing N-G clear studies do not supply a selection or SiLU acceptance budget. Reported average errors (`1e-5` exponential, `1e-4` GELU) and Table VI noise estimates are not uniform whole-decoder guarantees.

**N-N is not RMSNorm.** R's exact policy string is `row_head_ascending_tree_binary64_no_epsilon_reference_v1`; accepts `LayerNorm` only, shape `[32,1024]`, active width **768**, public gamma/beta owners, centered energy and `none_reject_zero_energy`. N Eq.(5) computes `z=n*x-sum(x)`, `v=sum(z²)`, `gamma*sqrt(n)*z/sqrt(v)+beta`. Width/padding and “no epsilon” belong to this clear reference only. They do not establish a checkpoint normalization or authorize any RMSNorm epsilon loss, zero-energy rule or numeric adoption.

## 4. Every epoch: full transformer, state and feedback obligations

Define source transition `(token_e, S_(e+1)) = F_M(input_e, positions_e, mask_e, lengths_e, S_e)` using all operations above, complete vocabulary and actual execution policy. Establish invariant for **every** prefill/decode epoch: same valid token history and absolute positions; matching per-layer post-RoPE K and unrotated V on all valid cells; matching validity/capacity and residual dependencies; matching complete last-valid logits, selection and feedback. Compare full transformer composition, not isolated op counts. HE implementation additionally needs accumulated error/selection-stability proof under separately accepted budgets; exact arithmetic identities alone cannot prove token equality near ties.

Persistent encrypted KV is a **separate unresolved profile**. Bind original request/model/layer/key-domain owner, `[B,Nk,C,d]` layout and valid-position map; append/read ranges, no future/padded reads, capacity exhaustion, ciphertext level/error evolution and any refresh. Charge old/new state overlap, views, packing conversions and cancellation/failure cleanup; retire keys/state/masks without cross-request reuse. Existing IR statefulness remains an obligation even if execution recomputes. No implicit fast cache, eviction, clear KV, capability or history owner.

A proposed **full-prefix fixed-count feedback profile** may recompute every prefix rather than persist encrypted KV, but is **not selected**. Must give a checked simulation relation: at every epoch reconstruct all logical `S_e` valid cells from same full prefix and prove same F_M output/state obligations, including positions and masks. Omitted physical state requires explicit checked-obligation equivalence, not deleting M state edges. Need admitted full-prefix history custody, one-use preprocessing for every repeated operation and cumulative work/owner bounds. Fixed count alone does not decide EOS, final output or token disclosure; D9 approval for other methods does not extend here.

## 5. Roles, data, freshness and undecided boundaries

N §II.A and Appendix D assume static computationally bounded semi-honest corruption of either client C or server S, privacy of both inputs/model; no malicious-result verification. Preserve both key domains. Matrix below records known contract, not assignment of extension roles.

| Data / phase | Known N owner/view and one-use requirement | Extension decision still outstanding |
| --- | --- | --- |
| Weights / publication | S owns W and server secret; C receives compressed `Enc_S(W)`, decompresses and computes encrypted UW (§III.B–C). Publication bound to model; transfer changes when model changes. | Tensor/checkpoint authentication, publication capacity/transport, tied embedding/model-private lookup custody and revocation; no plaintext W to C. |
| Mask U / offline UW | C chooses fresh U; C applies client RLWE layer to `Enc_S(UW)`; S removes **server** layer, retains `Enc_C(UW)`. Independent one-use pad per admitted input region; consume/retire on use or failure, never recycle on retry or prefix replay. | Concrete request/epoch/op/shard/key/layout binding and crash/cancel consumption protocol; who can prepare masked encrypted intermediates without reveal. |
| Online A−U | C knows source A and U in §III.C; S sees masked polynomial, combines `(A-U)W+Enc_C(UW)` under client key. | Hidden activations/feedback are already encrypted: C does not automatically know A. Choose proven non-interactive composition or separately approve interaction/protocol change; never decrypt hidden layers implicitly. |
| Noise / encoding | Appendix B: `(A,A*s_S+m+e)` → add `A*s_C+e'` → remove `A*s_S`, phase retains `m+e+e'`. C preserves supplied noise in `add_client_layer`; exact-ring diagnostics sample no keys/noise. | Secure samplers, distributions/encoding/rescaling and noise/security composition. Clear identity `(A-U)W+UW=AW` proves no encryption security. |
| Positions, lengths, padding, transcript shape | Logical M roots determine visibility and last-valid selection; N supplies no persistent-generation leakage contract. | Public versus encrypted custody; padded-capacity versus exact lengths; position/length leakage and failure/timing disclosure. Do not declare roots public merely to simplify HE. |
| Selection / result / history | N Alg.2 returns encrypted selection vector; C holds final decryption capability. N Table IV is one-word output. | Choose whole-vector final decoding vs policy-bound token-ID conversion; encrypted feedback vs approved client interaction; C/S history holder, per-token vs final-only output, fixed count vs EOS/stopping/cancellation disclosure. Full logits release changes model leakage. No choice made. |
| KV / retirement | No N persistent-state protocol; M has explicit Key/Value state input/output. | Persistent encrypted state or checked full-prefix simulation (§4), owner/location/layout, append/read, recovery/refresh and retirement. Neither profile implies decryption or a fast cache. |

Appendix D's pads are **uniform one-time polynomials in `R_Q`**, not small real masks. Need proof that chosen polynomial-to-CKKS encoding, modulus/scale evolution and dual-key transformation preserve input/model privacy and correctness with noise; bounded masks or clear cancellation algebra are insufficient. Matrix symbol A is unrelated to Appendix B's RLWE polynomial A. No clear identity proof may be promoted to model privacy, authenticated transport or admitted HE execution.

## 6. Admission, planner and concrete follow-up boundary

Retain original `OriginalSourcePlan` root and first factory plan plus configuration/raw owners throughout derived lowering; debit its **same cumulative construction-work ledger** for construction, including failed attempts and preceding work. This construction quota is not the retained original reader/controller execution ledger and cannot fund or authorize a new selector or derived execution. Keep sticky burn/nonrefundable reservations and actual backing guard coverage; a fresh caller ledger, equal digest or byte count cannot replace source custody. Full-prefix epochs, branching gate/up/residual values, persistent KV, original weight/key publication, preprocessing/masks, conversions and refresh require cumulative admission and peak live backing including old/new overlap. Logical source/planner quotes are not RSS, physical capacity, crypto work or whole-request admission.

Shared backend planner/ROSETTA work owns HE qualification. Lower semantic operations into existing typed `PublicCiphertextGraph` with explicit ordered parents, per-value owner/key/scheme/layout/level/error and declared capabilities; no shadow NEXUS planner. D's `PublicEncryptedDagSelection { operation_index, capability_index }` selects an existing capability for a **graph operation**, not a vocabulary token. Checked DAG establishes declared metadata consistency only, not ciphertext execution/security/capacity. Do not mint executable authority from this document, method labels, source hashes or checked metadata; do not adopt Poulpy or change default/backend/parameters.

**CANDIDATES ONLY, not independently scoped implementation tickets:** model-neutral exact semantic binding/checker for existing M DTO inputs/attrs/state/output; conditional reciprocal residual derivation `rho^32/a`; or original-source centered-energy/commutable **clear** algebra reuse. These candidates have explicit source formulas and require no invented numerical profile, but existing implementations must be reused, not duplicated. Astra62 holds numerical-join/checker dispatch until a concrete existing-validator join is frozen; parent review and a separate scoped implementation instruction remain required. No complete Qwen SiLU/RMSNorm/softmax/selection HE unit is fully source-supported yet. A future refusal/obligation checker must leave unknowns unresolved and must not silently become method execution.

Outstanding decision records must name values, source gap and selected option: (1) exact RMS/SiLU/softmax operand intervals, inverse scaling/seed/count and coefficient/evaluation identities from §3; (2) per-op mathematical and HE total error targets/statistics and full-model quality criteria—**no numeric tolerances proposed here**; (3) complete-vocabulary layout, selection tie/separation/decoding rule compatible with execution policy; (4) roles, history, disclosure/termination/interaction and state profile from §§4–5; (5) concrete mask/security and backend capability/chain/rescale/refresh/admission contracts. Missing values block the dependent unit; do not invent defaults, broaden domains or substitute sampled errors.

### 6.1 Current factual Q/K producer contract — no authenticated derived join

The authentic source → lookup → first-normalization prefix already exists. An authentic derived Q/K join does **not** follow from it. This bounded current-source inspection records the required producer relation, not a new implementation, numerical profile, capability or live request instance. Reading the source-factory contract is not authenticating an instantiated request.

Use `S=maximum_key_sequence` below (the visible-domain bound called `K` in §2), not cache capacity `C`:

- **Q** is `layer.l.rope_q`: `RotaryEmbedding([q_input,input.positions])` → `[B,Nq,T,d]`. `q_input` is the actual `q_norm` when configured, otherwise `q_heads` reshaped from the original `q_linear`; it is not the first norm directly. Bind `theta`, `rotary_dimensions=d`, `pairing=split_half`, `position_policy=sequential_absolute`, both layouts `batch_heads_sequence_feature`, and `tail_policy=unchanged`. Unscaled coefficient identity is `pllm.numeric.rope.q30.libm.v1`. `dense_rope_attributes` changes coefficient profile when `frequency_scaling` is present and copies the **complete object**; bind both exactly. No silent scaled/unscaled selection or numerical adoption.
- **K** consumed by attention is `layer.l.key_view`, not `rope_k` alone: `CacheSuffix([key_append,input.positions,input.attention_mask,input.sequence_lengths])` → `[B,Nk,S,d]`, `state_kind=Key`, `axis=2`, `maximum_sequence=S`, `semantics=visible_valid_prefix`. Bind the exact auxiliary-name attrs `absolute_write_positions_input=input.positions`, `padding_mask_input=input.attention_mask`, `valid_lengths_input=input.sequence_lengths` and their ordered edges.
- `key_append` consumes `[rope_k,input.positions,input.attention_mask,input.sequence_lengths]` in prefill; decode prepends the **original named Key state**. `rope_k` itself consumes configured `k_norm` or reshaped `k_heads`, plus original positions. Bind initialize/append mode, state identity/kind, `[B,Nk,C,d]`, capacity and `state_layout=batch_kv_heads_sequence_feature`; retain the **full** `attention_domain` object (`layout`, `maximum_sequence=S`, `includes_current=true`). Shape/length equality is not cache-history custody.
- **Consumer** is `AttentionScores([rope_q,key_view])` → `[B,Nq,T,S]`, `group_size=Nq/Nk`, with the exact GQA head mapping in §2. Following `AttentionScale` stores `head_dim=d`, **not** a source-supplied binary64 scale factor. Source BF16 → binary64/protected-scaling conversion is not defined by these edges. N's existing one-head square Algorithm 5 cannot automatically become rectangular GQA/cache-faithful attention.

### 6.2 Existing custody and held authority

Current private ownership is
`RawFirstNormActivation → AdmittedRawNormSpec → OriginalReservation<RawLookupActivation> → RawLookupActivation → PrivateSourceInstance → AdmittedRawInput`.
See [`raw_first_norm.rs`](../../crates/pllm-compiler/src/replicated_decoder/raw_first_norm.rs) (`RawFirstNormActivation`, `first_bf16_norm`, `validate`) and [`raw_clear_producer.rs`](../../crates/pllm-compiler/src/replicated_decoder/raw_clear_producer.rs) (`AdmittedRawNormSpec`, `AdmittedRawInput`, `token_lookup`, `reserve_original_key`). The chain retains actual gamma/lookup output, ordered tokens, reader root, selected prefill/decode graph and immutable original plan; borrowed words or matching labels cannot reconstruct it. Output/gamma/lookup payloads wipe and free before their covering charges and upstream owners retire.

`OriginalReservation` and `reserve_original_key` at `raw_clear_producer.rs:270–323` remain **TokenLookup-associated**. New funding does not authorize Linear, RoPE, cache or attention. [`raw_owner.rs`](../../crates/pllm-compiler/src/verified_source/raw_owner.rs) `original_weight`/`RawReader::weight` can resolve original weights within their existing scope; this does not authenticate a first-norm → Linear producer relation. Original construction provenance in [`original_source.rs`](../../crates/pllm-compiler/src/original_source.rs) explicitly does not certify weights, tokens, feedback, runtime or drain.

Existing protocol-specific `Invocation` ownership is not a raw Q/K issuer. For example, [`rotary_handoff.rs`](../../crates/pllm-compiler/src/replicated_decoder/rotary_handoff.rs) retains `Compiled`, `SuspendedWorker` and `Executor`; `CoordinatorDrained<Invocation>::restore` finishes the executor and restores owned outputs into the original worker. [`body_handoff.rs`](../../crates/pllm-compiler/src/replicated_decoder/body_handoff.rs) retains original runtime/source/last-use custody and requires global drain. Compiled-program identity and drained runtime `Values` authority remain specific to that protocol and restoration path; neither supplies arbitrary raw Q/K, positions, masks, lengths or cache-history authority.

### 6.3 Minimum shared successor selector — selected lifetime direction, proposed implementation

**No selector construction is approved here.** USER71554 selects the common
lifetime DESIGN direction now recorded in [Maverick's selected selector lifetime
direction](maverick-qwen-extension-contract.md#selected-selector-lifetime-direction);
both documents are updated in this tranche. That shared design, its exact ordered
sequence and five open gates govern the model-/method-neutral norm → Linear
contract; it is neither funded/proven construction nor an existing API. Do not
create a parallel NEXUS helper or treat standalone `projection_custody`/
`window_linear` as authority. The following retain requirements on proposed
implementation, not a construction ticket:

1. Consume the **whole original first-norm owner**. Authenticate requested Linear successor in its retained original graph, including the original norm input edge, ordered inputs, shape and exact weight/bias semantics. Bind requested weight, optional bias and direct sealed scale indices to that same original source inventory and backing owners. Caller indices are untrusted selection data; names, digests and token authority are insufficient.
2. Selection performs **no source DATA read, numeric execution or publication** and grants no such authority. One bounded lock-aware metadata `check_header` is explicit, not tensor/source-payload access. Retain the complete first-norm/gamma/lookup/tokens/reader/controller chain through selection retirement; never extract/re-attach a reservation or replace it with budget membership. `OriginalReservation` remains TokenLookup-associated. A later reader/executor needs its own authentic, separately scoped admission.
3. Original COLLECTION funding must cover frame/control/refusal/post-charge cleanup slots before construction; EACH authentic issuer separately prepays selector ENTRY on the original reader/controller EXECUTION `WorkLedger` (currently absent). Neither collection funding, `OriginalSourcePlan` construction quota nor a fresh caller ledger replaces that credit. Debit original cumulative work before byte reservation/semantic construction; fund all simultaneous controls and cleanup. [`raw_owner.rs`](../../crates/pllm-compiler/src/verified_source/raw_owner.rs) `admit_reader` and `admit_primitive` are existing scoped ingredients, not a new successor license or permission to invoke relocking/scanning helpers wholesale. Preserve earlier debits and sticky refusal; no ledger reset or work refund.
4. Bound original graph count `G`, source tensor count `T` (distinct from query length in §2), and complete attributes before scans. Candidate form `C0 + Cg*G + Ct*T` requires a frozen complete control/validation/cleanup census and rates; **neither census nor rates is frozen or approved**. Extra backing/file traversal and containing layouts must be covered, not hidden in those symbols. Logical work is not a physical CPU/RSS bound.
5. Follow the shared proposed `RawHost`/`NativePublisher` confinement: private non-Clone borrowed `SelectorFrame<'h>` plus `selection_live` under the SAME original-root mutex, authentic frame/activation/root and entry-credit checks before latching, genuine native owner AND actual outer lease retained through cleanup. Non-Clone is not Arc uniqueness; native aliases and multicall retirement require root exclusion beyond a Python gate. Unlock **before any relocking destructor**; guard/closure → transient root alias → partial handles → completed selection → remaining activation. `reader_live` may clear earlier, but retirement must check BOTH flags before Ready; clear `selection_live` only after scoped aliases/destructors end. New-charge refund may precede outer partials, so collection coverage continues through final aliases/control tail. Exclusive lifecycle APIs and this cleanup discipline are proposed, not compiled or approved construction; ambiguous cleanup proves no release/recovery.

USER71483's logical-atomic reservation-call choice remains narrowly accounting:
internal `fetch_update` retries/callbacks belong to one invocation, not separate
calls/peak updates/`fetch_sub`/loops/destruction; no physical retry bound or blanket
NEXUS rates follow. USER71554 separately selects the common lifetime direction,
not complete cleanup funding, frozen layout/census or implementation approval.
Even a future approved selector would establish only authentic successor
selection. The real consumer remains `AttentionScores([rope_q,key_view])`:
authentic RoPE/cache/history custody and a separately approved numeric producer
join are required, including projection, conversion/scaling and genuine positions/
mask/lengths owners from §6.1. Common norm → Linear lifetime grants NONE of that
derived-Q/K execution. Token custody alone is insufficient. No numerical profile,
cap, security/source-ownership claim, weights, route, implementation or execution
is selected here.

**Evidence boundary:** existing reported 1088/13740 regression figures are unchanged and **UNRUN** in this documentation task; no fresh regression result is claimed. Source inspection is not execution evidence. All selector construction and derived-Q/K authority remain held pending separate approval and evidence.

### 6.4 Proposed original-request ordered Q/K issuer — DESIGN ONLY

**Selected DESIGN direction only:** a private consuming transaction retains the
original request, source/controller, execution ledger, compiled graph and backing
owners. It publishes one inseparable, non-Clone pair only after both authentic
producer chains, authentic key-view construction, successful global drain and
original-worker restore. Original source and cumulative ledger are never
reconstructed or replaced. The representation candidate below remains UNSELECTED;
no issuer implementation, overlap contract, adoption, funding or execution is approved.

**Existing mechanics; not finished authority.**
[`rotary_creation`:22–108](../../crates/pllm-compiler/src/replicated_decoder/rotary_creation.rs#L22)
captures original identity and ordered projection/reshape creation, NOT completed
Q/K chains or authentic KV. An original-execution adapter binding the following
entire relation is still **MISSING**.
[`lower_dense_decoder`:1720–1842](../../crates/pllm-models/src/lib.rs#L1720)
uses the SAME layer input norm → separate Q/K projections → **head reshape** →
optional Q/K RMSNorm over head width `d` → RoPE. K then follows authentic
`key_append` → `key_view` before `AttentionScores([rope_q,key_view])`
([1843–1937](../../crates/pllm-models/src/lib.rs#L1843)). Proposed issuer must bind:

- Original ordered inputs and operation ordinals at EVERY step; original request,
  layer, epoch, selected prefill/decode graph and generation/history identity.
  Retain source weight, explicit optional bias and all gamma owners, exact shapes,
  reshape coordinate order, norm attributes and complete RoPE coefficient/scaling
  objects. Same names, dimensions or digests cannot substitute for these owners.
- Full original positions, masks, valid lengths and prior history/state custody.
  Prefill key append is `[rope_k,positions,mask,lengths]`; decode is
  `[original_Key_state,rope_k,positions,mask,lengths]`; key view is
  `[key_append,positions,mask,lengths]`. Bind complete state/attention-domain attrs
  ([helpers:2133–2245](../../crates/pllm-models/src/lib.rs#L2133)); no parallel NEXUS cache.
- GQA mapping `floor(qhead/(Nq/Nk))`; capacity `C`, visible extent `S`, actual
  valid length and per-query causal visibility are DISTINCT. Current/future,
  padding and unwritten cells need authentic visibility checks, not shape inference.

[`compact_request_history`:1–2](../../crates/pllm-compiler/src/replicated_decoder/compact_request_history.rs#L1)
explicitly issues no completion bit or executable storage; growing-prefix history
is NOT KV-completion authority. Key-view history and every physical backing and
coordinate map must be authenticated, including views spanning multiple owners.

[`ValueEdge`:1–36](../../crates/pllm-compiler/src/clear_value_binding.rs#L1)
checks retained owner/graph identity, leaving semantic checks to adapters;
[`Words`:1–20](../../crates/pllm-compiler/src/replicated_decoder/views.rs#L1)
contains references, not shares. Neither these, compiled spans, digests nor actual
exported words mint standalone source or finished-producer authority. Existing
[`raw_first_norm`:1–14,59–103](../../crates/pllm-compiler/src/replicated_decoder/raw_first_norm.rs#L59)
has a fixed first-norm profile; it does NOT authorize head-width Q/K norm,
BF16/binary64/protected conversions or scaled RoPE. Source identity, producer
completion, numerical qualification and security remain separate obligations.

**MISSING production drain issuer:**
[`rotary_handoff`:1–13,243–281,319–341](../../crates/pllm-compiler/src/replicated_decoder/rotary_handoff.rs#L1)
has private `CoordinatorDrained`/consuming invocation mechanics, not a production
router-drain issuer. Proposed issuer must authenticate three DISTINCT roles AND
actual router lease, globally fence new traffic and drain queues, in-flight work
and native callbacks at matching boundaries, retaining lease through begin AND
restore. Local-ready checks and caller assertions cannot mint this authority.
[`body_handoff`:1–15,125–151](../../crates/pllm-compiler/src/replicated_decoder/body_handoff.rs#L1)
explicitly separates local readiness from global drain. Proposed private
`SuccessfulDrained` and `CancelledDrained` authorities must remain distinct:
cancelled global quiescence NEVER permits Q/K publication. Existing
[`Invocation::cancel`:290–293](../../crates/pllm-compiler/src/replicated_decoder/rotary_handoff.rs#L290)
removes the executor before cancelling the suspended Worker. Exact funding,
numerics and cache/position policies remain **OPEN**; reuse shared admission/transport.

**UNSELECTED representation candidate: owning exports snapshot.**
[`Executor::finish`:739–753](../../crates/pllm-compiler/src/replicated_decoder/body_machine.rs#L739)
consumes a completed executor and returns original runtime plus owned `Values`.
[`Values`:802–825](../../crates/pllm-compiler/src/replicated_decoder/body_machine.rs#L802)
owns `Arc<Plan>` and `Vec<Option<Secret>>`; the whole aggregate can move, with
`Secret` zeroizing its storage. This is not a borrow across restore or cloned
`Words` labels. Existing
[`rotary_handoff::restore`:319–341](../../crates/pllm-compiler/src/replicated_decoder/rotary_handoff.rs#L319)
uses finish plus its compiled output map; it does NOT retain these exports.
[`body_handoff`:33–53,230–283](../../crates/pllm-compiler/src/replicated_decoder/body_handoff.rs#L230)
copies ALL exports, replays original retirement, erases output `Values`, then
releases Worker. Reusing that helper cannot establish retained-export custody.

A NEW, separately approved consuming restoration contract would keep an ordered
guard owning **Values → original runtime → suspension**, with original compiled,
source and request roots continuously held outside it. Fallible copy of ALL
exports and original-ledger replay only BORROW guard fields; move fields ONLY
after success. As with existing `RestoreOwners`, no custom `Drop` is needed to
move whole fields on success; on failure/unwind exports erase before runtime,
Worker cancellation and source cleanup. Never expose a partially restored Worker.

On success, proposed non-Clone `RestoredPair` would own **Values first, Worker
second, then original compiled/source/request roots**, with authenticated Q/K
output maps and consumer interval. Publish Q/K inseparably only after successful
global drain AND restore. No independent Worker advance, extraction of either
value or Worker, or detached coordinate authority; a private consuming consumer
transition must retain the complete pair. These names/APIs are schematic private
proposals, not implemented authority. Common restore boundary must precede the
EARLIEST consumer of EITHER value; refuse otherwise. Original retirement is
replayed unchanged, not delayed or replaced by pinning.

Retaining Values alongside restored Worker adds overlap, not free aliasing.
Separate approval and full-layout funding must cover retained Plan, all live
Secret backings, coordinate maps, restored copies, guard/transition/result slots,
consumer lifetime and failure cleanup, without discounting existing owners or
debits. This documentation edit does NOT select or approve that overlap.

Existing [`View`:37–79](../../crates/pllm-compiler/src/replicated_decoder/body.rs#L37)
and [`Builder::scores`:965–1015](../../crates/pllm-compiler/src/replicated_decoder/body_families.rs#L965)
accept layout/dimension inputs, no execution-pair authority. **Private consuming
adapter is MISSING.** Partial-copy failure must drop outputs → runtime →
suspension/cancel Worker before source custody release; existing ordered guard
and invocation field order supply reusable mechanics, not Q/K completion proof
([guard:33–53](../../crates/pllm-compiler/src/replicated_decoder/body_handoff.rs#L33),
[invocation:276–281](../../crates/pllm-compiler/src/replicated_decoder/rotary_handoff.rs#L276)).

**Evidence:** source reread only; tests/builds **UNRUN**. No HE execution,
backend/default change, activation or implementation/adoption approval.

## 7. Independent review and future tests (not results)

- [ ] Reject centered-energy→RMSNorm or GELU→SiLU substitution; lost epsilon/weight/offset, optional Q/K norm, bias or residual edge; N-G remains unadopted.
- [ ] Reject reset/omitted RoPE positions or coefficient profile, wrong GQA grouping, future/padding/unwritten-key visibility, dropped attention-value product or invalid last-token selection.
- [ ] Reject truncated vocabulary, assumed Boolean/multi-hot→ID conversion, invented near-tie tolerance, silently public logits/history/lengths or client layer decryption.
- [ ] Reject implicit cache/history holder, erased state obligations, mask reuse/bounded-real-pad proof, fresh caller source ledger, uncharged retained/overlap backing or metadata→activation/API authority.
- [ ] **Future models tests:** per-op independent dense formulas and mutated edges/attrs, epsilon and weights, Q/K norms, GQA, absolute positions, masks/padding, both residuals, complete head/tied weights, execution-policy ties; compare every epoch's full-prefix and stateful transformer outputs **and all valid KV state**, not one layer only.
- [ ] **Future compiler tests:** original-root/consumed-work continuity, sticky refusal, original backing lifetimes, append/read/retirement and prefix-simulation obligations; key/layout/scale/error capability mismatch, multi-ciphertext full-vocab selection, one-use mask failures/retries and unsupported operator refusal. HE tests require separately qualified backend, never mocked-metadata execution claims.
- [ ] **Future `nexus_gelu` regression tests:** retain original endpoint, polynomial/candidate and centered-energy distinctions; add no Qwen SiLU alias or newly adopted budget. Existing study results remain separate; no tests/build/native/weights/network/terminal run for this document.

**STOPEDIT.** One contract only; parent independent Astra review before any method implementation. Documentation is not encrypted execution evidence and makes no whole-trained-decoder claim.
