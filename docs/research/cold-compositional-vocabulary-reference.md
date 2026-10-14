# Cold compositional tied vocabulary/output head

Status: bounded synthetic correctness screen, **not Qwen compression quality or
wire savings**. Consumer CPU/GPU, software privacy only; no TEE. No serving changes,
real tensor loads, training cohort, GPU execution or timing measurements.

## Construction and attribution

For public books `C[m,k,h]`, public token codes `I[v,m]`, and positive public
scales `s[v]`, define `E[v] = s[v] sum_m C[m,I[v,m]]`. Client receives **all**
books, indices and scales before any private token lookup. Lookup gathers locally;
full head computes `D[m,k] = dot(hidden,C[m,k])`, then gathers **every vocabulary
score** `L[v] = s[v] sum_m D[m,I[v,m]]`. No top-k substitution. Codebook evaluation
needs no full reconstructed dense matrix. This is additive, not disjoint-subspace
product quantization, and does not assert small matrix rank: 1,024 basis vectors
already exceed hidden width 896. Unique codes do not imply good fitted weights.

Primary sources, specifications only:

1. Shu and Nakayama, [Compressing Word Embeddings via Deep Compositional Code
   Learning](https://arxiv.org/html/1711.01068v2), 2017, equations 1–4 and sections
   3–5. Explicit additive codebooks, discrete codes, reconstruction fitting and
   Gumbel-softmax training. Section 4 says reconstructed embeddings differ and
   downstream parameters need retraining. Section 5.3 explicitly excludes large
   softmax/recurrent layers from embedding storage results. Those results do not
   validate a tied full Qwen head or four 256-entry books. **No literature novelty
   claimed.** Authors also credit earlier additive quantization.
2. Press and Wolf, [Using the Output Embedding to Improve Language
   Models](https://arxiv.org/abs/1608.05859v3), 2017: input/output tying is established
   practice, not a new contribution here.
3. [Pinned Qwen2.5-0.5B-Instruct configuration](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/raw/7ae557604adf67be50417f59c2c2f167def9a775/config.json)
   sets `vocab_size=151936`, `hidden_size=896`, `tie_word_embeddings=true`.

Evidence records SHA-256 and lengths of HTTP response bodies inspected during
this screen; mutable arXiv HTML can change on later fetch. No downloaded upstream
implementation, vendoring, building or linking. Local source hashes bind script,
tests, this note, architecture and prior research; evidence cannot hash itself.

Prior PLLM findings matter: `packed-embedding-reference.md` proves token/head share
one 136,134,656-byte i8 matrix and embedding-only relocation releases zero matrix
bytes. Its frozen lower-bit variants fail complete-logit/KV fidelity.
`predictive-weight-codec-reference.md` reports exact lossless predictors losing to
zlib1: about 121.495 MB compressed matrix, with 607,744 scale bytes additional.
Our raw 136 MB denominator is therefore **not current wire baseline**.
`ARCHITECTURE.md` distinguishes bounded research operators from integrated SDK
contracts. No existing compiler numeric/protocol identity is changed here.

## Numeric contracts: three different claims

Probe delivery: little-endian fp16 centroids, uint8 indices, little-endian f32 row
scale; fp16 expands exactly to f32. Lookup left-adds components in f32 from zero,
then multiplies scale. Dense reference head separately left-adds feature products
in f32. Factored head left-adds feature products per centroid, then left-adds
gathered scores and multiplies row scale. No BLAS, FMA or parallel reductions.
This explicitly changes reduction order, **even for the same tied real matrix**.

Retained deterministic evidence checks **3,462 lookup values with zero bit
mismatches** and 2,019 complete-logit values across five fixtures:

| Fixture | Dense-vs-factored bit mismatches | Other measured numeric result |
| --- | ---: | --- |
| Structured dyadic | 0 / 672 | Exact tied lookup/head |
| Random float-order | 523 / 672 | Maximum absolute error 1.90735e-6 |
| Greedy additive fit | 496 / 672 | Source RMSE 0.442442; 4 / 7 token winners change |
| Cancellation | 1 / 1 | Absolute disagreement 139,008 |
| Low-margin adversary | 0 / 2 against reconstructed head | RMSE about 5e-5 still flips source winner |

Fit source complete-logit worst error is 2.412575, distinct from its dense-vs-
factored rounding error of 4.76837e-7. Structured synthetic artifact charges 799
bytes, including 188 metadata bytes and 611 framed compressed bytes; raw payload
is 1,536 bytes. Fitted artifact charges 1,253 bytes. Neither extrapolates Qwen wire.

* Integer/dyadic structured fixture: all rows and logits match independent scalar
  reconstruction/dense head exactly because all intermediate values fit exact
  binary arithmetic. This is representability, not learned checkpoint quality.
* Random fp16 books/f32 scales: lookup still matches scalar reconstruction but
  factored logits have bit mismatches. Cancellation fixture makes disagreement
  explicit. A tolerance check is not evidence of exact complete-logit semantics.
* Tiny six-iteration greedy residual k-means fit on 96×12 independent normal
  rows: freeze fp16 centroids at each component and reassign against delivered
  values. Report RMSE, complete-logit error and token-choice mismatches. This is
  neither paper training nor Qwen fitting/distillation. Adversarial nearly tied
  tokens flip winner despite very small reconstruction error.

For exact real arithmetic, `|delta L[v]| <= ||delta E[v]||2 ||hidden||2`.
Mean RMSE does not bound worst token error or protect arbitrarily small logit
margins. Embedding changes also alter hidden/KV trajectories; same-hidden head
comparison misses that feedback. Tying means both lookup **and** head use one
delivered model, not keeping original head while replacing embeddings.

Existing W8A8 path additionally quantizes activations/weights and dequantizes
integer sums. Generic fp16 additive books cannot replace that contract exactly.
`round(a+b)` differs from `round(a)+round(b)` (ties-to-even: 0.5+0.5), and saturation
does not distribute either (100+100 under signed symmetric int8 clipping).
Changing per-row quantization scales introduces another mismatch. Exact integer
additive books with wider accumulators might preserve integer products **if every
original i8 row is certified reconstructed exactly**, scales remain identical,
and original cast/multiply order remains; no such Qwen certificate exists.
Reconstructing each row, requantizing and running original integer head restores
only reconstructed-model numerics and retains O(VH) work. Full dense expansion
also restores dense reduction order, but loses thin-client memory advantage.
Thus plausible route is a newly trained/distilled tied model plus explicitly
versioned numeric identity, not transparent replacement of existing checkpoint.

## Structural bill, not measured transfer saving

| Qwen-sized owner | Raw bytes |
| --- | ---: |
| Four 256×896 fp16 books | 1,835,008 |
| Four uint8 codes/token | 607,744 |
| One f32 scale/token | 607,744 |
| Candidate total, before metadata/framing | **3,050,496** |
| Original i8 matrix entries only | 136,134,656 |
| Original matrix plus f32 row scales | 136,742,400 |
| Optional expanded f32 codebooks | 3,670,016 |
| Optional dense f32 reconstruction | 544,538,624 |
| One complete f32 logit row | 607,744 |

Raw candidate is about 44.63× smaller than raw i8 entries (44.83× including
baseline scales), **not** whole request. One factored hidden row uses 917,504
centroid MACs rather than 136,134,656 dense MACs, plus 607,744 gather/add operations
and 151,936 scale multiplications. These are algebraic counts, not CPU/GPU speed.
Implementation includes first addition to zero. Lookup costs four vector gathers,
four additions and final scale per coordinate; indices stored token-major.

Evidence charges actual synthetic payload, zlib1 frame with eight-byte raw/encoded
length header, and canonical descriptor (contract, shape, vocabulary and SHA-256).
This demonstration is **not** integrated transport, authenticated artifact parser
or security proof. Production must authenticate trusted model revision, numeric
contract, all tables/scales and decoder binding before use; zlib checksum/hash
without trusted commitment is insufficient. Compression can cover public books,
codes/scales/metadata; compressing secret-dependent traces is not automatically
safe. Private token-indexed downloads, server gathers or fallback requests are
forbidden unless separately protected and padded under explicit leakage contract.

Cold request ledger must charge, without cross-request amortization:

`candidate artifact + trusted manifest/auth + tokenizer/config + body weights or
public remote setup + cryptographic parameters/keys/correlations + preparation
frames/corrections + every online ingress/reply + final response/state transport`.

Each direction and role counts separately. Head/lookup local operation contributes
zero online *head* transfer after entire artifact arrives, not zero whole online
traffic. Other protected decoder boundaries, keys and setup do not disappear.
Subtract only actually replaced tied-artifact delivery once, not twice and not
raw bytes from a compressed wire total. All other cold terms and candidate
Qwen-sized compressed delivery are unknown here. Historical unrelated serving
bills cannot become measured savings by arithmetic substitution.

Client persistent owners: authenticated compressed storage (if retained), decoded
books, all codes/scales, descriptor/tokenizer/config, body runtime, private KV.
Setup owners: received bytes, decompressor, copies, hashing and expansion. Online
owners: hidden rows, centroid-score table `4*256*N*4`, complete logits
`151936*N*4`, local gathers and accumulation temporaries; chunk vocabulary by
public boundaries to avoid full gather copies. GPU upload/staging and CPU copies
are additional owners. No private-dependent host/server fetch. Local memory access
is private only against remote observer; side-channel-resistant client execution
is outside this software-network threat model. Publisher fitting/source arrays
and training weights remain separate server storage/CPU costs, not erased costs.
Quality, peak RSS, CPU/GPU seconds, latency, full cold bytes and wire savings stay
JSON null. Eight MiB synthetic extra-array envelope is conservative admission
estimate, not measured process RSS; imports/interpreter excluded.

## Gates and bounded replay

1. Cheap exactness falsifier: stream a public small fixed row sample through any
   proposed fit; compare original W8 integer rows/scales exactly, then original
   complete-logit float bits. One mismatch rejects transparent exact substitution.
   Current synthetic counterexamples already reject algebra-only proof.
2. Approximate-model route: fit public training rows, freeze all codes/scales/book
   dtype and contract; distinguish fit residual from held-out calibration. Evaluate
   complete logits, prefill/decode KV, margins and full feedback trajectories on
   independent public prompts. Distillation should include tied lookup and full
   head/logit loss, not embedding MSE alone. Full vocabulary must ultimately be
   coded: held-out token rows need public offline assignment, never private fetch.
3. Only after acceptable independently assessed quality: serial complete cold SDK
   pairs, same workload/privacy contract, every setup/link/storage owner, client
   lifetime RSS/CPU, provider setup, GPU copies, throughput/TTFT. No SOTA claim.

Potential parent numeric scan (not implemented/run): **one serial worker**, fixed
4×256 books, ≤512 source rows per chunk and 128 calibration hidden rows; no full
f32 matrix or V×K×H distance tensor. Distance formulation by norm/dot with
512×256 scores limits scratch; extra-array budget ≤64 MiB beyond already resident
source, **not** host admission or complete checkpoint RSS. Full assignment pass
costs roughly `4*151936*256*896 = 139,401,887,744` MACs, multiplied by fit passes;
fit wall time and source residency unknown. This is not a cheap unconstrained
training recommendation. Start ≤1,024 fixed rows, no checkpoint scan until parent
admits memory/headroom and exclusive host. No unsupported scan CLI implied.

Existing installed Python runs only bounded synthetic work:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 target/frontier-sdk-env/bin/python -m pytest -q -o pythonpath=. tests/test_cold_compositional_vocabulary.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 target/frontier-sdk-env/bin/python scripts/probe_cold_compositional_vocabulary.py --verify docs/evidence/cold-compositional-vocabulary-synthetic.json
# To make new evidence, output path must not exist:
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 target/frontier-sdk-env/bin/python scripts/probe_cold_compositional_vocabulary.py --output /private/var/folders/3d/16hqtng12ng_bfz1n99jg04w0000gn/T/opencode/cold-compositional-vocabulary-replay.json
```

Strongest conclusion: tiny delivery and no dense client matrix are structurally
possible for a **structured tied model** with full vocabulary output. Neither
existing Qwen quality, original W8A8 exactness, whole-request network reduction
nor speed follows. Float factoring and lossy fitting have concrete negative cases.
