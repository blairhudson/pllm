# Cold latent prompt: small client encoder (#4)

## Verdict and scope

**Plausible task-bounded project hypothesis; not validated Qwen optimization.**
Client could read a new private prompt with a genuinely small encoder and turn
it into k hidden-width vectors, then run a compatible decoder through existing
software-private stages. Fewer prefill rows can remove substantial traffic, but
encoder delivery, local work, incompatible positions/numerics, and lossy detail
retention can reverse that advantage. No literature-novelty claim: project
hypothesis is combining small client ownership, cold delivery, and protected
decoding. No real checkpoint downloaded, loaded, trained, or benchmarked here.
No TEE, SDK change, installation, wheel build, timing, or concurrent cohort.

Owned outputs:

- `scripts/probe_cold_latent_prompt.py`
- `tests/test_cold_latent_prompt.py`
- `docs/evidence/cold-latent-prompt-synthetic.json`
- this note

## Primary sources: what they establish, what they do not

1. Ge et al., **In-context Autoencoder for Context Compression in a Large
   Language Model**, [arXiv:2307.06945v4](https://arxiv.org/html/2307.06945v4),
   ICLR 2024. Sections 2.1–2.3 describe LoRA-adapted LLM encoder, appended memory
   tokens, frozen target decoder, autoencoding/continuation pretraining and
   instruction fitting. Section 3.1 uses 512 context tokens, 128 memory slots,
   rank 128. **1% added parameters is not a 1%-size encoder**: encoder still
   runs LLM backbone. Section 3.2.1/Figure 5 show smaller slot counts losing
   reconstruction quality. Table 3 reports BLEU 99.3 for normal text but 3.5
   patterned random and 0.2 completely random text. Their EM metric is exact
   matching *prefix fraction*, not full-string match. Appendix A trains on eight
   A100 80GB GPUs; Appendix B profiles on A100, not consumer CPU. Section 3.3.2
   includes compression time; its cached benefits do not establish first-prompt
   client encoder benefit. Frozen compatible decoder is possible, but does not
   prove small independent encoder works with arbitrary untouched Qwen.
2. Chevalier et al., **Adapting Language Models to Compress Contexts**,
   [arXiv:2305.14788v2](https://arxiv.org/html/2305.14788v2), EMNLP 2023.
   Section 3 introduces learned summary tokens, accumulation, randomized
   segments, and language-model loss. Its positional paragraph omits absolute
   positions for OPT summary vectors, but applies RoPE normally for relative
   models. Section 4.1 uses OPT-2.7B with 50 summary vectors; section 4.3 adapts
   Llama-2-7B attention via LoRA and summary embeddings. Table 3 and section
   4.3 explicitly retain gap to full context; section 5 classification results
   are not exact-copy guarantees. Section 6 corpus precomputation is unrelated
   to cold private first prompt. Compression still uses large adapted LM.
3. Mu, Li, Goodman, **Learning to Compress Prompts with Gist Tokens**,
   [arXiv:2304.08467v3](https://arxiv.org/abs/2304.08467v3), NeurIPS 2023.
   Abstract and method describe attention-mask training of LLaMA-7B /
   FLAN-T5-XXL to produce reusable gist state. Reported up-to-26x compression,
   40% FLOP reduction and 4.2% wall-time speedup are distinct metrics. Gist
   attention KV prefixes are not interchangeable with input soft embeddings;
   caching repeated instructions is not our new unrelated prompt comparison.
4. Hu et al., **LoRA: Low-Rank Adaptation of Large Language Models**,
   [arXiv:2106.09685v2](https://arxiv.org/abs/2106.09685v2), ICLR 2022.
   Frozen base weights plus trainable low-rank updates save trainable parameters
   and optimizer state, not all base inference weights or encoder compute.
   A trained update can merge into provider matrices, but changes source/numeric
   identity and requires new preprocessing correlations for those matrices.

These sources support learnable compression and compatible models, not privacy
of embeddings, arbitrary-model portability, or consumer cold-start feasibility.

## Privacy and actual integration seam (not implemented)

**Never send plaintext soft vectors to provider.** They intentionally retain
private prompt information. Compression is not encryption; learned inversion,
task inference, and exact recovery are allowed decoder objectives. HTTPS only
protects transit, not provider observation.

Client owns tokenizer, encoder, latent buffer, first normalization, nonlinear
operations, KV, final head/sampling and text decoding. For a hypothetical latent
boundary, replace embedding lookup with `[1,k,896]` client-owned activations;
perform first RMSNorm locally and feed resulting rows into ordinary first QKV
prepared linear stage. This must preserve the complete prepared mechanism:

- `crates/pllm-core/src/prepared_stage.rs`: `StageLayout::new` validates
  input/output geometry, wire/activation widths, scales and row bounds;
  `prepare_inner` lines 105–161 consumes `MaskRows::take`, calls
  `offset_transport::quantize_mask_pack`, and returns packed queries, retaining
  output masks and row scales in one-use `StageQuery`.
- Lines 216 onward call `unmask_dequantize_into`; `finish` consumes state.
  Binding must include checkpoint/adapter, latent encoder/config, row positions,
  numeric plan and one-use tickets. Failures burn masks, not retry same mask.
- Conceptually Client sends `a-r` in supported finite ring/field. Preparation
  sends correction `Wr-s` only to Inference, which computes
  `W(a-r)+(Wr-s)=Wa-s`; Client adds its output mask `s` to recover `Wa`.
  Client never receives the correction. This is explanatory algebra;
  actual offset transport includes quantization/scales and residue frame rules.
  Preparation→Inference correction `Wr-s` is charged for every executed row.
- `python/pllm/runtime/masked_runtime.py:114–129` explicitly labels its seeded
  local correlation factory a correctness baseline, **not private preprocessing**.
  Do not deploy that shortcut or disclose masks to Inference. Existing security
  assumptions, authenticated transport and Preparation/Inference non-collusion
  remain required; this proposal does not strengthen malicious correctness.

First-stage masking alone is insufficient if later hidden states, KV, final
logits or generated tokens are exposed. Continue existing protected placement
for every stage, refresh masks, and keep result decoding local. Public k,
generation length, shape and traffic still leak length/timing metadata; fixed
padding has explicit charged cost. Actual masked field payload remains random;
no additional entropy-code compression saving credited.

Required source/interface changes: add explicit latent-prefill boundary to
compiler/runtime, bypass token embedding only at initial prefill, preserve
token-ID embedding for generated continuation, prohibit provider-readable
latent serialization/logging, bind adapter/source identity, update state/history
serialization and admission. `dense_gated_source.rs` source config and decoder
lowering currently describe ordinary token models, not this contract.
`packed_embedding.rs` quantizer is for token-table representation, not proof
that learned latent outputs satisfy its existing bounds.

Numerics must fit encoder output/channel ranges with held-out saturation tests,
then apply first RMSNorm and stage-specific activation scaling. Toy Q12 int16
vectors are not production W8A8 calibration. Refit changes scales/manifests,
range proofs, correlation inventory and potentially every adapted matrix.

## Positions, KV, output semantics

Choose training convention explicitly: memory positions `0..k-1`, generated
tokens begin at k. Do not claim reconstructed n-token KV or original n-token
RoPE positions; memory is a new sequence and decoder needs training with this
convention. Separate instruction text or delimiters consume additional protected
rows. Our arithmetic assumes entire prompt compressed into k slots with no
extra textual query; literature often compresses context while leaving query
uncompressed. Any real experiment must count both.

`crates/pllm-core/src/kv_cache.rs` sequential contract rejects gaps/overwrites.
Jumping from k cached memory rows to original position n is not free, supported
behavior. Client persists newly computed latent KV for subsequent decoding,
not original full-prompt KV; identity includes encoder, adapter, position policy,
prompt digest, quantizer and source. Every fresh first request starts empty;
no prefix reuse credit. Retain private prompt/history locally for UX and possible
full-prefill fallback. A fallback adds *both* failed compressed request and full
request, not only the cheaper one. Last sampled token is pending, so m outputs
execute m−1 decode rows. Head/logits/sample/tokenizer decoding remain local;
vocabulary output is normal tokens, not inverse encoding of latents.

## Executable quality falsification

Probe fits width-eight linear autoencoder by uncentered PCA on chunk histograms
of 256 unique length-eight prompts, common vocabulary IDs 0–7; tied transpose
decoder. Encoder is learned 16×8 token embedding table, pool per chunk to k
slots. Boundary rounds to signed int16 Q12. Toy decoder snaps decoded histograms
to known chunk-count grid and repeats chunk argmax. This deliberately optimistic
grid decoder has no language-model quality claim. Held-out 128 common prompts
are disjoint; another 128 inserts one unseen rare token ID 8–15. Train/test
tasks: majority classification, exact-copy and first-position extraction.

| k | Held-out majority | Exact strings | Correct first token | Rare token copied |
|---:|---:|---:|---:|---:|
| 1 | 128/128 | 0/128 | 39/128 | 0/128 |
| 2 | 128/128 | 0/128 | 61/128 | 0/128 |
| 4 | 128/128 | 0/128 | 72/128 | 0/128 |
| 8 | 128/128 | 128/128 | 128/128 | 0/128 |

Full-prompt oracle copies 128/128, including rare split; oracle is not checkpoint
baseline. Exact collision pairs reverse token order inside chunks: identical
representation but different copy/first-position answers for k<8. No decoder
can answer both exactly from that representation. Rare failure persists at k=8
because trained subspace has zero support for unseen IDs. This falsifies this
specific toy encoder's task-general adequacy, **not all learned compression**.

Finite-precision bit budget `k*d*b` is only an upper bound. Here even smallest
latent has 128 bits versus 32 bits to encode eight vocabulary-16 IDs, yet loses
order: raw bit capacity alone is not sufficient. At Qwen width 896, few slots
can have more bits than token IDs, so dimensional reduction alone does not prove
information-theoretic impossibility. Actual trained mapping and task matter.
Peak added-array conservative allowance is 8,454,144 bytes, below 128MiB;
includes 32× dense input/PCA workspace allowance. Not RSS measurement.

## Cold capacity, delivery and logical memory

Evidence arithmetic is **illustrative dense schedule**, not measured production
traffic, terminal-pruned scheduling, packed/residue codecs or inferred timings.
Qwen-shaped config: 24 layers, hidden896, intermediate4864, QKV output1152,
vocab151936. Per layer groups QKV 896→1152, O 896→896, gate/up 896→9728,
down 4864→896. Hypothetical 4-byte field words: inputs 181248 words/row,
outputs 304128 words/row; two output-sized terms charge return and corrections.
That is 3,158,016 bytes per dense executed row across three directed links.

Ledger charges common cold tokenizer, tied embedding/head **once**, boundary
scales/metadata; extra encoder download; 12MiB fitting/manifest artifacts;
2MiB changed adapter delivered to Preparation; 1MiB per-request seed/setup on
Client→Preparation and another 1MiB Preparation→Inference; every masked input,
output, correction, position/admission and final control. Decoder adapter already
owned by Inference is not an invented duplicate download. Local public-data
training is not free compute: optimizer/data disk and energy are outside serving
ledger and must be reported separately. If training/data/model delivery uses
other network links, add them; current fitting ledger is assumption, not source
inventory measurement. Consumer GPU encoder upload/copies count local work, not
remote application bodies.

For new 150→32 input, eight output request:

- 16MiB encoder: **341,189,552 bytes** net projected saving.
- 128MiB encoder: **223,749,040 bytes** saving, but Low logical admission fails.
- 1GiB encoder: **715,775,056 bytes loss** for cold request.
- Strict encoder-delivery ceiling: **357,966,767 bytes**, after 14MiB extras.

Evidence sweeps n=32/150/512, k=16/32, encoder=16/128/1024MiB. Equal n=k
saves no rows and loses extras. Horizon formula requires unrelated prompts,
each empty KV; only artifact delivery amortizes. Strict-positive request count
is `floor((encoder+fit+adapter)/per_request_row_saving)+1`; cold objective is H=1,
not horizon break-even. Client latency requires `encoder_cost + latent_private
prefill_cost + decode_cost + added_delivery_cost < ordinary_cold_cost`; fewer
rows alone cannot demonstrate speedup. Never combine dense estimate with
historical 493.58MB cold/repeat pair as measured saving.

Logical owner screen (not RSS): encoder; tokenizer4MiB; tied head/embedding
136,134,656 bytes at i8 or twice at bf16; scales/metadata; 32MiB workspace
allowance; 4MiB one-window correlation allowance; float32 logits; int32 soft
buffer; ordinary float32 four-byte active KV `2*24*2*64*(k+m−1)*4`.
No hypothetical KV quantization change is credited. For k32,m8:
16MiB encoder/i8 boundary totals **199,240,704 bytes**, within Low256MiB.
Bf16 boundary totals **335,375,360**, fails Low. 128MiB encoder/i8 totals
316,681,216, fails Low. 1GiB encoder/i8 totals 1,256,205,312, logically fits
Medium2GiB but fails cold bytes. All shown scenarios logically fit Medium;
runtime/allocator/page copies/transport buffers can still exceed RSS budgets.
Fitting artifacts12MiB are disk owners, provider adapter2MiB and Preparation
copy2MiB are separate non-client owners. Paging or omitted head does not make
delivery/local head work disappear. This is admission screen, not memory promise.

## Minimal genuine model experiment: future gate, not executed

Use `Qwen/Qwen2.5-0.5B` base decoder, public
[config](https://huggingface.co/Qwen/Qwen2.5-0.5B/raw/main/config.json): hidden896,
24 layers, 14 query/2 KV heads, head_dim64, intermediate4864, vocab151936,
RoPE theta1e6, tied head. **Before any download** pin repository revision,
config/tokenizer/checkpoint SHA256 and licensing. Mutable `main` config here
only establishes shape, not reproducible checkpoint identity. Build small
position-aware encoder: learned token table vocab×32, position table512×128,
projection32→128, two Transformer layers (width128, four heads, FFN512),
32 learned pooling queries and output projection128→896. Roughly 5.4M
parameters, 10.8MB fp16 weights (i8 needs validated scales); tokenizer/decoder
boundary remains much larger. Encoder learns full prompt including task and
role delimiters; a `[decode]` marker, if used, adds one protected row.

Bound work: 4096 public synthetic prompts, lengths32/150/512, 512 validation,
512 disjoint test (unseen seeds and exact strings), balanced common/rare/random
copy, position lookup, bounded arithmetic and paraphrase tasks. Cap 1000 updates,
batch1 with accumulation4, max512 input and64 target, deterministic seed4.
Frozen decoder first; fit encoder using teacher-forced response CE plus bounded
teacher-logit distillation. If frozen target fails, separately label rank8 Q/V
LoRA compatible decoder variant, recompile merged weights and refit numeric
range artifacts. Teacher responses are from full prompt; no provider-private
training examples. Never optimize each online private prompt or cache its teacher
answer—would erase cold benefit.

Estimated lab envelope, not measurement: decoder about0.5B params (~1GB bf16,
~0.5GB i8 before scales); encoder optimizer/master/grad state ~90MB for Adam;
batch1 checkpointed decoder differentiation and vocabulary logits need several
GB. Reserve consumer GPU 12–16GB VRAM, host RAM16GB, disk4–8GB for bounded
weights/data/artifacts. Decoder must backpropagate to encoder inputs despite
frozen parameters. CPU-only possible for correctness, not promised practical
training latency. Use one frozen decoder for teacher and student serially,
no simultaneous teacher copy. Native quantized serving calibration remains
separate from bf16 training. Low serving budget plausible only with validated
i8 boundary and bounded workspace; full encoder-as-Qwen is not small.

Pass gates, predeclare before experiment: majority/task accuracy ≥95%; rare
copy and positional extraction ≥99%; report exact-copy separately without
masking misses by semantic score. Compare full-prompt original, full-prompt
adapted decoder, and compressed compatible decoder; adapters can improve or
degrade baseline independently. Test 16/32/64 slots. Gate raw BF16 first, then
numeric compatibility/saturation, then tiny prepared-private latent prefill
versus same latent clear path (complete logits/KV), then ordinary installed
private serving with cold delivery and empty cache per unrelated prompt.
Record encoder CPU/GPU work, all links, setup/corrections, full source artifacts,
logical owners, actual peak RSS/swap, TTFT, decode and complete time under
exclusive parent cohort. No protected path/model quality gate passed here.

## Replay and checks

```sh
target/frontier-sdk-env/bin/python -m pytest -q -o pythonpath= tests/test_cold_latent_prompt.py
target/frontier-sdk-env/bin/python scripts/probe_cold_latent_prompt.py --output docs/evidence/cold-latent-prompt-synthetic.json
```

Nine tests: disjoint corpus, independent order collision, learned common identity
and rare failure, deterministic task screen, independently computed all-link
arithmetic and strict break-even equality, zero-row-saving/memory controls and
three invalid inputs. Evidence contains SHA256 for probe/tests and actual
privacy/numeric/source seams. JSON deterministic in tested installed NumPy
environment; different LAPACK bases can rotate degenerate subspaces, so quality
metrics/rounded boundary should be rechecked rather than assuming cross-platform
bitwise learned artifacts. No claim of learned toy as checkpoint quality.
