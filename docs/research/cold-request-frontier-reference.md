# Seven cold single-request network screens

## Decision

Seven Sol 6.1 investigations explore software-private execution on ordinary
consumer CPUs/GPUs under the [frozen contract](../plans/cold-request-frontier-seven.md).
They produce bounded executable references, negative controls and replayable
evidence. **None establishes a new complete-request traffic saving, matched
language-model quality or Low/Medium client qualification.** Model changes,
restricted output contracts and local execution are explicit alternatives.

The best near-term implementation candidate is a **public finite-grammar boundary**:
retain only reachable output-head rows and retrieve private prompt embeddings
through an independently operated private lookup. It can preserve the original
model's explicitly restricted output contract without retraining, but its real
tokenizer, sampler, lookup transport and complete source binding are missing.
General JSON with arbitrary strings does not supply a small vocabulary union.

The highest-upside learned direction is a **small private prompt encoder**. It
targets the dominant prefill bill rather than only eight-token decode. A compatible
hierarchical model can additionally reduce global prefill/decode rows; both need
new trained models and held-out quality evidence. Serializer-aware integer training
is a smaller, concrete alternative to choosing nominal precision blindly.

## First-request bill: historical evidence, not candidate performance

`scripts/probe_cold_request_accounting.py` extracts startup plus the **first**
request from all 30 clients in the frozen five-way Qwen2.5 cohort. It checks raw
directed-counter conservation, cached attribution, actual input/output counts,
three replicates, cold/warm separation and continuous TCP-counter windows. It
excludes each later repeat and does not count startup twice.

Control medians, decimal MB:

| Covered first-request item | 150 inputs + 8 outputs | 288 inputs + 8 outputs |
| --- | ---: | ---: |
| Public client delivery | 124.905 | 124.905 |
| Known prefill material and online bodies | 335.686 | 644.418 |
| Known decode material and online bodies | 16.373 | 16.373 |
| Other known setup bodies | 0.003 | 0.003 |
| Unattributed bodies | 0.139 | 0.139 |
| Total covered application bodies | 477.105 | 785.838 |
| Native TCP streams, separate measurement | 478.654 | 787.877 |

Provider checkpoint and pre-positioned client-tokenizer distribution remain
unknown. These are client-cold/provider-source-cached samples, not all-parties-cold
measurements. Their process peaks also include the repeat; no first-request-only
RSS is extracted. Known prefill/decode attribution retains its residual rather
than inventing complete per-token rates.

Even **free deletion of all public delivery** removes at most 26.18%/15.89% of
these covered bodies. Free deletion of all known decode removes only 3.43%/2.08%.
For a hypothetical 20% reduction at 150 rows, replacing the entire public-delivery
bill leaves only 29.48 MB for every new artifact, lookup and setup body. At 288
rows that deletion cannot reach 20% even with a zero-cost replacement. These are
optimistic budgets within one historical body window, not new protocols or
physical-wire bounds. Raw vocabulary bytes cannot be subtracted from compressed
delivery counters byte-for-byte.

## Seven findings

| ID | Direction | Checked result | Decision / next gate |
| --- | --- | --- | --- |
| 1 | [Compositional vocabulary](cold-compositional-vocabulary-reference.md) | Additive codebooks support exact lookup and structured heads on constructed integer data. Floating-point association changes logits and can change winners; fitting/requantization changes the model. | Potentially large boundary-artifact reduction, but no unchanged-Qwen identity. Require pinned weight fitting, full head/lookup numerics, held-out quality and actual encoded delivery. |
| 2 | [Public grammar boundary](cold-grammar-boundary-reference.md) | Qwen-shaped raw tied table plus scales is 136.742 MB. A hypothetical 256-row head, maps/descriptors and 24 fixed private lookup slots total 0.296 MB. Exhaustive tiny restricted-distribution and causal/EOS controls pass. | Strongest restricted-task candidate. This is raw representation/lookup accounting, not saved wire bytes. Replicating the complete table to two cold workers reverses the saving; a real consumer and public tokenizer reachability proof are absent. |
| 3 | [Serializer-aware training](cold-wire-objective-reference.md) | A trained one-bit synthetic row budget reduces four covered stage bodies 2,695→2,623 B, with 96/96 held-out argmax but nonzero error. Two-bit targeting drops to 74/96. Native layouts, serializers and masked reconstruction agree with independent oracles. | Train against actual public row-width cliffs and the global widest-row input bound. Float shrink followed by max-normalized requantization saves nothing. Scan pinned integer rows before any model training. |
| 4 | [Private latent prompt](cold-latent-prompt-reference.md) | Learned pooled toy preserves majority but loses order and rare tokens. Ordinary exact copy is 0/128 at 1/2/4 slots and 128/128 at eight; rare-token recovery still fails at eight. Encoder download can reverse the assumed row savings. | Highest-upside prefill hypothesis, not a validated compressor. A small position-aware encoder and compatible decoder need disjoint task/copy/rare-token tests, calibration and protected latent-prefill binding. Never transmit plaintext latents to a provider. |
| 5 | [Hierarchical block decoder](cold-block-decoder-reference.md) | Causal feedback, local sampling/EOS, partial blocks and overissued material are explicit. Independent exact marginals still give 0.5 total-variation error on a correlated two-token counterexample. | Requires a trained global/local architecture, not skipped target verification. For public weights, existing client-local nonlinear/KV ownership can accompany masked global linears; provider-resident secret-shared execution is a distinct, more demanding path. Charge every local module and sampled-block feedback. |
| 6 | [Reusable local core](cold-shared-core-reference.md) | Defined tying matches copied unrolling; posthoc tying changes logits. Unique weights shrink, but depth work and KV do not. Pinned Huginn runtime files total 15.648 GB; 32-loop, 4,096-context BF16 KV alone is 11.419 GB. | The examined published artifact fails the short cold-download case. A genuinely smaller trained shared core could be a different local topology; upstream source remains specification/oracle, not a runtime dependency. |
| 7 | [Execution-ready provider artifacts](cold-provider-artifacts-reference.md) | Signed role-specific synthetic imports preserve quantized weights/scales and integer outputs; tampering, source/binding substitution, allocation preflight and owner retirement are checked. | Useful provisioning mechanism only when required provider artifacts are missing. Trusted publisher signatures do not prove derivation from absent BF16 sources. Production source assurance and complete artifact delivery need an explicit contract. |

These findings do not establish literature novelty. Primary sources and exact
adaptation limits are retained in each reference. Numerical examples from different
toy models, raw artifact estimates and historical transport cohorts are not pooled.

## Follow-up order

1. **Public finite-grammar boundary:** freeze one genuinely public finite language
   and pinned tokenizer; derive every reachable token without private-prompt
   selection. Check source-locked head rows, full restricted float32 sampling and
   private prompt lookup before SDK integration. Compare both provider-resident
   and provider-cold bills, retaining table replication and lookup compute.
2. **Private prompt encoder:** freeze a small model/download budget and task-quality
   contract. Train on public data only; compare full original, full adapted and
   compressed adapted inputs separately. Exact-copy, rare-token and positional
   failures must remain visible. A large backbone with small LoRA adapters is
   still a large encoder.
3. **Public integer-width scan:** identify real near-threshold rows, global-max
   blockers and common divisors before constrained stage/block training. Retain
   scales through export and price actual serializers plus compressed artifacts.
4. **Hierarchical model:** establish useful trained global/local quality and the
   complete local download first. Reuse the prepared public placement where
   compatible; do not require unimplemented protected provider nonlinear work
   merely to execute a client-owned nonlinear operator.

Provider artifacts are a parallel provisioning opportunity under all-parties-cold
conditions. Vocabulary factorization and a new small shared core remain larger
model-design experiments. None is automatically selectable by the current SDK.

## Evidence and replay

The seven linked references contain exact commands and source/native locks. The
historical extraction has its own artifact:
[`cold-request-accounting.json`](../evidence/cold-request-accounting.json).
The [manifest](../evidence/cold-request-frontier-manifest.json) binds each result
without merging their evidence identities. No checkpoint values, training cohort,
whole-response timing or candidate process-RSS measurement was run in this screen.

Parent validation passes **157 tests**; all eight reports reproduce exactly in
fresh interpreter invocations against the installed environment. Ruff passes for
all eight probes and their tests. The manifest/local-source audit checks 54
digests with zero mismatches; the 435-page docs-content check also passes. These
are correctness/accounting checks, not inference-performance measurements.

Run bounded references against the existing installed validation environment.
Ordinary source-only fixture tests do not establish SDK integration:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 target/frontier-sdk-env/bin/python -m pytest -o pythonpath=. --noconftest -q tests/test_cold_compositional_vocabulary.py tests/test_cold_grammar_boundary.py tests/test_cold_wire_objective.py tests/test_cold_latent_prompt.py tests/test_cold_block_decoder.py tests/test_cold_shared_core.py tests/test_cold_provider_artifacts.py tests/test_cold_request_accounting.py
target/frontier-sdk-env/bin/python scripts/probe_cold_request_accounting.py --output target/cold-request-accounting-replay.json
```

Real checkpoint/training and fresh-process serving follow-ups run serially under
the existing allocation, host-reserve, swap and measurement-lock contracts. Low
256 MiB and Medium 2 GiB remain whole-client limits; toy array bounds and weight
payloads do not qualify either tier.
