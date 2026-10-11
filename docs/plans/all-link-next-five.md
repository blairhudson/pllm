# Five further all-link traffic investigations

These investigations follow the measured
[placement and restoration work](../research/all-link-frontier-reference.md).
They explore existing open checkpoints, exact numeric execution and complete
directed traffic costs. Useful incremental improvements and reproducible
negative results both count as findings.

The starting checkout is `deb908768e8fc1ea0cc2dd4337af8d939b4c5b91`.
Five Sol 6.1 workers prepared independent probes and lightweight functional checks.
Checkpoint imports, fitting, cryptographic cost samples and performance cohorts
ran serially on the shared host after review.

## Selected hypotheses

The numbers retain their positions in the preceding ten-idea discussion.

| Idea | New mechanism | First decision gate |
| --- | --- | --- |
| 1: Decode epochs | Finish a bounded execution, prepare fresh material while idle, then resume exact state with a small further output allowance | Exact logits/KV and pending-token continuity; fewer issued correction bytes than existing demand allocation after charging admissions, unused rows and pauses |
| 2: In-flight prefix sharing | Execute a common prefix once before any completed-cache entry exists; fork independent continuations within one trusted client | Exact divergent branches, cancellation isolation, bounded state ownership and fewer actual stage rows; already-completed cache hits gain no invented saving |
| 4: Predictive public-weight coding | Encode exact public weight residuals against a bounded token-structured predictor or dictionary | Beat the current bounded zlib frames after charging every predictor/index byte, fitting, decoding and memory owner |
| 5: Exact residual delegation | Decompose integer weights as `W = A B + E`, evaluate a small client factor and a private remote residual, then reconstruct before original dequantization | Smaller complete input/correction/output bodies after factor delivery, with exact integer results and at least 90% of original body MACs remote in both phases |
| 7: Narrow private ingress | Send an eight-bit masked input and privately repair its carries in the wide result ring | A complete concrete repair protocol must beat current stage-packed/residue bodies; exact arithmetic and transcript checks precede a cost claim |

### Initial conditional estimates

These were prioritization hypotheses before the probes, not measurements:

| Selected idea | Initial estimate and condition |
| --- | --- |
| Decode epochs | About 36% fewer bodies for an illustrative 150-input/512-cap/80-used workload, before additional admissions and pauses |
| In-flight prefix sharing | About 50% across eight 150+8 requests sharing 96 input tokens, with one shared public bundle |
| Predictive coding | A net 10–20% bundle reduction would remove 2.5–5.1% of the anchor pair |
| Residual delegation | A 2–5% amortized target, conditional on useful integer residual widths and cheap factors |
| Narrow ingress | 24-to-8-bit input packing offers 12.2% gross; complete private carry repair must fit inside that budget |

The other five proposals remain separate hypotheses: verified binary deltas on
model/profile updates; sub-tensor demand pruning; quantization-certified partial
replies; oblivious exact-exception repair; and cross-client encrypted microbatches.
Their proposal estimates do not inherit evidence from these five probes.

## Shared control and evidence boundaries

The machine-readable control is
`docs/evidence/research-all-link-frontier.json`, not rounded discussion values.
Its first complete Qwen2.5 W8A8 cold/exact-repeat pair contains:

| Application-body category | Bytes |
| --- | ---: |
| Preparation to Inference corrections | 139,060,689 |
| Inference to client online replies | 139,123,150 |
| Client to Inference online inputs | 90,443,518 |
| Cold public bundle delivery | 124,905,091 |
| Remaining control and seeds | 42,799 |
| **All directed bodies** | **493,575,247** |

The source is `Qwen/Qwen2.5-0.5B-Instruct` revision
`7ae557604adf67be50417f59c2c2f167def9a775`; the body fingerprint is
`5d631be30158b3ea2a72cae355ce09a3b304bbfddde48b5758c339c346a34974`.
The pair contains two 150-input/eight-output requests. A changed allowance,
branch workload or artifact lifetime requires its own control. These historical
application bodies do not become newly measured physical wire or cold provider
checkpoint distribution.

Each probe must bind its actual runtime/native build, imported source files,
checkpoint or public fixture, control and measurement window. Measurements use
fresh workers through `scripts/frontier_probe_support.py`, the exclusive
`~/.cache/pllm/frontier-probe.lock`, post-reserve memory admission and swap guard.
The dedicated interpreter is `target/frontier-sdk-env/bin/python`; installed-wheel
tests override the checkout path with `pytest -o pythonpath=`. Environment changes
and source mutation cannot overlap a cohort.

Public artifacts, setup and failed or rejected work remain charged. Resource
projections, exact local execution, protected operator frames and complete SDK
responses retain distinct scopes. Unknown measurements remain unknown. Existing
one-use material burns on consumption, cancellation or failure; a saved decoder
state never authorizes reuse of masks, tickets or provider leases.

## Results and priority

All five bounded screens are complete. The real in-flight-prefix cohort remains
explicitly memory-admission blocked. Hypothetical proposal percentages have not
been promoted into measured all-link gains.

| Priority | Finding | Decision |
| --- | --- | --- |
| 1: Decode epochs | Two fresh pinned Qwen workers preserve exact final logits/KV/output in 15 comparisons. At 17/128 outputs, E=4 cuts covered stage bodies 55.22%; full-cap use grows 1.00%. E=32 saves 49.31% with no added early-stop pause, but full-cap use grows 0.10%. | Strongest follow-up: same-generation SDK yield/resume, then charge authenticated admissions, fresh inventories, all-link streams, CPU/RSS and pause latency. Natural EOS and stochastic continuity remain gates. |
| 2: In-flight prefix sharing | Three generated families preserve exact divergent branches and failure isolation. Two shared-prefix branches reduce packed tensor projections about 13%, identical prompts about 44%. No-match splitting grows them about 17%; completed-cache hits gain zero. | Conditional concurrency candidate. Price split/pruning overhead before selection. Real eight-branch 150+8 cohort requires unchanged 6 GiB post-reserve admission and was blocked twice. |
| 3: Residual delegation | Twelve of 96 checkpoint groups at ranks 0–4 yield only 27,662 projected net bytes per fresh 150+8 request after raw factors/maps, from two groups. Generated decoder state remains exact. | Low priority. No full-body extrapolation; current result does not justify native factor kernels and a new serving composition. |
| 4: Predictive weight coding | Twelve fresh workers check 1.63 billion matrix entries exactly. Token-parent, row-delta and global dictionary increase charged artifact bytes 0.18%, 4.70% and 6.43% over bounded zlib1. | Tested predictors fail the byte gate. Keep zlib control. Other fixed predictors would require a new independently charged hypothesis. |
| 5: Narrow ingress | Exact toy checks and all 96 checkpoint stages support the concrete cost veto: fully fused private carry repair saves zero; separate repair grows bytes. Leaking shortcuts fail transcript checks. | No integration candidate. This is a bound on the examined constructions, not universal cryptographic impossibility. |

Epoch, generated-prefix, residual and codec workers record zero observed swap
growth. Prefix RSS includes the clear
executor; epoch and narrow-ingress screens establish no isolated-role RSS or timing
benefit. Codec timings include import/oracle work. Residual byte projections are
neither serialized private transport nor a measured response saving. These scopes
cannot be pooled into a new serving result or multiplied together.

Contracts, complete evidence and exact replay commands:

* [Decode epochs](../research/decode-epochs-reference.md)
* [In-flight prefix sharing](../research/inflight-prefix-reference.md)
* [Predictive weight coding](../research/predictive-weight-codec-reference.md)
* [Residual delegation](../research/residual-delegation-reference.md)
* [Narrow private ingress](../research/narrow-ingress-reference.md)

The installed-wheel lightweight set passes 143 checks; three generated native
prefix tests are deliberately skipped there because the parent runs those three
families through their complete serial CLI workers. Ruff, 85 Node and five Python
documentation checks, content validation and the static docs build pass. The
publication audit verifies 13,894 file hashes, with the narrow-ingress report
explicitly retaining its pre-cohort research-prose digest alongside the updated
publication digest. Runtime, source and control hashes agree. SDK promotion remains
separate from these bounded screens.

## Subsequent SDK integration

The bounded epoch and pending-prefill contracts are now explicit installed SDK
options, with native admission, fresh inventory/session ownership and ordinary
benchmark targets. Generated Qwen2/Qwen3/Llama controls preserve exact logits/KV,
natural EOS and stochastic continuity; gateway controls compose both lifecycles.
The four-target installed-wheel example completes, including genuine pending joins
and cleanup audits. Historical probe percentages remain separate from complete
SDK traffic and resource measurements. Contracts, gates and replay:
[epoch and prefix SDK reference](../research/epoch-prefix-sdk-reference.md).
