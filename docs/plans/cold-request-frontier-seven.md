# Seven consumer-hardware cold-request investigations

Completed bounded results and the parent first-request accounting audit are in
[the research reference](../research/cold-request-frontier-reference.md).

## Objective

Reduce total network delivery for one fresh request. This screen explores the
seven software-only directions proposed after the
[fixed-tier objective screen](../research/tier-objectives-reference.md).
Ordinary consumer CPUs/GPUs are the target; TEE and attestation-dependent
execution are excluded.

New checkpoints, training, constrained output interfaces and fully local
execution are admissible research hypotheses. Each must state its changed
contract. They cannot inherit the current checkpoint's quality, complete-logit
semantics, remote-compute fraction or measured costs.

## Cold-start and comparison contract

The first-request bill charges every required missing artifact and every
request-related directed transfer:

```text
first-request bytes = missing public artifacts and source delivery
                    + setup and authentication
                    + fresh preparation and helper material
                    + online client and peer messages
                    + framing and lifecycle controls
```

Client-cold/provider-resident and all-parties-cold are distinct deployment
conditions. A provider's cached source cannot disappear from an all-parties-cold
bill; provider download savings cannot reduce a client-cold bill in which those
providers were already resident. Pre-positioned client tokenizers, encoders,
codebooks and keys are charged whenever the candidate requires them. There is no
later-request amortization, private prefix reuse or other caller's cached state.

Raw tensor/storage counts, serialized application bodies, TCP streams and
physical wire have different identities. Subtracting raw matrix bytes from a
compressed transport total is not a complete saving. Missing costs remain
unknown. Publisher training/compilation compute is recorded separately from
per-request loading and execution; it is not silently treated as runtime work
or as free candidate creation.

The existing [five-way screen](../research/sota-combinations-reference.md)
provides historical workload context: 150/288 input tokens and eight outputs.
Its cold control medians are 478.65/787.88 MB of native TCP streams, excluding
checkpoint and pre-positioned tokenizer distribution. Its client peaks cover
the cold-plus-repeat process lifetime. Neither field supplies a fully cold,
single-request comparison for these new hypotheses.

Unchanged numeric contracts require complete logits and active-state parity.
Restricted-output contracts require parity against the explicitly restricted
full-model oracle, including sampling and termination. Changed checkpoints need
disjoint held-out task-quality comparisons; identical token counts or a small
synthetic toy do not establish equivalent model quality.

## Investigations and ownership

| ID | File stem | Mechanism | Primary falsification gate |
| --- | --- | --- | --- |
| 1 | `cold_compositional_vocabulary` | Additive codebook token embeddings and tied output head | Complete lookup/logit computation, fitting error and cold artifact cost |
| 2 | `cold_grammar_boundary` | Public constrained-vocabulary head plus private prompt embedding retrieval | Exact constrained sampling and complete private-lookup setup/traffic |
| 3 | `cold_wire_objective` | Train against actual public residue widths and communicating dimensions | Serializer-certified byte thresholds and held-out quality |
| 4 | `cold_latent_prompt` | Client-private learned prompt compression into soft-token vectors | Encoder download, protected ingress and lost task information |
| 5 | `cold_block_decoder` | Remote block representations with client-local token expansion | Causal feedback, generation quality and complete local-decoder cost |
| 6 | `cold_shared_core` | Downloadable weight-tied recurrent-depth model | True trained tying, per-depth state, cold bytes and client computation |
| 7 | `cold_provider_artifacts` | Authenticated execution-ready role-specific weight artifacts | Exact source/numeric binding and genuinely cold provider provisioning |

Each Sol 6.1 worker owns its uniquely named probe, tests, research reference and
synthetic evidence. Workers implement bounded references and independent negative
controls; they do not modify shared SDK/native contracts, install dependencies,
build wheels or commit other workers' changes. Upstream papers and source are
specifications and provenance, not runtime dependencies.

Small synthetic correctness runs may proceed independently. Real-checkpoint
value loading, sustained training and timing cohorts require serial parent
admission, host reserve/swap checks and the shared frontier measurement lock.
Source/probe hashes and exact replay commands bind retained evidence.

## Resource and privacy requirements

Low and Medium retain 256 MiB and 2 GiB whole-client maxima. Codebook/model bytes
and logical scratch estimates do not qualify a process tier. SDK imports,
tokenizer, state, temporary conversions, overlapping owners and device/disk
storage must be priced, with whole-client peaks measured separately.

Soft-token and block representations are private activations. Their small size
does not authorize plaintext provider delivery. Grammar rows may be public only
under an explicit public output contract; private prompts cannot determine a
visible token-row fetch set. Fresh protocol material retains the existing
one-use, binding, burn and non-collusion requirements.

The fully local shared-core direction explicitly changes remote-compute
placement. Every direction retains client CPU/RSS, provider CPU/RSS, first-output
delay, decode rate and all-link traffic as separate costs. A completed research
screen does not establish an SDK component, a privacy proof or serving SOTA.
