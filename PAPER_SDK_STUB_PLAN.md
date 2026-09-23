# Paper-to-SDK capability plan

Status: proposal, 23 September 2026. The paper landing map is
`pllm-paper-module-landing-map-2026-09-23.md`; the 84-entry source/fingerprint
inventory is `docs/data/research/paper-library.json`. `papers/` is an ignored
local research cache; run `uv run python scripts/sync_paper_library.py --fetch`
to populate it, never during documentation builds. The existing 24 locked
research records in `docs/data/research/papers.json` are separate. A source in
the chronology is **not** an implementation, an SDK class, or an eligible
inference profile.

## The first stub is a proposal, not an executable component

Add a validated, immutable `CapabilityProposal` record in `pllm.research` (and
an SDK documentation view of it) before adding methods. Its fields should
include source ID and PDF hash, source/review status, proposed capability
family, semantic operator or lifecycle, input/output type and numeric domain,
state/freshness policy, role/placement and trust assumptions, proposed
dependencies, first bounded experiment, measured evidence IDs, and missing
promotion gates. The existing 24 locked records keep their identities; the
additional bibliographic entries must not be silently converted into reviewed
source locks. Ripple remains an unverified *lead*, and MOAI/MoZZarella lack
verified local full texts.

Generate **proposed capability** callouts on paper pages and family-level SDK
landings from these records. Label them `source only`, `bounded prototype`,
`compiled`, `whole-model executable`, or `measured` with independently checked
axes. Existing six scoped paper/component citation callouts keep their exact,
reciprocal links. Do not expose proposal IDs in the runtime component registry,
`Pipeline` selectors, `pllm.components` discovery, or runnable examples.

## Land proposals in shared capability contracts

| SDK home | Mapped research examples | First reusable contract |
| --- | --- | --- |
| `pllm.correlation` and `pllm.preparation` | PCFs, ring/field generators, silent OT, VOLE, MoZZarella | Correlation kind, algebra, share holders, authentication, expansion cost, one-use recipient and consumer |
| `pllm.nonlinear` and compiler numerics | Compact, LLAMA, SIRNN, ReDASH, FLUTE, LogRow, FSS/DPF lookup | Exact signed domain, approximation, protected selection, conversion and per-lane material bound |
| `pllm.modeling`, `pllm.passes`, and `pllm.kernels` | CipherPrune, Ditto, MPCFormer, ATLAS, SHAFT | Model-neutral semantic transform, numeric/quality invariant, operator schedule and artifact binding |
| `pllm.protocols`, `pllm.roles`, and `pllm.schedulers` | Bifrost, NEXUS, SIGMA, FuseFSS, Euston, THOR, MOAI | Topology, online parties, privacy contract, required offline material, message and state lifecycle |
| `pllm.verification` and Rust `pllm-assurance` | Maverick, EMVP, LAMP, DeepProve, matrix coding | Exact checked claim, soundness/attempt budget, challenge secrecy, trusted verifier and failure burn |
| `pllm.state` and `pllm.passes` | MPCache, Cachemir, non-autoregressive GPT | Valid-prefix/state handoff, token feedback, protected selection and generation quality |
| `pllm.research` and `pllm.verification` review records | Breaking Euston, Game of Arrows, Carnival, SoK, Ripple | Negative controls or provenance only; no executable candidate from a survey, attack, or unverified lead |

The homes are capability families, **not paper-named models, profiles, or
compiler switches**. Different trust topologies cannot share an executable
contract merely because they perform matrix multiplication. A new family is
created only when input/output, lifecycle, or trust semantics genuinely differ.
Static provider manifests may be discoverable, but native provider runtime
loading is not yet available; a provider proposal must not imply it can run.

## Promote one vertical slice at a time

1. **Source gate:** verify the primary paper and authors, exact PDF/version
   hash, license/provenance, claimed protocol and role assumptions. A
   bibliographic record is insufficient for a security claim.
2. **Reference gate:** implement one bounded method independently in the
   existing Rust/Python stack; test against exact/independent numeric and
   adversarial controls, account for setup and retained material. Upstream
   artifacts are oracles/provenance, never linked dependencies.
3. **Composition gate:** register a typed component only after its input,
   output, trust, state, numeric, one-use and resource contracts are enforced.
   Bind its serialized selection, lineage and implementation digest to a
   `Pipeline` and fail closed on missing operators or incompatible roles.
4. **Execution gate:** demonstrate the complete required path, including
   preparation, transport, recurrent state and error/cancellation burns. A
   scalar primitive or semantic plan is not whole-decoder support.
5. **Evidence gate:** check parity and generation quality on an exact locked
   model, then benchmark against `baseline.masked_linear_cpu` with the same
   body, tokens, output cap, warm state, environment and trust/assurance
   contract. Record full latency, TTFT, throughput, protocol bytes, offline
   material, client work and failures. Price, energy, WAN/GPU and external SOTA
   remain unmeasured until an eligible cohort exists. Promotion requires
   human review; search cannot waive these gates.

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

These slices create tested reusable contracts, then option pages and examples
under `/sdk/components/` and `/sdk/inference/` once executable. Source-only
proposals can appear in the paper chronology and a clearly non-runnable SDK
research index without pretending that all 84 papers provide an installed
Python API.
