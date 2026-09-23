# Paper-to-SDK capability plan

Status: staged paper-linked SDK symbols, 23 September 2026. The paper landing map is
`pllm-paper-module-landing-map-2026-09-23.md`; the 84-entry source/fingerprint
inventory is `docs/data/research/paper-library.json`. `papers/` is an ignored
local research cache; run `uv run python scripts/sync_paper_library.py --fetch`
to populate it, never during documentation builds. The existing 24 locked
research records in `docs/data/research/papers.json` are separate. A source in
the chronology is **not** an implementation or an eligible inference profile.
The packaged `python/pllm/components/planned_methods.json` binds every paper to one
importable Python symbol. The [generated research method roadmap](https://pllm.run/sdk/components/research-method-roadmap/)
lists those symbols and their next gates.

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
groups the missing first gates. The six scoped implemented-component citations
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

These slices create tested reusable contracts, then executable option pages and
examples under `/sdk/components/` and `/sdk/inference/`. The 84 paper-linked
symbols are already installed as importable, fail-closed declarations; their
presence does not mean that any corresponding method can run or join search.
