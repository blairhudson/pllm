# Paper-to-SDK capability plan

Status: SDK composition search, its matched finalist scorecard and the first
Qwen Slalom/SmoothQuant adaptation controls are implemented. Maverick and
SIGMA/FuseFSS have bounded native component references with explicit unresolved
resource/security gates. The paper landing map is
`docs/research/paper-module-landing-map-2026-09-23.md`; the 85-entry source/citation
inventory is `docs/data/research/paper-library.json`. `papers/` is an ignored
local research cache; run `uv run python scripts/sync_paper_library.py --fetch`
to populate it, never during documentation builds. The existing 24 locked
research records in `docs/data/research/papers.json` are separate. A source in
the chronology is **not** an implementation or an eligible inference composition.
The packaged `python/pllm/components/planned_methods.json` binds every paper to one
importable Python symbol. The [generated research method roadmap](https://pllm.run/sdk/components/research-method-roadmap/)
lists those symbols and their next gates.

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

Each delivery ends with:

1. A paper-to-method contract stating **full reproduction**, **scoped adaptation**
   or **component reference**, with deviations from the cited source.
2. A reusable executable SDK capability, bounded native admission and real role
   transport for the claimed scope, including failure/cancellation behavior.
3. A repository Python `Experiment` factory, pinned public calibration where
   needed, and immutable source/configuration/evidence identities.
4. A Qwen2.5-0.5B matched cohort against both the relevant baseline and the best
   eligible SDK combination. Qwen3-4B follows after memory admission. Comparable
   model/source/workload/lifecycle and trust contracts are explicit.
5. Quality-gated cohorts for methods that change numerics or semantics. Such
   methods cannot enter today's exact-body/output scorecard by suppressing its
   identity checks. Record quality, bytes on every link, setup/material, CPU/GPU,
   memory and failures separately; unknown measurements remain unknown.

The meta-study should first establish useful head-to-head **methods**, then compose
compatible methods through SDK search. Do not relabel the fastest PLLM stack as an
individual paper's reproduction or mix original-paper speedups into PLLM scores.

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
preprocessing lower bound still block an executable SDK method.

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
