# Documentation hierarchy

This document defines the target public documentation hierarchy for
`https://pllm.run`. It is the migration contract for routes, navigation,
generated publication records, redirects, and hierarchy tests.

## Rules

1. A top-level navigation URL is its useful landing page. Do not add a redundant
   `start` child.
2. Public URLs describe user concepts and tasks, not repository folders,
   implementation phases, or Python package internals.
3. Every canonical page belongs to one sidebar location or an explicit
   non-navigation allowlist.
4. One concept has one canonical page. Replaced URLs redirect permanently.
5. Guides teach use, compatibility, trust, support, and failure boundaries.
   Reference pages enumerate APIs and signatures.
6. Importable, constructible, compiler-accepted, runtime-bound, whole-model
   executable, tested, and evidenced are separate support claims.
7. Public slugs use lower-case kebab-case.
8. Relevant component and option pages link to narrowly scoped research sources;
   paper pages link back to those pages. A citation never implies reproduction or
   transfer of a paper's security or performance claims.
9. Generated Python-reference routes remain stable except when their hierarchy is
   malformed. Source layout should converge on canonical route ownership instead
   of accumulating route-map exceptions.

## Top navigation

| Label | Landing page |
| --- | --- |
| Learn | `/learn/` |
| CLI | `/cli/` |
| SDK | `/sdk/` |
| Research | `/research/` |

## Learn

`/learn/` is the Start page. The former `/learn/start/` layer is removed.

```text
/learn/
├── installation/
├── first-private-request/
├── first-local-benchmark/
├── inspect-a-plan/
├── integrations/
│   ├── local-gateway/
│   ├── responses-api/
│   ├── chat-completions/
│   ├── openai-python/
│   ├── openai-agents/
│   ├── codex/
│   └── opencode/
├── concepts/
│   ├── privacy-and-threat-models/
│   ├── parties-and-offline-work/
│   ├── masked-linear-inference/
│   ├── garbling/
│   ├── numeric-semantics/
│   └── research-and-evidence/
```

Architecture, trust, privacy-assurance, and evidence-claim explanations belong
under Learn concepts rather than a second hidden Learn hierarchy.
The Learn sidebar orders Start, Core concepts, then Working with PLLM.

## CLI

CLI keeps one task-oriented guide group and one command-reference tree.

```text
/cli/
├── private-inference/
├── provider-roles/
├── benchmarking/
├── inspect-and-research/
└── reference/
    ├── gateway/
    ├── serve/
    ├── benchmark/
    ├── config/
    ├── components/
    └── dev/
```

Command children follow the actual parser hierarchy. CLI reference pages may be
generated, but their public paths must remain command-shaped.

## SDK

SDK navigation teaches tasks and selectable capabilities. It does not mirror
`pllm` module locations or the old Build, Pipeline, Operate, and Research source
folders.

```text
/sdk/
├── experiments/
│   ├── define/
│   ├── files/
│   ├── resolve/
│   └── identity/
├── models/
│   ├── sources/
│   │   ├── hugging-face/
│   │   ├── safetensors/
│   │   ├── gguf/
│   │   ├── mlx/
│   │   ├── ollama/
│   │   └── tiny/
│   └── families/
│       ├── qwen2/
│       ├── qwen3/
│       ├── qwen3-5/
│       ├── phi-4-mini/
│       └── gemma-4/
├── inference/
│   ├── compare/
│   ├── masked-linear/
│   ├── verified-masked-linear/
│   ├── proprietary-guarded/
│   ├── proprietary-blinded/
│   ├── direct-fhe/
│   └── customize/
├── components/
│   ├── protocols/
│   ├── preparation/
│   ├── verification/
│   ├── nonlinear/
│   ├── schedulers/
│   ├── passes/
│   ├── state/
│   ├── correlation/
│   ├── kernels/
│   └── roles/
├── run/
│   ├── clients/
│   ├── local/
│   ├── gateway/
│   ├── embedded-gateway/
│   ├── provider-roles/
│   ├── plan-bound-runtime/
│   └── lifecycle/
├── plans/
│   ├── lower/
│   ├── inspect/
│   ├── transform/
│   ├── coverage/
│   ├── compile/
│   └── concepts/
├── evaluate/
│   ├── benchmark/
│   ├── search/
│   ├── evidence/
│   ├── assurance/
│   └── metrics/
├── extend/
│   ├── component/
│   ├── provider/
│   ├── model-adapter/
│   ├── native-plugin/
│   ├── agents/
│   └── upstreaming/
└── reference/
    ├── python/
    ├── components/
    ├── native/
    ├── schemas/
    ├── status/
    └── releases/
```

### Inference options

Each runtime-recognized profile has one learning page:

| Profile | Page |
| --- | --- |
| `MaskedLinearCpu` | `/sdk/inference/masked-linear/` |
| `VerifiedMaskedLinearCpu` | `/sdk/inference/verified-masked-linear/` |
| `ProprietaryGuarded` | `/sdk/inference/proprietary-guarded/` |
| `ProprietaryBlinded` | `/sdk/inference/proprietary-blinded/` |
| `DirectFHEProfile` | `/sdk/inference/direct-fhe/` |

Each page states use and non-use cases, weight ownership, trust boundary, roles,
preparation, optional dependencies, default components, permitted overrides,
support gates, runnable examples, API links, evidence, papers, and limitations.

### Component options

Each registered built-in component receives a learning page below its capability
category. Category pages compare options and explain compatibility. Option pages
must not imply that a component is a complete runtime profile.

Current categories and options:

- Protocols: masked linear, guarded linear, blinded linear, direct FHE, secure
  linear preview, and cleartext control.
- Preparation: model-aware corrections, BFV correlations, and HE-authenticated
  preprocessing.
- Verification: Freivalds and linear integrity.
- Nonlinear: arithmetic-garbled SiLU, binary-table multiplication, and R03 CRT
  multiplication.
- Schedulers: bounded elements, scalar, independent lanes, and chunked lanes.
- Passes: KV-cache eviction.
- State: client-local KV.
- Correlation: seeded expansion.
- Kernels: CPU.
- Roles: inference.

Every option page reports constructibility, serialization, profile acceptance,
compiler acceptance, runtime binding, whole-model execution, end-to-end tests,
and evidence separately.

### Metrics

Metric options live below `/sdk/evaluate/metrics/`: latency, throughput,
communication, memory, energy, accuracy, perplexity, and cost.

## Research

Research keeps exactly three primary destinations. Supporting PLLM-research pages
nest below the PLLM paper rather than competing with those destinations.

```text
/research/
├── papers/
│   └── <chronological paper pages>/
├── whitepaper/
└── pllm-paper/
    ├── methods/
    ├── compositions/
    ├── clean-room/
    ├── publications/
    ├── backlog/
    ├── recipes/
    └── records/
```

Paper routes use kebab-case. The timeline lists public papers newest year first,
and its sidebar groups paper pages chronologically.

## Research citations

Use a scoped citation admonition on each relevant option page. It records:

1. relationship type;
2. exact PLLM component and bounded scope;
3. adapted idea;
4. excluded or unreproduced work;
5. current execution and evidence status; and
6. reciprocal paper URL.

Initial relationships include masked linear and Freivalds with Slalom; arithmetic
garbling with Dash, ReDASH, and Garbling Gadgets; half-gate concepts with Half
Gates; KV-cache eviction with MPCache; and fixed-point conversions with ReDASH.
Options without a defensible source are labeled PLLM-specific infrastructure.

## Migration

Migration proceeds in verified vertical slices:

1. Flatten Learn and install permanent redirects.
2. Replace SDK implementation-phase hierarchy with task and capability roots.
3. Add profile pages and component category/option pages from checked public API
   data.
4. Consolidate duplicate Search, Benchmark, Assurance, Models, and Deployment
   pages.
5. Normalize Research nesting and paper slugs.
6. Move source files to match canonical ownership and remove obsolete route-map
   exceptions.
7. Regenerate publication artifacts and validate links, examples, navigation,
   search, sitemap, redirects, and production output.

No old path remains a second canonical page. Redirects preserve external links
without preserving the old taxonomy.
