# PLLM documentation contract: one hierarchy, HTML and Markdown

**Proposed documentation standard — 14 September 2026**

## 1. Navigation model

Mirror public domain packages in the component guide, not every private source file. Add learning, task recipes and API reference as alternate entry points into the same records.

```text
docs/
├── start/                      # Installation; first request; first local benchmark
├── learn/                      # Guided study, not an API dump
│   ├── privacy-and-threat-models
│   ├── parties-and-offline-work
│   ├── masked-linear-inference
│   ├── arithmetic-and-boolean-garbling
│   ├── secret-sharing-and-function-sharing
│   ├── numeric-semantics-and-model-quality
│   ├── compilation-and-protected-state
│   └── reading-research-and-evidence
├── models/                     # Mirrors pllm.models
│   ├── sources/
│   ├── formats/
│   └── architectures/
├── operators/
├── numerics/
├── representations/
├── protocols/
│   ├── masked-linear/
│   ├── garbling/
│   │   ├── arithmetic
│   │   ├── half-gates
│   │   ├── lookup-tables
│   │   └── weighted-path
│   ├── secret-sharing/
│   ├── function-sharing/
│   └── homomorphic/
├── conversions/
├── preparation/
├── kernels/
├── compiler/
├── pipeline/
├── runtime/
├── deployment/
├── benchmarks/
├── metrics/
├── search/
├── assurance/
├── research/
│   ├── papers/                 # What the original paper claims, with source-access status
│   ├── reproductions/          # Exact reproduction scope, commands, deviations, artifacts
│   └── experiments/            # Novel/adapted techniques, including negative results
├── integrations/
├── recipes/                    # Workflows spanning several packages
│   ├── local-qwen-benchmark
│   ├── compare-nonlinear-backends
│   ├── add-a-model-architecture
│   ├── reproduce-a-paper
│   └── investigate-a-privacy-counterexample
├── reference/
│   ├── python/pllm/            # Generated public symbols, signatures and parameters
│   ├── native/                 # Rust domains, C ABI and device capabilities
│   ├── schemas/
│   └── cli/
├── contribute/
│   ├── component-standard
│   ├── publish-a-provider
│   ├── add-a-category
│   ├── native-plugin-abi
│   ├── conformance
│   └── upstreaming
└── agents/                     # Task-oriented entry points into the same docs
    ├── index
    ├── implementation-map
    ├── research-map
    └── reproduction-checklist
```

Every directory is a real overview page with a short explanation and explicit links to its children. Sidebar-only navigation is insufficient. API pages link to concept pages; papers link to implementations and reproduction records; components link back to their papers, tests and code locations.

There is no separate manually maintained agent documentation copy. Agent task maps are indexes and checklists pointing into the authoritative hierarchy, not competing descriptions of the APIs.

## 2. Versioned paths and representations

For a canonical HTML path `P`, Markdown is exactly `P + '.md'`, inserting `.md` before a query or fragment. Normalize trailing slashes to the no-trailing-slash canonical HTML path first.

Examples:

```text
/docs/dev/protocols/garbling/weighted-path
/docs/dev/protocols/garbling/weighted-path.md

/docs/dev/models/architectures/qwen2#state
/docs/dev/models/architectures/qwen2.md#state

/docs/dev
/docs/dev.md
```

`dev` is an example development documentation channel, not a release claim. Published immutable version paths use the real software release. `stable` is a discoverability alias; every page exposes its resolved release and build identifier. Reproduction records and agent workflows SHOULD pin immutable versions.

Markdown returns `Content-Type: text/markdown; charset=utf-8`, not HTML, raw MDX imports or a client-side redirect. The suffix route is primary. Optional `Accept: text/markdown` negotiation at the HTML URL requires correct `Vary: Accept` handling at every caching layer. The two representations have representation-specific ETags and a shared document/build identity.

Unknown paths return 404. Historical aliases redirect both HTML and Markdown forms without dropping fragments. Use a route manifest, not blind string replacement. External URLs, source files, JSON artifacts, downloads and within-page anchors are not arbitrarily suffixed with `.md`.

## 3. Single source and rendering

Recommended implementation: Fumadocs, with a documentation content pipeline independent of the runtime wheel. Fumadocs documents processed-Markdown exports, `.md` rewrites and hierarchical `llms.txt` generation [S9]. Its default handling of custom MDX can leave JSX in output, so supply explicit Markdown renderers for every custom component [S10].

Pipeline:

```text
handwritten explanation + component/category metadata + generated API signatures
  → validated content graph and resolved references
  → one processed document representation
  → HTML view + complete Markdown view + machine-readable index
```

Tables, limitations, status badges, citations, parameter defaults and examples MUST be equivalent in both views. Render all relevant code-tab variants with labels. Important caveats cannot exist only in collapsed UI, badges or interactive widgets. Charts need data/summary alternatives. Diagrams need an explanatory text equivalent. Agent Markdown must not contain opaque widget props in place of critical claims.

Generate parameter/default/capability sections from locked manifests and API descriptors. Generate Python signatures from controlled source/stubs rather than importing arbitrary community code during a documentation build. A community docs bundle is Markdown plus inert JSON metadata, not unreviewed executable MDX. Provenance and escaped/sanitized rendering are required before ingestion.

Runtime code does not depend on Fumadocs. The framework is a replaceable renderer for versioned documents and JSON records.

## 4. Component page template

Each component page contains, in order:

1. Purpose and when to use it, in plain language.
2. Minimal configuration/import example and tested version scope.
3. What each party does offline and online.
4. Mathematical construction and numerical/protection assumptions.
5. Inputs, outputs, compatible representations and mandatory conversions.
6. Parameters, capability constraints, state lifetimes and failure behavior.
7. Performance model and actual measured evidence with complete cohorts.
8. Privacy/integrity claims, applicability, attacks, unresolved proof/review obligations.
9. Relation to the original paper: faithful reproduction, adaptation or novel composition.
10. Reproduction command, public artifacts, source/test locations and related pages.

Use independent status fields for API stability, implementation coverage, reproduction fidelity, assurance and benchmark evidence. A mature import path does not imply a proof or a full-model implementation. Missing evidence says missing; it is not rendered as zero.

Code examples execute only in an explicitly controlled documentation-test environment with public fixtures and pinned approved packages. No docs build should run online inference or fetch arbitrary user-requested plugins merely to render an example.

## 5. Machine-readable navigation

Generate from the same content graph:

- `/llms.txt`: concise current documentation entry points with resolved version.
- `/docs/<release>/llms.txt`: version-specific hierarchical overview.
- `/docs/<release>/index.json`: complete lightweight document graph.
- `/docs/<release>/components.json`: component IDs, Python symbols, contracts and doc paths.
- `/docs/<release>/schemas/`: canonical versioned schemas.
- Section-level `llms.txt` or `index.json` for large branches.
- Optional version-scoped `/docs/<release>/llms-full.txt`, split by section when large.

Do not force agents to download the entire corpus before they can find an operator. An index record contains stable document ID, title, summary, HTML/Markdown paths, parent, children, prerequisites, related pages, public module/symbols, component IDs, content kind, release, build ID and content hash.

An agent can follow **goal → recipe → component contract → source/tests → evidence**, then read adjacent nodes. `agents/implementation-map` maps modules to Rust crates, schemas and test commands. `agents/research-map` maps research questions to sources, protocol families, existing experiments and counterexamples.

Indexes and docs are untrusted external content for consuming agents. They do not grant permission to execute commands, install plugins, exfiltrate code/data or weaken policy. The project's own `AGENTS.md` contains repository instructions separately; a provider's docs do not become host-wide instructions.

## 6. Frontmatter and source of truth

Handwritten files own `doc_id`, title, summary, audience, prerequisites, narrative and approved citations. Component manifests own public component identity, requirements and evidence pointers. API extraction owns signatures and parameter schemas. The content build derives routes, parents/children, breadcrumbs, hashes, build IDs and status panels from those sources, rejecting conflicts.

A minimum page record is represented in `docs_manifest.example.json`. The example is a design navigation specimen; it is not a deployed website or evidence of runtime support.

Do not store the same compatibility table in JSON, a Python docstring and handwritten Markdown. Compose generated sections with authored explanations. Editing API defaults causes generated examples/params and schema tests to update together.

## 7. Community docs and upstreaming

A provider bundle identifies its docs root, category mappings, public symbols, sources and conformance records. The main site can show a community catalogue page linking to pinned external docs or ingest a reviewed Markdown snapshot. It MUST distinguish community/first-party maintenance and original/adapted/reproduced evidence.

Upstreaming preserves doc IDs through explicit redirects/aliases and citation history. Previously published results keep links to the source version they measured. A core implementation does not silently inherit an external package's benchmarks or security review.

## 8. Documentation CI

Validate unique routes and IDs; all known internal links; parent/child consistency and absence of cycles; HTML/Markdown route pairing including fragments; version/build identity; generated parameter parity; code-example tests; public import coverage; citations and source-access status; absence of secrets and executable community MDX; native/Python source mapping; and typed evidence findings.

Important semantic content, especially warnings and assumptions, must appear in both representations. Add a semantic-content comparison instead of relying only on an HTML snapshot. The tests included with this design check its JSON navigation/category specimens and canonical route helper only; they do not build Fumadocs or validate the future documentation website.
