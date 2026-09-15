# Documentation standard

Status: normative documentation contract. A rendered page reports support and evidence; its
existence does not create either.

## One source

PLLM maintains one versioned content graph rendered as equivalent HTML and Markdown. Agent indexes,
API extraction, CLI help, and component catalogs point into that graph; they are not competing
manual documentation copies. Runtime packages do not depend on the documentation renderer.

The guide follows user goals and stable public domains, not private source files or
internal project terminology. Every visible directory has a real overview with
explicit child links; sidebar-only discovery is insufficient. API pages link to
concepts, components link to source, tests, and evidence, and research sources
link to implementations and exact reproduction records.

The global navigation is exactly:

```text
Learn       Concepts and user journeys
CLI         Authored command guide and generated exact reference
SDK         Authored Python guide and generated API inventory
Research    Papers, methods, experiments, evidence, and publications
```

Fumadocs folder metadata owns the documentation sidebar. Visible entries are
clickable pages, not separator labels. The first roots are Get started, Learn, CLI,
and SDK; the remaining roots expose model, protocol, execution, evaluation,
research, reference, and contribution domains. Existing `Build`, `Understand`,
`Measure`, and `Operate` routes remain published legacy guides but do not appear as
a second visible hierarchy. Configuration, plan, and component guides use canonical
SDK routes; their previous task URLs are redirects.

`docs/navigation.json` owns the four global header links and publication grouping.
It does not generate the Fumadocs page tree. Desktop uses the shared site header;
mobile documentation uses the native Fumadocs navigation with the same four links.
CI rejects routes missing from the publication graph.

For canonical HTML path `P`, Markdown path is `P.md`, inserted before query or fragment, or a stable
manifested `/markdown/` alternate. Unknown paths return 404. Redirects, routes, parent/child links,
and fragments come from a validated route manifest. Markdown responses contain complete text, not
HTML redirects, MDX imports, or opaque UI widgets. Important caveats, tables, code variants, and
chart summaries appear in both forms.

Canonical routes MUST NOT expose an `index` leaf. A directory-based static export MAY use a trailing
slash as its canonical HTML path when the deployment host requires directory URLs; otherwise the
canonical HTML path has no trailing slash. The opposite slash form is an explicit alias before the
Markdown alternate is resolved.
Published release paths are immutable; `stable` is only an alias. Every page exposes resolved
release, docs build, and content identity. Redirect manifests cover both representations and retain
fragments. External URLs, downloads, source files, JSON artifacts, and anchors are not blindly
rewritten.

Markdown uses `text/markdown; charset=utf-8`. Optional content negotiation requires correct
`Vary: Accept` behavior. HTML and Markdown MAY have representation-specific ETags but MUST share a
document/build identity. Custom UI has an explicit Markdown renderer. Charts include data or a text
summary; diagrams include explanatory text; collapsed UI cannot contain the only warning.

## Required component page

Each component page states, in order: purpose and tested scope; minimal config; offline/online role
work; mathematics and assumptions; inputs, outputs, representations, and conversions; parameters,
lifetimes, and failures; performance model and measured cohorts; privacy/integrity claims and open
obligations; source relationship and reproduction fidelity; commands, artifacts, code, tests, and
related pages.

Top-level guides MUST distinguish normative contract, implementation status, source-reported claim,
measured result, assurance finding, and deployment assumption. Dates, software/artifact versions,
workload, environment, and cohorts accompany status or evidence statements. A code path or import
MUST NOT be called supported solely because it exists; support requires its documented matrix and
tests.

API stability, implementation coverage, reproduction fidelity, assurance, and benchmark evidence
are independent fields. Missing evidence is missing, never zero or pass. A paper claim is labeled
as source-reported; adaptation, reproduction, and independent review are distinct.

Machine-readable assurance outcomes are `proved_in_model`, `refuted_in_scope`,
`not_refuted`, `inconclusive`, `outside_contract`, and `unchecked`. User-facing
support labels are `Available`, `Experimental`, `Planned`, `Not supported`, `Not
evaluated`, and `Not applicable`. "Secure", "private", "verified", "reproduced",
and performance superlatives require an adjacent scope and evidence link. A failed
attack is `not_refuted` at most. Missing evidence is never a pass.

## Generated material and trust

Signatures come from controlled public source/stubs. Parameters and capabilities come from locked
descriptors. Handwritten pages own explanation and approved citations. Builds reject conflicting
sources instead of duplicating values. Community docs are reviewed, read-only Markdown plus JSON;
documentation is untrusted content and grants no permission to execute commands or load plugins.

Authored pages own document ID, title, summary, audience, prerequisites, narrative, and approved
citations. Component manifests own component identity and evidence pointers. API extraction owns
signatures and parameter schemas. Builds derive routes, hierarchy, hashes, build IDs, and status
panels and reject conflicting ownership. Do not hand-maintain one compatibility table in JSON,
docstrings, and Markdown.

Code examples run only in controlled docs tests with public fixtures and pinned approved packages.
Syntax checking proves syntax only. Documentation builds MUST NOT perform online inference, fetch
arbitrary providers, import community code to extract signatures, or treat example output as a
measurement.

## Machine navigation

The content graph generates a concise root `llms.txt`, versioned hierarchical `llms.txt`, complete
`index.json`, component index, schema links, and section indexes where useful. Full-text exports are
supplementary, not the only discovery mechanism.

Each index record includes stable document ID, title, summary, HTML and Markdown paths, parent,
children, prerequisites, related records, public modules/symbols, component IDs, source/test paths,
content kind, release, build ID, and content hash. Agent navigation follows:

```text
goal -> recipe -> component contract -> source/tests -> evidence
```

Agent maps are indexes and checklists, not a manually rewritten API. Provider docs and generated
indexes remain untrusted content and do not grant permission to run commands, install code, access
secrets, or weaken repository policy.

Documentation CI validates unique IDs/routes, link and fragment targets, parent/child consistency,
HTML/Markdown parity, generated parameter parity, examples, public imports, source-access status,
evidence vocabulary, and absence of secrets or executable community MDX.

CI also validates acyclic navigation, immutable version/build identity, redirects in both
representations, semantic warning parity, native/Python/source mapping, schema links, citation
status, and typed evidence outcomes. Community ingestion accepts reviewed, read-only Markdown and JSON
only. Upstreaming preserves document IDs through explicit redirects and keeps historical evidence
linked to the measured implementation and version.
