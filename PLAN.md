---
agent: devin-local
session: cobalt-slip
created: 2026-09-18T02:30:47Z
---
# PLLM deep-dive analysis and prioritized roadmap

> Historical audit snapshot. `COMPONENT_COMPOSITION_PLAN.md` supersedes its
> profile-driven compiler and runtime proposals; profile names below describe the
> repository state observed when this audit was written, not current interfaces.

One-sentence summary: PLLM is an unusually well-governed, honestly-documented, ~10-day-old codebase whose prepared masked-linear runtime works end-to-end today (Qwen2.5-0.5B), whose new compiler/component architecture is deliberately fail-closed and ~2/3 implemented (live coverage report: `rms_norm`, `softmax`, `greedy_token_selection`, `token_feedback` missing; whole-decoder scheduling absent), and whose biggest streamlining needs are (a) finishing exactly one executable profile, (b) unifying `serve`/`gateway`/`benchmark`/SDK onto one model-spec + role-topology core, (c) lowering the existing runtime arms (BFV, proprietary protocols, plaintext backends, tiny models) into the typed component system so old and new compose declaratively — sklearn-style optionality — (d) building the provider/plugin surface the research community is promised, and (e) attacking the 60–430 MB/request client traffic that blocks the VirtualDC vision (whose marketplace/placement layer is explicitly *out* of PLLM's scope — PLLM only emits the capability/evidence vocabulary VirtualDC routes on).

---

## 1. Verification performed (all commands run, results observed)

| Check | Command | Result |
|---|---|---|
| Python tests | `pytest -m 'not he and not sdk'` | **563 passed** (~16s). 28 deselected (he/sdk extras not installed) |
| Rust tests | `cargo test --workspace` | **All pass** — ~90 unit + 7 integration suites across 7 crates, incl. compile-fail doc-tests |
| Clippy | `cargo clippy --workspace --all-targets` | Clean (CI uses `-D clippy::correctness`; CONTRIBUTING asks `-D warnings`) |
| Ruff | `ruff check python/pllm scripts tests` | Clean |
| Docs content check | `node scripts/check-content.mjs` | 147 pages, 0 errors |
| Docs tests | `npm test` (docs/) | **47 pass after `npm run generate`** — failed before regen on stale golden manifest (see §8.3) |
| Generated reference check | `generate_developer_reference.py --check` | **Failed before regen**: `components.mdx` + `python/pllm/index.mdx` were stale vs the staged WIP (see §8.3) |
| CLI smoke | `pllm --version`, `components list`, `config show examples/pllm.yaml`, `lower_model`+`coverage()` | Works; live coverage dump in §4.3 |
| Example resolution | `Experiment.resolve()` on `examples/pllm.yaml`, `examples/benchmarks/*.yaml` | Both resolve |
| **Doc bug found** | `docs/.../start/first-private-request.mdx` tutorial Experiment | **Fails `resolve()`** — omits required `pllm/inference` component (§8.1) |
| **Doc bug found** | `deploy/README.md` says `pllm build` | **Command does not exist** (§8.1) |

Environment: macOS arm64 (NEON active, no AVX2), Python 3.13 venv, Rust 1.85.1 pinned. Working tree contains a **large staged-but-uncommitted WIP** (~17k lines: `dense_qwen_mlp*`, `rms_norm*_protected`, `provenance_primitives`, `boolean*`, `attention.rs`, `kv_cache.rs`, `rope.rs`) — someone/something is mid-implementation on exactly the coverage gaps listed below.

---

## 2. What PLLM actually is (as-built, not as-described)

Two systems share one `pllm.run` distribution (maturin/PyO3, abi3-py311):

**A. The prepared masked-linear runtime — works today.** Three roles: Client (owns plaintext, seeds, activation scales, attention state, sampling), Preparation (trusted; expands seeds→masks `r`,`s`, computes `W·r−s` offline, pushes corrections to Inference over WS, idle online), Inference (untrusted; holds public body, consumes one-use tickets, returns `W·x−s`). HTTP packed prefill + persistent-WS decode; u16/u24/u32 ring selected per stage from exact bounds; token lookup + output head are client-local; memory-only inventory with burn-on-reserve semantics. Validated end-to-end on `Qwen/Qwen2.5-0.5B-Instruct`; retained evidence: 30-token prompt → TTFT ~0.98s, 16-token gen in ~2.6–7.4s online, **63–433 MB client I/O per request** (Apple M5 loopback). Surfaces: `pllm serve inference|preparation`, `pllm gateway` (loopback Responses + Chat Completions APIs, verified against OpenAI SDK 3.8.0, Agents SDK 0.22.2, Codex 0.154, OpenCode 1.18.31 — 17 conformance tests), `pllm benchmark run`, `pllm dev dashboard` (OTLP-instrumented, SQLite history).

**B. The compiler/component research architecture — fail-closed, ~2/3 built.** `lower_model(config)` → immutable `ModelPlan` (semantic IR, 5 adapters: Qwen2, Qwen3, Qwen3.5-text, Phi-4-mini, Gemma4-E2B/E4B text); `plan.apply(ComponentRef)` → digest-bound lineage (only `KvCacheEviction` exists); `plan.coverage(profile)` → honest per-operator report; `pllm.compile(request)` → `CompiledPlan` (logical/execution/lock/region-program artifacts) that can execute only bounded regions: wrap32 linear, checked permutations, residual adds, output head, physical-last token, Q14→Q7 rescale, and experimental one-use Q7 garbling (SiLU ≤128 el; gated-multiply ≤4 lanes; dense-table 2.39 MB vs R03-CRT compact 245 kB single-lane payloads). `coverage().complete` is hardcoded `false` — by design, nothing silently falls back.

**Also present (parallel runtime arms — intended optionality, currently flag-driven rather than componentized; see §5.2):** BFV/TenSEAL HE correlation (`packed_bfv`, `tiled_bfv`, `bfv_correlations`), proprietary-weight protocols (`guarded_engine`, `blinded_engine`, `proprietary_engine`/direct-FHE, `secure_transformer`), a two-party secret-sharing MPC substrate (`shared_*`, 8 modules, ~4k lines, disconnected from the serving path), backend adapters for plaintext reference arms (`vllm`/`ollama`/`llama.cpp`/`mlx-lm`/generic-OpenAI), and `chat.py` (interactive REPL, orphaned). **Removed-by-decision:** `market.py`/`market_server.py`/`provider.py` (CentralMarket/QuoteBook/provider-agent prototype) — placement/offers/settlement belongs to VirtualDC, not PLLM (§5.2, P1.8).

---

## 3. What's done well — genuinely strong

### 3.1 Claims discipline (the project's superpower)
Every surface — README, docs, paper, schemas, code — separates *support claims*: adapter-lowering ≠ checkpoint-import ≠ compiler-coverage ≠ runtime-execution ≠ quality ≠ privacy-evidence ≠ deployment. The status page (`/sdk/reference/status/`) uses 6 explicit labels; `CapabilityStatus` component enforces vocabulary; the paper's IMPLEMENTATION-STATUS.md is candid; `decoder_coverage` reports `complete:false` forever until genuinely closed. For a privacy product this honesty is the moat — keep it sacred.

### 3.2 Design governance
`design/` is a real normative contract (MUST/MUST NOT/SHOULD, authority ordering, acceptance gates). `schemas/` holds 24 versioned JSON-Schema records. Research registry (`docs/data/research/papers.json` + 24 note files + source locks with sha256) generates the papers catalog — data-driven, not hand-maintained MDX. Repo tests enforce: GitHub Actions SHA-pinned, no `pull_request_target`, OIDC-only PyPI publish, no `PYPI_TOKEN`, wheel-contents checks, single-namespace packaging, mkdocs-absence, docs-example `ast.parse`.

### 3.3 Code quality in the core path
- `configuration.py`: frozen dataclasses, canonical JSON digests (`pllm.configuration.v1\0` domain separation), duplicate-key rejection in JSON *and* YAML, 1 MiB doc cap, cycle detection, `__`-path `with_params`, strict `_fields` validation.
- Rust: no TODOs anywhere; `fail-closed` is the idiom; `zeroize` on labels/secrets; redacted `Debug`; `try_reserve` for allocation safety; Rayon pool per executor; runtime AVX2/NEON dispatch; `py.detach` around every native call (GIL released); immutable `Matrix` ownership.
- PyO3 boundary is byte-oriented and coarse — matches the design rule "no Python callbacks per scalar/gate/label".
- CI: 3×Python matrix, 5-target wheel matrix (incl. zig/manylinux), wheel smoke test in a clean venv, `twine check --strict`, docs build+typecheck+content+tests.

### 3.4 Documentation surface
147 pages; honest landing page; 4 top areas (Learn, CLI, SDK, Research) with generated CLI references, generated Python inventory, generated component inventory, per-paper pages with provenance/status/boundaries, clean-room workflow, reproduction checklist, agent implementation map. Versioned `releases/0.1.0` publication manifest with per-page content hashes; llms.txt/llms-full.txt for agent consumption; markdown mirrors of every page.

### 3.5 Benchmark/evidence plumbing
`benchmark run` = real 3-role loopback with sanitized JSON out (no prompts/tokens/seeds in reports or SQLite archive; matrix grouping only by identical model fingerprint+workload+warm state; `--save-best` writes canonical Experiment). `pllm-bench` produces canonical `pllm.benchmark_report.v1` with scalar-oracle cross-checks (`OracleMismatch` = benchmark failure, not a warning). `pllm-assurance` runs weakened-fixture negative controls and prints its own limitations.

---

## 4. Current state vs. the mission

Mission: composable SDK for private-inference research → multi-party pipelines → benchmark CPU/Memory/Network/Disk/privacy → deploy preparation as trusted SaaS/enterprise over untrusted inference (VirtualDC home-GPU nodes).

| Objective | Status |
|---|---|
| Modular composable components | **Contract yes, mechanism no.** 9 hardcoded descriptors; `Pipeline.components` accepts arbitrary IDs but only builtins are typed; **no `pllm.providers.v1` entry-point loading** (designed, unimplemented); no `pllm-plugin.json`; no native ABI (honestly documented as absent) |
| Compose into pipelines | `Pipeline`/`Experiment` immutability + digests work; `baseline.masked_linear_cpu` is the only resolvable profile; `research.single_evaluator` exists as coverage target but resolution is incomplete |
| Benchmark vs objectives | Latency/bytes/rows instrumented; **Memory/Disk/Energy/cost metrics absent** (network is the only measured resource; CPU partially via threads knob) |
| Prompts never leave client | Achieved for the masked-linear path under honest-but-curious + non-collusion; client-local attention/norm/softmax/sampling makes this credible |
| Untrusted inference providers | `research.verified_masked_linear_cpu` verifies every prepared public linear result with authenticated one-use Freivalds material; the default baseline remains unverified, and no TEE, attestation, or arbitrary-malicious-provider claim follows |
| Trusted preparation SaaS | `serve preparation` exists + systemd unit + env templates; no HA, no tenancy, no durable material store (in-memory only), no rate/quota, no TLS provisioning |
| Performance | Rust kernels fine; **traffic is the wall**: 63–433 MB/request for 0.5B — unusable on home uplinks; decode-per-token ticket+masked-vector round trips add WAN latency |

---

## 5. Gaps, unfinished work, and defects (catalogued)

### 5.1 The big one — compiler coverage & whole-plan execution
Live `coverage("research.single_evaluator")` on Qwen2:

| Operator | Level | Blocker |
|---|---|---|
| token_lookup, rope, kv_cache_append, attention_scores/scale/values, causal_mask, silu, multiply | primitive | reference exists; no distributed executor |
| reshape, linear, residual_add, output_head, last_token | executable_region | whole-decoder scheduling unavailable |
| **rms_norm, softmax, greedy_token_selection, token_feedback** | **missing** | no reviewed decomposition |

The staged WIP adds `rms_norm*`/`attention`/`rope`/`kv_cache` primitives — i.e., someone is closing this gap now. Still absent after WIP: **whole-decoder scheduling** (regions exist but no executor chains them into a plan-locked run), `softmax`, token selection/feedback, length-aware last-token. Until one profile is `complete:true`, the SDK story ("compose methods → compile → benchmark") cannot produce a single end-to-end private generation, and every paper reproduction is a leaf, not a pipeline.

### 5.2 Multiple runtime stacks exist as *unintegrated components*, not legacy (streamlining target #1)

**User-corrected framing (decision): these stacks are intended optionality — sklearn/keras/deepeval-style composable arms — not dead code.** The defect is that optionality lives in imperative kwargs/env/flags (`OpenAI(privacy_mode=, correlation_mode=)`, `GatewayConfig(proprietary_protocol=, tenseal_path=, backends=)`, `serve --protocol`, `--tiny`, `PLLM_KERNEL_BACKEND`) instead of typed `Pipeline.components`. The work is declaration+dispatch (descriptors, slots, engine-seam conformance), not deletion or reimplementation.

- *Current arm*: masked-linear prepared inventory path (`transformer_client/engine`, `preprocessing_inventory`, `preparation_*`, `stage_protocol`, `correction_*`).
- *HE-correlation arm*: `packed_bfv`, `tiled_bfv`, `bfv_correlations`, `he_authenticated_preprocessing` — alternative `preparation-provider` behind the `he` extra; `correlation_mode="bfv"|"local-test"` and a `POST /correlations/bfv` endpoint already exist → cheapest second arm to componentize.
- *Proprietary-weights arm*: `guarded_engine`, `blinded_engine`, `proprietary_engine`/direct-FHE, `secure_transformer`, `authenticated_mpc`, `linear_integrity` — different privacy contract (model-confidential), `ProprietaryProtocol` enum is already a proto-taxonomy; keep composable, mark experimental, don't polish ahead of need (VirtualDC's first market is open weights).
- *2PC MPC substrate*: `shared_*` (~4k lines, different party topology; `InferenceEngine.execute_stage` seam exists) — salvage `BurnLedger` + durable SQLite material store (needed for prep durability / VirtualDC churn), then decide whether it earns a `research.shared_mpc_2pc` topology arm.
- *Reference arms*: `backends/` adapters (vllm/ollama/llama.cpp/mlx-lm) + `masked_runtime` bigram + `engine.py` protocols — plaintext/naive baselines; componentize as `cleartext-reference`/`naive` arms so benchmark comparisons get an honest denominator.
- *Model-source components*: `tiny_gemma`/`tiny_llama`/`synthetic` — `--tiny` becomes a declarative `pllm/tiny-*` model-source arm.
- *Out of scope for PLLM (user decision)*: `market.py`/`market_server.py`/`provider.py` — placement/offers/settlement is **VirtualDC's layer**; remove the three files + their tests in `test_product.py` (nothing else references them; ~600 LOC). PLLM keeps only the *vocabulary* VirtualDC routes on: component capabilities, `required_host_features`, resolved-plan role requirements, `measurement.v1` evidence digests.
- *Orphan shells*: `chat.py` (no entry point), `runtime/cli.py` (dashboard/benchmark subprocess shim — fold into `_cli`), `responses.py`/`official.py` overlap, dead `python/pllm/models/` dir (only `__pycache__`).

Cost of status quo: ~40% of `python/pllm/runtime` is non-public-path code reachable only via flags; contributors can't tell which stack a paper should land on; no arm can be selected declaratively in an Experiment.

### 5.2b `serve`/`gateway`/`benchmark`/SDK model+serving paths are 4 separate topologies (streamlining target #2)

The user's requirement: `pllm benchmark`, `pllm serve`, and the SDK must share underlying components; SDK must be able to load a model and serve it; HF pull is one path.

Verified seams:
- **Loaders are already unified at the bottom**: `runtime/hf_hub.resolve_huggingface_source` (prefers `huggingface_hub.snapshot_download` when installed → falls back to `hf_download` custom parallel ranged downloader) → `runtime/loaders.py` (`load_hf_directory`/`load_mlx_directory`/`load_gguf`) → `ModelManifest` → `runtime/models.py` `transformer_stage_plan`/`gemma4_stage_plan` → `TransformerEngine`. One loader stack serves hf/mlx/gguf.
- **Model spec is fragmented at the top**: (a) CLI flags `--model`/`--model-id`/`--revision`/`--hf-cache-dir`/`--tiny` on serve/gateway/dashboard/benchmark; (b) HTTP `PUT /v1/runtime/models` body `{model_id, path|hf_model|mlx_path|gguf_path, revision}` (multi-tenant dynamic load — good); (c) declarative `Pipeline.model: ModelSpec` (used only when `--experiment-config` is passed to benchmark); (d) Rust `pllm-models` `ModelSpec` for semantic lowering — a second HF-config parser that must converge with `loaders.py`/`transformer_stage_plan` (two manifests, two stage-plan sources of truth).
- **Topology is re-implemented 4×**: `serve` (single roles via `create_app`/`create_preparation_app`), `gateway` (in-process trio in `http_gateway.py`), `dev dashboard` (spawns 3 role subprocesses via `_spawn`), `benchmark run` (spawns `dev dashboard` as a subprocess then drives it over HTTP — a benchmark → dashboard-subprocess → 3-role-subprocesses nesting, ~4 process layers for what should be `build_roles(plan) → attach telemetry → drive workload`).
- **No public `pllm.serve*`/`pllm.load_model` API**: SDK consumers get `create_app`/`create_preparation_app` (raw FastAPI factories) + must assemble uvicorn themselves; the CLI's role-orchestration logic is private to `_cli/app.py`/`dashboard.py`. `create_openai_client` factories exist but only *consume* endpoints.
- `runtime/cli.py` bypasses the hub-aware `hf_hub` wrapper and calls `hf_download` directly — minor layering leak.

Alignment target (one core, three fronts):
- `pllm.Model` = the single typed spec (`hf`, `path`, `mlx`, `gguf`, `tiny`, `bundle` kinds) — unifying `ModelRef`/`ModelSpec`; CLI flags are constructors of it, the HTTP body is its serde, `Pipeline.model` already is it.
- `pllm.load_model(ref) -> ModelManifest` public; `resolve_huggingface_source` is the only HF-pull entry.
- `pllm.serve_inference(config)`/`pllm.serve_preparation(config)`/`pllm.serve_local(experiment|model|resolved_plan)` → handle objects (urls, health, `close()`); `serve`/`gateway`/`dashboard`/`benchmark run` all build on the same `build_roles(plan)`/`LocalTopology` — gateway = serve_local + OpenAI-compat HTTP front; dashboard = serve_local + UI + OTLP; benchmark = serve_local + telemetry + workload driver.
- `OpenAI(base_url=handle.url)` stays the consuming client.

### 5.3 Component system is a façade, not an ecosystem
- `_BUILTIN_DESCRIPTORS` + `_component_from_spec` hardcode 9 components in one file — adding a component = edit 3 places in `configuration.py` + Rust `resolve_experiment` validation + regenerate 2 doc inventories.
- `pllm/inference` is required by `baseline` profile but has **no descriptor/class** — invisible in `components list`.
- No external-provider discovery (`importlib.metadata` entry points never read), no manifest schema enforcement, no parameter-schema validation against descriptors (a `ComponentRef("pllm/cpu", {"threads": "x"})` survives Python-side until Rust resolution).
- `Plan.apply()` supports exactly one transformation kind (model-graph); no conversion/kernel/preparation/metric application surface.
- ComponentDescriptor fields `artifacts`/`evidence` exist but are always empty — the generated components.mdx is a table of "not recorded".

### 5.4 Schemas are canonical but unenforced
`schemas/*.schema.json` (24) are validated only by fixtures in one CLI test. Nothing checks that `pllm.compile` accepts exactly `compile-request.schema.json`, that `decoder_plan.v1` emitted by Rust matches `decoder-plan.schema.json`, or that `benchmark_report.v1` matches `measurement/evidence-bundle` schemas. Schema↔code drift will silently happen.

### 5.5 Documentation defects found (concrete, fixable)
1. `docs/content/docs/start/first-private-request.mdx` — tutorial Experiment **fails `resolve()`** (missing `inference` component). First-run breakage.
2. `deploy/README.md` — tells operators to run `pllm build` (nonexistent).
3. `design/cli.md` status table stale — lists `serve`, `benchmark run` as "Unavailable" (they're shipped; `reference/status.mdx` is correct).
4. `ARCHITECTURE.md` — documents only 3 crates (`pllm-core`, `pllm-python`, `pllm-models`); 5 others (`types`, `compiler`, `bench`, `assurance`, `garble`) undocumented there.
5. README mentions "historical evidence under `research/`" — dir doesn't exist (it's `docs/evidence`, `docs/data/research`); ruff `extend-exclude = ["research"]` also points at a nonexistent dir.
6. CONTRIBUTING docs flow says `npm install` + `bun run dev`; README says `bun install --frozen-lockfile` but only `package-lock.json` is committed — pick one tool.
7. `docs/content/docs/reference/status.mdx` "Checked 16 September 2026" — manual freshness date, already needs the chunked-lanes row (regenerating fixed it).
8. Nav/source split (`docs-routes.mjs` EXACT+PREFIX route maps) is clever but a second routing layer to maintain; several `EXACT_ROUTES` entries exist only to rename slugs — a lint rule mapping content-dir → canonical URL would reduce drift.

### 5.6 Testing/measurement gaps
- `integration`/`slow` pytest markers declared but **unused** (dead markers).
- No WAN/latency injection, no concurrent-client tests, no provider-fault-injection (wrong-answer detection), no memory/disk/energy measurement harness — the benchmark counts only latency/traffic/rows.
- `results/*.xml`, `.coverage`, `.pytest_cache` litter (ignored, fine) but signal ad-hoc local runs rather than a results pipeline.
- `he`/`sdk` extras untested locally (deselected); fine in CI.
- OpenResponses contract pinned to `2026-04-24`; conformance tests exist but no matrix of *gateway×backend×protocol* behaviors.

### 5.7 Security/ops hardening gaps (for the stated trust model)
- All auth is bearer API keys; no mTLS, no provider identity/attestation, no signed manifests; WS push auth = one shared key.
- `verify correct execution` = an opt-in full-model trusted-client adaptation now authenticates preparation projections and verifies bounded integer `W·x` results before dequantization; the baseline profile remains intentionally unchanged.
- Preparation is Python — masks/seed material in GC memory; "physical zeroization … has not been established" (paper admits). For a SaaS prep service this matters; consider Rust-side material handling or mlock/secure enclaves later.
- Inventory in memory only — restart loses all prepared rows (documented); no spill-to-disk with AEAD for node churn on VirtualDC.
- `market.py` auth model is a single admin key; ProviderOffer `input_privacy` is unverified self-attestation — **resolved by removal** (P1.8: placement is VirtualDC's layer; node claims must instead be backed by signed `measurement.v1` evidence, see P4.18).

### 5.8 DX friction
- First `uv sync` requires Rust toolchain + maturin build (~minutes); `PLLM_REQUIRE_RUST` semantics undocumented in README dev section.
- 3 doc generators run by 3 commands (`generate_developer_reference.py`, `generate-research-papers.mjs`, `generate.mjs`+`discover-pages.mjs`) — a single `docs:regen` alias would have prevented today's stale-golden failure.
- `.pyi` stubs cover only 6 modules; `pllm.runtime` has no stub (users get `Any`); no `mypy`/`pyright` in CI.
- `examples/` mixes config demos with live-network scripts; no `examples/README` index; no `first-request` runnable script matching the tutorial.
- Error taxonomy is good (`UsageError`, `ResolutionError`, `LocalIOError`) but `-m pllm` errors still print argparse usage for `pllm.runtime.cli` paths — two CLI surfaces with different error styles.
- `config/gateway.example.json`/`gateway.compose.json` describe backend passthrough (vllm/ollama/llama.cpp/mlx) — looks like a *plaintext* gateway config and will confuse newcomers about what PLLM is for.

### 5.9 Performance-relevant gaps (for VirtualDC economics)
- **Traffic**: 63–433 MB client↔inference per request (mask+correction transport of every body stage). Preston/Slalom-style techniques (Freivalds verify, LPN mask reuse-with-verification, R24 Maverick) target exactly this; currently unimplemented.
- Client-side attention/norm/softmax in NumPy = the "client-heavy" tax; fine for privacy but CPU-bound on the client device.
- `Matrix` retains source + int8 copy ("one extra signed byte per weight" documented); bundle schema-2 dedup helps.
- No GPU kernel path (`capabilities()["gpu"]=False`); VirtualDC's pitch is idle gaming GPUs — no wgpu/CUDA crate exists yet.
- No disk metrics for bundle cache/inventory; inventory is memory-only by design.

---

## 6. Prioritized roadmap

Ordering rationale: the mission-critical path is **one complete plan-compiled profile** (everything else — search, evidence, multi-arm comparisons — hangs off it), then **unification** (one model-spec + role-topology core; existing arms lowered into components), then **the provider surface** (plugins + conformance kits), then **VirtualDC economics** (traffic ↓, verification, the requirements/capability contract VirtualDC routes on), then **ecosystem** (more papers, CLI breadth).

### P0 — Land one executable profile + fix broken docs (days)
1. **Finish the in-flight WIP** (rms_norm/rope/kv/attention primitives), then implement `softmax`, `greedy_token_selection`, `token_feedback`, length-aware `last_token`, and — the actual gap — **whole-decoder scheduling** that chains regions under a `PlanLock`. Target: `coverage().complete == true` for dense Qwen2 on `research.single_evaluator`, executed end-to-end on the real 0.5B checkpoint, with the coverage report generated from code (not hardcoded false).
2. **Fix the two confirmed doc bugs**: add `inference` component to `first-private-request.mdx` (and add a `pllm`-level runnable `examples/first_request.py` that the test suite `exec()`s so this can't regress — docs currently only `ast.parse` snippets); replace `pllm build` in `deploy/README.md` with the real verify step.
3. **Refresh stale docs**: `ARCHITECTURE.md` crate table (8 crates), `design/cli.md` status table (or generate it from `_cli/app.py` like the public reference), README `research/` → `docs/evidence`/`docs/data/research`, ruff exclude.
4. **One regen command**: `scripts/docs_regen.{sh,py}` = `generate_developer_reference.py` + `npm run generate` + `--check` mode; wire into CI and `scripts/release.py check`.

### P1 — Unify serve/benchmark/SDK + componentize the existing arms (1–3 weeks)
5. **One model spec, one loader path**: `pllm.Model` absorbs `ModelRef`/`ModelSpec` as the single typed spec across CLI flags, `PUT /v1/runtime/models` body, `Pipeline.model`, and SDK. Public `pllm.load_model(ref) -> ModelManifest`; `resolve_huggingface_source` is the sole HF-pull entry (stop `runtime/cli.py` calling `hf_download` directly). Converge `loaders.py`/`transformer_stage_plan` with Rust `pllm-models` so stage plans have one source of truth (compiler plan becomes the runtime manifest, or the runtime manifest is generated from it).
6. **One topology builder**: extract `_cli`/`dashboard.py` role-orchestration into `runtime.servers` — `build_roles(plan)`/`LocalTopology` returning handles (urls, health, `close()`). Rebase: `serve` = single role; `gateway` = serve_local + OpenAI-compat HTTP front; `dashboard` = serve_local + UI + OTLP; `benchmark run` = serve_local + telemetry + workload driver (kills the 4-layer subprocess nesting). Public SDK: `pllm.serve_inference(config)`, `pllm.serve_preparation(config)`, `pllm.serve_local(experiment|model|resolved_plan)` → handles; `OpenAI(base_url=handle.url)` consumes.
7. **Componentize the existing arms (sklearn-style optionality, not quarantine)**: descriptors + profile slots + params for: BFV-correlation prep (`preparation-provider`, `required_host_features=("tenseal-backend",)`); guarded/blinded/secure/direct proprietary protocols (`protocol-method`, `experimental`, with `corruption_model` declared); plaintext backends + `masked_runtime` (`cleartext-reference`/`naive` arms — the benchmark denominator); `tiny_*`/`synthetic` (`model-source`); `pllm/inference` (required-but-undescribed today). Map today's flags/kwargs (`privacy_mode`, `correlation_mode`, `--protocol`, `--tiny`, `PLLM_KERNEL_BACKEND`) → component params or presets that emit a `Pipeline`. **Class-layer reorg (user requirement)**: move component classes out of `configuration.py` into top-level family modules (`pllm.protocols`, `pllm.preparation`, `pllm.kernels`, `pllm.nonlinear`, `pllm.schedulers`, … — flattened per §7.5 decision; `pllm.components` keeps only generics/machinery); complete class coverage (`Inference`, `BFVCorrelations`, `Guarded*`, `CleartextLinear`, `TinyModel`, `LinearIntegrity`); `pllm.components.get(identity) -> class` resolver; derive `_BUILTIN_DESCRIPTORS` from the classes' `describe()` (they're already per-class — remove the parallel hand-maintained dict); `Pipeline.components` accepts instances, strings remain the serialized form via `to_spec()`/`from_spec()`.
8. **Remove market subsystem from PLLM (decided)**: delete `market.py`, `market_server.py`, `provider.py` + market/provider tests in `test_product.py` (~600 LOC, zero other references). Placement/offers/settlement belongs to VirtualDC. PLLM retains the routing *vocabulary*: component capabilities, `required_host_features`, resolved-plan role requirements, `measurement.v1` evidence digests — VirtualDC queries these, doesn't get a service from PLLM.
9. **Unify CLI/service entry**: fold `runtime/cli.py` into `_cli` internals (dashboard/benchmark call the role builder directly, not `python -m pllm`); one error taxonomy.
10. **Category-driven `resolve_experiment` + threat-model fields + class-first authoring**: profiles become `pllm.profiles.*` `Pipeline` subclasses whose `__init__` signature IS the slot contract — each slot a typed kwarg annotated with its **category ABC** (`ProtocolMethod`, `PreparationProvider`, `KernelBackend`, `InferenceRole`, …; one ABC per `pllm/<category>` in its family module — the sklearn-mixin analogue); `isinstance` checks fire at construction (IDE + mypy catch wrong-category arms), descriptor checks at `resolve()`. Slot names (`baseline.masked_linear_cpu` → `MaskedLinearCpu`, `reference.plaintext` → `PlaintextRef`, `research.single_evaluator` → `SingleEvaluator`, +`BfvPrepared`/`ProprietaryGuarded`/`TwoPartyMpc`/research cohorts) remain identity strings for serde/digest only. `with_params` takes slot kwargs (`preparation=BFVCorrelations(...)`) and sklearn `slot__param` nested syntax. Add `corruption_model`/`party_topology`/`produces`/`consumes`/`requires_trusted_dealer` to descriptors; compiler rejects topologically/threat-incompatible compositions. Validate `ComponentRef` params vs `parameter_schema` at construction.

### P2 — Provider surface for the research community (2–4 weeks)
11. **Implement PCS-1 discovery**: `pllm.providers.v1` entry-point loading for `pllm-plugin.json` descriptors (inert manifest only — no code import), merged into `list_components()`/`get_component()` with duplicate-ID failure; version the descriptor schema against `schemas/component-descriptor.schema.json`. Third-party components sit beside builtins identically (the sklearn third-party-estimator story).
12. **Component authoring kit + conformance suites**: `pllm new component <category>` scaffolder (stubs + descriptor + conformance skeleton); per-category contract tests (the `check_estimator` analogue — `pllm/protocol-method` = masked stage round-trip + burn-once + freshness + malformed-input rejection; `pllm-assurance` crate is the seed). **Conformance kits matter more than adding components** — every new arm multiplies the matrix.
13. **Enforce schemas in CI**: golden test asserting Rust-emitted `decoder_plan.v1`, `benchmark_report.v1`, `assurance_report.v1`, `region_program.v1` validate against `schemas/*` (one parametrized test; `jsonschema` already present).
14. **Type surface**: `mypy --strict` (or pyright) on `pllm`'s public modules; complete `.pyi` coverage (generate `_native.pyi` from PyO3 stubs); ship `py.typed` (already shipped — keep).

### P3 — VirtualDC economics: traffic, trust, fleet ops (4–8 weeks)
15. **Attack client traffic** (the metric that makes VirtualDC viable or not): **[partial]** the R07-derived Freivalds verifier is implemented end to end as `research.verified_masked_linear_cpu`, with matched tiny functionality evidence and real-Qwen evidence unavailable. It verifies correctness but does not reduce online traffic. R24-Maverick LPN masking/batch verification remains the traffic-reduction work.
16. **Measure the real objectives**: extend `benchmark run` to record peak RSS (ru_maxrss/psutil), bundle-cache disk bytes, correction-store memory, per-stage kernel ms, and (optional) `powermetrics`/RAPL energy — produce a `pllm.measurement.v1` row per objective; add WAN emulation (tc/netem or a delay-injecting transport shim) + a 2-host docker-compose topology so "loopback ≠ WAN" gets real numbers.
17. **Provider integrity**: wire `linear_integrity` (exists, orphaned) or the new verify component into the prepared path so a wrong `W·x−s` is *detected*, not just undocumented; signed model manifests + checkpoint-digest pinning in `serve` (bundle fingerprint already exists — surface it in `pllm serve` output + dashboard).
18. **Preparation as a service**: durable (AEAD, rollback-protected) material store option; multi-inventory scheduling; rate limits/quota; health/metrics endpoints; a `deploy/kubernetes/` or `deploy/systemd` HA pair; document SaaS-vs-enterprise placement matrix (both exist, both need the same "sees no prompts" proof — that IS the marketable claim).

### P4 — PLLM→VirtualDC interface contract + node runtime (8+ weeks, parallel with P3)

Placement/marketplace is VirtualDC's layer — PLLM's job is to *emit* what a higher-level router needs, not to route.

19. **Capability & requirements export**: `pllm capabilities` (host features this node can serve: kernel-backend, tenseal-present, gpu, memory, arch) + resolved-plan `role_requirements()` (roles, per-role component IDs, host features, model fingerprint, material lifetimes) — the data contract a VirtualDC scheduler consumes. Design note: `design/` contract for "what a plan requires" vs "what a node offers" — this is the seam between PLLM and VirtualDC.
20. **Evidence for routing decisions**: `measurement.v1` records carry enough to price/schedule (per-role CPU/RSS/bytes/energy, region-tagged) — VirtualDC's marketplace maps offers↔requirements on these digests; signed evidence bundles so a node's self-reported capability is backed by an assurance record, not a boolean (`input_privacy: true` self-attestation dies with market.py).
21. **Node runtime readiness** (what PLLM must be for a VirtualDC node): `wgpu` kernel-backend component (portable to Metal/Vulkan for gaming PCs — `capabilities()["gpu"]=False` today); durable AEAD material store (salvage `shared_*`'s SQLite BurnLedger) so node churn ≠ total inventory loss; `pllm serve` installer profiles (systemd exists → launchd + `pllm-node` packaging); attestation hooks in `capabilities()` output.
22. **Research backlog execution** in registry order: R01–R03 fidelity completion, R18 activation approximation (cheap win — replaces garbled SiLU cost), R05/R06 conversions, R23 protected cache (structural transform exists), then full-system comparisons R12–R22 in isolated adapters — each lands as a component in the unified taxonomy from P1/P2, so every paper is benchmarkable against every other arm.

### P5 — Community/polish (continuous)
23. `pllm init`/`--set`, `model lower`, `plan check|compile|show`, `prepare/run/chat`, `benchmark search|compare`, `assure run` — the CLI's own documented-but-`Not supported` surface (status page already lists them; they're the natural contribution on-ramp).
24. Delete or finish: `chat.py` entry point (wire as `pllm dev chat` or drop), dead `python/pllm/models/` dir, `results/` lane conventions, unused pytest markers, `config/*.json` backend examples (move under a clearly-labeled `reference/` dir or document as plaintext-passthrough dev tooling).
25. Windows/macOS dev docs parity; `CONTRIBUTING` vs `README` tool unification (bun vs npm).

### P6 — Publications & docs (parallel with P1–P2)
26. **arXiv pre-print revision** of `paper/manuscript.md` → "PLLM: A High-Performance Private LLM Multi-Party Inference Runtime and Autonomous Research Harness" — restructured intro (private-inference opportunity → research fragmentation → sklearn/keras toolkit precedent → PLLM), new background/related-work section citing **all 24 tracked papers** mapped to component families, new component-model section, expanded `references.bib` (currently 17 entries, only `ball2017garbling` from the R-corpus). Full amendment spec in Appendix D.1. Constraint: **no VirtualDC mentions** (already true — keep it).
27. **2-page business whitepaper rebuild** of `paper/whitepaper.md` — currently ~3pp with LaTeX math and no figures; target is jargon-minimized prose + 4 diagrams (three-role data-flow, privacy-boundary map, component-slot swap, research loop), honest "status today" box. Spec in Appendix D.2.
28. **Docs-site consolidation + gap fill** — remove 3 empty content roots (`server/`, `components/`, `client/`), consolidate ~13 index-only sections (<20-line stubs: `recipes`, `search`, `representations`, `conversions`, `metrics`, `kernels`, `assurance`, `deployment`, `pipeline`, `runtime`, `preparation`, `compiler`, `numerics`), fix duplicated paragraph in `recipes/index.mdx`, document the `docs-routes.mjs` prefix map (file path ≠ public URL is a contributor trap), ship the §9 glossary as a docs page, and fill plan-driven gaps (component-authoring guide, capability/evidence-export reference, env-var reference). Full audit in Appendix D.3.

---

## 7. Target design — old → new structure

This section is the design for the P0–P4 changes: how the pieces fit together today vs. the target, with SDK examples for each surface. Names below (`serve_local`, `build_roles`, `role_requirements`, `serve_inference`/`serve_preparation`, `benchmark_run`, `load_model`, profile classes, new component IDs) are proposed; existing names (`Model`, `Pipeline`, `ComponentRef`, `Experiment`, `resolve()`, `OpenAI`, `create_app`) are as-shipped.

### 7.1 The spine — one core, many fronts

**Today (as-built): model spec has 4 shapes; topology is implemented 4×; the SDK has no load/serve surface.**

```
model spec (4 independent shapes)
┌──────────────────┬───────────────────────┬─────────────────────┬──────────────────┐
│ CLI flags        │ HTTP body             │ Pipeline.model      │ Rust ModelSpec   │
│ --model/--tiny   │ PUT /v1/runtime/      │ Model(source)       │ pllm-models      │
│ --model-id       │ models {hf_model|     │ (read only by       │ (2nd HF config   │
│ --revision       │  path|mlx|gguf,       │  benchmark          │  parser — must   │
│ --hf-cache-dir   │  revision}            │  --experiment)      │  converge)       │
└────────┬─────────┴───────────┬───────────┴──────────┬──────────┴────────┬─────────┘
         │                     │                      │                   │
         └──────────────► hf_hub.resolve_huggingface_source ──► hf_download (fallback)
                                      │
                          loaders.py: hf/mlx/gguf → ModelManifest
                                      │  runtime/models.py: *_stage_plan
                                      ▼
                          TransformerEngine (inference role)

topology (4 parallel implementations of "client + preparation + inference")
serve ────────────────► create_app / create_preparation_app   (one role each)
gateway ──────────────► http_gateway.py                       (in-process trio + HTTP front)
dev dashboard ────────► dashboard.py                          (3 role subprocesses + UI + OTLP)
benchmark run ────────► benchmark_cli.py                      (subprocess→dashboard→3 roles; 4 layers)
SDK ──────────────────► OpenAI(privacy_mode=, correlation_mode=)  (arm selection by kwargs;
                                                                 no load_model / no serve)
```

**Target: typed specs in → resolved plan → one role builder → fronts are thin.**

```
inputs — three fronts, one typed spec
┌──────────────┬─────────────────────┬──────────────────────────┐
│ CLI flags    │ YAML/JSON spec      │ SDK objects              │
│ (constructors│ Experiment          │ Model, Pipeline,         │
│  of specs)   │ (serde of same)     │ ComponentRef             │
└──────┬───────┴──────────┬──────────┴────────────┬─────────────┘
       └──────────────────┴──── Model ────────────┘
                  hf:// path mlx gguf tiny bundle

                    Experiment { name,
                      pipeline: Pipeline { profile,
                        model: Model,
                        components: {slot → ComponentRef} },
                      deployment, budget }
                                 │ resolve()
                    category-driven: slot→category match,
                    params vs parameter_schema,
                    corruption_model/party_topology checks
                                 ▼
            ExperimentProfile { roles, topology, requirements,
                           privacy_contract, model_fingerprint }
                                 │ build_roles(plan)      ← runtime.servers
            ┌────────────────────┼──────────────────────────┐
            │                    │                          │
   serve_inference/       serve_local(exp|model|plan) role_requirements()
   serve_preparation      → LocalTopology            capabilities()
   (single-role handles)     {handles}               → VirtualDC (P4.19)
                                 │
        ┌────────────────┬───────┼───────────┬───────────────┐
        │                │       │           │               │
     gateway          dashboard  benchmark   SDK             tests
   = serve_local    = serve_   = serve_    OpenAI(base_url=
   + HTTP front      local+UI  local+       topo.gateway_url)
                     +OTLP     telemetry
                               +driver
```

Rules: **flags are constructors of specs; specs serde to YAML/HTTP; resolution produces roles; roles are built once. Components and profiles are Python classes — `pllm/…` and `baseline.*`/`research.*` strings are serialization, not authoring; slots are typed kwargs** (sklearn/keras pattern; §7.2). `OpenAI` stays the consuming client — it never selects arms.

### 7.2 Component map — old arm → new component

**Authoring rule (user requirement — sklearn/keras pattern): components *and profiles* are Python classes; identity strings are serialization only.** A profile is a `Pipeline` subclass whose `__init__` signature **is** the slot contract — every slot is a typed kwarg annotated with its category ABC, so `MaskedLinearCpu(model=…, linear=MaskedLinear(), preparation=ModelAwareCorrections(), inference=Inference(), kernels=Cpu(threads=8))` gives IDE completion for every slot *and* fails at construction (and statically, via mypy/pyright) when a wrong-category component is passed. `components={…}` dicts and `pllm/…`/`baseline.*` strings remain the **wire/digest form** (`to_spec()` output, YAML, `from_spec()`) — never the authoring form — exactly as keras serializes `Dense` to `"keras.layers.Dense"` + config while you author `Dense(64)`.

The sklearn mapping is exact and deliberate:

| PLLM | sklearn/keras analogue | Purpose |
|---|---|---|
| `ComponentRef` | `BaseEstimator` | universal base — `get_params`/`with_params`/`to_spec`/`describe`/`component_id` |
| category ABCs — `ProtocolMethod`, `PreparationProvider`, `KernelBackend`, `InferenceRole`, `CorrelationSource`, `NonlinearProtocol`, `LabelScheme`, `StateProtocol`, `VerificationScheme`, `Conversion`, `NumericTransform`, `PlanPass`, `Codec`, `Adversary`, `Metric`, `ModelSource`, `ProtectedScheduler` | mixins — `TransformerMixin`, `ClassifierMixin`, … | one ABC per `pllm/<category>` living in its family module; slot annotations type against these |
| `pllm.profiles.*` profile classes — `MaskedLinearCpu`, `PlaintextRef`, `SingleEvaluator`, `BfvPrepared`, `ProprietaryGuarded`, `NexusHe`, `Bumblebee2pc`, `OpenWeightVerified` | `keras.Sequential` / specialized `Pipeline` subclasses | `__init__` signature = slot contract; `profile` attr = identity string; required slots = required kwargs, optional slots = defaults |
| `pipe.with_params(preparation=BFVCorrelations(…))` | `set_params(preparation=…)` | whole-slot swap; returns a clone |
| `pipe.with_params(preparation__poly_degree=8192)` | `set_params(step__param=…)` | nested param tweak — sklearn's `__` convention |
| `Pipeline.from_spec(…)` / subclassing `Pipeline` with a `slots` declaration | `Pipeline(steps=[…])` generic path | YAML-driven and third-party profiles |
| category conformance suite | `check_estimator` | gates composition (P2) |

Class-layer fixes needed today:
- All component classes are crammed in `configuration.py`; `pllm.components` re-exports only **7 of 9** (`Cpu`, `MaskedLinear`, `ModelAwareCorrections` missing — top-level `pllm.*` only). Reorganize into **top-level family modules** (flattened per §7.5 decision): `pllm.protocols`, `pllm.preparation`, `pllm.kernels`, `pllm.nonlinear`, `pllm.correlation`, `pllm.schedulers`, `pllm.conversion`, `pllm.state`, `pllm.verification`, `pllm.sources`, … — `pllm.components` keeps only generics/machinery (ComponentRef, ComponentDescriptor, registry).
- `pllm.components.get(identity: str) -> type[ComponentRef]` — the string→class resolver (keras `layers.get()` analogue); `cls.describe()` already produces the descriptor.
- **Registry derived from classes**: `_BUILTIN_DESCRIPTORS` is a hand-maintained dict parallel to the classes — replace with enumeration over the class registry (`describe()` is already per-class). Class = single source of truth for identity, params, descriptor.
- Every arm gets a class: `Inference` (required-but-undescribed today), `BFVCorrelations`, `GuardedLinear`/`BlindedLinear`/`SecureLinear`/`DirectFHE`, `CleartextLinear`, `TinyModel`/`BundleModel`, `LinearIntegrity`.
- Third-party: `pllm.providers.v1` entry points may ship real classes (trusted plugins — the authoring surface) and/or inert manifests (untrusted enumeration); a class IS the descriptor source so both resolve to the same `ComponentDescriptor`.

| Profile slot (typed kwarg on the profile class) | Category | Python class (target) | Ships today | Arms to componentize (P1.7) |
|---|---|---|---|---|
| `model` *(the Pipeline field — ModelSource-typed, not a component slot)* | `pllm/model-source` | `Model`, `TinyModel`, `BundleModel` | `Model(source)` — HF id string | `pllm/tiny-*` (replaces `--tiny`), `pllm/bundle` (pinned snapshot) |
| `linear` | `pllm/protocol-method` | `MaskedLinear`, `GuardedLinear`, `CleartextLinear`, … | `MaskedLinear` | `pllm/guarded-linear/v1`, `pllm/blinded-*`, `pllm/secure-*`, `pllm/direct-fhe` (proprietary arm); `pllm/cleartext-linear` (`backends/`, reference denominator); `pllm/shared-mpc-2pc` (topology arm — after salvage) |
| `preparation` | `pllm/preparation-provider` | `ModelAwareCorrections`, `BFVCorrelations` | `ModelAwareCorrections` (seeded) | `pllm/bfv-correlations/v1` (HE arm; `required_host_features=("tenseal-backend",)`), `pllm/he-authenticated-preprocessing` |
| `inference` | `pllm/inference-role` | `Inference` | required but **no class/descriptor** | descriptor + conformance |
| `kernels` | `pllm/kernel-backend` | `Cpu`, `Wgpu` | `Cpu` | `pllm/wgpu` (P4.21); `pllm/tenseal` host feature |
| `cache` | `pllm/state-protocol` | `ClientLocalKv`, `SecretSharedKv`, `MpcacheAttention` | client-local KV implicit today | R23 protected-cache variants; **`KvCacheEviction` is a `pllm/compiler-pass`**, applied via `plan.apply()` — it rewrites the plan's cache strategy rather than filling a slot |
| `nonlinear` | `pllm/nonlinear-protocol` | `BinaryTableGatedMultiplyQ7`, `R03CrtGatedMultiplyQ7` | both | R18 activation-approximation arm |
| `schedule` | `pllm/protected-scheduler` | `ScalarProtectedTensorSchedule`, `IndependentLanesProtectedTensorSchedule`, `ChunkedIndependentLanesProtectedTensorSchedule` | all three | — |
| `integrity` | `pllm/verification-scheme` | `LinearIntegrity` | — | `pllm/linear-integrity` (wire the orphaned module into the prepared path, P3.17); `pllm/assurance-check` stays a separate category for harness-level negative-control fixtures |
| `adversary`/`metric` | `pllm/adversary` / `pllm/metric` | fixtures, `Latency`/`Traffic`/`Rss`/`Disk`/`Energy`/`Quality` | — | weakened fixtures; objective metrics (P3.16) |

New descriptor fields to make composition *checked*: `corruption_model`, `party_topology`, `produces`/`consumes` (representation IDs), `requires_trusted_dealer` — compiler rejects a component needing a trusted dealer in a topology without one.

### 7.3 Research-backed component backlog — R01–R24 mapped + novel-to-PLLM

Every tracked paper maps to components in the slot taxonomy; papers are **provenance, not packages** (each contributes components to a stable capability family — the backlog's own rule). Source: `docs/data/research/papers.json` (each paper's `modules` field is its proto-component set) + `docs/content/docs/research/backlog.mdx` dependency order.

Legend: **[Rnn]** = derived from that paper · **[PLLM]** = novel to PLLM (future-paper candidates collected at the end) · *exists* = shipped today · *stack* = implemented code awaiting componentization · *tracked* = paper acquired, no implementation yet.

#### Family `pllm/protocol-method` — linear execution/delegation

| Component | Source | Status | What it contributes |
|---|---|---|---|
| `pllm/masked-linear` | **[PLLM]** (Slalom-adjacent) | *exists* | Seeded masked-linear over exact-bounds rings; PLLM's trusted-client/dealer adaptation — R07 motivates the family, the seeded variant is ours |
| `pllm/slalom-verify` | **[R07]** | tracked | Original blinding + preprocessed Freivalds-style verification boundary (`protocol.masked_linear`, `verify.preprocessed_linear`) |
| `pllm/maverick-matvec` | **[R24]** | planned | Delegated matrix-vector + LPN masking + batch verification against the prepared stage contract (`protocol.delegated_matvec`, `preparation.lpn_masking`, `gate.batch_verification`) — the traffic-crushing arm |
| `pllm/trapdoored-linear` | **[R16]** | tracked | Paper-specific trapdoor generation + delegation (`prepare.trapdoored_matrix`, `protocol.delegated_linear`) |
| `pllm/nexus-he` | **[R21]** | tracked | Full HE engine arm (`protocol.nexus`, `kernel.nexus_he`) — measured vs complete decoder graph |
| `pllm/bumblebee-2pc` | **[R22]** | tracked | Two-party-computation arm (`protocol.bumblebee`, `runtime.bumblebee`) |
| `pllm/guarded-linear` `pllm/blinded-*` `pllm/secure-*` `pllm/direct-fhe` | **[PLLM]** | *stack* | Proprietary-weights arms (existing engines; `corruption_model` differs — model-confidential not public-weight) |
| `pllm/cleartext-linear` | **[PLLM]** | *stack* | Plaintext reference arm over `backends/` (vllm/ollama/llama.cpp/mlx) — the benchmark denominator |
| `pllm/shared-mpc-2pc` | **[PLLM]** | *stack* | Two-party secret-sharing topology arm (salvage `shared_*`; different `party_topology`) |

#### Family `pllm/preparation-provider` — offline material assembly/delivery

| Component | Source | Status | What it contributes |
|---|---|---|---|
| `pllm/model-aware-corrections` | **[PLLM]** | *exists* | Seed expansion + `W·r−s` corrections + one-use tickets |
| `pllm/bfv-correlations` | **[PLLM]** | *stack* | HE-produced masking material (`required_host_features=("tenseal-backend",)`) |
| `pllm/he-authenticated-preprocessing` | **[PLLM]** | *stack* | Authenticated HE preprocessing variant |
| `pllm/carnival-preparation` | **[R08]** | tracked | Exact published parameterized variant — **quarantined comparison only**; Maverick attack must be reproduced before eligibility review (`prepare.subset_sum`) |
| `pllm/lpn-masking` | **[R24]** | planned | LPN mask generation for Maverick batch verification |

#### Family `pllm/correlation-source` — NEW (upstream crypto correlations consumed by preparation)

| Component | Source | Status | What it contributes |
|---|---|---|---|
| `pllm/ring-pcg` | **[R15]** | tracked | Ring PCG seed setup + expansion (`prepare.ring_pcg`) |
| `pllm/correlation-ole` | **[R15]** | tracked | OLE correlation generation |
| `pllm/authenticated-triple` | **[R15]** | tracked | Authenticated Beaver-style triples |
| `pllm/vector-ole` | **[R06]** | tracked | Vector-OLE for Duty-Free-Bits conversions (`prepare.vector_ole`) |
| `pllm/seeded-expansion` | **[PLLM]** | *exists (implicit)* | PRG-seed mask expansion inside model-aware-corrections — promote to a named correlation source so PCG/OLE sources can swap in |

#### Family `pllm/nonlinear-protocol` — protected nonlinear constructions

| Component | Source | Status | What it contributes |
|---|---|---|---|
| `pllm/binary-table/v1` | **[PLLM]** | *exists* | Dense one-use Q7 lookup table baseline |
| `pllm/r03-crt/v1` | **[R03]** | *exists* | Compact mixed-modulus Q7 (adaptation, not reproduction) |
| `pllm/dash-gates` | **[R01]** | tracked | Affine-label arithmetic gates: constant-mult, projections, tensorized evaluation (`garble.affine_label`, `kernel.label_tensor`, `gate.projection`) |
| `pllm/logrow-lookup` | **[R05]** | tracked | LogRow lookup with Boolean I/O contracts (`gate.logrow`) |
| `pllm/curl-wavelet` | **[R17]** | tracked | Wavelet-approximation lookup + probabilistic truncation (`gate.wavelet_lookup`, `gate.probabilistic_truncation`) |
| `pllm/odp-branching` | **[R10]** | tracked | Oblivious branching-program gates for encoded nonlinear paths (`gate.oblivious_branching_program`, `convert.branching_input`) |
| `pllm/fevbdd-eval` | **[R11]** | tracked | Encrypted traversal of compiled decision diagrams (experimental; public compile side is a compiler-pass, below) |
| `pllm/fss-sigma` | **[R13]** | tracked | FSS softmax/activation/normalization/truncation with phase contracts (`protocol.fss`, `gate.fss_nonlinear`, `prepare.fss`) |
| `pllm/fss-fused` | **[R14]** | tracked | Fused FSS helpers + offline key gen (`gate.fss_fused`) |
| `pllm/bumblebee-nonlinear` | **[R22]** | tracked | BumbleBee normalization/softmax/activation components for comparison |
| `pllm/nexus-nonlinear` | **[R21]** | tracked | NEXUS nonlinear operators |

#### Family `pllm/label-scheme` — NEW (garbling label/encoding families consumed by nonlinear constructions)

| Component | Source | Status | What it contributes |
|---|---|---|---|
| `pllm/affine-label` | **[R01]+[R03]** | tracked | Arithmetic-label ownership, modulus-specific offsets, mixed-modulus gadgets (`garble.arithmetic`, `gate.mixed_modulus`) |
| `pllm/boolean-halfgates` | **[R04]** | in-progress (WIP `boolean*.rs`) | Boolean labels, reviewed hash instantiation, batched ops (`garble.boolean`, `kernel.half_gates`) |
| `pllm/free-xor` | **[R04]** | in-progress | Free-XOR + circuit I/O conversions (`kernel.free_xor`) |

#### Family `pllm/conversion` — representation edges

| Component | Source | Status | What it contributes |
|---|---|---|---|
| `pllm/q14-q7-rescale` | **[PLLM]** | *exists* | Fixed-point Q14→Q7 rescale region |
| `pllm/r02-rescale` | **[R02]** | tracked | ReDASH signed base extension + scaling gadget (`convert.rns_base_extension`, `gate.scale`) |
| `pllm/dfb-bit-affine` | **[R06]** | tracked | Duty-Free-Bits bit↔affine conversions, direction-typed (`convert.bit_to_affine`) |
| `pllm/lookup-io` | **[R05]** | tracked | LogRow's explicit Boolean↔arithmetic conversion edges (`convert.lookup_io`) |

#### Family `pllm/numeric-transform` — approximation/quantization

| Component | Source | Status | What it contributes |
|---|---|---|---|
| `pllm/compact-fit` | **[R18]** | tracked | Compact fitting objective + published approximation profile, exact-vs-approximate semantics explicit (`numeric.piecewise_fit`) |
| `pllm/scalequant` | **[R02]** | tracked | ReDASH quantization policy (`numeric.scalequant_plus`) |
| `pllm/wavelet-encoding` | **[R17]** | tracked | Curl wavelet encoding/scale (`numeric.wavelet`) |

#### Family `pllm/compiler-pass` — plan-time transformations

| Component | Source | Status | What it contributes |
|---|---|---|---|
| `pllm/kv-cache-eviction` (transform) | **[R23]** | *exists* | MPCache structural `DecoderPlan` adaptation |
| `pllm/protected-cache-policy` | **[R23]** | tracked | Protected cache selection under explicit topology |
| `pllm/hycc-region-select` | **[R09]** | tracked | Hybrid region selection + conversion-cost model (`compiler.region_selection`, `compiler.conversion_costs`) |
| `pllm/fevbdd-compile` | **[R11]** | tracked | Public EVBDD/FEVBDD compilation, canonical residual normalization, variable-order search (`compiler.weighted_decision_diagram`, `compiler.variable_order`) |
| `pllm/fss-fusion` | **[R14]** | tracked | FuseFSS fusion pass (`compiler.fss_fusion`) |
| `pllm/approximation-profile` | **[R18]** | tracked | Frozen coefficient/interval hashes + tail policy (`compiler.approximation_profile`) |

#### Family `pllm/state-protocol` — NEW (protected persistent state)

| Component | Source | Status | What it contributes |
|---|---|---|---|
| `pllm/secret-shared-kv` | **[R23]** | tracked | Secret-shared KV state (`state.secret_shared_kv`) |
| `pllm/mpcache-attention` | **[R23]** | tracked | MPCache attention over shared state (`gate.mpcache_attention`) |
| `pllm/client-local-kv` | **[PLLM]** | *exists* | Client-local KV (the trust-simplest arm; baseline for comparison) |

#### Family `pllm/verification-scheme` — NEW (result-correctness verification)

| Component | Source | Status | What it contributes |
|---|---|---|---|
| `pllm/linear-integrity` | **[PLLM]** | *stack* | Existing integrity module — wire into prepared path (P3.17) |
| `pllm/freivalds-verify` | **[R07]** | tracked | Slalom's preprocessed verification state |
| `pllm/batch-verification` | **[R24]** | planned | Maverick batch verification (`gate.batch_verification`) |
| `pllm/open-weight-proof` | **[R19]** | tracked | Verifiable open-weight proof boundary + model-owner setup (`verify.model_owner_setup`) — full spec + artifact acquisition first |

#### Family `pllm/codec` — transport/material encoding

| Component | Source | Status | What it contributes |
|---|---|---|---|
| `pllm/u16-u32-packing` | **[PLLM]** | *exists* | Exact-bounds adaptive ring selection (u16/u24/u32 per stage) |
| `pllm/syndrome-codec` | **[PLLM]** | *design* | PLLM's own syndrome-based oblivious encoding (named in R20 registry entry as the comparison adaptation) |
| `pllm/oblivious-linear-code` | **[R20]** | tracked | Paper reproduction + domain certificates (`codec.oblivious_linear_code`, `codec.domain_certificate`) |

#### Family `pllm/kernel-backend`

| Component | Source | Status | What it contributes |
|---|---|---|---|
| `pllm/cpu` | **[PLLM]** | *exists* | Rust SIMD/Rayon |
| `pllm/piranha-gpu` | **[R12]** | tracked | Wrap original GPU kernels via narrow native interface, then selective Rust/CUDA ports (`kernel.modular_gpu`, `runtime.device_buffers`) |
| `pllm/nexus-he-kernels` | **[R21]** | tracked | NEXUS HE kernels |
| `pllm/wgpu` | **[PLLM]** | planned | Portable GPU (Metal/Vulkan) for VirtualDC gaming-PC nodes |

#### Families `pllm/adversary` + `pllm/assurance-check` + `pllm/metric`

| Component | Source | Status | What it contributes |
|---|---|---|---|
| `pllm/carnival-projection` | **[R08]** | tracked | The published Maverick projection attack as an adversary fixture (`attack.carnival_projection`) |
| weakened-fixture controls | **[PLLM]** | *exists* (`pllm-assurance`) | Negative controls in assurance runs |
| collusion/leakage-view fixtures | **[PLLM]** | planned | View-restricted leakage tests from the privacy contract |
| `pllm/metric/*` (latency, traffic, rss, disk, energy, quality, prep-cost, monetary-cost) | **[PLLM]** | partial | Objective metrics per the multi-objective search contract (P3.16) |

#### Families `pllm/model-source` / `pllm/inference-role` / `pllm/protected-scheduler` — all **[PLLM]**

`Model`/`TinyModel`/`BundleModel`, `Inference`, `Scalar`/`IndependentLanes`/`ChunkedIndependentLanes` schedules (exist; R01's `kernel.label_tensor` may later inform a tensorized-label schedule arm).

#### Papers that produce *benchmark cohorts*, not new slots

R19/R21/R22 are full-system comparison targets — they yield resolved **profiles** (`research.open_weight_verified`, `research.nexus_he`, `research.bumblebee_2pc`) measured under matched workloads with their own party topologies preserved in reports. No claims transfer across models/graphs/topologies (backlog rule). R12 similarly yields an `mpc_comparison` accelerator arm.

#### Dependency order (from registry `depends_on` + backlog priority)

```text
R24 (none) ──► delegated matvec + LPN + batch verify      [planned — traffic lever]
R23 (none) ──► protected KV cache                         [structural transform exists]
R03+R04    ──► R01 ──► R02                                [garbling + rescaling core]
R03+R04    ──► R05, R06, R09, R10                         [lookup, conversion, compiler, branching]
R07        ──► R08                                        [masked-linear → Carnival quarantine]
R13        ──► R14                                        [FSS → fused FSS]
R11, R12, R15, R16, R17, R18 (none)                       [independent foundations]
R19, R20, R21, R22 (none)                                 [comparison cohorts]
```

#### Novel-to-PLLM inventory — candidate contributions for a future PLLM paper

- Seeded-expansion offline correction (`W·r−s`) + burn-once ticket inventory + rendezvous delivery protocol
- Exact-bounds adaptive ring selection (u16/u24/u32 per stage)
- Plan-locked region-program scheduling + coverage-gated fail-closed compiler
- Plan-level privacy-contract synthesis (threat-model composition checking across mixed arms)
- Matched-cohort multi-objective Pareto benchmark methodology + sanitized evidence records
- PLLM syndrome codec
- Bounded one-use Q7 garbling region design (dense vs CRT-compact schedules)
- Trusted-client/dealer adaptation of Slalom's blinding boundary

These are the items where "PLLM did something the literature didn't" — they're the future PLLM paper's contribution list, and they should be marked `provenance: pllm` rather than `provenance: R##` in descriptors so claims stay honest.

### 7.4 Old → new SDK examples

**Load a model from HuggingFace**

```python
# OLD — no SDK path; flags or HTTP body only
#   pllm serve inference --model Qwen/Qwen2.5-0.5B-Instruct --revision <sha>
#   PUT /v1/runtime/models {"model_id": "...", "hf_model": "...", "revision": "..."}

# NEW — one typed spec, one loader entry
import pllm

ref = pllm.Model.hf("Qwen/Qwen2.5-0.5B-Instruct", revision="a8a9d45...")   # pllm.Model IS the model-source spec
manifest = pllm.load_model(ref)            # hf_hub.resolve_huggingface_source → loaders
manifest.fingerprint                        # checkpoint digest → plan locks
```

**Serve the two roles from the SDK**

```python
# OLD — raw FastAPI factories; you assemble uvicorn; model spec inside config flags
from pllm import create_app, create_preparation_app
app = create_app(...)                 # then: uvicorn.run(app, ...) yourself

# NEW — role handles with lifecycle
with pllm.serve_preparation(listen=("0.0.0.0", 8090)) as prep, \
     pllm.serve_inference(model=ref, listen=("0.0.0.0", 8080)) as inf:
    inf.url, prep.url, inf.health()   # .close() on exit; same roles `pllm serve` runs
```

**Local end-to-end (client + preparation + inference)**

```python
# OLD — three surfaces, three implementations:
#   pllm gateway --model ...        # in-process trio (http_gateway.py)
#   pllm dev dashboard --model ...  # 3 role subprocesses (dashboard.py)
#   pllm benchmark run --model ...  # subprocess → dashboard → 3 roles (4 layers)

# NEW — one builder; every front reuses it
exp  = pllm.Experiment.from_file("examples/pllm.yaml")
plan = exp.resolve()                                 # ExperimentProfile

with pllm.serve_local(plan) as topo:                 # build_roles(plan) — also takes exp or model
    topo.roles      # {"client": h, "preparation": h, "inference": h}
    client = pllm.OpenAI(base_url=topo.gateway_url, api_key=topo.client_key)
    resp = client.responses.create(model=exp.pipeline.model.source,
                                   input="hello", max_output_tokens=16)
```

`pllm gateway` = `serve_local` + OpenAI-compat HTTP front; `dev dashboard` = `serve_local` + UI/OTLP; `benchmark run` = `serve_local` + telemetry + workload driver. Same roles, same components, same code.

**Compose arms — kwargs/flags become components**

```python
# OLD — arm selection is scattered across kwargs/flags/env:
client = pllm.OpenAI(privacy_mode="proprietary", ...)   # proprietary arm via kwarg
settings = pllm.ClientSettings(correlation_mode="bfv")  # HE-correlation arm via setting
# pllm serve inference --protocol direct                # arm via flag
# pllm benchmark run --tiny                             # model source via flag

# NEW — arms are classes under top-level families (sklearn/keras style);
# identity strings are serialization only
from pllm.profiles    import MaskedLinearCpu, PlaintextRef, ProprietaryGuarded
from pllm.protocols   import MaskedLinear, CleartextLinear, GuardedLinear
from pllm.preparation import ModelAwareCorrections, BFVCorrelations
from pllm.kernels     import Cpu
from pllm.roles       import Inference

# a profile IS a Pipeline subclass — its __init__ signature IS the slot contract:
# every slot is a typed kwarg (annotated with its category ABC), so the IDE
# completes the slot list and rejects a wrong-category component statically.
base = MaskedLinearCpu(
    model=pllm.Model("Qwen/Qwen2.5-0.5B-Instruct"),
    linear=MaskedLinear(),                       # linear: ProtocolMethod
    preparation=ModelAwareCorrections(),         # preparation: PreparationProvider
    inference=Inference(),                       # inference: InferenceRole
    kernels=Cpu(threads=8),                      # kernels: KernelBackend
)

# HE-correlation arm — one slot swap by kwarg; identical workload/model ⇒ comparable
bfv = base.with_params(preparation=BFVCorrelations(poly_degree=8192))

# plaintext reference arm — a different profile class (different slot contract);
# the compiler refuses a *private* claim from this plan — the honest denominator
ref = PlaintextRef(model=base.model, linear=CleartextLinear(backend="vllm"),
                   kernels=Cpu(threads=8))

# proprietary-weights arm — today's --guard-* flags become typed params
guarded = ProprietaryGuarded(
    model=base.model,
    linear=GuardedLinear(max_rows_per_session=4096, output_dither=1e-3),
    inference=Inference(), kernels=Cpu(threads=8),
)

# nested param tweaks use sklearn's slot__param convention
tuned = base.with_params(preparation__poly_degree=8192, kernels__threads=4)

# serialization contract — class authors, string serializes (keras-style)
BFVCorrelations(poly_degree=8192).to_spec()
#   → {"component": "pllm/bfv-correlations/v1", "params": {"poly_degree": 8192}}
pllm.components.get("pllm/bfv-correlations/v1")     # → BFVCorrelations class
BFVCorrelations.describe()                          # → its ComponentDescriptor (registry is derived)

# YAML round-trip — the file keeps string IDs; Python keeps classes
#   preparation:
#     component: pllm/bfv-correlations/v1
#     params: {poly_degree: 8192}
exp = pllm.Experiment.from_file("experiment.yaml")   # strings → classes via components.get()
```

**Benchmark matched arms**

```python
# OLD — pllm benchmark run --model X  (one arm; dashboard path only)

# NEW — matched experiments differing only by component swap
report = pllm.benchmark_run(
    [exp_seeded, exp_bfv, exp_plaintext],   # same model fingerprint + workload required
    metrics=["latency", "traffic", "rss", "disk", "energy"],   # P3.16
)
# CLI equivalent: pllm benchmark run --experiment seeded.yaml --experiment bfv.yaml
```

**VirtualDC boundary — PLLM emits, VirtualDC routes**

```python
plan = exp.resolve()
plan.role_requirements()   # roles, per-role component IDs, host features, model fingerprint
node  = pllm.capabilities() # this node's host features (kernel-backend, gpu, tenseal, mem, arch)
# VirtualDC's control plane maps offers↔requirements and prices on signed
# measurement.v1 evidence — no market/placement code inside PLLM.
```

### 7.5 Package layout — old → new import structure

**Problems today (verified):** component classes live in `configuration.py` (the spec/serde layer — wrong home); six partial namespace packages each re-export 1–3 objects inconsistently (`pllm.components` has 7/9 classes, `pllm.kernels` only `Cpu`, `pllm.protocols`/`pllm.research` nearly empty, `pllm.pipeline`/`pllm.deployment`/`pllm.config` are thin re-export shells); `models.py` vs `modeling.py` + dead `models/` dir; `runtime/` is **68 flat modules** mixing engines/transports/model-loading/bench/gateway; dual import paths (`pllm.client.OpenAI` shim vs `pllm.runtime.client.OpenAI` canonical); generic names at package root (`official.py`, `chat.py`, `provider.py`).

**Rule set:** (a) module = *capability family / lifecycle phase*, never per-class file and never kitchen-sink; promote module→package when a family exceeds ~5 classes, public path stays stable via `__init__` re-export; (b) **one canonical home per object + at most one documented compat shim**; (c) top-level `pllm.*` = the *workflow* surface (`Experiment`, `Pipeline`, `OpenAI`, `load_model`, `serve_*`, `lower_model`, `compile`) — not a dumping ground for every class.

**Decision (user input, matching precedent): flatten — no `pllm.components` umbrella.** sklearn puts estimators under top-level family packages (`sklearn.linear_model`, `sklearn.preprocessing`, `sklearn.ensemble`); keras same (`keras.layers`, `keras.optimizers`, `keras.metrics`); deepeval same (`deepeval.metrics`). None have a `.<components>` level — the family IS the namespace. `pllm.components.protocol.MaskedLinear` carries a meaningless level vs `pllm.protocols.MaskedLinear`. **`pllm.components` survives only as the generics/machinery module** (base classes + registry — the `sklearn.base` role); concrete components live under top-level families.

Collision resolutions: `pllm.roles` (role *components*) vs proposed `pllm.runtime.roles` (server infra) → rename runtime one `runtime.servers/`; `pllm.kernels`/`pllm.protocols` already exist as thin packages → they become the real family homes; `pllm.pipeline`/`pllm.deployment`/`pllm.config` shells die (their types stay top-level); `pllm.research` is **promoted to a real namespace** — the research-harness API (source locks, references, fidelity tests, promotion — the §C.8 gates as functions); `pllm.models` name deliberately avoided (collides with `pllm-models` crate + `runtime.models`) → model-source components live in `pllm.sources`; `pllm.metrics` (components) vs OTLP telemetry stays distinct (`runtime.bench.telemetry`).

### Full public API hierarchy (the reference tree)

Tier 1 — workflow surface; Tier 2 — component families + profile classes; Tier 3 — machinery + documented internals. Everything below is `py.typed` + `.pyi`-covered; each component class exposes `__init__` (typed params), `get_params`/`with_params`/`to_spec`/`from_spec`, `describe() -> ComponentDescriptor`, `component_id` (the `pllm/…` string). Each family module also exposes its **category ABC** (`ProtocolMethod`, `PreparationProvider`, …) — the abstract base that profile slots type-annotate; `isinstance` checks fire at construction, descriptor checks at `resolve()`.

```
pllm — the package root; only the workflow surface lives at top level
│
│  TIER 1 — workflow surface (what users touch daily)
│
├── Experiment                    — the declarative unit of work: pipeline + deployment + budget; resolves into a runnable, digest-locked plan.
├── Pipeline                      — the model + component-slot map under a named profile; the object you swap arms inside.
├── Model                         — typed model-source reference (HF repo id, local path, MLX, GGUF, tiny, bundle).
├── Deployment                    — where the plan's roles run (today: local; VirtualDC topologies later).
├── ExecutionBudget               — workload bounds for a run: request count + input/output token caps.
├── ExperimentProfile             — the resolved, digest-bound output of Experiment.resolve(); what actually executes.
├── ConfigurationError            — raised when a spec is malformed or violates a contract.
├── ResolutionError               — raised when resolve() can't satisfy a profile (missing/wrong-category components).
├── UsageError                    — raised for invalid CLI/API usage.
├── LocalIOError                  — raised for local filesystem/cache failures.
├── loads_configuration(text)     — parse an Experiment from a string (YAML or JSON).
├── load_configuration(path)      — parse an Experiment from a file.
├── canonical_bytes(x)            — deterministic serialization used for digests, locks, evidence.
├── configuration_digest(x)       — SHA-256 identity of a spec, used everywhere provenance matters.
├── lower_model(spec)             — lower a model into the semantic IR (ModelPlan).
├── ModelPlan                     — immutable semantic decoder graph; carries coverage() and apply().
├── DecoderCoverageReport         — per-operator support report for a plan under a profile; complete=false fails closed.
├── compile(request)              — compile a plan + request into a CompiledPlan (logical/execution/lock artifacts).
├── CompiledPlan                  — the compiled artifacts; executes only regions with real coverage.
├── load_model(ref)               — resolve a Model (including HF pull) into a ModelManifest.
├── serve_inference(cfg)          — start an inference-role server; returns a RoleHandle.
├── serve_preparation(cfg)        — start a preparation-role server; returns a RoleHandle.
├── serve_local(experiment|ref)   — spin up client+preparation+inference locally from a resolved plan.
├── build_roles(plan)             — the one topology builder every front shares (serve/gateway/dashboard/benchmark/SDK).
├── LocalTopology                 — handle for a running local trio: role handles, urls, health, close().
├── RoleHandle                    — handle for one role server: url, health(), close().
├── capabilities()                — report what this node can host (kernels, tenseal, gpu, memory) — the VirtualDC advertisement.
├── benchmark_run(*experiments)   — run matched experiments through the instrumented topology → measurement records.
├── assure_run(experiment|plan)   — run the assurance suite against a plan: adversary fixtures, negative controls, integrity checks → detection report. (new)
├── OpenAI                        — synchronous consuming client (Responses + Chat Completions surface).
├── AsyncOpenAI                   — async consuming client.
├── ResponseStream                — streaming handle for incremental responses.
├── create_openai_client          — factory returning an official OpenAI-SDK client bound to a PLLM endpoint.
├── create_async_openai_client    — async variant of the same.
├── ClientSettings                — persisted client config (endpoints, keys, cache modes).
│
│  TIER 3a — pllm.components: generics/machinery ONLY (the sklearn.base role; no concrete components)
│
├── components
│   ├── ComponentRef              — base class of every component; identity + params; to_spec()/from_spec() serde.
│   ├── ComponentDescriptor       — immutable contract record: category, param schema, capabilities, host features, roles, provenance.
│   ├── get(identity)             — resolve a "pllm/…" string to its class (keras layers.get() analogue).
│   ├── list_components()         — enumerate all registered descriptors (builtins + third-party).
│   ├── categories()              — list the capability families the registry knows.
│   └── register(cls)/manifests   — pllm.providers.v1 glue for third-party components.
│
│  TIER 2 — component families + profile classes (each top-level namespace = one capability family;
│            every family module also exports its category ABC, listed first — slot annotations type against it)
│
├── profiles                      # Pipeline subclasses — a profile's __init__ signature IS the slot contract.
│   ├── MaskedLinearCpu           — baseline prepared masked-linear trio; required kwargs: model/linear/preparation/inference/kernels; optional kwargs: nonlinear, schedule, cache, integrity, adversary, metrics. [was "baseline.masked_linear_cpu" string]
│   ├── PlaintextRef              — cleartext reference profile; produces no privacy claim by construction. [was "reference.plaintext"]
│   ├── SingleEvaluator           — research compiler-execution profile for the new plan path. [was "research.single_evaluator"]
│   ├── BfvPrepared               — HE-correlation preparation arm (adds preparation-side crypto requirements). [planned]
│   ├── ProprietaryGuarded        — proprietary-weights arm (linear slot takes guarded/blinded/secure/FHE methods). [planned]
│   ├── TwoPartyMpc               — 2PC topology arm (client + one compute party; different party_topology). [planned]
│   ├── NexusHe / Bumblebee2pc / OpenWeightVerified — full-system research cohort profiles. [R21]/[R22]/[R19] [planned]
│   └── Pipeline (base)           — generic path: from_spec()/subclass + slots declaration for YAML & third-party profiles.
│
├── protocols                     # pllm/protocol-method — how linear work (matmuls) is executed or delegated.
│   ├── ProtocolMethod (ABC)      — the category interface every linear-protocol component implements.
│   ├── MaskedLinear              — PLLM's seeded masked-linear: untrusted inference computes on masked inputs with one-use correction tickets. [exists]
│   ├── CleartextLinear           — plaintext passthrough via a real backend — the zero-privacy reference arm all private arms are measured against.
│   ├── SharedMpc2pc              — two-party secret-sharing linear arm — a different party topology for comparison.
│   ├── GuardedLinear             — proprietary-weights arm: rate-limited, dithered linear queries while weights stay secret.
│   ├── BlindedLinear             — proprietary-weights arm using blinded evaluation.
│   ├── SecureLinear              — proprietary-weights arm using the secure-transformer construction.
│   ├── DirectFhe                 — proprietary-weights arm using direct FHE (strongest, slowest).
│   ├── MaverickMatvec            — [R24] delegated matrix-vector execution built for batch verification + low client traffic.
│   ├── TrapdooredLinear          — [R16] trapdoor-based delegation of linear computation.
│   ├── NexusHe                   — [R21] full-HE linear arm (single-evaluator comparison).
│   └── Bumblebee2pc              — [R22] two-party-computation linear arm for matched-system comparison.
│
├── preparation                   # pllm/preparation-provider — who produces offline material and how it reaches inference.
│   ├── PreparationProvider (ABC) — the category interface every offline-material producer implements.
│   ├── ModelAwareCorrections     — expands client seeds into masks + precomputes W·r−s corrections per stage; today's trusted offline worker. [exists]
│   ├── BFVCorrelations           — produces masking material via BFV homomorphic encryption instead of seed expansion.
│   ├── HeAuthenticatedPreprocessing — authenticated HE preprocessing variant (integrity on the material itself).
│   ├── CarnivalPreparation       — [R08] subset-sum preparation scheme; quarantined until its published attack is reproduced.
│   └── LpnMasking                — [R24] LPN-based mask generation enabling cheap batch verification.
│
├── correlation                   # pllm/correlation-source — raw cryptographic correlations that preparation providers consume.
│   ├── CorrelationSource (ABC)   — the category interface for correlation generators.
│   ├── SeededExpansion           — PRG-seed mask expansion; today's implicit method promoted to a swappable component.
│   ├── RingPcg                   — [R15] ring pseudorandom correlation generation: tiny seeds → large correlations.
│   ├── CorrelationOle            — [R15] oblivious-linear-evaluation correlation generation.
│   ├── AuthenticatedTriple       — [R15] authenticated Beaver-style multiplication triples.
│   └── VectorOle                 — [R06] vector-OLE material for Duty-Free-Bits conversions.
│
├── nonlinear                     # pllm/nonlinear-protocol — protected nonlinear ops (SiLU, softmax, comparison, lookup).
│   ├── NonlinearProtocol (ABC)   — the category interface for protected nonlinear constructions.
│   ├── BinaryTableGatedMultiplyQ7   — dense one-use Q7 lookup-table gate; PLLM's simple-but-large baseline. [exists]
│   ├── R03CrtGatedMultiplyQ7        — [R03] compact mixed-modulus Q7 gate (adaptation; much smaller payloads). [exists]
│   ├── DashGates                    — [R01] affine-label arithmetic gates: constant mult, projections, tensorized evaluation.
│   ├── LogrowLookup                 — [R05] lookup-table construction with Boolean I/O contracts.
│   ├── CurlWavelet                  — [R17] wavelet-approximation lookup + probabilistic truncation.
│   ├── OdpBranching                 — [R10] oblivious branching-program gates for encoded nonlinear paths.
│   ├── FevbddEval                   — [R11] encrypted traversal of compiled decision diagrams (experimental eval side).
│   ├── FssSigma                     — [R13] function-secret-sharing softmax/activation/norm/truncation with phase contracts.
│   ├── FssFused                     — [R14] fused FSS helpers + offline key generation.
│   ├── BumblebeeNonlinear           — [R22] BumbleBee normalization/softmax/activation for comparison.
│   └── NexusNonlinear               — [R21] NEXUS nonlinear operators.
│
├── labels                        # pllm/label-scheme — the label/encoding systems nonlinear constructions build on.
│   ├── LabelScheme (ABC)         — the category interface for label/encoding systems.
│   ├── AffineLabel               — [R01+R03] arithmetic-label ownership with modulus-specific offsets + mixed-modulus gadgets.
│   ├── BooleanHalfgates          — [R04] Boolean label ownership + half-gates evaluation (in progress).
│   └── FreeXor                   — [R04] free-XOR gadget + circuit I/O conversions.
│
├── conversion                    # pllm/conversion — typed edges between numeric representations.
│   ├── Conversion (ABC)          — the category interface for representation edges.
│   ├── Q14Q7Rescale              — fixed-point Q14→Q7 rescale used at garbling boundaries. [exists]
│   ├── R02Rescale                — [R02] ReDASH signed base extension + scaling gadget.
│   ├── DfbBitAffine              — [R06] Duty-Free-Bits bit↔affine conversions, direction-typed.
│   └── LookupIo                  — [R05] LogRow's Boolean↔arithmetic conversion edges.
│
├── numeric                       # pllm/numeric-transform — approximation/quantization policies.
│   ├── NumericTransform (ABC)    — the category interface for numeric policies.
│   ├── CompactFit                — [R18] piecewise-fit approximation profile with explicit tail policy.
│   ├── Scalequant                — [R02] ReDASH quantization policy.
│   └── WaveletEncoding           — [R17] Curl wavelet encoding/scale.
│
├── passes                        # pllm/compiler-pass — plan-time transformations applied via plan.apply().
│   ├── PlanPass (ABC)            — the category interface for plan transforms.
│   ├── KvCacheEviction           — [R23] MPCache-derived cache-eviction transform on the decoder plan. [exists]
│   ├── ProtectedCachePolicy      — [R23] protected cache selection under an explicit topology.
│   ├── HyccRegionSelect          — [R09] hybrid region selection + conversion-cost model.
│   ├── FevbddCompile             — [R11] public decision-diagram compilation + variable-order search.
│   ├── FssFusion                 — [R14] fusion pass merging compatible FSS gates.
│   └── ApproximationProfile      — [R18] freezes coefficient/interval hashes into the plan for reproducibility.
│
├── state                         # pllm/state-protocol — how persistent state (KV cache) is represented and shared.
│   ├── StateProtocol (ABC)       — the category interface for protected-state schemes.
│   ├── ClientLocalKv             — KV stays entirely client-local; the trust-simplest baseline arm. [exists]
│   ├── SecretSharedKv            — [R23] secret-shared KV state across parties.
│   └── MpcacheAttention          — [R23] attention over shared cache state.
│
├── verification                  # pllm/verification-scheme — proving the untrusted party computed correctly.
│   ├── VerificationScheme (ABC)  — the category interface for result-correctness proofs.
│   ├── LinearIntegrity           — existing integrity-check module to wire into the prepared path.
│   ├── FreivaldsVerify           — [R07] Slalom's preprocessed probabilistic verification of linear results.
│   ├── BatchVerification         — [R24] Maverick's batched verification of delegated matvecs.
│   └── OpenWeightProof           — [R19] verifiable open-weight proof boundary + model-owner setup.
│
├── codecs                        # pllm/codec — wire/material encodings.
│   ├── Codec (ABC)               — the category interface for wire/material encodings.
│   ├── U16U32Packing             — per-stage adaptive ring choice (u16/u24/u32) from exact bounds. [exists]
│   ├── SyndromeCodec             — PLLM's own syndrome-based oblivious encoding design.
│   ├── ObliviousLinearCode       — [R20] oblivious linear-code construction.
│   └── DomainCertificate         — [R20] certificate binding a codec to valid representation domains.
│
├── kernels                       # pllm/kernel-backend — the compute substrate a role runs on.
│   ├── KernelBackend (ABC)       — the category interface every compute backend implements.
│   ├── Cpu                       — Rust SIMD (AVX2/NEON) + Rayon CPU backend. [exists]
│   ├── Wgpu                      — portable GPU backend (Metal/Vulkan) for VirtualDC gaming-PC nodes.
│   ├── PiranhaGpu                — [R12] GPU MPC kernels wrapped via a narrow native interface.
│   └── NexusHeKernels            — [R21] NEXUS HE kernels.
│
├── adversary                     # pllm/adversary — attack/fixture components for assurance runs.
│   ├── Adversary (ABC)           — the category interface for attack/fixture components.
│   ├── CarnivalProjection        — [R08] the published projection attack as a reusable fixture.
│   ├── WeakenedFixture           — deliberately weakened variants used as negative controls. [exists via pllm-assurance]
│   └── CollusionView             — view-restriction fixtures simulating colluding/leaking parties.
│
├── metrics                       # pllm/metric — declared measurement objectives for an experiment.
│   ├── Metric (ABC)              — the category interface for objective measurements.
│   ├── Latency                   — end-to-end + per-stage latency (ttft, decode).
│   ├── Traffic                   — bytes per direction/role/stage.
│   ├── Rss                       — peak resident memory per role.
│   ├── Disk                      — bundle-cache + material-store bytes.
│   ├── Energy                    — per-request energy where measurable (RAPL/powermetrics).
│   ├── Quality                   — output quality vs cleartext reference (parity/perplexity).
│   ├── PrepCost                  — offline preparation time/material cost.
│   └── Cost                      — monetary cost under a priced-resource model.
│
├── sources                       # pllm/model-source — where weights come from.
│   ├── ModelSource (ABC)         — the category interface for weight sources.
│   ├── TinyModel                 — generated minimal weights for transport smoke tests (replaces --tiny).
│   └── BundleModel               — pinned snapshot bundle with digest for reproducibility.
│
├── roles                         # pllm/inference-role — the party slots a topology fills.
│   ├── InferenceRole (ABC)       — the category interface for inference-role components.
│   └── Inference                 — the untrusted inference-role component (required by baseline; no class today).
│
├── schedulers                    # pllm/protected-scheduler — how protected tensors are scheduled.
│   ├── ProtectedScheduler (ABC)  — the category interface for protected-tensor schedulers.
│   ├── ScalarProtectedTensorSchedule            — per-scalar scheduling. [exists]
│   ├── IndependentLanesProtectedTensorSchedule  — up to 4 independent lanes. [exists]
│   └── ChunkedIndependentLanesProtectedTensorSchedule — chunked lane scheduling. [exists]
│
│  TIER 3b — documented internals (power users; never required for Tier-1 work)
│
├── runtime
│   ├── client.py                 — canonical home of the OpenAI-compatible client.
│   ├── servers/                  — role server processes (inference_server.py, preparation_server.py).
│   ├── transport/                — wire framing, correction channels, rendezvous, WS transport.
│   ├── engines/                  — runtime engines + the InferenceEngine protocol every protocol arm implements.
│   ├── models/                   — model loading: HF pull, directory/GGUF/MLX loaders, safetensors, tiny generators.
│   ├── prep/                     — preparation internals: inventory, protocol, BFV correlations, ledger.
│   ├── mpc/                      — the shared_* two-party secret-sharing substrate.
│   ├── gateway/                  — OpenAI-compatible HTTP front + plaintext backend adapters.
│   └── bench/                    — benchmark driver, history, telemetry, dashboard.
│
├── native                        — public shim over _native: capabilities(), api_version() only.
├── serving.py                    — implementation home of the Tier-1 serving functions.
├── research/                     — promoted from thin shell to the research-harness API: lock_source, reference, fidelity_tests, assure helpers, promote (the §C.8 gates as functions).
└── _cli/                         — private: all CLI internals (folds runtime/cli.py in).
```

**Exposure rules (lower levels, meaningfully and consistently):**
- Every component class has exactly **one canonical import path**: `pllm.<family>.<Class>` — no top-level `pllm.MaskedLinear` (workflow surface stays clean); `pllm.components` never holds concrete components; moved objects keep at most one documented compat shim.
- `pllm.components.get("pllm/x/v1")` resolves any registered identity → class (built-ins + `pllm.providers.v1` third-party); `describe()` → descriptor; registry is *derived from classes*, never hand-maintained.
- `runtime.*` internals stay importable and documented as Tier 3 (power users/engines implement `InferenceEngine` there) but are never required for the Tier-1 workflow — anything users routinely need gets promoted (e.g., `create_app` → `serve_inference`).
- `pllm.native` exposes only the stable native introspection (`capabilities()`, `api_version()`); `_native` stays private — users never import it.
- No module named for its container (`pllm.models`, `pllm.utils`, `pllm.common`, `pllm.misc`) — names answer "which family/role", not "where things live".

**Old → new import map:**

| Old | New |
|---|---|
| `from pllm import MaskedLinear` · `from pllm.configuration import MaskedLinear` | `from pllm.protocols import MaskedLinear` |
| `from pllm.kernels import Cpu` | `from pllm.kernels import Cpu` (same path — becomes real family) |
| `ComponentRef("pllm/bfv-correlations/v1", {...})` | `from pllm.preparation import BFVCorrelations` |
| `pllm.components.get_component(id)` → descriptor | `pllm.components.get(id)` → **class**; `.describe()` → descriptor |
| `from pllm.pipeline import Pipeline` | `from pllm import Pipeline` (unchanged) |
| `from pllm.runtime.server import create_app` | `pllm.serve_inference(...)` preferred; `runtime.servers` canonical |
| `from pllm.runtime.client import OpenAI` | `from pllm import OpenAI` (canonical) |
| `from pllm.deployment import Deployment` | `from pllm import Deployment` (unchanged) |
| third-party: nothing | `from provider_pkg import TheirComponent` → registers via `pllm.providers.v1` |

### 7.6 What stays unchanged

- `OpenAI`/`AsyncOpenAI`/`ResponseStream` — the consuming client; gains no arm-selection kwargs (existing `privacy_mode`/`correlation_mode` kwargs get deprecation-by-status and forward to the corresponding component preset).
- `Experiment`/`Pipeline`/`ComponentRef`/`Model` types + YAML schema — already the convergent shape; `ModelRef`/`ModelSpec` merge into `pllm.Model` (`kind:` discriminator, `.hf()`/`.path()`/`.tiny()` constructors). Canonical component composition replaces profile identity in `pllm.experiment.v2`.
- `resolve_huggingface_source` + `loaders.py` — the loader core; `hf_download` stays the dependency-free fallback.
- `create_app`/`create_preparation_app` — become internals of `serve_inference`/`serve_preparation`; still exported for power users.
- `InferenceEngine`/`MaskedTransformerEngineProtocol` (`execute_stage`, `seeded_*`) — the engine seam every protocol arm implements for conformance.

---

## 8. Appendix A — findings inventory

### 8.1 Confirmed defects (evidence)
- `first-private-request.mdx` Experiment → `ConfigurationError: baseline profile requires inference component pllm/inference` (reproduced).
- `deploy/README.md:118` → `pllm build` → `invalid choice` (reproduced).
- `components list` shows 9 components; `baseline` profile requires a 10th (`pllm/inference`) that is undescribed.
- Before regen: `generate_developer_reference.py --check` failed (stale `components.mdx`, `python/pllm/index.mdx`); `npm test` failed (stale `public/releases/0.1.0` manifest `buildId`). Both pass after regen — pure freshness issues in the staged WIP, not code bugs.

### 8.2 File/area inventory
- Tracked files: 823 (docs 457 incl. generated mirrors; python 123; tests 58; crates 55; schemas 36).
- LOC: Python ~32k (biggest: `runtime/client.py` 3.3k, `runtime/server.py` 2.1k); Rust ~36.8k (biggest: `pllm-compiler/lib.rs` 4.9k, `pllm-models/lib.rs` 3.6k); tests ~13.9k; Rust tests ~7k.
- Wheel: `pllm_run-0.1.0a1` abi3, 2.6 MB `.so`, includes dashboard assets + py.typed + stubs (+ `market.py`/`provider.py`, slated for removal per P1.8).
- Git: 41 commits, all within Sep 8–17 2026; current WIP staged (dense_qwen_mlp/rms_norm/provenance/boolean garbling).

### 8.3 Generated-artifact map (must stay fresh; regenerate on change)
| Generator | Inputs | Outputs |
|---|---|---|
| `scripts/generate_developer_reference.py` | `python/pllm` API + `_cli/app.py` | `docs/content/docs/reference/python/pllm/index.mdx`, `reference/components.mdx`, `reference/cli/**` |
| `docs/scripts/generate-research-papers.mjs` | `docs/data/research/papers.json` + notes | `content/docs/research/papers/*.mdx`, `papers/index.mdx` |
| `docs/scripts/discover-pages.mjs` + `generate.mjs` | content tree + publication-registry | `docs/public/**` (markdown mirrors, manifests, llms.txt, search-index), `discovered-docs.mjs` |
| `docs/.next`/`next-env.d.ts` | next build | type artifacts (commit churn risk) |

### 8.4 Env-var surface (for docs consolidation)
~35 `PLLM_*` vars in use (gateway keys, rendezvous/prepared bounds, guard knobs, cache modes, HF paths) + `HF_*`/`XDG_*`. Canonical list exists only implicitly across `settings.py`, `_cli/app.py`, `config.py`, `server.py`, `preparation_server.py`, deploy README — candidate for a generated reference page.

### 8.5 Watch-items (not defects, but risks)
- Working tree carries ~17k uncommitted staged lines — commit or worktree-isolate before parallel contributors collide.
- `docs/package.json` uses npm lockfile; README/CONTRIBUTING disagree (bun vs npm) — CI uses `npm ci`, standardize on npm unless bun.lock is committed.
- `restack.toml` + `blairhudson/restack/actions/run` — external deployment dependency; document its role in `infra/README` (currently thin).
- Version skew: `market_server` reported `0.14.0` (resolves itself on removal); docs release is `0.1.0`; package is `0.1.0a1` — one release train should own remaining surfaces.

---

## 9. Appendix B — glossary of key terms

Grouped by theme (not alphabetically) so the relationships are visible. The single most-confused pair first:

### The spec types — Pipeline vs Experiment vs Profile

| Term | What it is | The short version |
|---|---|---|
| **Pipeline** | The *method stack* — a model plus its filled component slots. Authored via a **profile class** with typed slot kwargs; the `{slot → ref}` map is the internal/wire form, not the authoring form. | **What runs** — which components and how they're wired. |
| **Experiment** | `{name, pipeline, deployment, budget}` — wraps a Pipeline with execution context; the YAML-serializable, digestable unit that `resolve()`s and benchmarks. | **Where + how much** — Pipeline says what, Experiment says where it runs (Deployment) and how much work (Budget). |
| **Profile** | A `Pipeline` subclass (`pllm.profiles.*`) whose `__init__` signature IS the slot contract — which slots exist, which category ABC each accepts, which are required. Its `profile` attr is the identity string (`baseline.masked_linear_cpu`) for serde only. | The shape a Pipeline must fill — you instantiate the class, not the string. |
| **Category ABC** | The abstract base a family module exports (`ProtocolMethod`, `PreparationProvider`, `KernelBackend`, …) — what a slot's kwarg type-annotates against. | sklearn mixin analogue (`TransformerMixin`); `isinstance` at construction, descriptor check at `resolve()`. |
| **Deployment** | Where roles run (`local` today; VirtualDC topologies later). | Topology placement for an Experiment. |
| **ExecutionBudget** | Workload bounds: `requests`, `max_input_tokens`, `max_new_tokens`. | How much work an Experiment does. |

Rule of thumb: **you benchmark an Experiment; you swap arms in a Pipeline.** `Experiment` is the outer envelope — it can hold *any* Pipeline (masked-linear, BFV, proprietary, cleartext) as long as the Pipeline's own slots satisfy its Profile.

### Components

| Term | What it is | Note |
|---|---|---|
| **Component** | Three things, kept distinct: (1) the Python **class** you import (`MaskedLinear`) — the authoring surface; (2) the **ComponentRef instance** it creates (`identity + params`) — what sits in a slot; (3) the registered thing itself. The `pllm/…` **identity string** is serialization only. | keras `Dense` ↔ `"keras.layers.Dense"` analogy. |
| **ComponentRef** | Base class: `component` (identity) + `params`; `to_spec()`/`from_spec()` serde, `get_params()`/`with_params()` cloning, `describe()` → descriptor. | Every component class subclasses it. |
| **ComponentDescriptor** | Immutable contract record from `describe()`: category, parameter_schema, capabilities, required_host_features, role_eligibility, provenance, artifacts, evidence. | The registry stores these; resolution validates against them. |
| **Category** | The capability family a component belongs to (`pllm/protocol-method`, `pllm/preparation-provider`, …). | Determines which slots a component may fill. |
| **Family** | The Python namespace for a category (`pllm.protocols` ↔ `pllm/protocol-method`). | Import path = family; identity = category. |
| **Slot** | One named position in a Pipeline (`linear`, `preparation`, `inference`, `kernels`, `cache`, …) — surfaced as a **typed kwarg** on the profile class's `__init__`. | Exactly one component fills a slot; the kwarg's annotation is the category ABC it accepts. |
| **Arm** | Informal: one candidate component choice for a slot ("the BFV arm"). | An Experiment = a vector of arm choices; swapping one arm yields a matched comparison. |
| **Provider** | A distribution (package) that ships components, registered via `pllm.providers.v1`. | Built-ins live in `pllm.<family>`; third-party providers register the same way. |

### Plans — the three "plan" words

| Term | What it is | Note |
|---|---|---|
| **ModelPlan** | Immutable *semantic* decoder graph from `lower_model()` — the compiler's input. Carries `coverage()` and `apply()`. | Semantic, not executable. |
| **ExperimentProfile / ResolvedPlan** | Output of `Experiment.resolve()`: roles, topology, requirements, privacy contract, model fingerprint. | What `build_roles()` consumes. |
| **CompiledPlan** | Compiler output (logical/execution/lock/region-program artifacts); executes only covered regions, else fails closed. | From `compile(request)`, not `resolve()`. |
| **PlanLock** | The digest binding model + tokenizer + compiler + components + privacy + workload. | Guarantees a run matches its spec. |
| **Region** | A bounded executable slice of a decoder plan (wrap32 linear, residual, output head, …). | The unit of what can *actually run* today. |
| **Coverage / DecoderCoverageReport** | Per-operator support report for a plan under a profile; `complete=false` → fail closed. | Honest support, never silent fallback. |
| **ModelManifest** | Loaded weights + stage layout from `load_model()` — runtime-facing. | Distinct from ModelPlan (semantic). |

### Roles & topology

| Term | What it is | Note |
|---|---|---|
| **Role** | A party in the trust topology: client, preparation, inference (future: dealer, compute-party-2). | Filled by a role component; spawned by `build_roles()`. |
| **RoleHandle** | A running role: `url`, `health()`, `close()`. | From `serve_inference`/`serve_preparation`. |
| **LocalTopology** | The local trio (client+preparation+inference) built by `serve_local()`/`build_roles()`. | What gateway/dashboard/benchmark all wrap. |
| **Party topology** | Which roles exist + which may collude (client+prep+inference vs 2PC vs single-evaluator). | A descriptor field; composition-checked. |
| **build_roles(plan)** | The one topology builder every front shares. | `serve`/`gateway`/`dashboard`/`benchmark`/SDK all reuse it. |

### Crypto material vocabulary

| Term | What it is | Note |
|---|---|---|
| **Mask (r, s)** | Per-use blinding vectors expanded from client seeds; `x+r` is what inference sees. | Seeds never leave the client. |
| **Correction (W·r−s)** | Offline-computed product pushed to inference so `W(x+r) − (W·r−s) = Wx − s` unmasks correctly. | What preparation actually computes. |
| **Material / Inventory** | The pool of one-use prepared rows/tickets. | Memory-only today; durable AEAD store is P3/P4 work. |
| **Ticket** | A one-use burn token consumed per masked operation. | Burn-once semantics prevent reuse. |
| **Correlation source** | Crypto correlation generator (seeded PRG, ring-PCG, OLE, authenticated triples) that a preparation provider consumes. | Upstream of preparation — the swappable crypto layer. |
| **Preparation provider** | Assembles + delivers material (seeded corrections, BFV, …). | The trusted offline worker role. |

### Trust & assurance

| Term | What it is | Note |
|---|---|---|
| **Privacy contract** | Plan-level record: corruption model, party topology, leakage bounds. | Synthesized at `resolve()`; incompatible mixes rejected. |
| **Corruption model** | honest-but-curious vs malicious; who is trusted/untrusted. | Descriptor field on every component. |
| **Verification scheme** | Proof the untrusted party computed correctly (Freivalds, batch verification). | The missing trust layer; P3 pulls it forward. |
| **Assurance check** | Harness-level negative controls (weakened fixtures must *fail*). | Functional tests ≠ security proof. |
| **Adversary** | Fixture component implementing a published attack or leak view. | e.g., `CarnivalProjection`. |
| **Provenance** | `[Rnn]`/`[PLLM]` citation in a descriptor — where the method came from. | Papers are provenance, not packages. |
| **Conformance** | Per-category contract tests a component must pass. | The `check_estimator` analogue — gates composition. |

### Evidence

| Term | What it is | Note |
|---|---|---|
| **measurement.v1 / EvidenceBundle** | Sanitized, digest-bound benchmark record (no prompts/seeds). | What VirtualDC prices and routes on. |
| **Matched cohort** | Same model fingerprint + workload + warm state — required for a fair arm-vs-arm comparison. | `benchmark_run` enforces it. |
| **Pareto frontier** | Multi-objective comparison output; no single winner declared. | Latency vs traffic vs memory vs energy vs quality. |

### How they fit together — the 6-step flow

```text
Experiment { Pipeline { Model, slots→components }, Deployment, Budget }
      │ resolve()
      ▼
ExperimentProfile { roles, topology, requirements, privacy_contract, fingerprint }
      │ build_roles()
      ▼
LocalTopology { RoleHandles }  ──►  OpenAI(base_url=topo.gateway_url)
      │ benchmark_run()
      ▼
measurement.v1 (sanitized evidence)

parallel path: Model ─► load_model() ─► ModelManifest   (runtime weights)
               Model ─► lower_model() ─► ModelPlan ─► apply(pass) ─► compile() ─► CompiledPlan
```

---

## 10. Appendix C — complex CLI/SDK recipes

More involved combinations of the designed API (§7). Proposed names marked *(new)*; everything else maps to shipped types. These are the "how it all composes" examples the getting-started docs should eventually carry.

### C.1 Full private pipeline — declare → resolve → serve → request → evidence

```python
import pllm
from pllm.profiles    import MaskedLinearCpu      # a Pipeline subclass — its __init__
from pllm.protocols   import MaskedLinear         #   signature IS the slot contract
from pllm.preparation import ModelAwareCorrections
from pllm.roles       import Inference
from pllm.kernels     import Cpu

pipe = MaskedLinearCpu(                          # every slot is a typed kwarg —
    model=pllm.Model("Qwen/Qwen2.5-0.5B-Instruct"),  # IDE completes them; wrong-category
    linear=MaskedLinear(),                           # components fail isinstance at
    preparation=ModelAwareCorrections(),             # construction and in mypy/pyright
    inference=Inference(),
    kernels=Cpu(threads=8),
)

exp = pllm.Experiment(
    name="qwen-seeded-cpu8",
    pipeline=pipe,
    deployment=pllm.Deployment.local(root=".pllm/qwen-seeded-cpu8"),
    budget=pllm.ExecutionBudget(requests=4, max_input_tokens=128, max_new_tokens=32),
)

plan = exp.resolve()                                  # ExperimentProfile: roles + privacy contract
with pllm.serve_local(plan) as topo:                  # one builder, all fronts share it
    client = pllm.OpenAI(base_url=topo.gateway_url, api_key=topo.client_key)
    resp = client.responses.create(
        model=exp.pipeline.model.source,
        input="Summarize private inference in one sentence.",
        max_output_tokens=24,
    )
    report = pllm.benchmark_run(exp, topology=topo)   # instrumented run over the same roles
    print(report.measurement.digest)                  # sanitized evidence digest
```

CLI equivalent: `pllm benchmark run --experiment qwen-seeded-cpu8.yaml` — same resolved plan, same roles.

### C.2 Matched-arm benchmark — seeded vs BFV vs cleartext on identical workload

```python
from pllm.profiles    import PlaintextRef            # a different profile class =
from pllm.preparation import BFVCorrelations         #   a different slot contract
from pllm.protocols   import CleartextLinear
from pllm.kernels     import Cpu

bfv = pipe.with_params(preparation=BFVCorrelations(poly_degree=8192))   # slot swap by kwarg
ref = PlaintextRef(model=pipe.model, linear=CleartextLinear(backend="vllm"),
                   kernels=Cpu(threads=8))   # reference profile — no privacy claim by construction

report = pllm.benchmark_run(
    [exp.with_params(pipeline=bfv), exp.with_params(pipeline=ref)],
    metrics=["latency", "traffic", "rss"],            # declared objective set
    require_same_model=True,                          # matched cohort enforced
)
for row in report.pareto():                           # no single winner — frontier by objective
    print(row.arm, row.latency, row.traffic, row.rss)
```

CLI: `pllm benchmark run --experiment seeded.yaml --experiment bfv.yaml --experiment cleartext.yaml`.

### C.3 Two-host deployment — preparation trusted-SaaS, inference untrusted node

```python
# Host A (trusted prep SaaS / enterprise):
with pllm.serve_preparation(listen=("0.0.0.0", 8090)) as prep:
    print(prep.capabilities())            # host features it advertises

# Host B (untrusted inference node — a VirtualDC gaming PC):
with pllm.serve_inference(model="Qwen/Qwen2.5-0.5B-Instruct",
                          listen=("0.0.0.0", 8080)) as inf:
    print(inf.capabilities())             # what this node can host

# Client (anywhere):
plan = exp.resolve()
plan.role_requirements()                  # what each role needs — the VirtualDC routing input
client = pllm.OpenAI(base_url="https://inference.node:8080",
                     preparation_url="https://prep.saas:8090",
                     api_key=...)
```

VirtualDC consumes `role_requirements()` + node `capabilities()` + signed `measurement.v1` — PLLM emits, VirtualDC routes.

### C.4 Compiler-pass transform — apply, diff coverage, compile, run a region

```python
from pllm.passes import KvCacheEviction

plan = pllm.lower_model(pllm.Model("Qwen/Qwen2.5-0.5B-Instruct"))   # semantic ModelPlan
before = plan.coverage("research.single_evaluator")

plan2 = plan.apply(KvCacheEviction(observation_window=(1, 5)))      # new immutable plan
after = plan2.coverage("research.single_evaluator")
print(before.summary()["kv_cache_append"], "→", after.summary()["kv_cache_append"])

compiled = pllm.compile(pllm.CompileRequest(plan=plan2, profile="research.single_evaluator"))
# compiled executes only covered regions — anything else fails closed
```

### C.5 Third-party component — author → manifest → register → use

```python
# provider_pkg/__init__.py — a class IS the descriptor source
import pllm.components as pc

class MyApproxSilu(pc.ComponentRef):
    component = "provider/approx-silu/v1"
    def __init__(self, *, degree: int = 7): ...
    @classmethod
    def describe(cls): ...        # ComponentDescriptor: category=pllm/nonlinear-protocol,
                                  #   provenance="provider" or "R18", parameter_schema={...}

# pyproject: [project.entry-points."pllm.providers.v1"] → manifest JSON

# user side — same machinery as builtins:
pc.get("provider/approx-silu/v1")                    # → MyApproxSilu class
pipe = base.with_params(nonlinear=MyApproxSilu(degree=9))   # slot kwargs all the way down
exp2.resolve()                                       # category check + param validation + conformance
```

### C.6 Assurance run — attach integrity + an adversary fixture, verify detection

```python
from pllm.verification import LinearIntegrity
from pllm.adversary    import CarnivalProjection

armed = exp.with_params(pipeline=pipe.with_params(
    integrity=LinearIntegrity(),
    adversary=CarnivalProjection(),                # the published attack, as a fixture
))

result = pllm.assure_run(armed)                        # new surface (P5 CLI/API)
assert result.detected("carnival_projection")           # attack must be *caught*, not passed
assert result.privacy_contract.holds                    # honest check, not a test-green claim
```

### C.7 Custom metrics + evidence export for VirtualDC pricing

```python
from pllm.metrics import Latency, Traffic, Rss, Disk, Energy, Cost

report = pllm.benchmark_run(
    exp,
    metrics=[Latency(), Traffic(), Rss(), Disk(), Energy(), Cost(usd_per_kwh=0.12)],
)
bundle = report.to_evidence_bundle()                    # pllm.measurement.v1, digest-bound
bundle.sign(node_key)                                   # signed node self-report → VirtualDC
```

### C.8 The reimplementation gate as API — one paper's path to a component

```python
# R24 Maverick through the visible gates (docs/content/docs/research/backlog.mdx):
spec    = pllm.research.lock_source("r24-maverick", sha256=...)      # source + version pinned
refimpl = pllm.research.reference(spec)                              # clean-room reference
fidelity= pllm.research.fidelity_tests(refimpl)                      # source-scope checks
comp    = MaverickMatvec()                                            # reusable native component
fidelity.register(comp)                                               # fidelity → component link
plan    = exp.with_params(pipeline=pipe.with_params(linear=comp)).resolve()
                                                              # typed integration
review  = pllm.assure_run(plan)                                       # scoped security review
report  = pllm.benchmark_run(plan)                                    # matched benchmark
decision= pllm.research.promote(plan, review=review, report=report)   # documented promotion
```

Each gate is recorded on the component's descriptor (`evidence`, `artifacts`) — the generated components reference stops being a table of "not recorded".

---

## 11. Appendix D — publication & documentation plan

Two documents, two audiences, one discipline: the honest-claims register the current drafts already use. `paper/` already contains `manuscript.md` (302 lines → `paper.pdf`), `whitepaper.md` (142 lines → `whitepaper.pdf`), `references.bib` (17 entries), `header.tex`, pandoc lua filters, and web mirrors at `docs/content/research/{paper,whitepaper}.mdx`. Neither document mentions VirtualDC — **verified; keep it that way** (the whitepaper's "what comes next" must describe the runtime/harness roadmap, not fleet economics).

### D.1 arXiv pre-print — specific amendments to `paper/manuscript.md`

**Target:** `PLLM: A High-Performance Private LLM Multi-Party Inference Runtime and Autonomous Research Harness`. Pre-print for arXiv (`cs.CR` primary, `cs.LG`/`cs.DC` cross-list). The current draft is already conservative and well-structured — the revision is mostly *addition*, not retraction.

**Section-by-section amendment table:**

| Current | Amendment | Why |
|---|---|---|
| Title | → target title above (names runtime + harness explicitly) | User-specified; matches the one-line description used everywhere |
| Abstract | Keep the honest hedges (incomplete compiler, non-collusion, 9-run evidence). Add one sentence framing the component model as the contribution researchers consume. | The abstract currently describes *what exists* but not *why it matters to a reader*. |
| §1 Introduction (3 paras) | Expand to the 5-move arc: (a) the private-inference problem and opportunity — prompts/outputs are confidential data and hosted endpoints see everything; (b) the research-fragmentation problem — ~24 papers, each a monolithic stack with different models, hardware, threat models, metrics; techniques can't be composed or fairly compared; (c) the toolkit precedent — scikit-learn and Keras turned fragmented ML research into composable, swappable components and accelerated the field; privacy research has no equivalent; (d) PLLM's answer — runtime + typed component model + evidence-bound harness; (e) contributions + explicit non-claims (already drafted — keep). | This is the requested framing: *why PLLM exists*, not just *what it is*. The sklearn/keras analogy is the paper's thesis — a component model for AI-privacy researchers to experiment with and distribute innovations. |
| — (none) | **New §2: Background — the private-inference landscape.** Organize the field by technique family, not chronology: (1) HE linear layers — BFV/Fan-Vercauteren, Cheetah, Iron, THEX, NEXUS [R21], BumbleBee [R22]; (2) garbled circuits — Dash [R01], ReDASH [R02], gadgets [R03], half-gates [R04], LogRow LUTs [R05], Duty-Free Bits [R06]; (3) trusted-hardware masking — Slalom [R07], Carnival [R08]; (4) hybrid compilation — HyCC [R09]; (5) decision programs & BDDs — [R10], FEVBDD [R11]; (6) GPU MPC platforms — Piranha [R12]; (7) function secret sharing — SIGMA [R13], FuseFSS [R14]; (8) correlation generators — ring-PCG [R15]; (9) delegated linear algebra — trapdoored matrices [R16]; (10) encoding/approximation — Curl LUTs [R17], Compact [R18]; (11) verifiable outsourcing — [R19]; (12) ciphertext compression — [R20]; (13) MPC-friendly state — MPCache [R23]; (14) verifiable matvec — Maverick [R24]. Each subsection: 2–4 sentences on the technique + where it lands in PLLM's taxonomy (the §7.3 family map). | Satisfies "cite all of the research we are reimplementing" *and* supplies the private-inference background the user wants — one section does both jobs. |
| — (none) | **New §3: The component model.** Families as typed capability contracts (`pllm.<family>`), descriptors (category/params/capabilities/threat fields), immutable plans + lineage, conformance gates, matched-cohort benchmarks. Explicit sklearn/keras/deepeval comparison table: `ComponentRef` ↔ `BaseEstimator`, `with_params` ↔ `set_params`, category conformance ↔ `check_estimator`, `Pipeline` ↔ `Pipeline`, identity-string serde ↔ keras `get_config`. | The "distribute innovations" argument: a privacy researcher ships a *component* that composes with everything else, not a fork. This is the section that makes the paper more than a system report. |
| §2 Prepared inference protocol | Keep verbatim (well-written). Renumber. Optionally add one diagram: three-role sequence chart (seeds→prep, corrections→inference, ticket+masked→inference). | The protocol description is already arXiv-quality; a figure helps reviewers. |
| §3 Threat model and limits | Keep verbatim. | The honesty here is a strength — don't dilute it. |
| §4 Implementation status | Keep; split "component library and autonomous search" subsection into the new §3 (design) vs here (status). Update garbling paragraph when the staged boolean-gates WIP lands. | Separates *design* (§3) from *evidence of implementation* (§6). |
| §5 Historical evaluation | Keep verbatim, including both tables and all disclaimers. Add one sentence: future evaluations are matched-cohort component comparisons under the harness (§3), not single-system benchmarks. | Positions the 9-run study as baseline evidence, not the evaluation methodology. |
| §6 Conclusion | Light edit — end on the research-community ask (components, reproductions, profiles as contributions), not only the runtime roadmap. | Mirrors the new intro arc. |
| `references.bib` (17 entries) | **+24 entries for the full tracked corpus** — generate from `docs/data/research/papers.json` (each record already has title/authors/year/primary_url/sha256): `sander2024dash` [R01], `redash` [R02], `ball2017garbling` [R03, exists], `zahur2015halfgates` [R04], `logrow` [R05], `dutyfreebits` [R06], `tramer2019slalom` [R07], `carnival` [R08], `buscher2018hycc` [R09], `obliviousdp` [R10], `fevbdd` [R11], `watson2022piranha` [R12], `sigma-fss` [R13], `fusefss` [R14], `ringpcg` [R15], `trapdoored-mat` [R16], `curl` [R17], `compact` [R18], `openweight-verifiable` [R19], `oblivious-compression` [R20], `nexus` [R21], `bumblebee` [R22], `mpcache` [R23], `maverick` [R24]. **+3 toolkit cites**: `pedregosa2011scikit`, `chollet2015keras`, `deepeval`. **+background completes**: `juvekar2018gazelle`, `mishra2020delphi`, `knott2021crypten` (round out the HE/MPC landscape paragraphs). | One bib generated from the registry = citations can't drift from the tracked corpus; same source of truth as the docs papers pages. Verify venue fields against `primary_url` before submission. |

**arXiv mechanics checklist:** pandoc → standalone LaTeX (`--standalone`, embed `.bbl`); figures as committed PNG/TikZ under `paper/figures/`; no external `\input` beyond arXiv-supported packages; strip `web-date`/`web-note`/`edition` fields (docs-site concerns); metadata — `cs.CR` primary, `cs.LG`, `cs.DC`; submission tarball includes `header.tex`, figures, `.bbl`; run `paper/README.md` build steps and add a `make arxiv` target that emits the exact tarball. Also mirror to `docs/content/research/paper.mdx` (134-line web version must track the revision).

**Guardrails:** no VirtualDC; no performance claims beyond the retained 9-run study; "autonomous" stays qualified (gates are human/CI-enforced, search is planned); every `[Rnn]` citation marked as reproduction status honestly (mostly `reproduction_target` today — the paper cites the *sources*, not completed reimplementations).

### D.2 Two-page business whitepaper — rebuild of `paper/whitepaper.md`

**Current state:** 142 lines ≈ 3 dense pages; contains LaTeX math (`$c=Wr-s$`, `$W(x-r)+c=Wx-s$`), zero figures, semi-technical vocabulary ("masked matrix corrections", "semantic representation", "capability contracts"). Wrong register for the audience.

**Target:** exactly 2 pages, letter, single-column (drop `twocolumn` — diagrams need width), 10–11pt, generous whitespace. Audience: engineering leads, partners, procurement — *not* cryptographers.

**Page budget:**

| Page | Content |
|---|---|
| 1 | "What PLLM is" (3 sentences, no jargon) → "The problem" (hosted inference sees your data; TLS only protects the wire) → **Figure 1: three-role data-flow** → "How it works" in plain words (client keeps everything sensitive; a trusted prep service pre-computes reusable-looking "unmasking" values; the untrusted provider only ever sees masked numbers) → **Figure 2: privacy-boundary map** |
| 2 | "The research toolkit" (why a component library — the sklearn analogy in one plain sentence: "the way scikit-learn let every data scientist swap algorithms, PLLM lets privacy researchers swap techniques") → **Figure 3: component-slot swap** → **Figure 4: research loop** → "Status today" honesty box → "What comes next" (runtime completion, more components, broader evaluations — no VirtualDC) |

**Diagram specs** (commit as SVG/PNG under `paper/figures/`; also reused on the docs landing page and whitepaper web route):

1. **Three-role data-flow** — three boxes (Client [lock icon] / Preparation / Inference Provider). Arrows labeled in plain words: "random seeds (offline)", "pre-computed unmasking values", "one masked request per step", "masked result". Inside Client box: prompt, output, "model's sensitive numbers". No equations anywhere.
2. **Privacy-boundary map** — dashed boundary around the client. Inside: prompt, generated text, tokens, the model's internal numbers, decoding choices. Outside: model weights (public), masked numbers, tickets, timing. A "never crosses the boundary" checklist.
3. **Component-slot swap** — a pipeline strip `[input → linear math → prep → inference → output]` with each stage drawn as a socket; alternate components shown plugging in (seeded vs. encrypted vs. plaintext prep; CPU vs. GPU kernels). One-line caption: "swap techniques like parts — every combination gets measured the same way."
4. **Research loop** — circular: pin paper → reimplement → validate → compose → benchmark → keep the evidence. Caption: "new research becomes a component, not a rewrite."

**Jargon rules:** no math notation; define terms on first use in parentheses — "activations (the model's intermediate numbers)", "weights (the model's public parameters)". Avoid entirely: homomorphic, garbling, MPC, two-party, Q7, ring — or gloss once in a footnote. Keep the honesty: "Status today" box states plainly — working private runtime for supported open models on CPU; the compiler does not yet run a complete model; evaluation is a small historical loopback study; preparation must be trusted and must not collude with the provider. Business-friendly ≠ overclaiming — that's the brand.

### D.3 Docs-site audit — findings and dispositions

**Verified defects / removals:**

| Finding | Evidence | Disposition |
|---|---|---|
| 3 empty content roots | `server/`, `components/`, `client/` — 0 `.mdx` files each (created Sep 14) | Delete the dirs (and any meta.json/nav references) |
| Duplicated paragraph | `recipes/index.mdx` repeats "Use the pages in this section…" twice | Fix |
| ~13 index-only sections | single <20–40-line index with no children: `recipes` 20, `search` 19, `representations` 18, `conversions` 18, `metrics` 17, `numerics` 41, `compiler` 40, `preparation` 37, `runtime` 36, `pipeline` 36, `deployment` 34, `assurance` 34, `kernels` 33 | Either fill with real content (per gaps below) or fold into the parent hub page — an empty section is worse than no section |
| File path ≠ public URL | `lib/docs-routes.mjs` prefix map: `start/`→`/learn/start/`, `understand/`→`/learn/understand/`, `metrics/`→`/research/records/metrics/`, `protocols/`→`/sdk/pipeline/protocols/`, ~30 source dirs → 4 nav areas | Not a defect — but undocumented indirection is a contributor trap. Document the map in `CONTRIBUTING` + add a comment header in `docs-routes.mjs`; long-term consider co-locating source dirs by public path |
| Stale/broken pages | `start/first-private-request.mdx` experiment fails resolution (tracked, P0.4); `deploy/README.md` `pllm build` (P0.4) | Already in P0 — keep |

**Gaps vs. the rest of this plan (new docs to write):**

| Gap | Lands with | Source material |
|---|---|---|
| Glossary page (public-facing) | P6.28 | §9 glossary — adapt, don't copy verbatim |
| `load_model`/`serve_*`/`build_roles` SDK guide | P1.5–1.6 | §7.4 examples |
| Component-authoring guide (real one — current `contribute/add-a-category` is 14 lines) | P2 | §7.5 layout + Appendix C.5 recipe |
| Capability/evidence-export reference (what `capabilities()`, `role_requirements()`, `measurement.v1` emit) | P4 | §5.7/§7 contract — frame as integration boundary, no VirtualDC naming needed |
| Env-var reference page | P6.28 | §8.4 inventory — generate from `settings.py` like the CLI reference |
| Benchmark methodology page (`benchmarks/` is 1 page, 66 lines, for a harness whose whole point is measurement) | P3 | matched-cohort/Pareto rules from §7 + metrics catalog |
| Paper/whitepaper web mirrors | P6.26–27 | `docs/content/research/{paper,whitepaper}.mdx` must track both revisions |

**Keep-as-is (genuinely good):** the evidence/claims discipline pages (`understand/evidence-claims`, `privacy-assurance`, `trust-boundary`), the research backlog + per-paper cards (auto-generated from `papers.json` — the strongest part of the site), the honest capability framing throughout (`garbling/index` explicitly says "reference-only, unreviewed, excluded from executable profiles"). This register is the site's differentiator — consolidation must preserve it.

---

## 12. Appendix E — comparative benchmark recipes (competing research approaches)

Appendix C showed how the API composes; this appendix specifies **which comparisons matter** — pairs (and triples) of components that compete for the same slot, drawn from the cited corpus. Each recipe is a claim under test: the papers assert different things, and the matched-cohort benchmark is what adjudicates them *in PLLM's setting*.

**Comparison rules (what makes a fair fight):**
- **Matched cohort** — same model fingerprint, same workload (input lengths, output caps, request count), same warm/cold state, same host. `benchmark_run` refuses the cohort otherwise.
- **Same slot(s)** — arms differ by exactly the slots under comparison; everything else pinned identical.
- **Threat differences are reported, not hidden** — a 2PC arm and a 3-role arm are *not* trust-equivalent; the Pareto table carries `corruption_model`/`party_topology` columns so a cheaper-but-weaker arm is visibly weaker, never silently "faster".
- **Every arm gets the plaintext denominator** — `PlaintextRef` runs in every cohort as the zero-privacy floor; private arms report *overhead vs.* it, not absolute numbers alone.
- **Status honesty** — `[run]` = both arms exist today, `[partial]` = one arm stubbed/flag-driven, `[blocked]` = needs the P0/P1 component work first.

### E.1 The `preparation` slot — how offline material gets made

| Recipe | Arms | Claim under test | Differentiating metrics | Status |
|---|---|---|---|---|
| **E1** seeded vs HE | `ModelAwareCorrections` [PLLM] vs `BFVCorrelations` [BFV] | BFV preprocessing buys stronger independence at what prep latency/byte cost? | prep time/token, prep→inference bytes, host features | [partial] — BFV arm is flag-driven today |
| **E2** correlation sources | `SeededExpansion` [PLLM] vs `RingPcg` [R15] vs `CorrelationOle` [R15] vs `AuthenticatedTriple` [R15] | Do PCG/OLE correlations beat seed-PRG expansion per prepared row? | CPU/row, bytes/row, dealer requirement | [blocked] — needs `correlation-source` family (§7.3) |
| **E3** trust shape | `BFVCorrelations` vs `HeAuthenticatedPreprocessing` [HE-lit] | Authenticated preprocessing's malicious-security upgrade vs honest-but-curious BFV — worth the overhead? | prep time, material size, threat column | [blocked] |

### E.2 The `linear` slot — who computes Wx and how it's delegated

The 63–433 MB/request problem lives here — this is the highest-value comparison group.

| Recipe | Arms | Claim under test | Differentiating metrics | Status |
|---|---|---|---|---|
| **E4** delegation trio | `MaskedLinear` [PLLM] vs `MaverickMatvec` [R24] vs `TrapdooredLinear` [R16] | Three delegation schemes for the same Wx: which minimizes **client I/O** without losing exactness? | client bytes/request, inference CPU, one-use material cost | [blocked] — R16/R24 not implemented |
| **E5** FHE ceiling | `MaskedLinear` vs `DirectFhe` [proprietary arm] | Is direct FHE's stronger boundary worth its constant factor on CPU? | wall time, memory, energy | [partial] — direct arm exists as flag |
| **E6** proprietary guard | `GuardedLinear` vs `BlindedLinear` vs `SecureLinear` [proprietary arms] | Which weight-hiding construction has the best query-rate/security trade-off? | rows/session, dither noise vs quality, provider CPU | [partial] — flags today |

### E.3 The `nonlinear` + `labels` slots — protected SiLU, four paradigms

| Recipe | Arms | Claim under test | Differentiating metrics | Status |
|---|---|---|---|---|
| **E7** the big nonlinear bake-off | `BinaryTableGatedMultiplyQ7` vs `R03CrtGatedMultiplyQ7` [R03] vs FSS eval [R13/R14] vs `ApproximationProfile`-fixed poly [R18] vs garbled-SiLU [R01] | Four paradigms for the *same op*: garbled table vs CRT gadgets vs FSS vs polynomial approx — which wins per element? | payload bytes (2.39 MB vs 245 kB already measured!), eval ms/element, label-gen cost, accuracy bound | [partial] — two arms measured, FSS/R18 absent |
| **E8** label schemes | `AffineLabel` [R01+R03] vs `BooleanHalfGates+FreeXor` [R04] vs `LogRow` [R05] vs `DutyFreeBits` [R06] | R04's claim: ~50% comms cut vs classic garbling; R05/R06: fewer ciphertexts per gate — do they hold on our Q7 workload? | ciphertexts/gate, wire bytes, garble+eval ms | [blocked] — needs `label-scheme` family |
| **E9** lineage check | `Dash` [R01] vs `ReDash` [R02] | R02 asserts better scaling than R01 — reproduce the *improvement claim itself* in-harness | throughput vs element count curve | [blocked] |

### E.4 The `state` slot + `verification` slot + `codecs`/`kernels`

| Recipe | Arms | Claim under test | Differentiating metrics | Status |
|---|---|---|---|---|
| **E10** KV strategies | `ClientLocalKv` [baseline] vs `SecretSharedKv`+`MpcacheAttention` [R23], ± `KvCacheEviction` pass | Does MPCache's shared-KV decode actually cut client state? Does eviction compose with it? | decode-phase client RSS, bytes/token, eviction-accuracy bound | [partial] — eviction pass exists, shared KV absent |
| **E11** verification trio | `LinearIntegrity` [PLLM] vs `FreivaldsVerify` [R07] vs `BatchVerification` [R24] vs `OpenWeightProof` [R19] | Verification overhead vs assurance strength: cheap integrity counter → Freivalds spot-checks → batch proofs → full open-weight proof | verify ms/op, bytes of proof, soundness level (column) | [partial] — LinearIntegrity orphaned, others absent |
| **E12** transport compression | plain packed transport vs `SyndromeCodec` [R20] | R20's oblivious compression on real correction/ticket traffic — actual byte reduction? | wire bytes/request, codec CPU | [blocked] |
| **E13** kernel substrate | `Cpu(threads=N)` vs `Wgpu` vs `Tenseal` host feature | Is consumer-GPU integer masked-linear bandwidth-bound (VirtualDC feasibility question) or a real speedup? | tokens/s, energy/token, memory | [blocked] — wgpu is P4.21 |

### E.5 Full-system cohorts — profile vs profile (the headline bake-offs)

These vary the *whole pipeline*, not one slot — the comparisons papers actually fight over.

| Recipe | Profiles | Question | Metrics | Status |
|---|---|---|---|---|
| **E14** the field | `MaskedLinearCpu` [PLLM] vs `NexusHe` [R21] vs `Bumblebee2pc` [R22] vs `OpenWeightVerified` [R19] vs `PlaintextRef` | Four system philosophies (3-role masking, non-interactive HE, 2PC, verified outsourcing) on the *same model + workload* | TTFT, tok/s, client bytes, role count, threat columns | [blocked] — needs P0 profile + cohort profiles |
| **E15** garble-everything vs split | `SingleEvaluator` (R01–R06 garbling stack) vs `MaskedLinearCpu` | Can a single-evaluator garbled decoder beat the 3-role split on a real model — or is it a leaf result? | end-to-end generation cost | [blocked] — needs complete garbling coverage |
| **E16** party topology | `TwoPartyMpc` (shared-mpc substrate) vs `MaskedLinearCpu` | 2-party vs 3-party for the same Wx: which topology is cheaper at equal privacy? | total traffic, role coordination overhead | [partial] — substrate exists, no plan path |
| **E17** FSS lineage | `SigmaFss` [R13] vs `FuseFss` [R14] | FuseFSS's claimed efficiency over SIGMA on LLM ops — holds in matched cohort? | op latency, key/dataset bytes | [blocked] |

### E.6 The recipe pattern (code)

Every comparative recipe is the same shape — clone the base experiment, swap exactly the compared slots, run the cohort:

```python
import pllm
from pllm.profiles    import MaskedLinearCpu, PlaintextRef
from pllm.protocols   import MaskedLinear, MaverickMatvec, TrapdooredLinear, CleartextLinear
from pllm.preparation import ModelAwareCorrections
from pllm.roles       import Inference
from pllm.verification import LinearIntegrity, FreivaldsVerify, BatchVerification
from pllm.metrics     import Traffic, Latency, Rss

base = MaskedLinearCpu(model=pllm.Model("Qwen/Qwen2.5-0.5B-Instruct"),
                       linear=MaskedLinear(), preparation=ModelAwareCorrections(),
                       inference=Inference(), kernels=Cpu(threads=8))
dep, budget = pllm.Deployment.local(root=".pllm/e4"), pllm.ExecutionBudget(requests=8, max_new_tokens=16)
plaintext_floor = pllm.Experiment(name="floor", budget=budget, deployment=dep,
    pipeline=PlaintextRef(model=base.model, linear=CleartextLinear(), kernels=Cpu(threads=8)))

# E4 — delegation trio: only `linear` (and its paired verifier) differ
arms = {
    "seeded":    base,
    "maverick":  base.with_params(linear=MaverickMatvec(), integrity=BatchVerification()),
    "trapdoor":  base.with_params(linear=TrapdooredLinear()),
}
report = pllm.benchmark_run(
    [pllm.Experiment(name=n, pipeline=p, deployment=dep, budget=budget)
     for n, p in arms.items()] + [plaintext_floor],
    metrics=[Traffic(), Latency(), Rss()],
    require_same_model=True, require_same_workload=True,
)
report.pareto(columns=["arm", "client_bytes", "tok_per_s", "rss_peak",
                       "corruption_model", "party_topology"])
#            ↑ threat columns always printed — a cheaper arm that trusts more
#              shows up as cheaper *and weaker*, never just "faster"
```

### E.7 Sequencing against the roadmap

- **[run] today**: E3's two measured arms (dense-table vs R03-CRT payloads) — extend to eval-time numbers
- **[partial] → runnable after P1.7**: E1, E5, E6, E10, E11 (componentize the flag-driven arms; wire `LinearIntegrity`)
- **[blocked] → needs P2/P3 + paper implementations**: E2, E4, E7 (FSS/R18 arms), E8, E9, E12, E14, E15, E16, E17 — these land as the §7.3 research backlog executes (dependency order: R24+R23 roots, R03+R04 unlock six, R07 unlocks Carnival)
- **E13 (wgpu)** gates VirtualDC node economics — P4.21
- Every completed recipe emits a `measurement.v1` row per arm → the evidence corpus the arXiv paper's evaluation section grows into (Appendix D.1's "future evaluations are matched-cohort component comparisons" made concrete)
