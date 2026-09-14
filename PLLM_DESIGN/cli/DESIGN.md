# Composing and running PLLM across parties

Status: proposed API and CLI contract, 14 September 2026. The examples require the
planned PLLM uplift. This bundle does not implement or run private inference.

## 1. One composition, three entry points

Python objects, declarative YAML and locked plans feed the same native validation,
compilation, preparation and execution services. The CLI is not a second runtime.

Separate:

- `Pipeline`: model, numerical profile, named component bindings, privacy/role policy.
- `Deployment`: how required logical parties map to processes, endpoints and identities.
- `Experiment`: pipeline plus deployment, execution budget and optional measurement policy.
- `Runtime` / `Session`: live opaque native state, never cloned or serialized as configuration.

Named components are bindings into a profile/semantic graph, not sequential sklearn
transformers. Profiles supply the remaining required operators, conversions, numerical
manifest, role topology and eligibility constraints. Replacements must satisfy their slot
contracts. Component registration alone does not establish complete model coverage.

## 2. Starting profiles and support

`baseline.masked_linear_cpu` is the first end-to-end integration target. It retains the
archived client-local embeddings, head, attention, nonlinearities, state and sampling.
It is explicitly not the thin-client target. Its frozen numerical manifest must be
recovered before calling it a reproduction. Missing manifests produce E_NUMERIC_MANIFEST.

`research.single_evaluator` is the preferred architecture: thin Client, offline-only
model-aware Preparation, one online encoded evaluator, no HE. Missing attention,
normalization, rescaling, sampling or state implementations produce E_COVERAGE.
Neither profile name is evidence that the corresponding current wheel supports it.

The profile identities are versioned and resolved into a lock. Model source names may
be convenient in authoring, but execution uses checkpoint, tokenizer, numeric graph,
component/native artifact and plan hashes. No generated model hash or fake plan.lock
is included in this bundle.

## 3. Python example

`examples/private_app.py` creates an immutable `Experiment` with named `linear`,
`preparation` and `kernels` bindings. Constructors only create declarations. The guarded
`main()` starts the runtime, prepares one bounded request, and streams its output.

`Runtime.__enter__` resolves/validates/compiles as necessary and establishes the selected
deployment. It must not perform online inference. `runtime.prepare()` explicitly creates
and installs material for the budget. `runtime.session()` reserves a fresh allocation.
`session.generate()` consumes installed material only: it must not call Preparation.

The budget counts requests and bounds the complete tokenized input, including templates,
prior turns and control tokens. `max_new_tokens` is an output cap, not a promise of exact
length. A fixed-length benchmark has separate EOS/dummy-work semantics.

Unused material is retired under the selected lifecycle policy, not recycled onto a new
input. Reusing a public plan, a model or a Python configuration does not authorize reusing
execution material. Closing a context cancels/drains requests and retires exposed allocations.

## 4. Pure CLI

Create a starter configuration without writing Python:

```sh
pllm init --profile baseline.masked_linear_cpu \
  --model Qwen/Qwen2.5-0.5B-Instruct --output pllm.yaml
pllm plan check pllm.yaml
pllm run pllm.yaml
```

`init` writes a compact profile configuration with the documented starter budget (one
request, 128 input tokens, 32 output tokens) and four CPU threads. It does not download,
compile or start parties. It refuses to overwrite a file unless `--force` is explicit.
`run` prompts on a TTY. Noninteractive input requires `--input-file PATH` or
`--input-file -`; absent input fails instead of hanging.

The equivalent YAML is `examples/pllm.yaml`. IDs such as `pllm/cpu` are proposed PCS
component IDs; their exact native artifacts are resolved at compilation.

A default `run` performs a visible preflight preparation phase, then disconnects the
corresponding Preparation path before online execution. For reused prepared inventory,
`--prepared-only` forbids starting/contacting Preparation, including health probes/refill.
Insufficient material fails before exposing a newly assigned input when possible.

## 5. Execute Python-defined composition

Supported targets:

| Target | Meaning |
| --- | --- |
| `pllm.yaml` | Declarative Experiment |
| `private_app.py:experiment` | Explicit local file and public object |
| `myproject.pipelines:experiment` | Importable module and object |
| `myproject.pipelines:make_experiment --factory` | Explicit zero-argument factory |
| `build/qwen/plan.lock.json` | Locked public execution specification |

An object is an Experiment or Pipeline. A bare Pipeline requires deployment/budget
configuration from its profile or explicit options. Unknown required fields fail.

```sh
pllm run private_app.py:experiment --input-file question.txt
pllm config export private_app.py:experiment --output exported.yaml
pllm plan compile private_app.py:experiment --output build/qwen
```

This borrows the explicit module/object/factory convention familiar from Uvicorn, with
an additional local-file form. A file or factory is trusted Python code and executes on
the invoking host only. No remote code import occurs merely because a deployment is
remote. The export contains public component IDs/parameters, not pickles or callables.
Python objects not serializable into the supported contract are rejected with the path
of the offending field. Source/lock provenance is recorded; it is not a sandbox.

Every party receives its public plan slice and separately delivered secret material.
Do not send the Python source, a serialized closure or the whole coordinator environment
to the other parties. Installed native providers must be independently approved there.

## 6. Overrides and configuration identity

Use `Experiment.with_params(...)` and `--set PATH=VALUE` with the same public
`get_params(deep=True)` paths:

```python
changed = experiment.with_params(pipeline__components__kernels__threads=8)
```

```sh
pllm run pllm.yaml --set pipeline__components__kernels__threads=8
```

The `params` wrapper in serialized component nodes is an encoding detail and is omitted
from these paths. The category schema gives types. Parse scalars/arrays as safe data,
never Python expressions; do not use eval or arbitrary class-path construction.

Resolution precedence: versioned profile < object/file < explicit deployment replacement
< explicit `--set`. Secret values do not come from `--set`. Paths for identities name
role-local credential descriptors, not raw private keys. Unknown paths and invalid types
fail. CLI shell completion and parameter help come from those same schemas.

A locked plan cannot be altered in place with `--set`. Fork its public configuration and
recompile; invalidate incompatible prepared material. Changing a peer identity or native
implementation likewise requires a compatible newly authorized execution binding.

`--deployment remote.yaml` replaces placement, not protocol/privacy/numerical choices.
A forbidden co-location or missing role is a compatibility failure, not an auto-repair.
The coordinator can validate declared identities, not prove administrative non-collusion.

## 7. Local orchestration

`Deployment.local(root=...)` starts a native supervisor with three role-isolated
subprocesses for this profile. The Python API communicates with the trusted local Client;
Inference and Preparation are distinct native processes. Credentials and secrets are
separated by role; public locks/caches can be shared. This is a development simulation,
not three independently administered trust domains.

The sequence is:

1. Resolve public descriptors and check source/native/policy compatibility.
2. Resolve model/tokenizer/numerical artifacts and lock exact identities.
3. Compile party-specific plans, including required conversions and state.
4. Compute an explicit resource estimate and reject over-budget material.
5. Prepare locally at the Preparation process; send secret slices directly to recipients.
6. Receive readiness and close Preparation for the allocation.
7. Bind the Client request before exposure; execute the protocol; decode at Client.
8. Retire used/unused reservations and emit scrubbed metrics.

The baseline still exchanges masked intermediates with the Client. Only a fully covered
single-evaluator profile can keep that computation remote. The launcher never changes
protocol placement to make a model run.

## 8. Separate hosts

The example `remote.yaml` binds Client locally and attaches to remote Preparation and
Inference services. Its `.example.net` domains and identity paths are deployment examples;
operators must provision real endpoints and independent identities. Private key material
is deliberately not supplied in the bundle.

On Inference:

```sh
pllm party serve inference --listen 0.0.0.0:7443 \
  --identity /etc/pllm/inference.json --state-dir /var/lib/pllm/inference
```

On Preparation:

```sh
pllm party serve preparation --listen 0.0.0.0:7444 \
  --identity /etc/pllm/preparation.json --state-dir /var/lib/pllm/preparation
```

A role service starts only that role. It authenticates clients, authorizes requested models
and component artifacts, validates a role plan, and rejects unknown/unapproved providers.
The role-local identity descriptor points to TLS keys/certificates, trust roots and explicit
peer authorization. Endpoint hostnames and `peer_identity` must match verified credentials;
strings are not identity proofs. Do not bind externally without this configuration.

From Client:

```sh
pllm plan compile private_app.py:experiment \
  --deployment remote.yaml --output build/qwen
pllm prepare build/qwen/plan.lock.json --deployment remote.yaml
# Preparation may now be stopped after the READY receipt.
pllm run build/qwen/plan.lock.json --deployment remote.yaml \
  --prepared-only --input-file question.txt
```

Compilation can delegate heavy model scanning/bound analysis to model-aware Preparation;
the thin Client receives public model descriptors, bounds/manifests and its plan, not weights.
Their authenticity/assurance is checked according to the chosen trusted-Preparation policy.
The baseline intentionally loads its own client-required matrices.

Input/output plaintext never goes through either remote service or the generic coordinator.
Material moves on direct, authenticated role-to-role channels, not through a combined
plaintext material archive. Readiness receipts contain public allocation facts only.
Persistent client secrets live in the client's protected local store so the separate
`prepare` and `run` commands can share one authorized allocation safely.

No silent online refill is permitted. A garbled plan is bound to a bounded generation
schedule and fresh instances; a persistent service does not make the circuit reusable.

## 9. Benchmarks and search

`examples/benchmark.py` and `examples/benchmark.sh` express the same public workload:
one warmup, five measured requests, 32 total input tokens and exactly 16 output tokens
under fixed termination. Six fresh request allocations are explicitly budgeted. The native
benchmark constructs/verifies public input fixtures at the specified tokenizer boundary.
Each search candidate gets its own fresh six-request allocation and immutable plan.

The Python API uses `Benchmark` and `Study`; the CLI uses `bench run` and `bench search`.
Grid JSON values are data, not executable expressions. `Study` may accept either an
Experiment with a workload policy or an already specified Benchmark. This is a declarative
convenience only; timing, launches, traces, adversary hooks and hot search operations are native.

Preparation, warmup and online phases are measured separately; no warmup is dropped from
all-party costs. Invalid, over-budget or failed candidates remain visible in the results.
Privacy-contract changes are new cohorts, not a dimension optimized away in one search.

Assurance uses the same resolved plan and view-restricted public fixtures:

```sh
pllm assure run build/qwen/plan.lock.json --suite protocol-regressions
```

No successful suite implies universal privacy. Scoped proof, counterexample and
not-refuted findings retain their exact assumptions and budgets.

## 10. Command grammar

```text
pllm init
pllm run TARGET
pllm chat TARGET
pllm serve TARGET
pllm prepare TARGET
pllm config show|export TARGET
pllm plan check|compile|show TARGET
pllm party serve ROLE
pllm bench run|search TARGET
pllm bench compare RUN_A RUN_B
pllm assure run TARGET
pllm components list|show
pllm research cite
```

`serve` means the trusted client-side HTTP gateway and defaults to loopback.
`party serve inference` means the untrusted encoded-message endpoint. They are deliberately
separate. Gateway requests terminate inside the Client trust boundary; they are never
forwarded as plaintext to the encoded endpoint. No HTTP compatibility claim is made by this
bundle. `chat` retains session state and charges each turn against public budgets; it does
not perform hidden refill. Its input cap includes all prior context.

A Python CLI facade can use Typer while the native supervisor/executor owns process and
protocol work. `party serve` directly starts the native role executable. No per-scalar,
per-stage packet or benchmark-clock Python callbacks. Token-delta presentation is allowed.

## 11. Developer and agent ergonomics

Generated text goes to stdout; human progress/diagnostics to stderr. Machine modes expose
structured JSON for finite commands and JSONL for streams. Do not log private prompt/output
text, token IDs, secret seeds or payloads by default. A public-fixture benchmark may record
fixture IDs. Direct argv prompts are intentionally absent from the preferred examples.

Interactive `run` prompts only on TTY. `--no-input` fails on any unresolved question.
`--quiet`, `--no-color`, `--format` and schema-driven shell completion are consistent.
Ctrl-C triggers native cancellation and retirement, not recycling. Local hot reload creates
a new plan/epoch after draining; production role services never import changing Python files.

Diagnostics include stage/role, component and unmet requirement. For example:

```text
E_COVERAGE
  profile: research.single_evaluator
  operation: private token selection
  missing: compatible inference-local implementation
  fallback: forbidden by profile
```

Do not print invented successful compilation/performance transcripts. `plan check` reports
which checks require heavy validation or remain unresolved; it cannot certify coverage merely
from metadata. `plan compile` performs the required checks before preparing secrets.

Every CLI help page maps to the same versioned documentation graph, with a Markdown twin:
`/docs/dev/reference/cli/run` and `/docs/dev/reference/cli/run.md`.
The Python Runtime/Session API references link to those commands and vice versa.

## 12. Public and secret artifact separation

```text
build/qwen/
  experiment.resolved.json
  model.lock.json
  plan.lock.json
  roles/client.plan.json
  roles/preparation.plan.json
  roles/inference.plan.json
  compatibility.json

.pllm/qwen-local/
  client/           # owner-only live state, identity refs and sealed client material
  preparation/      # owner-only preparation secrets; never included in benchmark export
  inference/        # owner-only assigned evaluator material
  public/           # public receipts and redacted status

runs/<run_id>/       # scrubbed measurement/evidence artifacts only
```

Public locks are exportable. Secret state is not. An ordinary config clone, directory copy
or snapshot restore must never reauthorize consumed material. Recovery follows the runtime's
stated rollback policy; filesystem permissions alone do not solve arbitrary snapshot rollback.

## 13. Acceptance tests for implementation

- Python configuration and YAML resolve to the same logical plan on the same environment.
- CLI and Python overrides generate identical parameter values/digests.
- Loading a definition does not start runtime work; loading code is explicit trust.
- Remote role processes never execute the user's definition module.
- Local and attached deployments use the same protocol with changed placement only.
- Prepared-only mode makes zero Preparation connections after readiness.
- CPU/native requirements, numeric manifests, conversions and operator gaps fail explicitly.
- Noninteractive commands never wait for a prompt.
- Stdout never mixes generated text with progress; default evidence contains no prompts.
- Every benchmark repetition/candidate has fresh material, including warmups.
- Retrying a command cannot assign exposed material to a changed input.
- Community-provider CLI metadata and Python factories use one component contract.

These are required implementation tests, not results claimed by this specification.

## References

Prior supplied design: PACKAGE_STRUCTURE.md sections 4–5; COMPONENT_STANDARD.md sections
3–8; INTERFACES.md (runtime lifecycle, locks and role protocol). The original September 11
PLLM manuscript establishes the client-heavy masked-linear partition, not thin-client support.

External interface patterns checked 14 September 2026:
- scikit-learn pipelines and named nested parameters: https://scikit-learn.org/stable/modules/compose.html
- Uvicorn module/object and explicit factory pattern: https://uvicorn.dev/
- Python distribution/entry-point metadata: https://docs.python.org/3/library/importlib.metadata.html
- Typer command groups: https://typer.tiangolo.com/tutorial/subcommands/
- PyO3 coarse native interface: https://pyo3.rs/
