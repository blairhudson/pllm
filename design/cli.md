# Command-line interface

## Status and scope

This document is the normative target contract for the `pllm` command. A command shown in a
normative section is not thereby implemented. The [implementation status](#implementation-status)
section records the currently shipped command surface separately.

The CLI is an authoring and presentation facade over the same configuration, model lowering,
compiler, benchmark, and assurance services used by Python. It MUST NOT become a second compiler or
runtime. Given the same locked inputs, CLI and Python entry points MUST produce the same canonical
configuration and plan digests.

## Command grammar

Target grammar:

```text
pllm [GLOBAL_OPTIONS] init
pllm [GLOBAL_OPTIONS] config show|export TARGET
pllm [GLOBAL_OPTIONS] model lower TARGET
pllm [GLOBAL_OPTIONS] plan check|compile|show TARGET
pllm [GLOBAL_OPTIONS] prepare TARGET
pllm [GLOBAL_OPTIONS] run TARGET
pllm [GLOBAL_OPTIONS] chat TARGET
pllm [GLOBAL_OPTIONS] serve TARGET
pllm [GLOBAL_OPTIONS] party serve ROLE
pllm [GLOBAL_OPTIONS] benchmark run|search|compare TARGET...
pllm [GLOBAL_OPTIONS] assure run TARGET
pllm [GLOBAL_OPTIONS] components list|show [COMPONENT]
pllm [GLOBAL_OPTIONS] research sources|methods|recipes list|show [ID]
pllm [GLOBAL_OPTIONS] dev dashboard
```

`TARGET` MUST be one of these explicit forms:

| Form | Meaning | Trust effect |
| --- | --- | --- |
| `experiment.yaml` or `experiment.json` | Strict declarative experiment | Safe data parsing only |
| `path.py:object` | Public object in a local Python file | Executes trusted local Python |
| `package.module:object` | Public object in an importable module | Executes trusted local Python |
| `package.module:factory --factory` | Explicit zero-argument factory | Executes trusted local Python |
| `build/name/plan.lock.json` | Immutable compiled public plan | Verifies lock and artifacts |

Object and factory targets MUST resolve to a supported public configuration object. The CLI MUST
reject implicit factory calls, arbitrary expressions, pickle input, and class-path construction.
Before importing a Python target, the CLI MUST warn and obtain confirmation from an interactive
user. `--no-input` and noninteractive stdin reject Python execution unless `--trust-python`
explicitly approves it. Declarative file targets never execute target or provider code.
Remote placement MUST NOT cause source modules, closures, pickles, or the coordinator environment
to be sent to another role.

## Global behavior

- `--help` and `--version` MUST perform no model, device, plugin, network, or preparation work.
- Long options use kebab case. Abbreviated options MUST NOT be accepted. A short option MUST have
  one stable meaning across command families.
- Commands MUST identify unresolved required input before beginning expensive or irreversible work.
- `--no-input` MUST fail rather than prompt. A noninteractive process MUST never wait for input.
- `run` MAY prompt only when stdin is a TTY. Noninteractive prompt input uses `--input-file PATH` or
  `--input-file -`. Private prompt text SHOULD NOT be accepted directly in argv.
- `--quiet`, `--no-color`, and `--format human|json|jsonl` MUST have consistent meanings. Color and
  progress rendering MUST be disabled when their destination is not a TTY unless forced explicitly.
- Ctrl-C MUST request native cancellation, drain or retire affected allocations, and exit 130. It
  MUST NOT make exposed or reserved material reusable.
- Help and completion data MUST come from the same command and component schemas used for
  validation. Every command help page links to its HTML and Markdown documentation routes.

Generated response text goes to stdout. Human progress, warnings, and diagnostics go to stderr.
Finite machine-mode commands emit one JSON document to stdout; streams emit JSONL with one complete
record per line. Diagnostics, logs, telemetry, and evidence records MUST NOT contain prompt text,
output text, token IDs, payloads, credentials, seeds, or prepared material unless an explicit,
authorized capture policy names those fields. A requested response stream carries its output as its
declared payload, not as diagnostic metadata.

## Command families

### Configuration and model lowering

`init` writes configuration only. It MUST NOT download a model, discover a device, import a provider,
compile a plan, prepare material, start a process, or open a network connection. Existing output
requires `--force`.

`config show` renders canonical public configuration. `config export` writes a strict public JSON or
YAML representation. Neither command may serialize live handles or secrets. Loading a Python target
still executes that trusted local target and MUST be reported before execution in interactive mode.

`model lower` maps supported public model configuration into an immutable `ModelPlan`, equivalent to
Python `lower_model(...)`. Lowering model configuration MUST NOT imply that weights were loaded or
that complete protected execution coverage exists.

### Planning

`plan check` performs all cheap, metadata-complete checks and reports checks that require artifact
resolution or remain open. It MUST NOT convert unknown coverage into a pass.

`plan compile` resolves concrete components and artifacts and invokes the same compiler as Python
`compile(...)`. Its output is a new immutable directory; it MUST NOT modify a lock in place. All
`auto` choices, model/tokenizer identities, numeric semantics, component versions and digests, role
slices, and required capabilities are frozen before preparation. Unsupported operations,
conversions, roles, numerical policies, and required evidence fail closed.

`plan show` reads a public plan or lock without loading secret runtime state. Redacted views MUST say
which fields were omitted; absence and redaction are not equivalent.

### Preparation and execution

`prepare TARGET` allocates and installs material for the declared bounded workload. It emits a public
readiness receipt separately from role-local secret state. Preparation MUST finish before any input
is exposed to an assigned allocation.

`run TARGET` handles one bounded request. Convenience mode MAY perform a visible preparation
preflight and then execute. `--prepared-only` is a hard contract: the command MUST NOT start,
contact, health-check, or refill from Preparation. Compatible installed material is used or the
command fails. Material MUST NOT be replenished during generation unless a separately named policy
explicitly changes the execution contract.

Budgets cover complete tokenized input, including templates, prior turns, and control tokens, plus
the bounded generation schedule. Exhaustion, cancellation, retry, or early termination follows the
plan's retirement rules; no mask, label, share, correlation, or other one-use material is silently
recycled.

`chat` retains client-owned conversation state and charges every turn and accumulated context to
public budgets. It obeys the same no-hidden-refill and output contracts as `run`.

### Services and roles

`serve TARGET` is the trusted Client-side application gateway. Plaintext requests terminate inside
the Client trust boundary. It defaults to loopback and MUST require explicit authenticated
configuration before external binding.

`party serve ROLE` starts exactly one plan-execution role such as `preparation` or `inference`. It
accepts authenticated and authorized role plans and protocol messages, never arbitrary Python code.
It MUST load only providers authorized for that role and MUST NOT receive plaintext merely because
the trusted gateway supports a compatible application protocol.

These endpoints are not interchangeable. Placement changes process and endpoint locations; it MUST
NOT change protocol, numeric, privacy, corruption, preparation, or workload choices. Local
orchestration MUST preserve role-specific processes, credentials, stores, and channels. Co-location
tests protocol boundaries but does not establish administrative non-collusion.

Local and attached orchestration follow the same ordered contract:

1. Inspect public descriptors and check source, provider, and policy compatibility.
2. Resolve and lock model, tokenizer, numeric, component, and native artifact identities.
3. Compile role-specific plans, conversions, state effects, and resource requirements.
4. Reject preparation that exceeds declared resource or workload bounds.
5. Deliver secret slices directly to authorized roles and obtain a public readiness receipt.
6. Close the Preparation path for the allocation, then bind Client input before exposure.
7. Execute, decode only at the authorized Client boundary, retire reservations, and emit scrubbed
   records.

Role services use independently provisioned credentials. A target deployment may resemble:

```sh
pllm party serve inference --listen 0.0.0.0:7443 \
  --identity /etc/pllm/inference.json --state-dir /var/lib/pllm/inference
pllm party serve preparation --listen 0.0.0.0:7444 \
  --identity /etc/pllm/preparation.json --state-dir /var/lib/pllm/preparation
```

These commands require operator-provisioned authenticated transport and authorization. Example
hostnames, identity paths, and role strings do not establish peer identity.

### Benchmark and assurance

`benchmark run` invokes the same evidence-producing service as Python `benchmark(...)` for supported
scopes. Preparation, warmup, and online measurements are separate. Warmups count in all-party cost
and consume fresh material. Every repetition and every search candidate receives a fresh authorized
allocation. Failed, unsupported, and over-budget candidates remain in results.

`benchmark search` changes only declared search parameters. Numeric semantics, privacy contract,
role topology, workload, and evidence requirements are cohort constraints, not dimensions that an
optimizer may weaken. `benchmark compare` rejects or clearly separates incomparable cohorts.

`assure run` invokes scoped assurance work equivalent to Python `assure()` where supported.
Successful command execution means a report was produced, not that privacy was proved. Exact
outcomes and scopes use the evidence vocabulary in the component and documentation standards.

`components list` performs metadata-only discovery. `components show` reports identity, contract,
availability, trust requirement, and evidence pointers without loading provider code or native
libraries. An explicit later operation may approve and load a provider.

### Research metadata

`research sources|methods|recipes list|show` reads canonical public JSON records as inert data.
These commands MUST NOT import research scripts, upstream artifacts, provider code, native
libraries, or recipe commands. Returned source, method, and recipe records are immutable. Showing a
recipe never executes its acquisition or reproduction workflow. Distributions MAY carry an inert
snapshot when canonical records are outside package data, provided tests prove exact record parity.

`dev dashboard` is an explicitly non-normative developer surface for the existing loopback
benchmark dashboard. Its implementation is imported only after command validation. It is not
`benchmark run` and does not produce normative benchmark evidence.

## Configuration layering

For unlocked authoring input, later layers override earlier layers:

```text
versioned profile
  < declarative file or Python object
  < explicit deployment replacement
  < repeated --set PATH=JSON_VALUE options
```

`--set` uses the same `__`-separated paths as `with_params()`. Values are parsed against the category
schema as data, never evaluated as Python. Unknown paths, overlapping paths, duplicate map keys,
invalid types, and non-finite numbers fail. The serialized `params` wrapper is not part of a public
component parameter path.

Environment variables and a user configuration file MAY supply documented operational defaults
such as output format, endpoint, and cache location. They MUST NOT silently change protocol,
privacy, numerical semantics, roles, or a locked plan. Command-line values override operational
defaults. Every effective non-secret value and its source MUST be inspectable in dry-run output.

Locked plans reject `--set`. Changing configuration, deployment identity, provider artifact, or
other locked input creates and compiles a new plan and invalidates incompatible prepared material.

## Dry run and output contracts

All state-changing target commands MUST support `--dry-run`. Dry run resolves only through the last
side-effect-free stage available to that command and reports planned reads, persistent writes,
network peers, roles, provider loads, artifact fetches, and unresolved checks. It MUST NOT write
persistent output, fetch a model, load a native library, open a network connection, start a role,
reserve material, or execute a plan. A Python target may execute trusted local Python to construct
configuration; dry-run MUST make that exception explicit before loading it.

Machine output has a versioned schema. Diagnostics include a stable code, stage, role when known,
component or artifact when known, user-safe message, and structured details. Human wording may
improve without changing the code. A failure such as missing operator coverage MUST identify the
operator, rejected implementation or conversion, required contract, and whether fallback is
forbidden.

Exit classes are stable:

| Exit | Meaning |
| --- | --- |
| `0` | Command completed and produced its declared result |
| `2` | CLI grammar or option usage invalid |
| `3` | Target resolution, public validation, compatibility, coverage, or compilation rejected |
| `4` | Required source, artifact, dependency, or local I/O unavailable |
| `5` | Authentication, authorization, or deployment policy rejected |
| `6` | Runtime, transport, or role execution failed |
| `7` | Prepared material absent, incompatible, exhausted, or retired |
| `130` | Interrupted; cancellation and retirement requested |

An assurance counterexample or benchmark failure is normally typed report data and does not become
exit failure merely because its result is unfavorable. CI policy MAY request a nonzero exit with an
explicit `--fail-on` rule; the report itself remains unchanged.

## Security and secrets

- Shared configuration, plan locks, role plans, benchmark records, and assurance reports contain no
  raw credentials, private keys, prompt data, seeds, or prepared material.
- Secret values MUST come from role-local protected stores, file descriptors, or documented
  environment references. Preferred examples MUST NOT place them in argv or shared YAML.
- Identity strings and endpoint names are selectors, not proofs. Verified credentials and explicit
  authorization bind peers, model identities, plans, and providers.
- Paths read from provider metadata MUST be normalized, confined to the provider distribution, and
  checked against traversal and symlink escapes.
- `--force` controls replacement only; it MUST NOT bypass policy, lock verification, freshness,
  coverage, or trust checks.
- Dry-run, verbose logging, shell completion, crash reports, and telemetry obey the same redaction
  rules as execution.

## Examples

These are target-design examples, not transcripts proving implementation:

```sh
pllm init --profile baseline.masked_linear_cpu \
  --model Qwen/Qwen2.5-0.5B-Instruct --output experiment.yaml
pllm plan check experiment.yaml
pllm model lower experiment.yaml --batch 1 --max-input-tokens 128 \
  --max-new-tokens 32 --output build/model-plan.json
pllm plan compile experiment.yaml --output build/qwen
pllm prepare build/qwen/plan.lock.json --deployment remote.yaml
pllm run build/qwen/plan.lock.json --deployment remote.yaml \
  --prepared-only --input-file question.txt
```

```sh
pllm run private_app.py:experiment \
  --set pipeline__components__kernels__threads=8 --input-file question.txt
pllm benchmark run build/qwen/plan.lock.json --format json --output runs/qwen
pllm assure run build/qwen/plan.lock.json --format json
```

## Required tests

- Python, JSON, and YAML configuration resolve to identical canonical configuration and plan digests.
- CLI `--set` and Python `with_params()` produce identical typed values and digests.
- Definition loading performs no runtime work, and Python execution is explicit.
- Remote roles never import or receive the user's Python target.
- Local and attached deployments execute the same locked contract.
- Prepared-only mode makes zero Preparation connections after readiness.
- Noninteractive commands never prompt; stdout and stderr contracts hold in every format.
- Dry-run performs none of its prohibited side effects.
- Every benchmark warmup, repetition, and search candidate gets fresh material.
- Retry, cancellation, and interruption cannot assign exposed material to a changed input.
- Machine errors and exit classes remain stable under human-message changes.
- Examples are syntax-checked separately from integration and security tests.

These are acceptance requirements, not claims that they currently pass.

## Implementation status

Only rows marked **Shipped** are parser-visible. This table records the exact parser
surface checked by the generated CLI reference; target rows marked **Unavailable** are
intentionally not placeholder commands. `pllm._cli` owns the sole parser; local child roles invoke
`python -m pllm serve inference|preparation` rather than a second runtime parser.

| Command | Status | Current boundary |
| --- | --- | --- |
| `config show TARGET` | **Shipped** | Strict JSON/YAML or explicitly trusted Python `Experiment`; canonical public output only |
| `config export TARGET --output PATH [--force]` | **Shipped** | Strict public JSON/YAML; exclusive create unless forced; dry-run writes nothing |
| `components list|show` | **Shipped** | Built-in public `ComponentDescriptor` metadata only |
| `gateway` | **Shipped runtime facade** | Trusted local Responses and Chat Completions gateway; not yet the target-based `serve TARGET` contract |
| `serve inference|preparation` | **Shipped role facade** | Existing authenticated runtime roles; not yet generic `party serve ROLE` over a plan lock |
| `benchmark run` | **Shipped runtime facade** | Existing local benchmark orchestration and evidence output; `search` and `compare` are unavailable |
| `dev dashboard` | **Shipped, non-normative** | Existing loopback dashboard, lazy runtime import |
| `init` | **Unavailable** | Profile catalog unresolved |
| `--set PATH=JSON_VALUE` | **Unavailable** | Typed override/schema integration incomplete |
| `model lower` | **Unavailable** | Python lowering exists; the target-based CLI contract is not shipped |
| `plan check|compile|show` | **Unavailable** | Full lock and artifact CLI contract incomplete |
| `prepare`, `run`, `chat` | **Unavailable** | Prepared-material lifecycle is exposed through existing runtime facades, not these target commands |
| `serve TARGET`, `party serve ROLE` | **Unavailable** | Generic target role/authentication contract incomplete |
| `benchmark search|compare` | **Unavailable** | Cohort-safe search and comparison facade incomplete |
| `assure run` | **Unavailable** | Scoped assurance facade incomplete |
| `research sources|methods|recipes list|show` | **Unavailable** | Canonical records exist, but no parser-visible research commands ship |

Removed pre-0.1 top-level runtime, market, provider, simulator, configuration-secret, and native
build commands are not compatibility aliases. Runtime Python APIs remain separate and unchanged.
