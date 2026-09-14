# Proposed public and native interfaces

These are implementation contracts for PLLM. The working program included in this bundle is `validate_examples.py`, not the runtime/CLI described below.

## 1. User workflow

A benchmark command should perform model resolution, policy checks, compilation, budget estimation, party launch, preparation, online execution, validation and artifact capture. The user should not need to keep three terminals and independently edit three configurations.

Proposed commands:

```bash
pllm model inspect Qwen/Qwen2.5-0.5B-Instruct
pllm methods list --family masked_linear
pllm methods show garbled.weighted_path.experimental.v1
pllm bench run --config configs/qwen-local-baseline.yaml --output runs/baseline
pllm bench search --config configs/operator-search.yaml --output runs/search
pllm bench compare runs/baseline runs/search --output reports/comparison
pllm research cite --run runs/search --format bibtex
```

`model inspect` may resolve metadata; execution requires a pinned content lock. `bench compare` must reject claims of speedup between incompatible workload/numeric/privacy cohorts or prominently present them as different scenarios. It must not divide unrelated scalar and full-model measurements.

For separate machines, use the same public config with a deployment manifest containing role-to-agent bindings and opaque credential references. The control plane sends only role-specific public plans and sealed artifacts. Agent logs cannot serialize secret tensors. SSH is a launcher/control channel, not a substitute for protocol peer authentication.

Proposed Python usage:

```python
from pllm import Experiment, Benchmark

experiment = Experiment.from_file("configs/qwen-local-baseline.yaml")
report = Benchmark(experiment).run(output="runs/baseline")
print(report.summary())
report.write_json("runs/baseline/summary.json")
```

```python
from pllm import Experiment, Study

experiment = Experiment.from_file("configs/operator-search.yaml")
study = Study(experiment)
frontier = study.search(output="runs/search")
frontier.write_json("runs/search/frontier.json")
```

These familiar declarative objects do not hide the privacy contract. Calling `.run()` cannot silently change roles, numerical precision or leakage to make a plan execute. Public estimator-like parameters and named pipeline regions support cloning/search, but cryptographic material is never copied by cloning an experiment.

## 2. Semantic/protected type examples

A value descriptor must carry more than dtype and shape:

```text
value_id: layer.3.mlp.gate_input
shape: [batch, intermediate]
numeric:
  signed: true
  domain: integer
  encoding_moduli: [251, 241]
  scale: 4096
  rounding: ties_to_even
  range_certificate: bound.3.mlp.gate
protection:
  representation: arithmetic_label
  garbling_instance: execution_epoch
  label_layout: crt.2x18
  holder: inference
  plaintext_owner: client
lifetime:
  semantic_instance: [request, token, layer, op, element_range]
  material_usage: one_distinct_input
```

The concrete primes/layout are an example from a supplied experiment, not a default security parameter recommendation. A real method descriptor determines valid cryptographic parameters. Semantic value range and cryptographic label entropy are separate quantities.

## 3. Adapter contract

| Interface | Inputs | Outputs / checks |
|---|---|---|
| `inspect` | pinned or resolvable model source | names, format, features, shard map, aliases; no custom-code execution |
| `validate_tensors` | descriptor + read-only store | shape/dtype/alias validation; complete error list |
| `lower_semantics` | validated descriptor + tensor handles | semantic DAG, explicit state operations and constants |
| `reference` | trusted fixture + semantic graph | intermediate/output expectations with upstream implementation version |
| `conformance_cases` | architecture feature flags | cases covering prefill, decode, positions, masks, tied weights and cache behavior |

A `TensorStore` exposes read-only row/tile access and memory mapping where supported. Compiled weights are content-addressed by tensor data, quantization manifest and importer version. Source formats are adapters, not protocol choices.

## 4. Method and kernel lifecycle

A new technique registers the following contracts:

| Step | Required behavior |
|---|---|
| `describe()` | method ID/version, sources, operators, supported protections, numeric assumptions, parties, material lifetimes and evidence |
| `validate(region, contract)` | diagnostics for missing bounds, illegal input representation, trust assumptions or state behavior |
| `lower(region, target)` | candidate subplan with explicit conversions, kernels, communication and resource requests |
| `estimate(subplan, calibration)` | typed cost estimates with confidence/source; unknown cost remains unknown |
| `prepare(role_ctx, requests)` | role-specific material handles and public receipts; no secret data returned to the coordinator |
| `execute(role_ctx, schedule)` | only that role's actions; native opaque input/material/output buffers |
| `reference(public_fixture)` | pure semantic oracle in trusted tests, never a production fallback |
| `verify_fixture()` | arithmetic, protocol-view, negative and interoperability tests |
| `reproduction_recipe()` | pinned source artifact/version, command, expected functional outputs and evidence schema |

A Python reference may implement the lifecycle for research, but it must be marked as a reference and should not appear in performance-optimized defaults. Native implementations use the same public contract.

The native kernel is narrower:

```text
supports(KernelRequest) -> CapabilityDecision
prepare_public(PublicConstants) -> ImmutableKernelState
workspace_bytes(ShapeProfile) -> Bytes
execute(ExecutionContext, ImmutableInputs, MutableOutputs, Workspace) -> Completion
```

Kernel dispatch keys include arithmetic domain, precision certificate, value layout, weight layout, shape, device, instruction set and implementation version. The executor batches an entire region across the Python/Rust boundary, not one call per scalar. Stream/event completion is explicit on accelerators; launch time is not execution time.

## 5. Compiler diagnostics and locks

Use stable machine-readable errors:

- `E_MODEL_FEATURE`: unsupported architecture semantics.
- `E_NUMERIC_MANIFEST`: missing or inconsistent frozen numerical definition.
- `E_PRIVACY_CONTRACT`: candidate requires forbidden trust/party/leakage assumptions.
- `E_CONVERSION`: no eligible protected representation bridge.
- `E_COVERAGE`: missing operator/state/sampling implementation.
- `E_RANGE_CERTIFICATE`: required bound not established.
- `E_PARAMETER_ASSURANCE`: cryptographic profile not validated for selected method.
- `E_PREPARATION_BUDGET`: estimated material exceeds budget before allocation.
- `E_DEVICE_KERNEL`: missing exact supported kernel.
- `E_NO_VALIDATED_PLAN`: no complete candidate with qualifying evidence.

A compiler report includes the first blocking node, all incompatible candidates and the reason each was eliminated. An unsupported candidate is a recorded result, not silently omitted from the search.

`plan.lock.json` binds the source model/shards, tokenizer, semantic DAG, numeric graph, method/kernel/conversion IDs, public parameters, compiler commit, runtime ABI, party mapping, threat/leakage contract, public workload bucket and capacity assumptions. Device/profile plans may specialize prefill and decode separately. It never stores fresh seeds or active material.

Two levels of hashing are useful: a portable logical-plan digest and a hardware-specific execution digest. Secret material is indexed by a distinct execution epoch bound to both; it is not part of a publicly reproducible plan hash.

## 6. Role protocol

The native role runner consumes a role-specific plan. A binary envelope carries:

```text
protocol_version, session_id, logical_plan_digest, execution_digest,
role_from, role_to, sequence, request_id, op_id, material_id,
shape_id, codec_id, payload_length, authenticated_payload
```

Cryptographic peer identity comes from authenticated channels, not merely the `role_from` field. Canonical headers bind the expected configuration. The runtime checks dimensions, lengths, sequence and duplicate semantics before invoking a kernel. A public plan digest authenticates configuration only when verified against a trusted key; it does not prove correct execution by itself.

Preparation protocol: `RESERVE → PREPARE → INSTALL → READY`. Online: `BIND → SEND → EVALUATE → VERIFY/DECODE → RETIRE`. Receipts and errors contain no secret material. A sender binds immutable request bytes and an execution identity before possible transmission. Retry uses the same application payload or a new fully prepared execution, never a recycled pad on a changed input.

A checksum, consistency tag, ready acknowledgement and full malicious-security proof are different things and must be labelled separately.

## 7. Generative-state lifecycle

KV-cache, sampler state, encoded token feedback and EOS state are explicit objects, not incidental tensors. Each declares its representation, owner/holders, authorized read/update operations, position ranges, public capacity and epoch.

A garbled decode loop cannot reuse the same label wire for successive different tokens. Preparation must issue distinct compatible instances for the bounded generation schedule, including private state transitions and token feedback. An immutable KV value may be reused only under its method's justified fan-out/lifetime rule; secret semantic content must not be relabelled with a public shortcut.

Variable-length output requires an explicit leakage/termination policy. Fixed-length benchmarks pad/ignore EOS according to a declared workload profile. A private termination policy needs a corresponding protocol; do not confuse that with ordinary server-visible early stopping.

## 8. Artifact contract

```text
runs/<run_id>/
  experiment.resolved.json    # public parameters only
  model.lock.json
  plan.lock.json
  environment.json           # CPU/GPU, clocks, RAM, libs, affinity, network
  sources.json               # exact technique/artifact citations
  compilation.json           # coverage, candidates, estimated costs
  requests.jsonl             # client-side event times and public test IDs
  events.jsonl               # privacy-scrubbed phase events
  traffic.jsonl              # sender-counted bytes by direction/phase
  resources.jsonl            # per-party samples; unavailable != zero
  correctness.json           # declared graph parity
  quality.json               # model-quality comparison, distinct from parity
  inventory.json             # counts/rates/waste, no material payload
  summary.json
```

All-party resource totals are sums over actual processes/devices, including Preparation. Loopback roles sharing a CPU require recorded affinity/core limits and concurrency; local results are not extrapolated to WAN. Sequential prefill and decode phases must have explicit boundaries. A one-time cache warmup can warm code/weights but must not be confused with reusing one-time cryptographic resources on changed inputs.

## 9. Benchmark optimizer contract

Search domains are public component parameters: kernel tile/layout, thread count, fusion boundaries, permitted residue choices justified by bounds, gate backend and decision ordering. Changes to rounding/approximation or privacy are new comparison cohorts, not interchangeable values in one optimization run.

The initial optimizer enumerates compatible region plans, measures candidate regions and conversion boundaries, then performs Pareto beam search across the graph. It keeps global memory, material consumption and critical-path constraints. A region cost sum is only an estimate; top complete plans must be measured because overlap, contention and state scheduling are nonadditive.

Optimize at least online latency/TPS, total preparation+online cost, total traffic and peak memory subject to a quality/security contract. Store costs with measurement provenance and uncertainty. Use public calibration workloads for tuning and a frozen held-out public set for reporting.

A registry entry alone does not prove a protocol fits the preferred topology. Each new component requires a compatible conversion, state-lifetime and assumption analysis. The build should generate `methods show`, method documentation, configuration parameter docs and citations from these same metadata records.
