**Define the composition once. Run it from Python, from a declarative file, or by pointing the CLI at the Python object. Change deployment without rewriting the composition.**

The CLI should be a frontend to the same compiler and runtime—not a second implementation.

**PLLM_composition_CLI_design.zipZIP** ·

These are **proposed API/CLI contracts for the uplift**, not commands already implemented in the current release.

## 1. Four objects, with clear responsibilities


| Object                | Specifies                                                  |
| --------------------- | ---------------------------------------------------------- |
| `Pipeline`            | Model, numerical profile and named component bindings      |
| `Deployment`          | Where the required parties run and how they authenticate   |
| `Experiment`          | Pipeline + deployment + execution budget                   |
| `Runtime` / `Session` | Live native execution, prepared material and request state |


This preserves the distinction already established in the package design: configuration is cloneable; compiled plans are immutable; live cryptographic state is neither cloneable nor serializable as configuration. PACKAGE_STRUCTURE.mdMD

The shared lifecycle is:

```
Define → Resolve → Validate → Compile → Prepare → Execute → Measure
```

**Components are named bindings into a graph, not sequential pipeline steps.** The compiler handles placement, conversions and dependencies.

## 2. Compose and run entirely in Python

For the initial full-model example, use the masked-linear baseline. It deliberately retains client-side attention, nonlinearities, embeddings and output-head work; it is **not** the thin-client target. Pasted text.txtTXT

### `private_[app.py](http://app.py)`

```
from pllm import Experiment, Model, Pipeline, Runtime
from pllm.deployment import Deployment
from pllm.kernels import Cpu
from pllm.preparation import ModelAwareCorrections
from pllm.protocols.masked_linear import MaskedLinear
from pllm.runtime import ExecutionBudget


experiment = Experiment(
    name="qwen-local",
    pipeline=Pipeline.from_profile(
        "baseline.masked_linear_cpu",
        model=Model("Qwen/Qwen2.5-0.5B-Instruct"),
        components={
            "linear": MaskedLinear(),
            "preparation": ModelAwareCorrections(),
            "kernels": Cpu(threads=4),
        },
    ),
    deployment=Deployment.local(root=".pllm/qwen-local"),
    budget=ExecutionBudget(
        requests=1,
        max_input_tokens=128,
        max_new_tokens=32,
    ),
)


def main() -> None:
    with Runtime(experiment) as runtime:
        runtime.prepare()

        with runtime.session() as session:
            for delta in session.generate(
                "Explain private inference in two sentences.",
                stream=True,
            ):
                print(delta.text, end="", flush=True)

        print()


if __name__ == "__main__":
    main()
```

The profile supplies the rest of the graph, numerical manifest, placement rules and required conversions. The explicitly supplied components override its named bindings.

Importing this module **only constructs configuration**. Entering `Runtime` resolves, validates, compiles and establishes the deployment. `prepare()` installs fresh material. `session.generate()` consumes that material without calling Preparation.

An unavailable numerical manifest or unsupported operator must produce a specific error—not a substitute implementation.

### Changing a component parameter

```
faster_cpu = experiment.with_params(
    pipeline__components__kernels__threads=8,
)
```

Use sklearn-style nested parameter paths, but retain immutable `with_params()` rather than mutating active configurations. Sklearn’s named-component parameter convention is a good fit for inspection and search. [scikit-learn](https://scikit-learn.org/stable/modules/compose.html)

### Selecting the preferred architecture

The same composition mechanism should express the single-evaluator target:

```
from pllm.preparation import ModelAwareGarbling
from pllm.protocols.garbling import ArithmeticGarbling, BooleanHalfGates


single_evaluator = Pipeline.from_profile(
    "research.single_evaluator",
    model=Model("Qwen/Qwen2.5-0.5B-Instruct"),
    components={
        "linear": ArithmeticGarbling(),
        "nonlinear": BooleanHalfGates(),
        "preparation": ModelAwareGarbling(),
        "kernels": Cpu(threads=4),
    },
)

target = experiment.with_params(pipeline=single_evaluator)
```

The profile enforces the thin-client, offline-Preparation, single-online-evaluator contract. Selecting a nonlinear backend does not magically implement attention, sampling or state transitions: missing coverage remains a compilation error.

## 3. Run that Python definition from the CLI

The normal command should be:

```
pllm run private_app.py:experiment
```

For file input:

```
pllm run private_app.py:experiment --input-file question.txt
```

For a parameter override:

```
pllm run private_app.py:experiment \
  --set pipeline__components__kernels__threads=8 \
  --input-file question.txt
```

**The parameter path is identical in Python and the CLI.**

Support these target forms consistently:


| Target                                          | Meaning                                   |
| ----------------------------------------------- | ----------------------------------------- |
| `pllm.yaml`                                     | Declarative experiment                    |
| `private_[app.py](http://app.py):experiment`    | Object from an explicit local Python file |
| `myproject.pipelines:experiment`                | Object from an importable module          |
| `myproject.pipelines:make_experiment --factory` | Explicit zero-argument factory            |
| `build/qwen/plan.lock.json`                     | Already resolved and compiled public plan |


The module/object and explicit factory pattern is familiar from Uvicorn; PLLM adds the local-file and locked-plan forms. [Uvicorn](https://uvicorn.dev/?utm_source=chatgpt.com)

**Loading Python executes trusted code on the invoking machine.** The CLI must not serialize and ship the module to the other parties. It exports component specifications and party-specific plans, not Python closures or pickles.

### Export Python configuration for deployment

```
pllm config export private_app.py:experiment --output pllm.yaml

pllm plan compile private_app.py:experiment --output build/qwen
```

That gives researchers Python’s flexibility while operators can deploy a declarative, locked artifact.

## 4. The equivalent pure-CLI experience

A developer should be able to start without writing Python or hand-authoring YAML:

```
pllm init \
  --profile baseline.masked_linear_cpu \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --output pllm.yaml

pllm plan check pllm.yaml

pllm run pllm.yaml
```

`init` writes configuration only. It does not download weights or start services.

The generated configuration would look like this:

```
schema: pllm.experiment.v1
name: qwen-local

pipeline:
  profile: baseline.masked_linear_cpu

  model:
    source: Qwen/Qwen2.5-0.5B-Instruct

  components:
    linear:
      component: pllm/masked-linear
      params: {}

    preparation:
      component: pllm/model-aware-corrections
      params: {}

    kernels:
      component: pllm/cpu
      params:
        threads: 4

deployment:
  kind: local
  root: .pllm/qwen-local

budget:
  requests: 1
  max_input_tokens: 128
  max_new_tokens: 32
```

These component IDs are registry references, not import paths. Community implementations use the same structure and must pass the same compatibility checks. The existing component standard already requires discovery, approval and role-specific loading to be separate steps.

For noninteractive use:

```
pllm run pllm.yaml --input-file question.txt
```

Or stdin:

```
cat question.txt | pllm run pllm.yaml --input-file -
```

I would avoid encouraging private prompts in command-line arguments, which are easier to expose through shell history and process inspection.

### What local orchestration does

`Deployment.local()` launches separate native role processes automatically:

```
Python API or CLI
        │
        ▼
Native supervisor
        ├── Client
        ├── Preparation
        └── Inference
```

It establishes role-specific credentials, compiles the plan, performs preparation, waits for readiness, disconnects Preparation for that allocation, then executes.

**Three local processes exercise the distributed protocol boundaries; they do not establish independent administrative trust.**

The baseline still exchanges masked intermediates with the Client. The fully supported single-evaluator profile would keep those operations at Inference.

## 5. Run the same composition on separate machines

Only the deployment changes.

### Start the Inference party

Run on the inference host:

```
pllm party serve inference \
  --listen 0.0.0.0:7443 \
  --identity /etc/pllm/inference.json \
  --state-dir /var/lib/pllm/inference
```

### Start the Preparation party

Run on the preparation host:

```
pllm party serve preparation \
  --listen 0.0.0.0:7444 \
  --identity /etc/pllm/preparation.json \
  --state-dir /var/lib/pllm/preparation
```

Each command starts **only that party**. It accepts authenticated, authorized role plans—not arbitrary Python code.

Identity files reference locally provisioned certificates, keys, trust roots and peer authorization. No secrets belong in the shared experiment.

### `remote.yaml`

```
schema: pllm.deployment.v1
kind: attached

parties:
  client:
    launch: local
    state_dir: .pllm/qwen-remote/client
    identity: ./identities/client.json

  preparation:
    endpoint: wss://prep.example.net:7444
    peer_identity: preparation

  inference:
    endpoint: wss://infer.example.net:7443
    peer_identity: inference
```

The example hostnames and identity paths need actual deployment provisioning. `peer_identity` must match verified credentials; it is not trusted merely because the server reports that string.

### Compile and prepare from the client

```
pllm plan compile private_app.py:experiment \
  --deployment remote.yaml \
  --output build/qwen

pllm prepare build/qwen/plan.lock.json \
  --deployment remote.yaml
```

After the readiness receipt, **Preparation can be stopped**.

Then run:

```
pllm run build/qwen/plan.lock.json \
  --deployment remote.yaml \
  --prepared-only \
  --input-file question.txt
```

`--prepared-only` is a hard contract:

> Do not start, contact, health-check or refill from Preparation. Use already installed, compatible material or fail.

The public plan is reused. The cryptographic material is consumed according to its lifetime rules.

### Equivalent Python attachment

```
remote_experiment = experiment.with_params(
    deployment=Deployment.from_file("remote.yaml"),
)

with Runtime(remote_experiment) as runtime:
    runtime.prepare()

    with runtime.session() as session:
        for delta in session.generate(
            "Explain private inference in two sentences.",
            stream=True,
        ):
            print(delta.text, end="", flush=True)
```

For a separate Python process consuming existing preparation:

```
with Runtime(remote_experiment, prepared_only=True) as runtime:
    with runtime.session() as session:
        result = session.generate("Explain private inference in two sentences.")
        print(result.text)
```

Neither placement changes nor prepared-only mode may silently change the selected protocol.

## 6. Make preparation convenient without hiding it

There should be two deliberately different workflows:


| Workflow                                                     | Behavior                                                           |
| ------------------------------------------------------------ | ------------------------------------------------------------------ |
| `pllm run TARGET`                                            | Visible preparation preflight, followed by online execution        |
| `pllm prepare TARGET` then `pllm run TARGET --prepared-only` | Explicitly separated phases; Preparation can be unavailable online |


In Python, `runtime.prepare()` remains explicit in either case. The convenience CLI simply performs that call for the first workflow.

**Never replenish halfway through generation without an explicitly different execution policy.** Otherwise “offline Preparation” becomes a misleading name.

Budgets must cover the complete tokenized input—including templates and previous turns—and the bounded generation schedule. Exhausted material produces a useful error, not mask recycling or an automatic privacy downgrade.

## 7. Benchmark and search the same composition

### Python

```
from pllm import Benchmark, Study
from pllm.benchmarks import Decode
from private_app import experiment


benchmark = Benchmark(
    experiment.with_params(budget__requests=6),
    workload=Decode(
        input_tokens=32,
        output_tokens=16,
        termination="fixed",
    ),
    warmup=1,
    repeats=5,
)

report = benchmark.run(output="runs/smoke")
print(report.summary())

study = Study(
    benchmark,
    space={
        "pipeline__components__kernels__threads": [1, 2, 4, 8],
    },
)

study.search(output="runs/thread-search")
```

### CLI

```
pllm bench run pllm.yaml \
  --set budget__requests=6 \
  --input-tokens 32 \
  --output-tokens 16 \
  --warmup 1 \
  --repeats 5 \
  --output runs/smoke
```

```
pllm bench search pllm.yaml \
  --set budget__requests=6 \
  --grid 'pipeline__components__kernels__threads=[1,2,4,8]' \
  --input-tokens 32 \
  --output-tokens 16 \
  --warmup 1 \
  --repeats 5 \
  --output runs/thread-search
```

The six-request budget explicitly covers one warmup and five measurements. Every repetition and search candidate gets fresh material.

The benchmark constructs public fixtures of the specified token length and records preparation separately from online execution. Unsupported, over-budget and failed candidates remain visible.

Privacy assumptions and numerical semantics are constraints—not parameters the optimizer weakens to get a better score.

## 8. A CLI command map that mirrors the API


| CLI                              | Python concept                            |
| -------------------------------- | ----------------------------------------- |
| `pllm init`                      | Construct an `Experiment`                 |
| `pllm config show/export TARGET` | Inspect or serialize public configuration |
| `pllm plan check/compile TARGET` | Validate and compile the composition      |
| `pllm prepare TARGET`            | `Runtime.prepare()`                       |
| `pllm run TARGET`                | One request through `Session.generate()`  |
| `pllm chat TARGET`               | Stateful client session                   |
| `pllm serve TARGET`              | Trusted client-side API gateway           |
| `pllm party serve ROLE`          | One native protocol-party service         |
| `pllm bench run TARGET`          | [`Benchmark.run](http://Benchmark.run)()` |
| `pllm bench search TARGET`       | [`Study.search](http://Study.search)()`   |
| `pllm assure run TARGET`         | Scoped assurance suite                    |
| `pllm components list/show`      | Registry discovery and documentation      |


**Keep** `pllm serve` **and** `pllm party serve inference` **distinct.** The first receives plaintext inside the trusted Client boundary. The second accepts encoded protocol messages. They must not become interchangeable endpoints.

CLI parsing and object loading can use Python; compilation, preparation, process supervision, protocol execution and benchmark timing stay in Rust. That is consistent with the component standard’s coarse native boundary and prohibition on performance-path Python callbacks. COMPONENT_STANDARD.mdMD

## 9. Developer-experience rules worth enforcing

**One configuration resolver.** Python and CLI produce the same public specification and logical-plan digest. CLI `--set` uses the same typed parameter paths as `with_params()`.

**Predictable output.** Generated text goes to stdout; progress and diagnostics go to stderr. Finite commands support JSON; streaming commands support JSONL. Noninteractive commands never wait for a prompt.

**Immutable locks.** Model snapshots, numerical profiles, component versions and native artifacts are frozen at compilation. Changing a locked plan means creating and compiling a new configuration—not editing live execution state.

**Useful errors.** A failed composition should identify the blocking operation, rejected component and required conversion or assumption. No generic “incompatible pipeline”.

**Discoverable help.** Parameter completion and help come from component metadata. Every command links to its human documentation and Markdown twin—for example:

```
/docs/dev/reference/cli/run
/docs/dev/reference/cli/run.md
```

The supplied bundle includes Python, YAML, local-party and remote-party examples. Their **syntax and configuration specimens were checked; the proposed APIs and distributed runtime were not executed**.

**The experience to aim for is: start with** `pllm run`**, graduate to explicit preparation and deployment, and keep using the exact same composition throughout.**