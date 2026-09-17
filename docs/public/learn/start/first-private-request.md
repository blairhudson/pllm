# Inspect a private inference plan

Define a reproducible experiment, lower a semantic model plan, and check its execution coverage.

[View canonical HTML](https://pllm.run/learn/start/first-private-request/)

Document ID: `pllm.docs.start.first-private-request`  
Release: `0.1.0`  
Build: `sha256:2b873610e88902ce44935954b3c8be5ba82c51e9673ea0929e4c8f08e5a4bf62`  
Source hash: `sha256:0ef7daf4f6dbbf65af60e4ffcbfaa269a11e3c25f7c564b69ec68e38ae0b728a`

## Define the experiment

```python
from pllm import Experiment, Model, Pipeline
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
    budget=ExecutionBudget(requests=1, max_input_tokens=128, max_new_tokens=32),
)

print(experiment.configuration_digest())
print(experiment.to_spec())
```

The [SDK configuration model](/sdk/configuration/) keeps this declaration
inspectable without starting a service or loading a model. You can inspect or
export an equivalent checked-in JSON or YAML file:

```bash
pllm config show examples/pllm.yaml --format json
pllm config export examples/pllm.yaml --output experiment.resolved.json
```

Python configuration targets execute local code. In noninteractive use, you must
explicitly allow them with `--trust-python`.

## Lower and compose

The [`lower_model` model-planning path](/sdk/build/models/) accepts public model
configuration and workload limits. It does not accept a model repository name or
load weights:

```python
import pllm
from pllm.components import KvCacheEviction

config = {
    "model_type": "qwen2", "hidden_size": 896, "intermediate_size": 4864,
    "num_hidden_layers": 24, "num_attention_heads": 14,
    "num_key_value_heads": 2, "vocab_size": 151936,
    "max_position_embeddings": 32768, "hidden_act": "silu",
    "rms_norm_eps": 1e-6, "rope_theta": 1000000,
    "tie_word_embeddings": True,
}
plan = pllm.lower_model(config, batch=1, max_input_tokens=128, max_new_tokens=32)
composed = plan.apply(KvCacheEviction())
coverage = composed.coverage("research.single_evaluator")
print(composed.digest, coverage.complete, coverage.to_dict())
```

This plan describes work that could precede an inference request. The reported
compiler coverage is currently incomplete, so the example does not perform
private text generation. Semantic planning, runtime support, model quality,
benchmark results, privacy evidence, and deployment readiness are separate
status checks.

Inspect component metadata with `pllm components list`. See
[runtime](/sdk/pipeline/runtime/),
[protocol interfaces](/sdk/pipeline/protocols/), and
[current support](/sdk/reference/status/). To send an executable request from a
compatible client, continue to the [integration guides](/learn/integrations/).
