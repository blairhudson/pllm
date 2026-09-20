# Search

Generate deterministic experiment candidates and rank only benchmark results with compatible evidence cohorts.

[View canonical HTML](https://pllm.run/sdk/research/search/)

Document ID: `pllm.docs.measure.search`  
Release: `0.1.0`

`pllm.search` searches immutable public configuration, not live sessions or secret
material. A search space names exact `Experiment.get_params(deep=True)` paths;
invalid paths, duplicate values, invalid combined configurations, and excessive
cardinality fail before evaluation.

Candidate generation does not benchmark or rank anything. Evaluate candidates
with a function that returns `BenchmarkResult`, then use an explicit metric
direction or Pareto objective. PLLM rejects comparisons across incompatible
model, workload, environment, numeric, privacy, or warm-state cohorts.

## Python SDK example

```python
from pllm import Deployment, ExecutionBudget, Experiment, MaskedLinearCpu, Model
from pllm.kernels import Cpu
from pllm.search import GridSearch, SearchSpace

experiment = Experiment(
    "thread-search",
    MaskedLinearCpu(
        Model("Qwen/Qwen2.5-0.5B-Instruct"),
        kernels=Cpu(threads=1),
    ),
    Deployment.local(root=".pllm/thread-search"),
    ExecutionBudget(requests=1, max_input_tokens=32, max_new_tokens=4),
)
space = SearchSpace(
    experiment,
    {"pipeline__kernels__threads": [1, 2, 4]},
)
candidates = GridSearch("docs-threads", space).candidates()

assert [candidate.parameters["pipeline__kernels__threads"] for candidate in candidates] == [1, 2, 4]
assert len({candidate.configuration_digest for candidate in candidates}) == 3
```

API: [`pllm.search`](/sdk/reference/python/pllm/#objects-and-signatures)

Persist the exact candidate experiment with each observation. Do not select a
winner from latency alone when token counts, model fingerprints, environments,
or other evidence-cohort fields differ.
