# Compositions and recipes

Learn how PLLM records research recipes and changes to a model plan.

[View canonical HTML](https://pllm.run/research/compositions/)

Document ID: `pllm.docs.research.compositions`  
Release: `0.1.0`  
Build: `sha256:2e8eacabaf27e4c40f6b41814c3af934ab5186546825b569c6c9c1cb0a7b4db4`  
Source hash: `sha256:855a770f860f8566718697ac5e1811aa1582524b5d82afd78b425920f9ba865f`

Research workflows describe source acquisition, target model operations,
required evidence, review gates, and failure policy. They are documentation, not
executable package records. Start with the [research backlog](/research/backlog/)
and choose a workflow under [research recipes](/research/recipes/).

A [composed plan](/sdk/plans/) records the method, value formats, conversions, numeric policy,
parties, threat model, visible information, state lifetime, workload, component
version from the [component inventory](/sdk/reference/components/), and coverage.
Matching shapes and data types are not enough. A change to the source,
implementation, configuration, plan, workload, or cohort creates new plan
history. Components contribute declared capabilities; they do not establish
runtime coverage, privacy, fidelity, or benchmark parity by their presence alone.

MPCache is one example: `pllm/kv-cache-eviction` structurally adapts a compatible
plan through code in `crates/pllm-models/src/cache.rs`. It is not a protected
MPCache runtime. See [method implementations](/research/methods/) and the public
[component APIs](/sdk/components/).

Method and lifecycle choices remain separate, so research configurations can
declare and later recombine them:

```python
from pllm.components import (
    IndependentLanesProtectedTensorSchedule,
    R03CrtGatedMultiplyQ7,
)

method = R03CrtGatedMultiplyQ7()
schedule = IndependentLanesProtectedTensorSchedule(max_elements=4)

assert method.describe().category == "pllm/nonlinear-protocol"
assert schedule.describe().category == "pllm/protected-scheduler"
```

These declarations are immutable configuration inputs. The baseline Experiment
profile rejects them because it does not execute this protected region. The
specialized native region test path consumes both selections; full-decoder
Experiment resolution and matched benchmarking remain deferred until compiler
coverage is complete.
