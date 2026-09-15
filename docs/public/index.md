# PLLM — Private LLM Inference

Private language model inference. Client and server guides, the protocol, and reproducible research.

[View canonical HTML](https://pllm.run/)

Document ID: `pllm.home`  
Release: `0.1.0`  
Build: `sha256:1e873bcc571b7b7717ee66b62aff1ec3e361ead12088e6b707b77e88ee008328`  
Source hash: `sha256:dabcf8a2aacfcc2e1912f645ceff604724110e336e3650dd261e39d686976141`

# Keep your data private. Open compute to the world.

PLLM is a high-performance private LLM multi-party inference runtime and autonomous research harness.

[Start building →](#start-building)
[Run research →](#run-research)

[Learn how PLLM works](/learn/)

## Separate the compute from the data.

Remote AI usually means one provider gets both the work and the data. PLLM separates them.

When providers do not need the full request, more providers can compete on price, location, speed, and energy use. Users get more choice. Less private data is exposed to one company.

These are goals. Speed, cost, quality, privacy, and energy use must each be measured.

## Split the work across clear roles.

Client
Holds plaintext prompts, context, model state, masks, and output.

Preparation
Creates one-time protected work before inference.

Inference
Runs masked model stages without receiving the seeds.

Neither remote service receives the full request. This depends on both services following the protocol and not colluding.

[Learn how private inference works](/learn/masked-linear-inference/)

## Start building.

Install PLLM from PyPI. Run the real client, preparation, and inference roles on one machine, or add the SDK to a Python project.

**CLI**

**Python SDK**

**Inspect**

```text
uv tool install pllm
pllm dev dashboard --tiny --no-open
```

```text
uv add pllm
uv run python - <<'PY'
from pllm.components import get_component

component = get_component("pllm/masked-linear")
print(component.component, component.lifecycle_phase)
PY
```

```text
pllm components list
pllm research methods list
```

The tiny dashboard checks transport with generated weights. It does not measure model quality or production performance.

[Install PLLM](/learn/start/installation/)
[Use the SDK](/sdk/)
[Run a real model](/learn/start/first-local-benchmark/)

## Run autonomous research.

Generate an `AGENTS.md` for Codex, Claude Code, OpenCode, or another coding agent. The file explains PLLM and tells the agent how to find strong methods, add them as components, test them, and run a fair benchmark.

**Agent guide**

**Research catalog**

```text
pllm research agents --output AGENTS.md
```

```text
pllm research sources list
pllm research methods list
pllm research recipes list
```

- Find the strongest relevant methods.

- Add them to PLLM.

- Test correctness, privacy, and model quality.

- Compare them under the same conditions.

Comparisons must use the same model, workload, privacy rules, numeric settings, and hardware. Automation cannot invent claims or publish results without review.

[Research overview](/research/)
[Research workflow](/research/)
[Benchmarks](/sdk/research/benchmarks/)

## Continue with PLLM.

[Documentation](/)
[Source](https://github.com/probabilistic-alchemy/pllm)
[Technical paper](/research/paper/)
[Whitepaper](/research/whitepaper/)
[Benchmarks](/sdk/research/benchmarks/)
