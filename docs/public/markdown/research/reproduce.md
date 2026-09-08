# Reproduce the study

Keep the experiment environment separate from the serving runtime.


The complete lifecycle experiment is included under `research/lifecycle/`. It has its own `pyproject.toml`, tests, trained small checkpoint, and raw results. The source pins Python 3.13 and recorded numerical dependencies.

## Install and test

```bash
cd research/lifecycle
uv sync --group test
uv run pytest -q
```

This resolves actual HE dependencies. The study does not use a plaintext substitute when an HE package is missing.

## Recreate lifecycle results

Inspect the runner help before allocating large matrices:

```bash
uv run python -m pllm_study.lifecycle --help
uv run python -m pllm_study.benchmark --help
uv run python -m pllm_study.planner --help
```

The bundled result files record each command's parameters and timing components. Start with the small tests and checkpoint. Full synthetic Qwen stages can exceed the memory and preparation budget of a laptop.

## Compare like with like

Match the same integer matrix, batch, modulus, thread count, and inclusion of conversion and decryption. Run the control on the same host. Separate setup from steady preparation without removing setup from the total request trace.

Record hardware, memory limit, CPU quota, dependency versions, model hash, tokenizer revision, prompt length, output length, numerical error, and discarded correlations. Do not multiply speedups from different scopes into a new claimed language model rate.
