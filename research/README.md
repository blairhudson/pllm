# Research and evidence

`lifecycle/` contains the independent preparation and inference study. It has its
own `pyproject.toml`, test suite and executable entry points. `evidence/` contains
recorded measurements used by the manuscript. These are historical data, not
results newly produced by the repository packaging workflow.

The application server and the study do not share one fully integrated execution
backend. The study includes later coefficient preparation work that is not
advertised as the application provider's default engine.

Run the study separately:

```bash
cd research/lifecycle
uv sync --group test
uv run pytest
```

Tests containing real HE use the actual TenSEAL backend. Full stage benchmarks
can require substantial CPU, memory and preparation time. Read
`lifecycle/REPORT.md` before running model-sized experiments.

The measured large Qwen stages use synthetic matrices at actual model shapes.
The trained generation checkpoint is small. No complete 27B generation rate,
GPU performance, or general malicious security claim follows from these data.
