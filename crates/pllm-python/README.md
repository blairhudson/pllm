# PLLM Python binding

PyO3 exposes the numeric core as `pllm._native`. The Python package is in
`python/pllm`. Build from the repository root so Maturin reads the root
`pyproject.toml` and packages Python and Rust together:

```bash
uv sync
uv run maturin develop --release
uv build
```

The binding accepts immutable bytes, validates shape and representation, and
releases the Python interpreter lock during numeric execution. Buffer conversion
is part of the interface cost and must remain inside complete call benchmarks.
