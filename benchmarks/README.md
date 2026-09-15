# Benchmarks and profiles

```bash
uv run python benchmarks/rust_migration.py --cpp-reference --threads 1 --output results/rust.json
uv run python benchmarks/rust_migration.py --cpp-reference --threads 4 --output results/rust-4.json
uv run python benchmarks/rust_migration.py --qwen --cpp-reference --profile results/native.prof
uv run python -m pstats results/native.prof
```

Matrix snapshots are compiled before repeated timing; their cost is reported
separately. Each call includes Python validation and buffer conversion. The C++
control is the exact previous source under `reference/`, not a rewritten slow
baseline. Every output is checked before its timing is accepted.

The optional `--allow-missing-rust` flag records a missing native build and runs
only available controls. It never invents a Rust measurement or speedup. Large
shape results are synthetic matrix tests, not language model TPS.

The lifecycle suite in `research/lifecycle/` measures actual encrypted
preparation and decoding separately. The historical paper records those earlier
measurements, not the new native implementation.

Compare client codecs, quantization, masking, and OS random generation with
`uv run python benchmarks/client_native.py`. Each timed call includes Python
validation and buffer conversion. Every returned sample is checked outside the timer.

## Live protocol dashboard

Run a real three-role seeded-preparation chat with Qwen2.5-0.5B-Instruct:

```bash
uv run pllm dev dashboard
```

The first run downloads the public model if it is not already cached. The
loopback dashboard starts isolated inference and preparation service processes,
runs the PLLM client in the dashboard process, and displays OTEL
process metrics, HTTP traces, protocol traffic, TTFT, and generation throughput.
Completed runs retain unused inventory rows instead of preparing another batch;
the next run refills only when its exact row requirement exceeds that remainder.
Use another local or Hugging Face public-weight checkpoint with:

```bash
uv run pllm dev dashboard --model /path/to/Qwen2.5-0.5B-Instruct --model-id Qwen/Qwen2.5-0.5B-Instruct
```

Prompts and activations are not attached to OTEL records; custom metrics contain
byte counts and stage IDs. For a fast transport-only smoke test, use
`uv run pllm dev dashboard --tiny`; its generated random weights do not produce
meaningful language.
