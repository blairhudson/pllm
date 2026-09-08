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
