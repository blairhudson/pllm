# Start here: integrate, do not replace blindly

## Decisions to carry into PLLM

1. **PLLM is a privacy-focused composable inference compiler/runtime and evidence system.** It is not one masking protocol plus switches.
2. **Rust owns execution.** Python is a friendly API and configuration/reporting layer. Native code owns the compiler passes that scan model data, numeric kernels, crypto, material allocation, transport, scheduler, benchmark clocks and adversary hooks.
3. **One online evaluator is the preferred deployment.** Model weights at offline Preparation are allowed. HE and multiple online workers require separately authorized benchmark profiles.
4. **No untyped composition.** An arithmetic label, additive share, masked integer and Boolean label need explicit conversions. Security, numerical quality and topology are constraints, not optimization rewards.
5. **Evidence is a product feature.** Every compiled plan and result identifies its method, source version, correctness scope, privacy assumptions and missing proof/review obligations.

## First commands: no Rust required

Run from the extracted package root:

```bash
export PYTHONPATH="$PWD/python"
python -m pllm_uplift research list
python -m pllm_uplift research show R01
python -m pllm_uplift validate configs/operator-smoke.json
python -m pllm_uplift assure --output evidence/local-assurance.json
python -m pytest -q tests
python tools/check_package.py
```

On PowerShell, use `$env:PYTHONPATH="$PWD/python"` instead of `export`. The assurance command requires the native Z3 shared library for the SMT checks. Absence is recorded as unavailable and returns an error; it never produces a pass. `--fixtures-only` runs only the public negative controls.

## Native build and benchmark

On a machine with Cargo and Rust installed:

```bash
cargo test --manifest-path native/Cargo.toml --all-targets
cargo run --release --manifest-path native/Cargo.toml -p pllm-bench --bin pllm-native-lab -- matrix 64 18 7 24 32
cargo run --release --manifest-path native/Cargo.toml -p pllm-bench --bin pllm-native-lab -- assure
python -m pip install maturin
python -m pip install .
python -m pllm_uplift bench --dim 64 --lanes 18 --repeats 7 --bits 24 --tile 32 --output evidence/native-kernel.json
```

These commands run the included **public synthetic native kernel experiment**, not a full model. There is deliberately no Python performance fallback. An unavailable extension fails with installation instructions. Full `pllm bench run`, multi-party launch, model loading and twenty paper implementations are specified integration deliverables, not implemented commands in this staging package.

## Bring it to the existing repository

From this package root, `python tools/inspect_repository.py /absolute/path/to/your/pllm --output evidence/repository-map.json` makes a read-only file/dependency inventory. It does not read credentials, source contents, Git history or private datasets. Review [MIGRATION.md](docs/MIGRATION.md) before moving code.

Keep the production `pllm-inference` distribution, `pllm` import, CLI and supported Responses client API. This package uses the temporary `pllm_uplift` namespace to prevent accidental overwrite. Merge native modules into the existing Rust workspace and PyO3 extension after parity tests, rather than publishing a competing runtime.

## What counts as done

A registry card is not a reproduction. A reproduced equation is not a reproduced paper. A reproduced paper is not automatically a secure adapted protocol. Native parity is not a full-model speedup. Default eligibility requires full functional coverage, applicable assurance evidence, deployment benchmarks and resource/quality compliance for the requested contract.

Read `docs/RESEARCH_PORTFOLIO.md` for the twenty implementation targets and `backlog/tasks.json` for their dependency-linked work packages.
