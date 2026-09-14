# Evidence from this handoff

- `test_results.json` / `pytest.xml`: 76 Python contract, schema, CLI, archive, fixture and formal-wrapper tests passed; no tests skipped in the authoring environment.
- `formal_results.json`: ten native Z3 checks met their declared SAT/UNSAT expectations. Three are explicit counterexamples; two UNSAT results establish leakage identities in unsafe constructions. All are scoped specifications, not a complete implementation proof.
- `attack_fixture_results.json`: six deliberately weakened public fixtures, plus the finite ideal uniform-mask control. Not attacks against an installed PLLM version.
- `combined_assurance.json`: repeat run of the public fixture/formal campaign through the actual staging CLI.
- `package_check.json`: source-card, recipe, legacy-hash and dependency-graph checks.
- `target_admission.json`: preferred full-model target is explicitly blocked by missing model/numeric locks and unimplemented coverage.
- `build_status.json`: no Rust/Cargo/maturin toolchain; native source was not compiled, wheel-built or benchmarked here. No new model or GPU run.

Earlier results live under `legacy/` and retain their source-reported scope. Do not merge their performance numbers into new native evidence.
