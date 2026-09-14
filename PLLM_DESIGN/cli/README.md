# PLLM composition and CLI design

This is an additive API/CLI specification for the current PLLM uplift, not a functioning
private-inference release. The included Python and shell examples require the proposed
interfaces to be integrated. No model, private inference, native compilation or network
party has been executed by this bundle.

Read **DESIGN.md**. The examples show:

- `private_app.py`: one immutable Python Experiment and guarded local execution.
- `pllm.yaml`: the equivalent declarative experiment.
- `local.sh`: pure CLI local lifecycle.
- `from_python.sh`: CLI loading/exporting a trusted Python-defined experiment.
- `inference.sh`, `preparation.sh`: role-specific remote services.
- `remote.yaml`, `remote_prepare.sh`, `remote_run.sh`: attached deployment and strict offline preparation.
- `benchmark.py`, `benchmark.sh`: matched Python/CLI benchmark and search declarations.
- `composition.py`: preferred single-evaluator composition, still coverage-gated.

The primary example uses the client-heavy masked-linear baseline intentionally. Missing
archived numerical manifests or native functionality must fail, not invent a reproduction.
The single-evaluator target cannot fall back to that baseline to satisfy a thin-client policy.

Remote endpoint names and identity file locations illustrate the deployment contract; real
operators provision their endpoints/credentials independently. No secrets are distributed.

`syntax_checks.json` records only Python AST parsing, YAML parsing/field checks and Bash
syntax checks. None is a runtime, API-compatibility, privacy or deployment test.
