# PLLM schemas

Canonical schemas use JSON Schema draft 2020-12 and distinct stable `$id` values. They describe
public records only. Live runtime, session, secret, and prepared-material handles are never JSON.

`experiment.schema.json` matches production `pllm.experiment.v1`. Plan schemas define accepted 0.1
architecture contracts; their presence does not claim compiler implementation. Evidence schemas
keep measurements and scoped assurance outcomes separate from claims.

Research provenance uses separate source-record, upstream-artifact-lock, method-record, and
reproduction-recipe schemas. Recipe source acquisition and workflow execution are independent.

Run `python3 research/validate.py` for dependency-free repository integrity and experiment-fixture
checks. Production YAML parity is covered by `tests/test_configuration.py`.
