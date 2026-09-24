# PLLM schemas

Canonical schemas use JSON Schema draft 2020-12 and distinct stable `$id` values. They describe
public records only. Live runtime, session, secret, and prepared-material handles are never JSON.

`experiment.schema.json` matches production `pllm.experiment.v2`. Plan schemas define accepted 0.1
architecture contracts; their presence does not claim compiler implementation. Evidence schemas
keep measurements and scoped assurance outcomes separate from claims.
`decoder-session.schema.json` covers only the public plan/body commitments supplied before a
prepared session; it does not serialize session handles, one-use material, or client secrets.

Research provenance uses separate source-record, upstream-artifact-lock, method-record, and
reproduction-recipe schemas. Recipe source acquisition and workflow execution are independent.

Run `uv run --frozen --all-extras pytest tests/test_configuration.py -q` to validate the
canonical schemas and experiment fixtures.
