# Validation records

`python-reference.log` and its JUnit file record the normal suite with the
explicit Python numeric reference. `he-lane.log` and `he/` record real HE cases
executed by `scripts/run_test_lane.py` in separate processes. These are not
compiled Rust test results.

`docs-checks.log` records Node documentation tests and internal links.
`typescript-syntax.json` records syntax transpilation only. It is not a semantic
framework type check or a Next.js build. `repository.json`, `syntax.json` and
`import-and-cli.json` record source, packaging and namespace checks.

`summary.json` is the machine readable account of what ran and what did not.
See `../VALIDATION.md` for interpretation.
