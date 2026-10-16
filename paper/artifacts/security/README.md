# Boundary security audit evidence

`boundary-security-checks.json` records twelve bounded checks from
`tests/test_paper_boundary_security.py`: exact small-ring algebra, all provider
coalitions in an ideal-XOR toy model, independent installed-native arithmetic,
counterfactual randomness/codec failures, actual cancellation/replay custody
limits, and three independent-review witnesses: coin-dependent timing leakage,
causal body simulation/conditional-mask correlation, and packed-input ambiguity.
No model/checkpoint is loaded, and no service, build or benchmark starts.

The theorem now covers only an untimed admitted honest protocol with
cryptographic-coin-independent semantic leakage. Actual durations (including
serialized `server_ns`), backpressure and errors are outside that theorem. The
bounded timing example establishes a composition counterexample, not an actual
runtime timing exploit.

`boundary-security-checks.pre-review.json` preserves the original nine-check
snapshot byte-for-byte. It is superseded: its test hash identifies the earlier
test version and is not expected to match the current file. Replay below updates
only the current report.

Replay using the existing environment:

```sh
.venv/bin/python -m pytest tests/test_paper_boundary_security.py -q
PYTHONPATH=python .venv/bin/python tests/test_paper_boundary_security.py \
  --output paper/artifacts/security/boundary-security-checks.json
```

The current JSON hashes the audit, test, relevant source files and installed native
extension. Hashing both does not establish that the binary was built from the
recorded source revision. All fixtures are public/synthetic. Fresh DPF random
keys remain ephemeral and are not written to evidence. The ideal-XOR model is
not the actual AES DPF and is not cryptographic-security evidence.

Interpretation and theorem conditions: `paper/SECURITY-AND-CORRECTNESS.md`.
