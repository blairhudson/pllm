# Assurance

Evaluate a specific privacy, integrity, numeric, implementation, or deployment claim.

[View canonical HTML](https://pllm.run/sdk/research/assurance/)

Document ID: `pllm.docs.assurance`  
Release: `0.1.0`  
Build: `sha256:1e873bcc571b7b7717ee66b62aff1ec3e361ead12088e6b707b77e88ee008328`  
Source hash: `sha256:0e618f172f701848159a2103a6650e9abe5800ae2d74719f63a6e8a338780732`

An assurance record identifies the exact plan, component versions, threat model,
protected values, visible information, corruption and collusion assumptions,
numeric policy, environment, procedure, result, and limitations.

Reference tests, formal analysis, cryptographic review, transport hardening,
deployment inspection, and leakage tests answer different questions. Missing
assurance is not a failed test, but it is not a pass. Evidence for participants
that follow the protocol does not establish security against malicious behavior.

A publication claim must cite the relevant records and keep their limitations.
See [reading research and evidence](/learn/reading-research-and-evidence/).

## Python SDK example

```python
from pllm import assure

report = assure()
print(report.schema_version, report.to_dict()["limitations"])
```

Report scope is bundled assurance fixtures, not universal privacy proof.
