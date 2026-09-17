# Assurance

Evaluate a specific privacy, integrity, numeric, implementation, or deployment claim.

[View canonical HTML](https://pllm.run/sdk/research/assurance/)

Document ID: `pllm.docs.assurance`  
Release: `0.1.0`  
Build: `sha256:bab6f73b33765ac11862794abf645d4eb324a2fd52abc43cc2a9cd21c09e27c7`  
Source hash: `sha256:13f22bea16c6bc03779bbf5003402ef7eba8dd47d0b615ef73e086341603f93d`

An assurance record identifies the exact plan, component versions, threat model,
protected values, visible information, corruption and collusion assumptions,
numeric policy, environment, procedure, result, and limitations.

Reference tests, formal analysis, cryptographic review, transport hardening,
deployment inspection, and leakage tests answer different questions. Missing
assurance is not a failed test, but it is not a pass. Evidence for participants
that follow the protocol does not establish security against malicious behavior.

A publication claim must cite the relevant [research evidence](/research/evidence/)
and keep its limitations. See
[reading research and evidence](/learn/reading-research-and-evidence/) for the
conceptual boundary and [publications and claims](/research/publications/) for the
review requirements.

## Python SDK example

```python
import pllm

document = pllm.assure().to_dict()
findings = {finding["id"]: finding for finding in document["findings"]}
assert document["ideal_uniform_control"]["outcome"] == "proved_in_model"
assert findings["affine_label_reuse"]["outcome"] == "refuted_in_scope"
```

API: [`pllm.assure`](/sdk/reference/python/pllm/#objects-and-signatures)

Report scope is bundled assurance fixtures, not universal privacy proof.
