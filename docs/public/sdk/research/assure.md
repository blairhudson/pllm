# Assure

Run a defined assurance check and report exactly what its result supports.

[View canonical HTML](https://pllm.run/sdk/research/assure/)

Document ID: `pllm.docs.measure.assure`  
Release: `0.1.0`  
Build: `sha256:1e873bcc571b7b7717ee66b62aff1ec3e361ead12088e6b707b77e88ee008328`  
Source hash: `sha256:9af947d02e7699a9242a080ce026e29c78a15a25b8b1af23dda5338c36fc293a`

`pllm.assure()` runs implemented native fixtures, including negative controls, and returns `EvidenceReport`. Report production is not universal privacy proof. Each finding retains model or observed view, assumptions, resource bound, implementation/refinement boundary, and exact outcome.

CLI `assure run` is unavailable because scoped assurance command/output contract is incomplete. Python support applies only to fixtures present in current native package. See [privacy semantics](/learn/understand/privacy-assurance/).

## Python SDK example

```python
from pllm import assure

report = assure()
print(report.to_dict()["limitations"])
```

Result covers bundled fixtures only, not a production runtime attack.
