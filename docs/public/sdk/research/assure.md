# Assure

Run a defined assurance check and report exactly what its result supports.

[View canonical HTML](https://pllm.run/sdk/research/assure/)

Document ID: `pllm.docs.measure.assure`  
Release: `0.1.0`  
Build: `sha256:2b873610e88902ce44935954b3c8be5ba82c51e9673ea0929e4c8f08e5a4bf62`  
Source hash: `sha256:026d5431c2c14d5a4f2d88c51e875ff73c403eaaa00f5e9d3872c157da4e79d9`

`pllm.assure()` runs implemented native fixtures, including negative controls, and returns `EvidenceReport`. Report production is not universal privacy proof. Each finding retains model or observed view, assumptions, resource bound, implementation/refinement boundary, and exact outcome.

CLI `assure run` is unavailable because scoped assurance command/output contract is incomplete. Python support applies only to fixtures present in current native package. See [privacy semantics](/learn/understand/privacy-assurance/).

## Python SDK example

```python
import pllm

report = pllm.assure()
document = report.to_dict()
findings = {finding["id"]: finding for finding in document["findings"]}
assert findings["missing_truncation_carry"]["outcome"] == "refuted_in_scope"
assert "production runtime not attacked" in document["limitations"]
```

API: [`pllm.assure`](/sdk/reference/python/pllm/#objects-and-signatures)

Result covers bundled fixtures only, not a production runtime attack.
