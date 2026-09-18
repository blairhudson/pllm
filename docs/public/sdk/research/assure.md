# Assure

Run a defined assurance check and report exactly what its result supports.

[View canonical HTML](https://pllm.run/sdk/research/assure/)

Document ID: `pllm.docs.measure.assure`  
Release: `0.1.0`  
Build: `sha256:87cbf81718b764da3fb4871c21f12e5943efd717a5d5854e5a0cecf969808585`  
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
