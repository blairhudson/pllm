# Compiler

Learn how PLLM turns a model plan and component choices into a checked execution plan.

[View canonical HTML](https://pllm.run/sdk/pipeline/compiler/)

Document ID: `pllm.docs.compiler`  
Release: `0.1.0`  
Build: `sha256:1e873bcc571b7b7717ee66b62aff1ec3e361ead12088e6b707b77e88ee008328`  
Source hash: `sha256:ce1397fe7dd5f0980df6541073962ade70e8f3b9f162056a30c766f83b0da39f`

Compilation combines a model plan, component choices, workload limits, and policy.
It checks operator support, value formats, conversions, numeric settings, party
placement, preparation, native kernels, and assurance requirements. The result is
an executable plan with a content digest.

The coverage report names every missing operator, conversion, party role, numeric
bound, assurance record, or compatible component version. Required gaps stop
compilation instead of triggering a hidden fallback. All model families use the
same semantic model format.

Current research profiles cannot execute a complete listed model plan. A model
adapter can therefore be available while full private inference remains unsupported.

## Python SDK example

```python
import json
from pathlib import Path

from pllm import compile

request = json.loads(Path("compile-request.json").read_text(encoding="utf-8"))
plan = compile(request)
print(plan.execution_plan_digest)
```

Compilation rejects unsupported coverage rather than hiding it behind a fallback. Complete model compilation remains unsupported for current research profiles.
