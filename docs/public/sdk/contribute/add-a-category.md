# Add a category

Extend the semantic component graph when an existing category cannot express a real contract.

[View canonical HTML](https://pllm.run/sdk/contribute/add-a-category/)

Document ID: `pllm.docs.contribute.add-a-category`  
Release: `0.1.0`  
Build: `sha256:1e873bcc571b7b7717ee66b62aff1ec3e361ead12088e6b707b77e88ee008328`  
Source hash: `sha256:51ac44a9659b02824c90f9cd9975cef347d25b17a8fe2e80072c6d7dcb3cf248`

Add a category only when existing operator, representation, protocol, conversion, preparation, kernel, compiler, runtime, deployment, metric, search, or assurance contracts cannot express the new capability without ambiguity.

Define ownership, descriptor schema, composition rules, lifecycle, validation,
compatibility, evidence, public API, CLI discovery, generated reference,
documentation, and migration policy together. Do not name a reusable category
after one paper or model.

## Python SDK example

```python
from pllm.components import list_components

categories = sorted({component.category for component in list_components()})
print(categories)
```

Inspect existing categories before proposing a new contract.
