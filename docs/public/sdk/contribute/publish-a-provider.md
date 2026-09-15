# Publish a provider

Expose a component implementation without importing it during core discovery.

[View canonical HTML](https://pllm.run/sdk/contribute/publish-a-provider/)

Document ID: `pllm.docs.contribute.publish-a-provider`  
Release: `0.1.0`  
Build: `sha256:bf56c232413fe9b57bc2befe009368956690afaf0d708914c7620c3b7d5d9531`  
Source hash: `sha256:d58223f0fe5e9e5722a33d4717e34cd783e5c0f0293bc7e9d64e914a711dff76`

A provider package publishes signed or digest-addressed descriptor metadata, implementation artifacts, host requirements, supported component versions, representations, roles, and evidence references. Discovery reads static metadata before any trusted code executes.

Provider identity does not imply operator identity or trust. Loading requires explicit policy, compatibility validation, and artifact verification. Distribution, component, provider, deployment, and evidence versions remain separate.

## Python SDK example

```python
from pllm.components import get_component

descriptor = get_component("pllm/masked-linear")
print(descriptor.provider, descriptor.distribution, descriptor.artifacts)
```

Discovery reads metadata only and does not load provider code.
