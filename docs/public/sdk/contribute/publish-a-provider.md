# Publish a provider

Expose a component implementation without importing it during core discovery.

[View canonical HTML](https://pllm.run/sdk/contribute/publish-a-provider/)

Document ID: `pllm.docs.contribute.publish-a-provider`  
Release: `0.1.0`  
Build: `sha256:426652b6512bb11e794ef7caf6e150d2b19f0c9f7b933d041295b3bacebb441a`  
Source hash: `sha256:28cc67df58ecba3cbf725acd0f36daaffa145628bd30b496c0188c9d22656cca`

A provider package publishes signed or digest-addressed descriptor metadata, implementation artifacts, host requirements, supported component versions, representations, roles, and evidence references. Discovery reads static metadata before any trusted code executes.

Provider identity does not imply operator identity or trust. Loading requires explicit policy, compatibility validation, and artifact verification. Distribution, component, provider, deployment, and evidence versions remain separate.

No public Python API publishes or registers provider packages. See [current support](/sdk/reference/status/)
for the static built-in discovery boundary.
