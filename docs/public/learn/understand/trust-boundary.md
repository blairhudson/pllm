# Trust boundary

Understand what the client, preparation service, inference service, and operators can access.

[View canonical HTML](https://pllm.run/learn/understand/trust-boundary/)

Document ID: `pllm.docs.understand.trust-boundary`  
Release: `0.1.0`  
Build: `sha256:fa1208fc732ce6403c8c82d95417818355eb7253231b6d031bfc704a15c0a95f`  
Source hash: `sha256:fef1568144f0f8f9a318f91dd19b04b81e8baf86b003617ab8f5d5754c7fdcac`

Client owns plaintext prompts, token identities, decoded output, and authorization decisions. Preparation and inference are distinct roles in protected methods that require them; co-location tests protocol wiring but does not prove non-collusion. Provider describes component implementation. Operator controls deployed process, credentials, storage, and policy. These names are not interchangeable.

Endpoints and identity strings are selectors, not authentication proof. Placement cannot change protocol, numeric policy, leakage, corruption assumptions, or output policy. Timing, traffic shape, public model identity, shapes, and approximate sequence length may remain visible depending on selected contract.

The public CLI can start each role, but command availability does not establish
operator independence. See [provider roles](/sdk/operate/provider-roles/) and
the [local gateway](/learn/integrations/local-gateway/).
