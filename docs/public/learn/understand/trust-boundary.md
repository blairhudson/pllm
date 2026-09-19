# Trust boundary

Understand what the client, preparation service, inference service, and operators can access.

[View canonical HTML](https://pllm.run/learn/understand/trust-boundary/)

Document ID: `pllm.docs.understand.trust-boundary`  
Release: `0.1.0`

Client owns plaintext prompts, token identities, decoded output, and authorization decisions. Preparation and inference are distinct roles in protected methods that require them; co-location tests protocol wiring but does not prove non-collusion. Provider describes component implementation. Operator controls deployed process, credentials, storage, and policy. These names are not interchangeable.

Endpoints and identity strings are selectors, not authentication proof. Placement cannot change protocol, numeric policy, leakage, corruption assumptions, or output policy. Timing, traffic shape, public model identity, shapes, and approximate sequence length may remain visible depending on selected contract.

The public CLI can start each role, but command availability does not establish
operator independence. See [provider roles](/sdk/operate/provider-roles/) and
the [local gateway](/learn/integrations/local-gateway/).
