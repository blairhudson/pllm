# Trust boundary

Understand what the client, preparation service, inference service, and operators can access.

[View canonical HTML](https://pllm.run/learn/understand/trust-boundary/)

Document ID: `pllm.docs.understand.trust-boundary`  
Release: `0.1.0`  
Build: `sha256:1e873bcc571b7b7717ee66b62aff1ec3e361ead12088e6b707b77e88ee008328`  
Source hash: `sha256:2149462a616b1532c066c03b11559a4d4e13c79271afc32a217d5715a026bb67`

Client owns plaintext prompts, token identities, decoded output, and authorization decisions. Preparation and inference are distinct roles in protected methods that require them; co-location tests protocol wiring but does not prove non-collusion. Provider describes component implementation. Operator controls deployed process, credentials, storage, and policy. These names are not interchangeable.

Endpoints and identity strings are selectors, not authentication proof. Placement cannot change protocol, numeric policy, leakage, corruption assumptions, or output policy. Timing, traffic shape, public model identity, shapes, and approximate sequence length may remain visible depending on selected contract.

No current generic party-service CLI exposes this target topology. See [provider roles](/sdk/operate/provider-roles/).
