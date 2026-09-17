# Half-gates

Boolean AND-gate representation and the obligations surrounding free-XOR circuit execution.

[View canonical HTML](https://pllm.run/sdk/pipeline/protocols/garbling/half-gates/)

Document ID: `pllm.docs.protocols.garbling.half-gates`  
Release: `0.1.0`  
Build: `sha256:426652b6512bb11e794ef7caf6e150d2b19f0c9f7b933d041295b3bacebb441a`  
Source hash: `sha256:149e42548d78ad08f4d8c22390f3013eb618a1b1af9f8d5165c41aa6e1789f21`

Half-gates reduce encrypted table material for Boolean AND gates under compatible label and hash assumptions. A component contract must bind circuit identity, wire labels, correlation assumptions, evaluator material, one-time use, serialization, and output decoding.

PLLM documents this category for composition and research mapping. Presence in the taxonomy does not state that a reviewed native implementation or executable profile exists.

No public Python API currently exposes a half-gate component. See
[research evidence](/research/evidence/) for the evidence boundary; taxonomy alone
does not provide executable SDK support.
