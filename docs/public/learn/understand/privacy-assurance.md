# Privacy and assurance semantics

State a specific privacy claim and interpret the evidence that applies to it.

[View canonical HTML](https://pllm.run/learn/understand/privacy-assurance/)

Document ID: `pllm.docs.understand.privacy-assurance`  
Release: `0.1.0`  
Build: `sha256:50bdd8d40f8b1f362adbabe08e845d37acfffd39aacc3351d7706456b26c4b1a`  
Source hash: `sha256:f4ba911c9dd49cb8fd9ebb3ad589b9f2cbd0703c283f8bbb103adb4f64a90a00`

Privacy claim must name protected values, observer, roles, corruption and non-collusion assumptions, leakage, workload, implementation, and evidence. “Private” without adjacent scope and evidence is not a valid implementation claim.

Allowed assurance outcomes are `proved_in_model`, `refuted_in_scope`, `not_refuted`, `inconclusive`, `outside_contract`, and `unchecked`. Successful test or failed attack is at most `not_refuted` for recorded budget unless formal result says otherwise. Missing, unavailable, skipped, and not applicable differ.

Current generated descriptors contain no applicable assurance evidence. Research method lifecycle currently records assurance as `unchecked`. See [assure](/sdk/research/assure/) and [claim taxonomy](/learn/understand/evidence-claims/).
