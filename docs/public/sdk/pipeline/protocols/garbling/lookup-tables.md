# Lookup tables

Bounded function tables with explicit domains, indexing, leakage, and material size.

[View canonical HTML](https://pllm.run/sdk/pipeline/protocols/garbling/lookup-tables/)

Document ID: `pllm.docs.protocols.garbling.lookup-tables`  
Release: `0.1.0`  
Build: `sha256:06781cb06588c2919922b5252154662ef566cf6cbec9ad8049f32aa70887b75e`  
Source hash: `sha256:b4de26e167a6b5b6538882089e360fdb8610b0bf90dc40bf3fd0cae0476354fa`

A lookup-table component maps a finite encoded domain to an encoded range. Its contract records signed interpretation, table cardinality, invalid indices, material ownership, one-time use, communication, and whether selection is data-oblivious under the stated threat model.

Tables are useful for nonlinearities only when domain bounds and total storage remain explicit. Approximation and model-quality evidence belong to the numeric policy, not the table name.

No public Python API currently exposes a garbled lookup-table component. See
[research evidence](/research/evidence/) for the evidence boundary; no lookup-table
component is currently published through the SDK.
