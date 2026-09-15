# Compositions and recipes

Learn how PLLM records research recipes and changes to a model plan.

[View canonical HTML](https://pllm.run/research/compositions/)

Document ID: `pllm.docs.research.compositions`  
Release: `0.1.0`  
Build: `sha256:1e873bcc571b7b7717ee66b62aff1ec3e361ead12088e6b707b77e88ee008328`  
Source hash: `sha256:20270d54411c507f0efa499fac5353e728404cf5820dc96109b7455508e2aabb`

Recipes describe source acquisition, target model operations, required evidence,
review gates, and failure policy. Inspecting a recipe returns a read-only record;
it does not run third-party code or a PLLM workflow. See the
[Research APIs](/sdk/research/) and [CLI](/cli/) documentation for callable
interfaces.

A composed plan records the method, value formats, conversions, numeric policy,
parties, threat model, visible information, state lifetime, workload, component
version, and coverage. Matching shapes and data types are not enough. A change to
the source, implementation, configuration, plan, workload, or cohort creates new
plan history.

The generated [recipe catalog](/research/records/method-catalog/#compositions-and-recipes)
currently marks every workflow as `not_executed`.
