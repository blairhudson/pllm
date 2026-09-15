# Compositions and recipes

Learn how PLLM records research recipes and changes to a model plan.

[View canonical HTML](https://pllm.run/research/compositions/)

Document ID: `pllm.docs.research.compositions`  
Release: `0.1.0`  
Build: `sha256:65f4ad316621284cc28b60b1a825a6d9c6d1f108cbb5032aee0983de1e5c4966`  
Source hash: `sha256:fdeba24345277f49d2ffc511853562098ad13790a132d2bd3fe63a65d22da36d`

Recipes describe source acquisition, target model operations, required evidence,
review gates, and failure policy. Inspecting a recipe returns a read-only record;
it does not run third-party code or a PLLM workflow. Use
[`pllm research recipes list`](/cli/reference/research/recipes/list/) to discover
records and [`pllm research recipes show`](/cli/reference/research/recipes/show/)
to inspect one.

A [composed plan](/sdk/plans/) records the method, value formats, conversions, numeric policy,
parties, threat model, visible information, state lifetime, workload, component
version from the [component inventory](/sdk/reference/components/), and coverage.
Matching shapes and data types are not enough. A change to
the source, implementation, configuration, plan, workload, or cohort creates new
plan history.

The generated [recipe catalog](/research/records/method-catalog/#compositions-and-recipes)
currently marks every workflow as `not_executed`.
