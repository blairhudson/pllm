# Admission audit of the v2 decoder reference

This independent review found two blocking admission defects after the v2
functional suite passed. `local-v1/` and `local-v2/` retain their original bytes
and test outcomes. Those outcomes do not establish repaired admission.

1. The compiler constructed semantic graphs before bounding source expansion.
   A small configuration could declare `num_hidden_layers = 2^32 - 1` and
   reach unbounded per-layer graph construction. This huge input was **not
   executed**; source inspection established the ordering defect.
2. The Python facade called `dict(tensors)` before native cardinality checks.
   A bounded lazy-mapping probe supplied 15 entries against 14 declared source
   tensors. All 15 entries were read before rejection. The resulting
   library-owned copy was not covered by excluding caller-owned buffers.

The review also exercised four tiny Qwen2/Qwen3 full/seeded executions, including
repeated tokens, a seven-token vocabulary and differing Qwen3 head width.
Arithmetic, complete final KV and pending-token checks passed. An injected final
verification failure returned no result and retired KV/coefficient/native owners.
AST comparison found the 32 shared local-operator method bodies unchanged.

These checks cover the bounded in-process reference. They do not qualify official
checkpoint import, full-width admission, independent roles, transport, batched
prefill, complete cryptographic security or serving costs.
