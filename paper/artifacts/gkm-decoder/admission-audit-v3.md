# Admission repair review

This record follows [the original v2 findings](admission-audit-v2.md).
It does not change the `local-v1` or `local-v2` evidence.

## Findings and repairs

1. **Graph construction before resource admission.** The native compiler now
   passes a caller-selected expansion budget through the existing source-mapping
   registry before a graph builder can run. Checked layer, width, vocabulary,
   projection and activation/state limits bound the admitted expansion. Unpriced
   source mappings reject this entry point. The ordinary lowering path retains
   its prior behavior. Sentinel-callback tests prove rejection before graph
   construction, including the original extreme layer count.
2. **Source mapping copied before cardinality checks.** The Python facade now
   checks cardinality before reading values and accesses only declared names.
   It does not enumerate arbitrary source keys. Its configuration encoder bounds
   depth, bytes and container traversal before constructing the encoded snapshot.
   Mapping, serialization and contract-identity regressions cover these paths.

The follow-up native review found two more pricing defects in the shared budget
API: Qwen2 could price an ignored `head_dim`, and Qwen3 could omit its wider query
projection. Neither bypassed the fixed GKM envelope, but both violated
caller-selected ceilings. The repair resolves head width through each registered
source contract and charges the complete query projection. Each original
256-element counterexample now rejects before lowering. A 512-element control
reaches lowering, enumerates the real graph's largest activation/state shape,
and preserves the ordinary plan exactly.

The independent read-only recheck found both native follow-up findings closed
with no remaining admission blocker in that reviewed path. Parent review covers
the facade repair. This is an engineering review of the bounded composition;
it is not independent certification of the cryptographic construction.

## Evidence and scope

`local-v3/` retains the repaired source, guarded cached wheel build, native tests,
installed-package identity checks and fresh-environment Python regressions.
Its verifier ties native-test source snapshots to the build and freezes executed
test-file identities separately. Original attempts and earlier verifier controls
remain retained.

The graph budget prices semantic activation/state elements and separate source
dimension ceilings. It does not establish measured RSS or bound every caller
allocation. Full-width Qwen, official source authentication, original-model
quality, prefill batching, transport and independent worker processes remain
outside this bounded reference.
