# Five next communication experiments

Completed results: [next-five-network-methods.md](next-five-network-methods.md)
and [the evidence report](../evidence/next-five-network-2026-10-03.md).
The generated-prefix experiment uses a new message-history cohort with its own
matched 402.05 MB control. It is not ranked against the historical repeat/extend
cohort below; the request contexts and token counts differ.

These are five new, independently falsifiable capabilities, starting from the
combined runtime controls in `docs/evidence/combined-compatibility-qwen25-2026-10-03.md`.
The three-request, 24-output-token lean prepared control costs 310,964,801
covered setup-inclusive bytes and 117,752,640 online bytes. Its tenfold targets
are 31,096,480 and 11,775,264 bytes. The seeded/packed offset control and its
reuse-enabled variant remain separate topology controls. Each new result must
state whether it applies to fresh prompts, conversations, or model delivery.

| Method | Hypothesis | Required gate |
| --- | --- | --- |
| Generated-state canonicalization | Executed generated tokens under prefix-canonical arithmetic can become useful fresh-prefix state | Exact logits and every KV bit versus full prefill; pending tokens excluded; source, numeric, ownership and verifier lineage remain bound |
| State equivalence across placements | Lossless placement/transport changes can retain client state | Compare full semantic/numeric/weight contracts; source or verification mismatch rejects; cross-placement continuation equals fresh execution |
| Masked worker aggregation | A separately masked peer result can be combined at one worker before client download | Fresh independent output mask, one-use context-bound frames, exact packed reconstruction; count peer traffic and client CPU as well as client bytes |
| Token-local projection memoization | Compiler-derived token-local work repeats even across unrelated prompts | Generic dependency analysis; exact integer/float outputs; bounded private storage; no unpriced preparation waste or hidden access-pattern leakage |
| Progressive head precision | Public high weight bits plus conservative residual bounds can eliminate vocabulary rows before private refinement | Native integer bounds including float32 rounding; exact greedy tie handling; fixed private retrieval budget, client storage and worker scanning charged |

Python owns immutable probe configuration, orchestration, reports and independent
oracles. Rust owns arithmetic, packed codecs, mask expansion, one-use state and
head bounds. Research reports never contain prompts, token IDs, activations,
cache keys, masks or credentials. No result changes the numeric contract or a
live cache admission rule merely because a few prompts agree.

Run bounded tiny checks first, then a cached pinned checkpoint cohort. Do not run
100-request sweeps. Preserve failures and cost vetoes; a client-link improvement
with unchanged all-link traffic does not count toward the all-link tenfold goal.
