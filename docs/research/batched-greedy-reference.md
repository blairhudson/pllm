# Exact batched greedy verification gate

The bounded local reference uses the compiled `prefix_f32` continuation schedule
to verify candidate tokens in blocks. Prefill emits its already-known first
selection immediately. Later target blocks include pending accepted tokens;
the first rejected proposal terminates acceptance, and the unused KV suffix is
erased. Rejected target work remains charged. Failure retires the runtime rather
than rewinding the external stage callback.

The target head runs once for every evaluated row, including rejected rows. The
ordinary single-final-prefill-row optimization cannot apply to verification.
The reference keeps the same body projections, exact numeric contract and final
pending-token convention as serial greedy generation. It is an in-process
diagnostic, not an SDK provider protocol or a new speculation component.

## Pinned Qwen2.5 screen

One public 39-token prompt is decoded for eight and 32 outputs using pinned
Qwen2.5-0.5B W8A8. A real bounded n-gram drafter searches only client-owned history
and falls back to repeating its last token. A separately labeled perfect-draft
oracle reads the serial target's future selections and is only an optimistic
batching control. Neither is a model-quality cohort.

All eight candidate cases match the serial output selections, **24,309,760 logit
words** and **2,850,816 retained KV words**, compared as uint32 bits. Retired
speculative suffixes are zero. Generated Qwen2/Qwen3/Llama tests additionally
force rejection at different positions and exercise widths 2, 4, 8 and 16.

| Outputs | Drafter / proposed width | Accepted / proposed | Target decode rows | Body stage calls | Projected online arithmetic, MB |
| ---: | --- | ---: | ---: | ---: | ---: |
| 8 | Serial | — | 7 | 768 | 73.34 |
| 8 | N-gram / 2 | 0/11 | 18 | 768 | 90.88 |
| 8 | N-gram / 4 | 0/18 | 25 | 768 | 102.04 |
| 8 | Perfect oracle / 2 | 4/4 | 7 | 384 | 73.34 |
| 8 | Perfect oracle / 4 | 5/5 | 7 | 288 | 73.34 |
| 32 | Serial | — | 31 | 3,072 | 111.61 |
| 32 | N-gram / 2 | 0/59 | 90 | 3,072 | 205.67 |
| 32 | N-gram / 4 | 0/114 | 145 | 3,072 | 293.36 |
| 32 | Perfect oracle / 2 | 20/20 | 31 | 1,152 | 111.61 |
| 32 | Perfect oracle / 4 | 24/24 | 31 | 768 | 111.61 |

These whole-response projections count both input and output arithmetic words
at the existing stage ring width. They exclude envelopes, tickets, provider
reservations, unused burned inventory, compression, row residues, physical wire,
checkpoint distribution and the client boundary. The report separately counts
correction arithmetic and extra vocabulary-head work. It does not claim measured
network savings: even the perfect oracle reduces calls without reducing these
arithmetic bodies.

Local same-process CPU includes all target rows and the drafter, but excludes
model loading, provider execution and network pacing. Serial eight/32-output CPU
is 3.97/7.86 seconds. Width-four n-gram CPU is 5.23/16.38 seconds; the perfect
oracle uses 3.36/5.65 seconds. These are single fixed-order samples, not a matched
fresh-process serving or TPS comparison.

## Gate and replay

The cheap real drafter accepts nothing in this prompt. It supplies no reason to
activate remote speculation. A useful independently evaluated drafter, bounded
draft memory, provider attempt/rollback contracts, burned preparation accounting
and complete capped-response measurements remain required. The perfect oracle
cannot stand in for any of them.

```sh
python scripts/probe_batched_greedy.py --output batched-greedy-qwen.json
```

Evidence: `docs/evidence/research-batched-greedy-qwen25.json`, with source,
continuation, implementation and environment commitments. Tests:
`tests/test_batched_greedy_verification.py` and `tests/test_greedy_verification.py`.
