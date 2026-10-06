# Public SiLU mapping and complete coefficient-lookup costs

**Decision: no-go for decoder/SDK promotion.** Four fixed public profiles cover
the checked Qwen inputs, but none preserves every held-out selection. Vector
coefficient lookup improves the matching complete-block control while leaving
uncompressed one-use material far above the resource gate.

This is a bounded Rust reference with a separate clear-checkpoint numeric
diagnostic. It does not execute a protected Qwen decoder or reproduce FuseFSS's
GPU system.

## Source and numeric contract

The pinned [FuseFSS paper](https://arxiv.org/abs/2606.09551), Appendix I.2 and
Appendices N/O, describes scalar-to-vector interval lookup followed by remaining
share-based products and conversions. Its PDF SHA-256 is
`6109399761ab0cc1ca6eb617fec1a811cbe59d474455ce3549467e5a5b77cbf0`.
PLLM independently instantiates the public coefficient lookup with the universal
DCF from [FSS for Mixed-Mode Secure Computation](https://eprint.iacr.org/2020/1392),
Figure 1 and Section 4.1/Figure 14. Public arithmetic coefficient deltas need no
Boolean-predicate conversion in this particular lookup. This is an adaptation,
not the paper's general packed-comparison compiler or its published SiLU fit.

The four immutable profiles are fixed before checkpoint evaluation:

- Input interface: signed Q16 in a 32-bit ring; gate in `[-48,48]`, up in
  `[-128,128]`. Float-to-Q16 conversion occurs only in the trusted clear diagnostic.
- Internal fractional bits: eight or nine. Both inputs use exact signed
  ties-to-even rescaling.
- SiLU: 16 or 64 uniform mathematical secants on `[-8,8]`; slope uses Q14 and
  intercept uses the internal fraction plus 14 bits. No checkpoint, calibration
  prompt or held-out value fits these coefficients or selects these bounds.
- Explicit tails: zero below `-8`, identity at/above `+8`. Values outside the
  declared input domain reject; they are never clipped or silently passed to
  the original operator.
- Round the affine result by 14 bits, multiply by the rescaled up input, then
  round by the internal fractional width. Every division is ties-to-even.

The native profile digest binds coefficients, interval boundaries, ranges, tails
and representations. Exhaustive internal gate-domain checks cover signed
overflow and tails at the up-domain extremes. Those public numeric bounds are
caller obligations for secret-shared inputs: a protected floating-point bridge
and secret range enforcement are not implemented.

## Complete shared block

Each party owns only its input, intermediate and output shares. An in-process
trusted dealer issues fresh masks, interval keys and two Beaver triples. The
complete six-round sequence is:

1. Rescale gate and up from Q16 in one batched round.
2. Select both affine coefficients from the hidden rescaled gate.
3. Multiply the selected slope by the gate; add the shared intercept locally.
4. Rescale the affine result.
5. Multiply the activation share by the up share.
6. Rescale the gated product and retain opaque output shares.

The scalar control uses independent masks/keys for slope and intercept, batched
in the same round. The vector candidate uses one mask/key for both coefficients
and shares the interval evaluations. Both retain four rescalings, two products
and six rounds; fusion does not erase those dependencies.

Each bounded issuance preflights its combined allocation estimate under 64 MiB
and at most 256 lanes. Profile, layout, lane count, plan/session/operation,
tensor offset, fresh issuance nonce and phase bind the frames. Party handles
are non-cloneable; completion, cancellation, malformed frames, replay and
out-of-order calls burn remaining material. These checks are not authenticated
transport, malicious-output verification or independent cryptographic review.

## Fresh held-out numeric gate

The fixed twelve public prompts are distinct from the preceding Q7 gate's
prompts. All 24 MLP layers are overridden in the pinned Qwen2.5-0.5B W8A8
compiled decoder. Each prompt checks prefill and three teacher-forced decode
positions against the original W8A8 trajectory. Candidate KV remains its own;
the next input token comes from the original control.

| Public profile | Prefill matches | Same-token decode matches | Worst absolute logit error |
| --- | ---: | ---: | ---: |
| Q8, 16 pieces | 10/12 | 32/36 | 4.7350 |
| Q8, 64 pieces | 9/12 | 35/36 | 3.1176 |
| Q9, 16 pieces | 10/12 | 34/36 | 4.9807 |
| Q9, 64 pieces | 10/12 | 35/36 | 4.0648 |

No profile rejects a checked input. The control's maximum absolute gate/up
values are 36.60 and 101.11. Each candidate checks 63,737,856 native gated
outputs against an independent NumPy integer oracle; all agree with its encoded
profile. Nevertheless, no candidate has a bit-exact logit vector or full KV
snapshot at any of the 48 checked positions. Encoded correctness and expanded
range coverage therefore do not establish model fidelity. These comparisons
are against pinned W8A8, not upstream float32 or representative task quality.

## Matched complete-block costs

Sixteen fresh-process cases cover four numeric profiles, two lookup layouts and
one/64 lanes. Three fresh issuances per case check 1,560 shared outputs against
the independent integer oracle, including signed endpoints, tails and rounding
boundaries. The following rows use the same Q9/64-piece profile and 64 lanes:

| Cost | Separate coefficient lookups | Vector coefficient lookup |
| --- | ---: | ---: |
| Complete material per party | 404,736 B | 358,144 B |
| Coefficient keys within that total | 93,696 B | 47,104 B |
| Four rescalings within that total | 309,504 B | 309,504 B |
| Two Beaver triples within that total | 1,536 B | 1,536 B |
| Both directions' framed peer bodies | 6,056 B | 5,544 B |
| Peer rounds | 6 | 6 |
| Median issuance | 47.49 ms | 42.14 ms |
| Median local online execution | 726.76 ms | 399.70 ms |

Vector lookup cuts complete material by 11.51% and peer bodies by 8.45%.
Rescalings still occupy 86.42% of the vector layout's material. Local online
timing executes both party machines and reconstruction in one process; it is
not provider throughput, transport latency or a whole-response measurement.
At 16 pieces, vector Q9 online time is 161.02 ms with the same material and
peer sizes, but its held-out fidelity is worse. Timing samples describe this
CPU implementation, not the paper's GPU comparison.

## Geometry and go/no-go

The native semantic graph supplies MLP counts for hypothetical 39+8 and 39+32
requests. Every element receives new material; there is no reuse across rows,
layers, phases or responses. The Q9/64-piece, vector/64-lane projection is:

| Workload | Both parties' raw material | Framed peer bodies | Extra input/output share payload if nonresident |
| --- | ---: | ---: | ---: |
| 39+8 | 60.10 GB | 465.16 MB | 128.88 MB |
| 39+32 | 91.46 GB | 707.86 MB | 196.12 MB |

MLP material plus peer bodies already exceeds the historical prepared
whole-response covered-body controls of 116.84/178.97 MB. The separate-lookup
39+8 control requires 67.92 GB and 508.12 MB. These are raw-payload/geometry
gates, not measured full wire or same-numeric runtime rankings. Offline framing,
linear stages, attention, normalization, feedback and checkpoint distribution
remain additional. Compression or a distributed dealer is not assumed.

Even granting free coefficient lookup and free rescaling, this layout's two
uncompressed Beaver triples use 48 bytes per element across both parties and
their raw openings use another 32. At 39+8 that is **429.59 MB** before headers
or any other decoder operator. Smaller comparison keys alone cannot satisfy
the resource gate; changing correlation representation or the complete product
protocol would require a new measured construction.

The older Q7 block's 41.33 GB is retained as a different numeric contract;
the wider-domain block cannot inherit its identity or be ranked as its exact
replacement. All four new profiles fail both the predeclared selection gate
and the complete-cost gate. The checkpoint diagnostic observed zero new swap.

Further work needs a materially cheaper **complete** rescale/product protocol
and a fresh numeric hypothesis, tested on new held-out data. Tuning these fits
on the twelve reported prompts would turn them into calibration data. There is
no tensor/role/SDK promotion from this result.

## Reproduction

Configuration and measurements:

- `benchmarks/research/piecewise_gated_reference.py`
- `docs/evidence/research-piecewise-gated-reference-qwen25.json`
- `crates/pllm-core/src/piecewise_gated_reference.rs`
- `crates/pllm-garble/src/shared_affine_lookup_reference.rs`
- `crates/pllm-garble/src/shared_piecewise_gated_reference.rs`
- `tests/test_piecewise_gated_reference.py`

The recorded command was:

```sh
uv run python -m benchmarks.research.piecewise_gated_reference \
  --output docs/evidence/research-piecewise-gated-reference-qwen25.json
```

For a rerun, choose a new output path: the command rejects existing files.
It builds the native examples and uses the cached pinned checkpoint, with a
4 GiB headroom preflight and host-pressure guard. Evidence binds source files,
profile manifests, checkpoint/body identity and historical control hashes.
