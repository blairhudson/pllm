# Mixed arithmetic for complete gated blocks

The optional `backend="mixed"` on `CurlWaveletGatedMultiplyQ9`,
`LogRowGatedMultiplyQ9` and `CompactGatedMultiplyQ12` replaces expensive private
products with arithmetic sharing and explicit garbled conversions. Existing
numeric profiles, public ranges, exact rounding and default serialized component
identities are preserved. A backend change has its own execution configuration.

## Construction and ownership

`crates/pllm-garble/src/shared_circuit.rs` implements a bounded one-use
arithmetic-to-Boolean-to-arithmetic bridge. A trusted local dealer builds a
half-gates circuit that reconstructs additive inputs only under opaque labels,
evaluates the selected operation, subtracts fresh uniform full-ring output
masks and returns additive output shares.

The garbler receives input encodings, both sender pads for each fresh random OT,
and its output masks. The evaluator receives one independent random choice and
pad per input bit, evaluator material and output-label hash commitments. Its
masked choices and the garbler's two padded labels implement chosen-input OT
from those correlations. Only the garbler owns the free-XOR offset. Output
decoding never transfers both labels of a wire, which would reveal that offset.
The evaluator decodes only uniformly masked outputs; neither party receives an
intermediate plaintext value.

`shared_wide_product.rs` supplies independent one-use ring64 Beaver products.
`mixed_gated.rs` composes those primitives with the existing profiles:

- **Curl:** exact private index/up reduction, the shared Haar lookup, protected
  signed lifting from ring32 to ring64, one product and exact output rescaling.
- **LogRow:** private index reduction, the 4,096-row lookup and up rescaling in
  one bridge, one arithmetic product and exact output rescaling.
- **Compact:** private piece/coordinate/coefficient selection, fresh arithmetic
  Horner products with exact rounding after each, explicit zero/identity tails,
  up multiplication and final rounding. Every lane runs the same public schedule,
  including tails. Selection never reveals a piece-dependent coefficient.

Complete party-local material binds profile, plan, session, tensor index, public
schedule and fresh issuance. Party states are non-cloneable and retire on
completion, malformed frames, substitution, replay, cancellation or drop. A
combined 64 MiB allocation upper bound precedes the first issuance. SDK lanes
run serially and retire before the next; reports charge this serial schedule.

This is an independently implemented PLLM adaptation of the cited methods and
half-gates, not their paper systems. Random OT and triples come from an in-process
trusted dealer; distributed setup, malicious-input enforcement, authenticated
role transport, independent cryptographic review and compiler tensor scheduling
remain absent. Float conversion and public input-range checking remain client
operations. These bounded operators do not activate decoder Experiments.

## Accounting and evidence

The ordinary immutable operator report separates both material owners, public
tables, evaluator ciphertexts/constants, public instruction/profile metadata,
client sharing/boundary estimates and every direction of every peer phase.
Operation subtotals conserve complete material. Public profile construction,
fresh issuance, local online execution, clear evaluation and process CPU have
separate measurements. Keys have no transport codec in this slice; logical
payloads and local frames are not full-wire or peak-memory measurements.

`benchmarks/research/mixed_gated.py` freezes six profiles before measurement and
compares each original backend with its mixed counterpart in fresh processes.
The input fixture, independent integer oracle and numeric digest must agree.
The changed role ownership and serial schedules remain explicit. Full-width
projections cover only hypothetical 39+8-token Qwen MLP geometry.

A separate twelve-prompt pinned W8A8 Qwen diagnostic checks the unchanged
profiles on fresh held-out text. It is clear checkpoint execution: protected
synthetic parity does not turn it into a protected model run. No profile is fit
to these prompts. Fresh full-runtime controls must preserve every checked logit,
KV value and eight-token trajectory before candidate results are recorded.

### Matched complete costs

[The cost cohort](../evidence/research-mixed-gated-cost.json) checks **2,916
protected outputs in 36 fresh processes**, with 1/16/64 lanes, three samples
and both backends for each frozen profile. At 64 lanes:

| Profile | Mixed/original combined material | Mixed/original local online time | Mixed whole-MLP material | Mixed peer frames |
| --- | ---: | ---: | ---: | ---: |
| Curl, uncompressed | 1.729× | 2.006× | 368.39 GB | 35.38 GB |
| Curl, four Haar levels | 4.147× | 1.906× | 203.43 GB | 35.38 GB |
| LogRow | 0.751× | 0.848× | 254.79 GB | 33.81 GB |
| Compact, four-piece quadratic | 0.704× | 0.444× | 1.397 TB | 123.33 GB |
| Compact, eight-piece cubic | 0.719× | 0.482× | 2.112 TB | 145.37 GB |
| Compact, sixteen-piece cubic | 0.878× | 0.640× | 3.017 TB | 145.37 GB |

These decimal material totals charge both owners. The hypothetical geometry is
24 × 4,864 × (39 + 7), or 5,369,856 MLP elements; public instruction metadata,
profile data and client boundary estimates remain separate. The 64-lane control
has more per-lane frame overhead than the earlier 256-lane Curl cohort, so its
1.05 GB projection does not replace that historical 971.27 MB result.

Replacing Boolean products helps LogRow and Compact, but the conversions,
private selection and exact rounding still dominate complete costs. Curl's
existing interval-key arithmetic already has cheap products; generic garbled
conversions make it worse. Default backends therefore remain unchanged. The
garbler is a worker in the mixed layout and a client in the older Boolean
layout; retained material is not automatically network traffic or client RSS.

### Fresh numeric diagnostic

[The independent quality cohort](../evidence/research-mixed-gated-quality-qwen25.json)
checks twelve fresh public prompts under the existing frozen profiles:

| Profile | Prefill selections | Same-token decode selections | Exact eight-token trajectories |
| --- | ---: | ---: | ---: |
| Uncompressed Curl / LogRow | 10/12 | 34/36 | 9/12 |
| Four-level Curl | 7/12 | 34/36 | 5/12 |
| Four-piece quadratic Compact | 12/12 | 35/36 | 8/12 |
| Eight-piece cubic Compact | 10/12 | 36/36 | 9/12 |
| Sixteen-piece cubic Compact | 9/12 | 34/36 | 7/12 |

No candidate preserves a full logit vector or KV state. Each profile checks
128,876,544 encoded values against an independent integer oracle; the unchanged
control preserves all 48 logit/KV positions and all twelve trajectories. Observed
swap growth is zero. These are clear-checkpoint numeric checks, separate from
the synthetic protected cohort; both complete cost and fidelity still reject
recommended decoder execution.

## Reproduction

The installed-wheel example checks all three mixed components and exact replay:

```sh
python examples/benchmarks/mixed_gated.py --lanes 16 --samples 3 --output new-mixed-sdk.json
```

Repository measurements refuse to overwrite evidence:

```sh
uv run --no-sync python -m benchmarks.research.mixed_gated --mode cost --output new-mixed-cost.json
uv run --no-sync python -m benchmarks.research.mixed_gated --mode quality --output new-mixed-quality.json
```

The quality path requires four GiB beyond the host reserve and aborts on reserve
loss or new swap growth. Public source/profile digests, independent integer
counts and exact output-state comparisons remain separate from timing claims.
