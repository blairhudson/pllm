# Nonlinear component frontier

This screen independently runs Compact, LogRow and Curl references against one
public four-piece Compact Q7 profile. Seventeen fixed encoded inputs cover
`[-128,128]`, representing `[-1,1]`. Each evaluation receives fresh one-use
material. Different ownership and output contracts prevent a protocol ranking.

## Observed component costs

| Method | Material body bytes / element | Peer bytes / element | Encoded matches | Worst encoded error |
| --- | ---: | ---: | ---: | ---: |
| Compact polynomial half-gates | 843,392 | Unmeasured | 17/17 | 0 |
| LogRow masked-index lookup | 2,144 | Unmeasured | 17/17 | 0 |
| Curl, no compression | 4,100 | 136 | 17/17 | 0 |
| Curl, two Haar levels | 1,028 | 136 | 6/17 | 2 |
| Curl, four Haar levels | 260 | 136 | 2/17 | 7 |

Compact counts its evaluator ciphertext body; LogRow counts its evaluator
material body; Curl counts both parties' dealer-share payload. These are
different material scopes. The report retains issuance/local evaluation timings
and fresh-process CPU/RSS for each case. Curl's checked online times are only
microseconds, so they are especially sensitive to timer resolution and noise.
No latency ratio is treated as a complete-protocol speedup.

The Curl table pads unused public entries with the upper endpoint, then applies
Haar low-pass averaging with ties-to-even rounding. It uses this screen's Q7
profile, not the earlier Q16 SiLU table. Compression changes the function.
The truncated input index is formed outside the protected protocol. Compact and
LogRow return labels decoded by the trusted client; Curl retains additive output
shares until the test oracle reconstructs them.

## Complete-block gate

Compiler-derived Qwen2.5 **39+8-token** geometry has 5,369,856 gated elements.
Simply repeating these activation components projects the following lower bounds:

| Method | Activation material bodies | Activation peer bodies |
| --- | ---: | ---: |
| Compact | 4.53 TB | Unmeasured |
| LogRow | 11.51 GB | Unmeasured |
| Curl, no compression | 22.02 GB | 730.30 MB |
| Curl, two levels | 5.52 GB | 730.30 MB |
| Curl, four levels | 1.40 GB | 730.30 MB |

Protected input conversion/range enforcement, output representation conversion,
secret multiplication by the up input, final rescaling, authenticated transport,
offline metadata, other decoder operators and token feedback are additional.
These per-element layouts cannot claim batching savings that have not been
implemented. The fixed Q7 domain already fails the separate
[checkpoint gate](nonlinear-block-reference.md). There is no protected checkpoint
execution, complete-block cost or reviewed security claim in this historical screen.

## Installed operator components

The subsequent [complete Haar gated block](wavelet-gated-reference.md) has a
distinct wider Q18-to-Q9 contract, protected index/up/output rescaling and fresh
multiplication. Its full costs and new checkpoint cohort are separate from this
historical activation-only screen.

`pllm.nonlinear.CompactPiecewiseActivation`,
`pllm.nonlinear.CurlWaveletEncodedTable` and
`pllm.protocols.LogRowGarbledLookup` now use the ordinary immutable component
contracts and `pllm.metrics.benchmark_operator`. These are bounded operator
executors; whole-decoder binding remains separate. Public offline counts and
piece limits are replayable parameters; private input values are omitted from
reports and must be supplied again for custom-input replay.

The installed report improves the earlier **ciphertext-only** material scope:
Compact charges active constant labels alongside half-gate ciphertexts and
charges the garbler's input/output encodings. LogRow charges client zero labels
and its shared delta; all masked table bits remain in evaluator payload. Curl
retains both worker shares and all peer frames. Public metadata/profile and
client boundary estimates stay separate, with explicit role names. Serial lanes
burn material before the next issuance. Allocation ceilings are estimates,
not measured process peaks; physical wire and checkpoint quality remain unknown.

Public fitting/table construction, fresh issuance, online evaluation and clear
table-oracle timings are separate. Independent Python integer arithmetic checks
every output. Compressed Curl has its own numeric digest; its agreement with
the uncompressed profile and float64 SiLU errors appear separately.

```sh
python examples/benchmarks/activations.py --lanes 17 --samples 3 \
  --output activation-operators.json
```

The installed-wheel cohort is
`docs/evidence/research-activation-operators-sdk.json`. The earlier table above
remains immutable historical evidence, with its original narrower scope.

All **255/255** installed-wheel outputs match their declared integer contracts
across five configurations, seventeen lanes and three fresh samples. Per lane,
Compact retains 800 client bytes and 858,816 evaluator bytes, including 15,424
constant-label bytes. LogRow retains 304 client and 2,144 evaluator bytes.
Those are material payloads: Compact's additional fixed-width public instruction
metadata estimate is 1,292,190 bytes per lane, with 464 client-boundary bytes.
The two-/four-level Curl profiles match the uncompressed profile at 18/51 and
9/51 checked outputs respectively; their exact compressed-contract checks pass.

## Reproduce

```sh
uv run python -m benchmarks.research.nonlinear_frontier \
  --output docs/evidence/research-nonlinear-frontier-reproduction.json
```

The native example is `crates/pllm-garble/examples/nonlinear_frontier_probe.rs`.
The canonical report is `docs/evidence/research-nonlinear-frontier-qwen25.json`;
it locks the component sources and the separate domain/projection evidence.
