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
execution, complete-block cost, reviewed security claim or executable SDK choice.

## Reproduce

```sh
uv run python -m benchmarks.research.nonlinear_frontier \
  --output docs/evidence/research-nonlinear-frontier-reproduction.json
```

The native example is `crates/pllm-garble/examples/nonlinear_frontier_probe.rs`.
The canonical report is `docs/evidence/research-nonlinear-frontier-qwen25.json`;
it locks the component sources and the separate domain/projection evidence.
