# Universal interval keys for shared rescaling

PLLM independently implements the masked multiple-interval construction in
**FSS for Mixed-Mode Secure Computation**, Section 4.1, Lemma 1 and Figure 14.
One universal compact comparison key per bit width replaces separate threshold
keys within a single one-use helper. It reduces preprocessing material and
issuance time; comparison count and peer traffic remain unchanged.

This is a bounded native component reference. It does not activate a Pipeline
or supply a complete SIGMA/FuseFSS nonlinear operator or Qwen decoder.

## Published identity and ownership

Source: [Boyle et al., ePrint 2020/1392](https://eprint.iacr.org/2020/1392), pinned
PDF SHA-256 `dc94c63a77a5cb52cfd12b186a327d045de4a1fc662060c1de6412b334f4fcf4`.
Figure 1 supplies the existing compact DCF. Section 4.1 and Figure 14 explain
sharing one DCF across masked interval boundaries. No upstream implementation
is built, linked or vendored.

For a `k`-bit input, write `N = 2^k`, secret mask `r`, public boundary `b` and
masked opening `u`. A single DCF holds `gamma = (r-1) mod N`. Lemma 1 gives:

```text
[u < (r+b) mod N]
  = [(r+b) mod N > b] - [u > b]
    + [(u-b-1) mod N < gamma]
```

The evaluator computes public shifts of `u`, one universal DCF evaluation and
the public `u > b` term. The dealer combines `[(r+b) mod N > b]`
into the existing additive constant shares. No secret-dependent correction
appears in public metadata. Combining two boundary tests gives the published
closed-interval predicate, including wraparound and endpoint equality.

This reference lifts the interval algebra into the existing signed-helper
equations over `Z/(2^32)`. It retains the paper's floor helper and PLLM's
separate exact ties-to-even extension, with the **original input's** nonnegative
predicate. The [earlier helper note](shared-rescale-reference.md) specifies
those equations. It does not claim the paper's native bit-packed wire format.

A fused helper needs comparisons at the input width, discarded-bit width and,
for ties-even, discarded-bit width plus one. These remain separate keys. Seven
ties-even evaluations use at most three distinct keys; floor needs three
evaluations and at most two keys. Equal widths share only within the same lane
and opening phase. Each new lane, phase and invocation owns fresh one-use mask
material. The unfused second phase retains its independent mask and key.

Keys have no public clone, export or standalone evaluation API. Backend identity
is bound into the helper context. Start/advance take ownership before parsing;
malformed frames, wrong backend/context, replay and cancellation burn material.
The 64 MiB combined allocation preflight and 1,024-lane cap remain enforced.
Masks, constant shares, seeds and corrections have zeroizing owners. In-process
framing is not authenticated transport or malicious-output verification.

Independent integer checks cover all small masks, boundaries and inputs,
Figure 14 closed intervals and 32-bit endpoints. All three backends pass the
same exhaustive 2–6-bit signed-input/shift tests, negative ties, 32-bit boundary
tests and one-use failures. This is correctness evidence, not independent
cryptographic or side-channel review.

## Matched fresh-process costs

Reproduce with a new output path:

```sh
uv run python -m benchmarks.research.interval_rescale_reference \
  --output docs/evidence/research-interval-rescale-reference-reproduction.json
cargo test -p pllm-garble
```

The configuration runs **24 fresh native processes**, pairing independent
compact DCF keys against universal interval keys and alternating execution
order. Three width/shift pairs, two rounding modes and both layouts receive
five fresh 64-lane issuances. All **7,680 output pairs** match native and Python
integer oracles; observed new swap is zero. Pinned source resolution and native
build are outside these measurements. The two earlier cohorts remain historical.

Below: ties-even, bytes per party/lane, median elapsed milliseconds per complete
64-lane batch. Online time includes both local evaluations, framing,
reconstruction and retirement; it excludes issuance and input sharing.

| Width / shift | Layout | Independent → shared key bytes | Issuance ms | Online ms | Peer bodies |
| --- | --- | ---: | ---: | ---: | ---: |
| 16 / 7 | Unfused | 1,426 → 620 | 10.82 → 5.01 | 8.89 → 8.89 | 1,336 B |
| 16 / 7 | Fused | 1,718 → 758 | 12.83 → 5.74 | 11.30 → 11.03 | 668 B |
| 24 / 12 | Unfused | 2,108 → 906 | 15.88 → 7.03 | 13.83 → 13.73 | 1,336 B |
| 24 / 12 | Fused | 2,620 → 1,154 | 19.09 → 8.55 | 17.44 → 17.40 | 668 B |
| 32 / 12 | Unfused | 2,460 → 1,082 | 18.18 → 8.29 | 16.20 → 16.21 | 1,336 B |
| 32 / 12 | Fused | 2,972 → 1,330 | 21.62 → 9.81 | 19.93 → 19.88 | 668 B |

For fused 16/7 ties-even, keys fall **55.88%** and issuance time **55.25%**.
The small online difference is not evidence of a general online speedup: there
are still seven evaluations. Full process CPU for all five samples falls
**0.12456 to 0.08746 s**, and process peak RSS **2.06 to 1.95 MB**. These tiny
process measurements include the local dealer and both evaluators.

Payload accounting includes every 16-byte root, correction and byte control
(`20+22k` per compact comparison), plus the mask and three constant shares
per phase. Future offline transport framing and allocator overhead are extra.
The estimate separately prices allocation overhead. Raw timings, floor cases,
CPU/RSS and implementation/configuration hashes are retained in the report.

Fusion still costs more material and local CPU than unfused execution: shared
16/7 keys are 758 versus 620 bytes, and online time is 11.03 versus 8.89 ms.
Its benefit is one peer opening instead of two, 668 versus 1,336 body bytes.

## Whole-width gate

Compiler-derived Qwen2.5 geometry assumes one helper per semantic SiLU element.
This is a hypothetical count, not a Qwen W8A8 fixed-point mapping: no checkpoint
values, logits, KV or generation quality are evaluated.

| Workload | Shared-key layout | Both parties' one-use payload | Peer bodies at 64 lanes |
| --- | --- | ---: | ---: |
| 39+8 | Unfused | 6.66 GB | 112.10 MB |
| 39+8 | Fused | 8.14 GB | 56.05 MB |
| 39+32 | Unfused | 10.13 GB | 170.58 MB |
| 39+32 | Fused | 12.39 GB | 85.29 MB |

The 39+8 fused projection drops from **18.45 to 8.14 GB** of keys. Linear
extrapolation still gives about **482 seconds issuance** and **926 seconds
local online work**, before polynomial evaluation, other rescaling, attention,
feedback, distribution or complete response execution. These are projected
helper costs, not measured decoder compute, WAN or a whole-response speedup.

Tensor admission remains closed. Further isolated key shrinking cannot by
itself establish a viable decoder: complete-operator numeric mapping and cost,
independent cryptographic review, distributed issuance and authenticated
independent roles remain gates. Batch traversal and paper-native output widths
are candidates to investigate, not measured improvements.

Code: `crates/pllm-garble/src/interval_fss.rs` and `shared_rescale_reference.rs`.
Replay: `benchmarks/research/interval_rescale_reference.py`.
Evidence: `docs/evidence/research-interval-rescale-reference-qwen25.json`.
