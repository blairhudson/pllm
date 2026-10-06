# SIGMA / FuseFSS: exact shared-rescaling reference

PLLM independently implements one bounded fixed-point helper pair in Rust:
**signed rescaling plus the original input's nonnegative predicate**. The
unfused control follows SIGMA's truncate/reduce then sign-extension structure;
the fused control follows FuseFSS's mask-aware elimination of intermediate
openings. The initial cohort uses the same arithmetic prefix-DPF backend. A
[follow-up compact-DCF cohort](compact-rescale-reference.md) remeasures both
backends under these same contracts. This is a **component
reference**, not an implementation of either paper's complete system, optimized
comparison protocol, GPU backend or reported performance.

Sources, read as specifications only:

- [SIGMA](https://petsymposium.org/popets/2024/popets-2024-0107.php), Sections
  2.4 and 4.2.2–4.2.3. Pinned PDF SHA-256:
  `f911b715b0e0523fa4d77f21c28c6a0440c5be781e67b5b30419f20e505a6ebf`.
- [FuseFSS](https://arxiv.org/abs/2606.09551), Sections 3.5, 4.3, 4.5 and
  Appendix E. Pinned PDF SHA-256:
  `6109399761ab0cc1ca6eb617fec1a811cbe59d474455ce3549467e5a5b77cbf0`.

No upstream implementation is built, linked or vendored.

## Numeric and party contract

The descriptor fixes a two's-complement ring width `n` from 2 to 32, a public
shift `f` from 1 to `n−1`, and floor or ties-to-even rounding. Inputs and outputs
are additive shares. Reconstruction gives `rescale(x, f)` and `[x >= 0]`, with
no saturating arithmetic or sign ambiguity. The predicate refers to **x**, even
when a small negative input rounds to zero.

Paper arithmetic right shift uses floor. Exact ties-to-even is a separate PLLM
extension, not something attributed to the papers. Tests cover every input and
shift for 2–6-bit domains, negative/positive ties, full-width extremes, 32-bit
wraparound and random values, plus an independent Python integer oracle.

The in-process research dealer samples independent OS-random masks and point
keys for each lane and phase. Each non-cloneable evaluator owns only its own
input-mask, constant and function-key shares. A peer sees a masked opening and
its own keys. The dealer must remain trusted and separate from both online
views; this program itself runs all roles in one process and establishes no
operator independence or reviewed security claim.

The unfused path opens the masked original input, truncates to `n−f` bits, then
opens a freshly masked intermediate for sign extension and its sign predicate.
The fused path evaluates the final result from one opening of the original
input. Ties-to-even correction is evaluated on that first opening in both paths.

### Mask-aware equations

Let `u=(x+r) mod 2^n`, `H=2^(n−1)`, `theta=(r+H) mod 2^n`, and
`c=[r+H >= 2^n]`. The fused floor and predicate are:

```text
floor(x_signed / 2^f)
  = (u >> f) - (r >> f)
    - [u mod 2^f < r mod 2^f]
    + 2^(n-f) * ([u < theta] + c - 1)

[x_signed >= 0] = [u < theta] - [u < r] + c
```

The dealer shares the constants and privately programs the thresholds. Neither
threshold nor constant is published. The shared `[u < theta]` result serves both
outputs. For ties-to-even, let `L=2^f` and `t=x mod 2^(f+1)`. Increment the floor
exactly when `t` is in `[L/2+1,L)` or `[3L/2,2L)`. Four additional arithmetic
comparisons on the low `f+1` bits implement those ranges without opening a bit
or performing a secret multiplication. Negative ties use the same rule.

### Backend deviation and costs

An unsigned threshold is a disjoint union of binary prefixes. For a `k`-bit
comparison, the backend always issues `k` independent point-function keys of
depths 1 through `k`, including zero-payload keys. Threshold bits do not change
public key counts, depth, frame length or query order.

The two-party binary-tree DPF uses 128-bit seeds, AES-keyed counter expansion,
separate control bits and arithmetic output shares modulo `2^32`. A party's
depth-`j` key counts a 16-byte root, `j` 18-byte correction words and a four-byte
final correction. One comparator therefore costs **20k + 9k(k+1) bytes per
party**. No dense truth table or exponentially sized lookup is stored.

This deliberately straightforward construction is **quadratic in comparison
width**. SIGMA's optimized DPF comparison and FuseFSS's compatible production
backends are not reproduced. In particular, the unfused control uses arithmetic
shares instead of the paper's optimized bit-share/B2A implementation. Its
shorter-width comparisons can therefore beat fused full-width comparisons in
key size and local work. The result measures this backend's fusion tradeoff,
not a contradiction of the papers' performance results.

## Lifecycle and admission

At most 1,024 lanes enter a batch, with a combined dealer/two-party allocation
estimate capped at 64 MiB **before any key issuance**. The estimate includes
retained key payload, owner/allocator allowance and scratch; it is not a peak
measurement or an OOM proof. Each phase/lane also counts a four-byte mask share
and three four-byte constant shares. No offline distribution codec exists, so
offline headers and transport are additional, unmeasured costs.

Each online frame contains a 78-byte context/issuance/phase/role/count header and
four bytes per lane. Context binds the caller's plan, session, operator and
first tensor index to the numeric descriptor, layout and selected backend; this is not compiler
admission. Wire values, lengths and phase identities are checked before
evaluation. A handle burns **all remaining phases before parsing** a peer frame.
Malformed input, foreign issuance, wrong role/phase, replay, duplicate start,
cancellation and drop cannot recover usable key state. Retained masks, constants,
seeds and corrections are zeroized on release.

Frames are local research messages, not authenticated transport or malicious
output verification. A changed but well-formed peer value is not proven honest.
There is no distributed dealer, durable replay state, live share-resident
attention/KV, private token feedback or executable Pipeline selection.

## Measurements

Replay the actual invocation with a new output path:

```sh
uv run python -m benchmarks.research.shared_rescale_reference \
  --output docs/evidence/research-shared-rescale-reference-reproduction.json
cargo test -p pllm-garble
```

The saved Python configuration builds the native example, resolves the pinned
Qwen2.5 source and reads its configuration, then launches **12 fresh native processes**:
16-bit/seven-bit-shift and 24/32-bit/twelve-bit-shift cases, each with
floor/ties-to-even and unfused/fused
layouts. Each case has five samples of 64 synthetic public inputs. Every sample
uses new secret material; all **3,840 helper outputs** match both native and
independent Python integer oracles. Observed new swap is zero.

The table shows medians for a complete 64-lane batch. Online elapsed time includes
both evaluators, framed openings, trusted reconstruction and key retirement in
one process. It excludes issuance and input-share construction. Whole native
process CPU and peak RSS are separately recorded in the JSON for every case.

| Width / shift / rounding | Layout | Keys per party per lane | Peer bodies per batch | Issuance ms | Local online ms |
| --- | --- | ---: | ---: | ---: | ---: |
| 16 / 7 / floor | Unfused | 2,656 B | 1,336 B | 16.07 | 13.48 |
| 16 / 7 / floor | Fused | 6,196 B | 668 B | 36.46 | 32.71 |
| 16 / 7 / ties-even | Unfused | 5,888 B | 1,336 B | 35.31 | 29.98 |
| 16 / 7 / ties-even | Fused | 9,428 B | 668 B | 55.94 | 49.32 |
| 24 / 12 / ties-even | Unfused | 12,556 B | 1,336 B | 74.26 | 66.04 |
| 24 / 12 / ties-even | Fused | 21,012 B | 668 B | 122.95 | 112.56 |
| 32 / 12 / ties-even | Unfused | 17,628 B | 1,336 B | 103.69 | 94.01 |
| 32 / 12 / ties-even | Fused | 29,540 B | 668 B | 172.31 | 160.23 |

Clear rescaling of the same 64 values is below one microsecond in each recorded
median. Process peak RSS spans 2.54–6.96 MB and includes the dealer and both
parties; it is not client RAM or independent-role memory. Timings are native CPU
microbenchmarks, not 100/40 WAN or full-response comparisons. Source resolution
and the native build are outside these timings. All twelve cases,
raw samples and source digests are retained, including floor cases omitted from
the summary table.

### Qwen geometry gate

The semantic plan contains 24 SiLU tensors of width 4,864 per token. Placing
**one hypothetical helper per SiLU element** gives 5,369,856 evaluations for
39+8 tokens, or 8,171,520 for 39+32. This counts prefill plus `N−1` decode steps;
it is not a lowering of Qwen's existing floating-point W8A8 nonlinear path.

For the 16-bit, seven-bit-shift **ties-to-even** pair:

| Workload | Layout | Both parties' one-use key payload | Peer bodies, 64-lane batches |
| --- | --- | ---: | ---: |
| 39+8 | Unfused | 63.24 GB | 112.10 MB |
| 39+8 | Fused | 101.25 GB | 56.05 MB |
| 39+32 | Unfused | 96.23 GB | 170.58 MB |
| 39+32 | Fused | 154.08 GB | 85.29 MB |

These partial-layout projections already omit nonlinear polynomial evaluation,
attention, rescaling elsewhere, input/output conversion, token feedback and
offline transport. They cannot support tensor/decoder resource admission or
an SDK benchmark claim. The native peer-body saving is real for this helper;
it does not establish aggregate network or compute savings.

## Next gate

The independently implemented Figure 1 compact DCF now preserves this oracle
and failure contract; its separate matched cohort is in
[compact-rescale-reference.md](compact-rescale-reference.md). It reduces
material and local CPU but still projects 18.45 GB of fused keys at 39+8 tokens.
Independent review and complete-operator resource reductions remain necessary
before tensor/role admission. Qwen fixed-point numeric and generation-quality
coverage must precede a matched native 100/40 decoder benchmark.

Code: `crates/pllm-garble/src/shared_rescale_reference.rs` and `point_fss.rs`.
Replay: `benchmarks/research/shared_rescale_reference.py`.
Evidence: `docs/evidence/research-shared-rescale-reference-qwen25.json`.
