# Public range proofs and program-admission diagnostics

This slice tightens public-only bounds without changing the Q18 operator
equations, stochastic rounding, primitive payload ceilings or complete-program
sequence/AES limits. Full-Qwen execution remains blocked.

## Reusable implementation

- Native admission errors identify the selection, semantic operator/layer and
  public input intervals. No private input or activation sample determines a bound.
- Publisher and every importing worker independently derive exact maximum row
  L1 and squared L2 statistics from the original bounded source frames. Their
  immutable descriptor commitment prevents file-owner reuse with substituted
  ranges or certificates. The original source hashes remain independently checked.
- A partitioned proof bounds the existing normalized rsqrt polynomial. It checks
  all 65,537 normalized integers and sixteen power/final-rescale rounding choices
  with an independent integer oracle: 1,048,592 combinations.
- RMSNorm additionally certifies complete output-row squared norms, including
  every rounding at the input square, mean, normalization and public-gamma scales.
  The compiler uses Cauchy–Schwarz only for intact certified rows and original
  physical weight rows. Duplicate, partial and differently sized views reject
  certificate reuse. Complete ring64 sums still receive one final Q18 truncation.
- Private execution preserves six frozen generated identities. Public execution
  binds its changed proof metadata; numeric arithmetic and primitive work/payload
  comparisons retain separate integer-oracle checks.

The full derivation is in the
[contract](../../../../docs/research/puma-truncation-reference.md#public-range-certificates-and-semantic-admission-diagnostics).

## Pinned Qwen publication

All attempts use the same public four-input/two-output Experiment and checkpoint
as `../local-v7/qwen-memory-preflight.json`. They call the ordinary SDK source
publisher after fresh whole-topology memory admission and with its memory guard.
Four fixed public zero IDs supply geometry; no private prompt is used. No worker
process or online decoder is launched.

| Source publication | First rejecting semantic operator |
| --- | --- |
| Initial source publication | Unattributed `Bounds` error |
| Public diagnostic context | Layer 0, step 14, `AttentionScores` |
| Partitioned scalar RMSNorm and matrix row L1 | Layer 0, step 14, `AttentionScores` |
| Complete row norms and Cauchy–Schwarz | Layer 0, step 17, `Softmax` |

The row-norm publication reaches softmax with public encoded input interval
`[-1471675662507, 1486519245655]`. Its original schedule evaluates all five
exponential powers before masking shifted scores at or below `-14`. Those
eager powers do not fit the admitted ring64 range. This is a static-domain
rejection, not an observed private activation overflow or a quality result.

Guarded shared tail evaluation needs its own arithmetic-equivalence, schedule,
resource, cancellation and SDK checks. The current code retains rejection.
Later decoder operators and complete-program work remain unadmitted.

Original reports live in `publication-attempts/`; the final-build retry is
`validation/qwen-program-admission.json`. Process peaks and CPU cover source
admission and their launchers, not serving. These diagnostics do not establish
complete-client memory eligibility, latency, model quality or hidden-state parity.
The initial/context-only attempts do not contain their own installed-file audits;
the two proof attempts and final retry do. Every publication attempt records zero
observed swap growth. The final source publisher takes 12.59 seconds with a
271,171,584-byte process-lifetime peak, including the validation launcher.

## Validation and replay

`summary.json` records final conformance. `validation/` retains the installed-wheel
commands, JUnit output, native logs, package-file comparisons and dependencies.
`validation-inputs.json` and `inputs.tar.gz` preserve their source/test inputs.
`build/` retains the guarded cached-local build. Its monitored resource scope
does not include tests or serving. Counts overlap previous slices.

| Check | Result |
| --- | --- |
| Installed-wheel Python cases | 318 passed; seven optional skips |
| Native PUMA library | 78 passed |
| Native replicated compiler | 24 passed |
| Allocation, multiply, PRSS and truncation integration | 43 passed |
| Ownership doctests | Five passed |
| Clippy and public coverage inventory | Passed |
| Installed package files | All 367 match before and after validation |
| Source/test inputs | All 791 match before and after validation |

The 150 native checks include the five doctests. Clippy retains the existing
`needless_range_loop` and `nonminimal_bool` allowances. Optional Python skips
require Metal or explicitly configured real-checkpoint test paths; the source
publisher diagnostic has its separate actual-checkpoint receipt.

`earlier-attempts/` retains the test-fixture ordering panic and Clippy diagnostic.
The fixture had placed a computed tensor among the declared input nodes; moving
its source declaration before computation repairs the test. The runtime's
source-input contract remains unchanged. Clippy replaces a zero-length comparison
with `is_empty()`. Neither failed command supplies a passing-test count.

Verify from this directory:

```sh
shasum -a 256 -c SHA256SUMS
```

Rebuild/install the preserved inputs, then run the commands in
`validation/validation-commands.json`, adjusting interpreter and output paths.
Python tests use the installed wheel with `-I`; the repository supplies test
helpers. The archived validation launcher includes the original bounded SDK
publisher invocation. It requires the pinned checkpoint and fresh memory/disk
admission; a historical receipt never authorizes a new launch.
