# Complete attention-row and residual-energy admission

This slice tightens public PUMA bounds through unchanged protected arithmetic.
Pinned Qwen source publication clears post-attention RMSNorm, then rejects
layer-zero SiLU. Full-checkpoint execution remains unqualified.

## Reusable implementation

`Program::softmax_certified` proves a nonnegative complete-row mass from public
denominator partitions and the original stochastic reciprocal recipe. It retains
the guarded softmax instructions, truncations, work and party payload allowance.
The compiler uses that mass only for the complete ordered softmax row, after
ordinary attention-product admission.

Public source descriptors additionally bind the sum of squared absolute weight-row
sums. Each worker recomputes this statistic across the original source frames.
Whole-row product energy and the residual triangle inequality then bound the
RMSNorm input without assigning every coordinate its scalar maximum. Ordered word
provenance preserves these certificates across tensor chunks. Public proof metadata
counts against the existing compiler allowance and changes source/program
commitments; private-weight identities stay separate.

`Program::rms_norm_with_input_norm` uses the certified energy before the original
mean/epsilon/rsqrt schedule. Its range obligation, like a scalar interval, must be
proved by the caller. The decoder compiler derives it only from admitted operators
and bound source statistics. Private inputs never select a public range or proof
partition. Numeric operations, polynomial constants, rounding and work ceilings
remain unchanged.

Independent stochastic-integer checks cover every softmax/RMSNorm output in their
declared fixtures, row certificates, sparse 896-value rows, cancellation and
allocation failures. Complete-row tests reject partial, reordered, duplicated and
substituted provenance. Worker tests reject changed source statistics and reuse
under a different source commitment.

Six generated Qwen2/Qwen3/Llama controls exercise nonzero V/O products and private
feedback. Three use `V=O=8I`; three use `V=8I, O=diag(28,0)` to require the energy
proof in addition to mass. Zero Q/K and MLP preserve a separate signed-head oracle.
Both Qwen2 layouts also traverse three authenticated SDK workers and retire their
sessions. These controls do not establish full-checkpoint hidden-state parity or
generation quality.

## Pinned Qwen gate

The ordinary SDK source publisher uses the locked four-input/two-output Experiment
from `../local-v7/qwen-memory-preflight.json`, four fixed public zero IDs, fresh
whole-topology memory admission and the ordinary pressure watchdog. It reads
checkpoint values but launches no workers or online decoder.

The mass-only diagnostic reduces post-attention's encoded interval from
`[-1571860067,1571850851]` to `[-145907592,145898376]`, but still rejects RMSNorm.
With complete residual-energy admission, the first rejection moves to layer-zero
`Silu`, step 25 in selection zero, at `[-28618714,28618714]`. All retained source
publication attempts observe zero swap growth. Their CPU and process peaks cover
source admission and its launcher, not serving costs or a client-memory tier.

Whole-model numeric admission, complete-program work, independent hidden-state
oracles, source-model quality and matched response costs remain open.

## Evidence and replay

| Check | Result |
| --- | --- |
| Installed-wheel Python cases | 327 passed; seven optional skips |
| Native PUMA library | 83 passed |
| Native replicated compiler | 26 passed |
| Allocation, multiply, PRSS and truncation integration | 43 passed |
| Ownership doctests | Five passed |
| Clippy and public coverage inventory | Passed |
| Installed package files | All 367 match before and after validation |
| Source/test inputs | All 791 match before and after validation |

The 157 native checks include the five doctests. Clippy retains its recorded
`needless_range_loop` and `nonminimal_bool` allowances. Optional skips require
Metal or explicitly configured checkpoint paths; the pinned source-publication
diagnostic has its separate actual-checkpoint receipt.

`summary.json` derives validation counts from original JUnit and native logs.
`validation/` retains the installed-wheel checks, package comparisons, dependency
versions and full source-publication receipt. `build/` retains the cached wheel's
guarded build receipt. `inputs.tar.gz` preserves every tested source/test input
listed in `validation-inputs.json`; the archive is verified against that manifest.
`package-delta.json` compares installed files against the guarded-softmax slice.

`old-guarded/` preserves three generated `8I` sources rejecting RMSNorm on the
previous wheel. `old-mass/` preserves the exact current concentrated fixture
rejecting on the intermediate mass-only wheel. `earlier-concentrated/` retains the
initial `diag(35,0)` negative control separately; the passing fixture uses 28, not
35. These expected failures add no passing cases. The mass-only and first combined
checkpoint diagnostics retain separate receipts and build identities; they are
not pooled as performance samples.

Verify these read-only artifacts from this directory:

```sh
shasum -a 256 -c SHA256SUMS
```

Rebuild/install the preserved inputs, then run the commands in
`validation/validation-commands.json`, adjusting interpreter and output paths.
Python checks use the installed wheel with `-I`; the repository supplies test
helpers. Source publication requires the pinned checkpoint plus fresh host/disk
admission. Historical receipts authorize no new launch. Counts overlap earlier
slices and are not additive. Full equations and contract:
[reference](../../../../docs/research/puma-truncation-reference.md#complete-attention-row-and-residual-energy-proofs).
