# Exact-I/O traffic frontier

This screen investigates substantial all-link traffic reductions while preserving
an unchanged existing model's inputs, complete reconstructed logits/state,
positions, outputs and sampler behavior. The complete client target is 256 MiB;
predominantly remote body execution and each construction's privacy contract are
additional requirements. The [research plan](../plans/exact-io-traffic-frontier.md)
defines the shared gates and three Sol 6.1 investigations.

**No new serving protocol, traffic improvement or Low-tier qualification is
established.** These are bounded arithmetic/state references, public-source
analyses and conditional cost ledgers. Exact integer reconstruction is necessary
but does not by itself preserve every original floating-point decoder operation.

## What the construction screens establish

| Direction | Exact property checked | Decisive limitation | Next useful gate |
| --- | --- | --- | --- |
| Sparse-noise image compiler | `r=A*u+e`, `W*r=(W*A)*u+W*e` over the original ring | A public parity test rejects the proposed low-noise sensitivity; public delivery and client expansion remain substantial | A genuinely different, specified generator must survive complete-key security analysis and the complete budget |
| Transparent exact key-homomorphic expansion | Algebraically correlated keys reconstruct `W*x` | Public cross-index relations disclose private input relations; full-entropy tapes restore the original delivery cost | An explicit nontransparent correlated-key construction with exact, priced error repair |
| Exact weight circuits | `W=U*V+E` with an exact residual and unrestricted fresh masks | A small checkpoint circuit is unproved; discarding residuals changes outputs | Verify one source-bound coefficient certificate and its complete size/work before integration |
| Ordered nonlinear quotient | Certified identical subgraphs can share a value while preserving every original downstream rounding node | Frozen Qwen evidence supports no syntactic duplicate-row saving | Prove a nontrivial equivalence of complete original rounded subgraphs over an accepted public domain |
| Complete-state one-hot lift | A finite state machine, including its sampler, becomes an exact linear shared transition | Raw unary float32 encoding alone costs 34.36 GB of initial shares | A proven much smaller exact reachable representation, including private input injection |
| Exact rounding-cell fallback | Public tail bounds can certify one original integer-to-float32 result | The examined Qwen full-domain tail interval is at least 254 units versus rounding cells at most eight | A certificate for the complete original scale/nonlinearity chain, with protected checks and exact fallback fully charged |
| Additional exact reuse | Branch state and sampling agree; synthetic retention matches the actual Python cache | Corrected screened incremental network savings are zero; the control already publishes the tested intermediate cuts | Price only genuinely missing boundaries; delta restoration and operation receipts target local work |

Track references contain the construction details, role views, counterexamples
and replay commands:

- [Fixed-weight correlations](exact-correlation-frontier-reference.md)
- [Exact model adapters](exact-adapter-frontier-reference.md)
- [Exact state/work reuse](exact-reuse-frontier-reference.md)

## The stronger sparse-noise veto

The identity `W*(A*u+e)=(W*A)*u+W*e` is exact. Its input privacy does not follow
from that identity. For any public `n`-by-`d` matrix with `n>d`, binary elimination
on at most `d+1` rows supplies a nonzero public parity vector `h` satisfying
`h^T*A=0 mod 2`. Let its support have size `w`.

For `t` uniformly selected distinct noise coordinates with independent uniform
ring values, including zero, the opening parity is unbiased whenever the selected
coordinates intersect that support, and deterministic otherwise. The support law
and coefficients are input-independent and conditional on the fixed public `A`.
Two chosen inputs with different `h` parity therefore have parity-view total
variation

```text
binom(n-w,t) / binom(n,t) >= binom(n-d-1,t) / binom(n,t).
```

The chosen-pair probability gap is this total variation `p`; equal-prior guessing
succeeds with probability `(1+p)/2`, an advantage `p/2` above random guessing.
For one fixed input, parity distance from uniform is `p/2`. The signed-vector
primitive admits the chosen pair `0` and one coordinate equal to `1`. Reachability
through a particular checkpoint quantizer/model would require a separate proof;
this result is not a whole-decoder attack.

This is an efficiently computed single-opening test. At the originally proposed
cost coordinates `d=128,t=27`, the lower bound is 0.01406 for width 896 and
0.48301 for width 4,864. Fresh roots, labels and the independent output mask do
not remove it. These coordinates are rejected, not merely awaiting a performance
test. Other samplers or constructions need their own analysis; this is not an
impossibility theorem for computational correlation generators.

Even suppressing this one total-variation lower bound to `2^-128` requires at
least 413 or 2,387 selected noise coordinates respectively at `d=128`. These are
necessary conditions for this particular metric/test, not secure parameter
recommendations. The stipulated evaluator charges each selected coordinate's
`W*e` work; an alternative may exactly skip realized zeros but must price its own
complete evaluation. Uniformity of `u` is unnecessary for this parity argument.

## Comparator corrections

Fixed-256-row preparation overissues material for these workloads. The primary
correlation comparison must instead use the installed demand shape: one prefill
row for each terminal-pruned stage, 150 for other stages, and `N-1` decode rows.
The 150+8 and 150+64 cells remain separate. Under the explicitly modeled records,
the request-sized known-payload controls are 352.77 and 483.27 MB; the sparse-image
candidate is 697.42 and 778.80 MB before the declared unknown costs. Its additional
public delivery is about 477.37 MB. Neither cold cell wins even before security
admission. Physical wire, source-cold checkpoint delivery and complete-client RSS
remain unknown.

The first reuse report also omitted the installed `put_prefixes` policy. That
policy already selects intermediate completed-prefill boundaries, including the
8/10 and 48/50 boundaries used in the initial sibling fixtures. Their advertised
incremental 1,024-byte and 27.59–76.27% savings are withdrawn. Corrected v2 grants
both arms the same cuts, shared-block backing storage, LRU/entry bounds, admission
and scheduling opportunities: the three-sibling fixture executes 23 causal rows
in each arm, with **zero incremental network saving**. The 2/4/8/16-sibling scaling
cells likewise have zero additional savings. Differential checks exercise the
actual Python cache with synthetic tensors, not checkpoint decoder execution.

Identified on-disk v1 remains byte-identical and rejected for its comparator.
An earlier audit reported a stale replay with misrouted corrections, but that
earlier snapshot has not been recovered; it must not be identified with the
preserved v1 bytes, which already route corrections Preparation-to-Inference.
V2 records the available artifact hash and subsequent source changes explicitly.

An unscheduled 45-row boundary in a 47-row prompt leaves five gross rows beyond
the installed retained cut at 40. With an admitted eight-row grid the gap is at
most seven rows; eviction can invalidate that bound. Extra publication, retention,
eviction damage and the best explicit-prefix/waiting control remain unpriced,
so this is not a net checkpoint traffic saving or a substantial new reuse lead.

Authenticated delta-child restoration and application-operation receipts remain
potential local CPU/I/O improvements. They cannot beat an existing zero-provider-row
hit on provider traffic. A new stochastic invocation must still draw its own
sample; an application retry receipt is valid only for the same completed logical
operation.

## Research decision

The most concrete remaining mathematical gate is a **verified small fixed-weight
circuit with unrestricted fresh masks**. Independently check one source-bound
`W=U*V+E` certificate, including every residual coefficient, then charge the full
certificate, client arithmetic and streamed reads. Dense residuals recover the
existing expensive client computation; they do not pass the gate. A different
compact generator needs a new explicit construction and complete-key proof.

The remaining adapter gate is **source-proved equivalence of complete rounded
subgraphs**, not approximate rank, token agreement or omitted rounding. Neither
track currently has a useful full-checkpoint certificate.

Reuse already captures much of the easy exact opportunity. Further proposals
must improve the strongest applicable installed cache and lifecycle control.
Any surviving opportunity must then pass complete generated-decoder parity,
failure/cancellation ownership, conservative memory admission and matched
fresh-process all-link measurements before becoming an SDK choice.

## Bounded validation

The integrated focused suite passes **119 tests and 36 subtests**. Ruff passes for
all three probes and test files. Run the same suite with the repository root
available for the adapter's `scripts` import, while using the installed SDK for
the cache differential checks:

```sh
target/frontier-sdk-env/bin/python -I -m pytest -o pythonpath=. -q tests/test_exact_adapter_frontier.py tests/test_exact_correlation_frontier.py tests/test_exact_reuse_frontier.py
target/frontier-sdk-env/bin/python -I -m ruff check scripts/probe_exact_adapter_frontier.py scripts/probe_exact_correlation_frontier.py scripts/probe_exact_reuse_frontier.py tests/test_exact_adapter_frontier.py tests/test_exact_correlation_frontier.py tests/test_exact_reuse_frontier.py
```

Each linked track retains its frozen record, source hashes and replay procedure.
Independent read-only review reproduces the reuse/correlation v2 records and the
archived correlation v1 CLI byte-for-byte, and recomputes the request-sized
directed ledgers from frozen geometry. Parent review also reproduces the adapter
record byte-for-byte. Reuse v1's missing source snapshot remains explicit.
These checks establish bounded mechanisms and vetoes; they supply no new model
quality, serving performance or whole-client peak-memory measurements.
