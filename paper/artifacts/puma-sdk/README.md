# Bounded PUMA SDK conformance

`PumaThreePartyCpu` runs the approved source-derived Q18 decoder through the
native compiler, three authenticated workers, ordinary SDK/gateway and response
benchmarks. D9 requires fresh full-prefix execution for each fixed-count greedy
selection. Feedback stays shared; only final token IDs reach the client.

The [contract](../../../docs/research/puma-truncation-reference.md#bounded-sdk-composition)
defines the admitted geometry, D10 RMSNorm epsilon, numeric ranges, ownership
limits and semi-honest threat model. Generated Qwen2/Qwen3/Llama controls exercise
nonzero attention/MLP prefix dependence, private feedback and failure retirement.
Complete hidden-tensor oracles, full-checkpoint admission, source-model quality
and matched serving performance remain open.

The subsequent [quality/wide-contraction slice](local-v2/README.md) adds ordinary
final-token quality benchmarking and complete generated 4,864-wide MLP execution.
It retains its own build and tests. The common Qwen checkpoint still rejects
source admission; its quality and complete-model capacity remain open.

The [compact-compiler slice](local-v3/README.md) replaces expanded source
references with checked strided descriptors, shares immutable tensor programs
and retires construction views at last use. Frozen execution identities and
mixed-representation matrix controls retain the original arithmetic schedule.
Resident checkpoint ownership and full-Qwen capacity remain separate gates.

The [public-weight adaptation](local-v4/README.md) adds explicit public-checkpoint
execution with immutable file-backed Q18 coefficients and batched protected token
boundaries. Private-weight defaults keep their identity. That common Qwen metadata
check reached the compiler-view bound; full-checkpoint execution, range
admission, quality and practical resource measurements remain open.

The [activation-view and tensor-loop slice](local-v5/README.md) prices the live
compiler frontier and executes public lookup, rescaling and SiLU in bounded
loops with fresh per-iteration MPC state. That Qwen view-memory gate cleared;
its private lookup exceeded the declared-work allowance before source
values or worker launch. The separate installed-wheel suite passes 245 Python
and 20 native checks, with seven optional skips.

The [native work-budget slice](local-v6/README.md) derives consumed sequence
ordinals and PRSS cursor/AES bounds from each immutable instruction schedule.
All loop iterations
and full-prefix selections are charged under independently enforced original
ceilings. Qwen's public 4+2 metadata gate admits; 16+2 still exceeds the sequence
ceiling before source values. Complete-model work and ranges remain separate
execution gates. The installed-wheel checks pass 245 unique Python cases, 139
native core/compiler/integration checks and five ownership doctests, with seven
optional skips. Ordinary Qwen source preflight separately rejects the shared
reader's private-weight tensor limits; it supplies no execution result.

The [source-header slice](local-v7/README.md) derives shared reader bounds from
native-admitted source shapes, independently checked against the semantic plan.
Large F32/F16/BF16 fixtures preserve bounded conversion and malformed-header
rejection. Ordinary pinned Qwen 4+2 memory preflight now admits a 2.19 GiB
whole-topology estimate including 25% margin. Complete-model numeric/program
admission, checkpoint execution and measured resources remain separate gates.

The [public range-proof slice](local-v8/README.md) adds streamed matrix row
certificates, independently recomputed worker commitments, tighter RMSNorm
polynomial envelopes and complete-row norm bounds. Native semantic diagnostics
locate the failing public interval. Pinned Qwen publication passes its first
attention product but rejects the first softmax's eager exponential range;
worker execution and full-model qualification remain blocked.

The [guarded-softmax slice](local-v9/README.md) moves the shared tail predicate
before exponential evaluation in public-weight execution. Exact selection keeps
retained-domain rounding and zero tails, with the changed schedule and additional
product charged. Generated wide-score Qwen2/Qwen3/Llama controls execute private
feedback. Pinned Qwen publication clears softmax and rejects the next RMSNorm;
complete-checkpoint execution remains blocked.

The [attention-range slice](local-v10/README.md) certifies complete softmax mass
and residual energy under the original arithmetic and work budgets. Pinned Qwen
publication clears post-attention RMSNorm and reaches layer-zero SiLU's range
rejection. Generated nonzero-attention and concentrated-energy controls retain
private feedback; full-model execution and quality remain unqualified.

The [guarded-activation slice](local-v11/README.md) provides native SiLU/GeLU
programs with safe stand-ins for unused polynomials, preserving interior arithmetic
and zero/identity tails. Public-weight compilation selects guarded SiLU.
Fresh selection work remains charged. Generated Qwen2/Qwen3/
Llama feedback controls pass; pinned publication reaches layer-one RMSNorm and
rejects its propagated residual range before worker launch. Full-checkpoint work,
correctness, quality and resource qualification remain open.

## Retained evidence

[`local-v1/summary.json`](local-v1/summary.json) records:

| Check | Result |
| --- | --- |
| Broad installed-wheel suite, build v3 | 692 passed; one optional Metal skip |
| Final installed-wheel suite, build v4 | 138 passed |
| Native core/compiler checks | 165 passed |
| Compile-fail ownership doctests | 5 passed |
| Installed package files | All 367 match each wheel before and after its tests |
| Final formatting and Clippy | Passed; Clippy retains two recorded lint allowances |
| Shared full-Qwen factory preflight | Rejected during resolution; no serving result |

The only build-input change from v3 to v4 sorts two Rust module declarations.
[`source-delta.json`](local-v1/build-v4/source-delta.json) checks the module set
and remaining source bytes. All Python package files are identical; the rebuilt
native extension has a different hash. The final suite reruns the three-process
SDK, gateway, capped benchmark and native-boundary checks. The two suite counts
overlap and must not be added.

Both builds use the guarded **cached local build** path. Their receipts report
zero sampled build swap growth. This does not measure test/serving memory or
establish clean-room release provenance. Build v4 wheel SHA-256:

```text
b767a589253dfc80f868df7adad0280917456ef2af7385bc5f1814142e7dc917
```

The archive retains original build receipts, commands, observer/source hashes,
JUnit results, native logs, dependency versions and package-file checks.
Earlier fixture and compatibility failures remain in `earlier-attempts/`.
Their XML records are historical diagnostics, not accepted conformance results.

`inputs.tar.gz` retains 763 source/test inputs, including the concurrent native
modules present in this local build. `test-fixtures.tar.gz` adds ten referenced
fixture/oracle inputs. Both have per-file hash manifests. To reconstruct the v3
source view, replace only `crates/pllm-python/src/lib.rs` with
`build-v4/v3-lib.rs`; its hash is checked against v3's original source manifest.
These snapshots preserve the tested inputs independently of later build-cache
cleanup. Wheels and toolchain/dependency caches remain regenerable.

## Replay

Verify the retained evidence without model execution:

```sh
cd paper/artifacts/puma-sdk/local-v1
shasum -a 256 -c SHA256SUMS
```

From the repository root, with the wheel installed in the selected environment:

```sh
python -I -m pytest -q -o pythonpath=. -o junit_family=xunit1 \
  tests/test_replicated_*.py \
  tests/test_delegated_transport_binding.py \
  tests/test_delegated_transport_ownership.py tests/test_delegated_decoder.py
cargo test --locked --offline -p pllm-compiler --lib replicated_decoder
```

The repository root supplies test helpers and inventory scripts; it has no
second `pllm` package. The package-origin and file receipts identify installed
imports. Full commands for both suites and native checks remain in their JSON
records. The ordinary common-model CLI preflight and its original structured
rejection are retained under `build-v4/common-model-preflight.*`.

Response benchmarks charge client publication and all directed peer bodies.
Buffered final IDs provide no streaming first-token or N−1 decode-throughput
measurement. This package qualifies bounded integration, not the complete PUMA
paper system or any client-memory tier.
