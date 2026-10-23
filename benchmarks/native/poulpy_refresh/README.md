# Poulpy refresh qualification: blocked before allocation

**No encrypted qualification ran. The pinned nightly is installed; native
compilation/tests remain unrun and gated by conservative free/speculative host
memory. The standalone consumer dependency lock exists.** Python guard/oracle and
frozen-identity tests pass;
this is an admission-only implementation, not a completed two-refresh evaluator.

User approved a distinct torus qualification profile, Poulpy v0.9.0 generic
primitives, two sequential encrypted refreshes, one CPU thread, an 8-GiB full
owner envelope and ten minutes after separately bounded setup/build. Parent owns
serial qualification execution. This approval does not map torus widths to paper
RNS profiles, qualify production security, or authorize model inference.

## Why admission fails

The exact `n16_d35_k600_p19_c2s` preset and `NTT4x30Portable<Standard>` backend use:

* N=65536; radix 52; rank 1; raised width 1427.
* High-modulus keys: 7 gadget rows, 4-limb digits, 224 auxiliary bits, 33 limbs.
* 32 distinct rotation keys plus one conjugation key. Both affine rotations
  (1 and 8) are already in the DFT inventory.
* Raw coefficient words: 8 bytes. **Prepared words: 32 bytes**, not 8 or 16:
  `Q120bScalar = CrtWord<Primes30, u64>` has four u64 lanes.

The raw payload of one high-modulus key is
`65536 * 7 * 1 * 2 * 33 * 8 = 242,221,056` bytes. Prepared storage is four times
that. For the **automorphism keys alone**:

| Live payload | Bytes | GiB |
| --- | ---: | ---: |
| Raw | 7,993,294,848 | 7.4443359375 |
| Prepared | 31,973,179,392 | 29.77734375 |
| Both during borrowed `.prepare()` | 39,966,474,240 | 37.2216796875 |

These are lower bounds, not full owner estimates. Tensor and both switching
keys, secret owners, encoders, contexts, module tables, all simultaneous
raw/prepared copies, scratch, ciphertexts, decoding, allocator overhead and the
guard still need space. Counting only two ciphertexts or only compressed/raw
keys would miss the dominant live allocation. A lower bound already above the
ceiling proves rejection without creating keys or compiling a context.

Native `src/main.rs` uses upstream public preset/layout arithmetic to corroborate
the independent Python derivation when the parent eventually admits compilation
and runs it. It never creates a `Module`, key, ciphertext or scratch arena. Its
only accepted command is `admission-report`, always ending with exit code 78.
No environment variable or command-line override enables HE execution.

## Frozen contract and unfinished requirements

`contract.json` records the exact backend, preset, key inventory, intended owner
boundary, full-public binary-rational fixture, two affine maps, stage/width/scale
expectations, error thresholds, negative cases and build/runtime limits.
Unknown complete memory bounds and parameter-analysis receipts are **null and
blocking**, never zero. Its intended operation trace is not an execution receipt.

For slot `j`, the public fixture is
`((17j mod 129)-64)/1024 + i*((29j mod 129)-64)/2048` for `j<127`, otherwise zero.
The two intended maps are `2*conj(rot_1(x))+1/64-i/128`, then
`2*conj(rot_8(y))-1/32+i/64`. The independent final oracle is
`4*x[(j+9) mod 32768]+i/32`. Every one of 65,536 real/imaginary components is
checked; transformed padding is included and generally nonzero. No backend
plaintext evaluator participates in the reference.

Since admission already fails, no client-secret/evaluator implementation,
entry-width reduction or encrypted rejection tests were added as pretend working
code. They remain required before any successful two-refresh result. The native
test is layout arithmetic only. Source preset checks are not a lattice-estimator
receipt or a whole-circuit convolution/failure bound.

## Build-only handoff

The standalone consumer `Cargo.lock` is present and its frozen package identities
are checked without Cargo or native imports. No new resolution is needed for
that check. Native work still requires standard Cargo, the pinned nightly and
sufficient host headroom.

The standalone `[workspace]`, local `.cargo/config.toml`, dated
`rust-toolchain.toml`, explicit manifest path and explicit `rustup run` avoid root
Cargo/toolchain changes. Compiler is pinned to nightly-2026-05-14; compilation
uses one job under D33. Any later build must use the existing consumer lock with
`--locked`.
Frozen identities do not establish selected features, native build success or
complete solver allocation bounds. The initial blocked resolution described below
is historical, not the current lock state.

Operational budgets: resolution 300 s / 1 GiB, toolchain setup 300 s / 1 GiB,
both compilations together 900 s / 4 GiB; each retains 4 GiB physical host
reserve, 8 GiB disk reserve plus a 4-GiB build/download allocation. These are
abort ceilings/allocations, not compiler performance predictions. Guard checks
aggregate process-group RSS plus 64 MiB helper reserve, global swap-used **and**
swapout counters, deadline, bounded logs and observation gaps. For single-job
compilation only, [D33](../../../paper/methods/DECISIONS.md#d33-he-compile-only-host-memory-admission)
requires at least 8 GiB available from the existing
`pllm.runtime.benchmark_memory.host_memory()` accounting and normal memory
pressure. A pressure percentage is not a byte measurement. This supersedes the
free-plus-speculative-only compilation check, not the allocator/execution guards,
runtime caps or unproved HE admission. It permits test-binary compilation with
`--no-run`, not test execution, key generation, encryption or RNG changes.
Separately, [D35](../../../paper/methods/DECISIONS.md#d35-execute-tiny-he-storage-tests)
permits only the 31 compiled `staged_storage` ownership tests, with one test
thread and fresh D33 admission. They use tiny buffers, not HE keys, ciphertexts,
the solver or refresh. Allocator limits, resource guards and all encrypted
execution restrictions remain unchanged.
Existing swap is not additional capacity; additional swap invalidates an
attempt. Process groups are terminated and remaining descendants killed on
failure, including a leader that has already exited. Use ordinary Cargo commands
and tests; keep temporary logs and resource observations in ignored `target/`.
Git history and the existing dependency lock supply source/version history. No
receipt collection, source copy, archive or replay verifier is a completion gate.
Sampling is not a proof of missed peaks and cannot replace a complete
preallocation bound for HE.

Historical 2026-10-17 attempts stopped before any child or key allocation:
93,601,792 bytes (89.27 MiB) of conservative available physical memory was below
the 1-GiB setup allowance plus 4-GiB reserve, and the dependency lock and pinned
nightly were then missing. Later refusal checks and source formatting did not
change the byte bounds. The lock and toolchain are now present; those historical
refusals are not current dependency-state evidence. No cryptographic timing or
performance claim follows from them.

Focused tests (stdlib, no upstream/native crypto execution):

```sh
python3 -m unittest discover -s benchmarks/native/poulpy_refresh/tests -v
```

Ten tests cover source-derived key storage, missing-key/shape inventory drift,
fail-closed admission, independent exact affine composition, every padding slot,
NaN/shape rejection, conservative memory accounting, swapout despite decreasing
swap use, deadline enforcement after the final observation, and timeout cleanup
of a real lightweight parent/grandchild pair.
These are harness/contract tests, not backend illegal-key/entry/layout tests.

The lower bound already establishes refusal; no additional receipt/replay step is
required. Do not launch an upstream preset benchmark as a workaround: its convenience
setup itself bootstraps once and its eager keys exceed the envelope.

## What must change before encrypted execution

The eager resident-key profile cannot fit the approved cap. A different reviewed
key-residency design would need explicit accounting for key generation, storage,
import, bounded preparation, contexts and scratch, plus parameter qualification.
Poulpy's generic key-access traits are a possible basis for that investigation;
this is not evidence that such an implementation fits or meets ten minutes.
No larger memory cap, smaller/altered parameter preset, foreign backend or
source-method mapping is approved implicitly. Parent records decisions/indexing.


## Frozen dependency identities

The standalone consumer lock now exists. The ordinary Python dependency test
compares all seven frozen solver package versions/checksums and four immutable
Poulpy Git identities, without rewriting historical source manifests. Standard
consumer-only `cargo update -p cfg-if --precise 1.0.4` corrected the initial
`1.0.5` drift; standard `cargo fetch --locked` acquired the missing packages.
The ordinary identity test now passes. Root Cargo.lock was not changed by this
lane. Exact checksums remain in the normal consumer Cargo.lock and match the
unchanged frozen allocator contract.

All seven solver packages use the exact source ID
`registry+https://github.com/rust-lang/crates.io-index`. All four Poulpy packages
use `git+https://github.com/poulpy-fhe/poulpy?rev=<commit>#<commit>`, with both
commit fields equal to `ca3532570ff62b96599a74f7adb85de7726b0446`. The test rejects
missing/duplicate pinned packages and changed/missing versions, registry checksums
or source IDs; unrelated packages are outside this identity-only gate. Historical
`consumer_lock_exists`/`download_or_resolution_performed` fields in the unchanged
allocator contract describe its pre-resolution snapshot, not current lock status.

Earlier locked offline metadata for `aarch64-apple-darwin` succeeded. Its observed
solver identities and enabled features follow; the Python identity gate does not
rerun metadata or validate feature selection:

| Package | Version | Features |
|---|---|---|
| cfg-if | 1.0.4 | none |
| dashu-base | 0.6.1 | std |
| dashu-float | 0.6.2 | default, num-order, std |
| dashu-int | 0.6.2 | default, num-order, std |
| num-modular | 0.6.5 | none |
| num-order | 1.2.0 | none |
| static_assertions | 1.1.0 | none |

Poulpy's four packages remain 0.9.0 at
`ca3532570ff62b96599a74f7adb85de7726b0446`; portable backend features are exactly
`enable-ckks, enable-core`. Cargo fetched `astro-float-num 0.3.7` as locked.
Available sources and matching identities do not prove arithmetic allocation
capacities, whole-process bounds, or native compilation under the frozen nightly.
They do not promote parameter/security provenance; D28 remains design only.

Native compilation and encrypted qualification remain blocked. The identity
gate does not admit staged storage, secure sampling, or encrypted execution.
