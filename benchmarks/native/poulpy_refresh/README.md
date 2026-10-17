# Poulpy refresh qualification: blocked before allocation

**No encrypted qualification ran. Native compilation and standalone dependency
resolution are also blocked.** Python guard/oracle tests pass; this directory is
an admission-only implementation, not a completed two-refresh evaluator.

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

Use from the repository root after sufficient host headroom exists:

```sh
python3 scripts/qualify_rust_ckks_refresh.py resolve
python3 scripts/qualify_rust_ckks_refresh.py toolchain
python3 scripts/qualify_rust_ckks_refresh.py build
```

The standalone `[workspace]`, local `.cargo/config.toml`, dated
`rust-toolchain.toml`, explicit manifest path and explicit `rustup run` avoid root
Cargo/toolchain changes. Resolution uses already-installed Cargo 1.85.1 without
compiling dependencies. Compiler is pinned to nightly-2026-05-14; jobs never
exceed two. Build driver sequentially compiles the binary and `cargo test
--no-run`; it never executes native tests or HE code. Cargo.lock must exist before
the locked build. **No standalone Cargo.lock exists yet because the resolver was
not admitted; the dependency closure is not claimed frozen.**

Operational budgets: resolution 300 s / 1 GiB, toolchain setup 300 s / 1 GiB,
both compilations together 900 s / 4 GiB; each retains 4 GiB physical host
reserve, 8 GiB disk reserve plus a 4-GiB build/download allocation. These are
abort ceilings/allocations, not compiler performance predictions. Guard checks
aggregate process-group RSS plus 64 MiB helper reserve, global swap-used **and**
swapout counters, deadline, bounded logs and observation gaps. Free/speculative
pages alone count toward conservative memory admission; inactive pages and
existing swap are not treated as capacity. Additional swap invalidates an
attempt. Process groups are terminated and remaining descendants killed on
failure, including a leader that has already exited. Each attempt retains an
immutable freeze, command and result; launched commands also retain stdout,
stderr and bounded resource samples. Sampling is not a proof of missed peaks;
it cannot replace a complete preallocation bound for HE.

Initial receipts (preserved, no child launched):

* `receipts/20261017T121458-resolve-2b6f9ccd/`: conservative available physical
  memory 93,601,792 bytes (89.27 MiB), less than 1 GiB + 4 GiB reserve.
* `receipts/20261017T121523-build-041be5c7/`: dependency lock missing after blocked
  resolution. Compiler not launched and dated nightly not installed.
* `receipts/20261017T122803-qualify-57963441/`: parent replay of the refusal, with
  no child or key allocation.
* `receipts/20261017T123127-qualify-f7cd5fac/`: same refusal after native-source
  formatting, with unchanged byte bounds and current input hashes.

These freezes are historical inputs. The current contract clarifies that native
comparison is pending, not compiled; receipt statuses always report prelaunch
blockage. There are no timings from cryptographic work and no performance claim.

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

Parent may record the already-proven refusal using `python3
scripts/qualify_rust_ckks_refresh.py qualify`; that phase launches no child.
Do not launch an upstream preset benchmark as a workaround: its convenience
setup itself bootstraps once and its eager keys exceed the envelope.

## What must change before encrypted execution

The eager resident-key profile cannot fit the approved cap. A different reviewed
key-residency design would need explicit accounting for key generation, storage,
import, bounded preparation, contexts and scratch, plus parameter qualification.
Poulpy's generic key-access traits are a possible basis for that investigation;
this is not evidence that such an implementation fits or meets ten minutes.
No larger memory cap, smaller/altered parameter preset, foreign backend or
source-method mapping is approved implicitly. Parent records decisions/indexing.
