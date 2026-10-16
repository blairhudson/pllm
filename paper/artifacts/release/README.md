# Fresh-target release provenance

This directory records new local build/replay observations. The historical
cached-target supplement and prior conformance archives retain their original
identities. See [FORMAT.md](FORMAT.md) for inventory, cache and claim boundaries.

## Completed evidence chain

**Build:** [`2026-10-16-fresh-target-v2`](2026-10-16-fresh-target-v2/receipt.json).
**Fresh installation/replay:**
[`2026-10-16-fresh-environment-v1`](2026-10-16-fresh-environment-v1/receipt.json).
**Read-only verification:**
[`2026-10-16-verification-v1`](2026-10-16-verification-v1/result.json).

The completed build and later fresh replay are separate, hash-linked executions.
The build's original receipt remains `blocked` at its first offline install
attempt. The continuation installed that exact wheel successfully in another
new venv; it does not relabel or overwrite the failed install record.

- An observed empty unique Cargo target compiled **68 crates**, with **91 rustc
  invocations**, including `_native`, and no Cargo `Fresh` entries. Locked,
  offline Maturin used `target/frontier-sdk-env/bin/python`, two Cargo jobs and
  the existing registry source/archive cache. All source, Cargo dependency and
  recorded tool hashes matched before/after. The command completed in 55.62 s;
  this duration is build provenance, not a performance result.
- Pre-build headroom: 9.63 GiB available memory and 33.61 GiB free disk. During
  compilation, sampled available memory stayed at least 9.00 GiB and sampled
  simultaneous rustc processes never exceeded two.
- The newly created venv started with zero distributions and no PLLM import.
  It installed 69 exact-version dependencies plus the generated wheel offline.
  All **355 PLLM package files** matched the wheel, every imported PLLM module
  resolved under the new installation, and **120 active dependency requirements**
  passed. Installed dependency payload/RECORD checks passed before and after
  replay. Torch, Transformers, MLX and TenSEAL were absent.
- Generated Qwen2/Qwen3/Llama fixtures passed all nine tile comparisons:
  **180 array pairs / 88,776 values** were byte-identical over prefill and three
  prescribed decode steps. Six wrong-model/post-bind-mutation checks rejected.
  All **12 frozen boundary functions** passed, including 57,600 finite residue
  cases. The replay command completed in 8.59 s.
- Both recorded compiled native artifacts matched the wheel's native bytes.
  The fresh native digest differs from the historical cached-build native;
  historical conformance is not transferred to the new binary.
- Archive and live-input verification passed. Deliberately corrupted logs,
  a resealed nonempty-target claim, and resealed unequal trajectory bytes were
  rejected. The verifier left both original archives byte-identical.

| Identity | SHA-256 |
|---|---|
| Wheel | `3b63ae30631f0b54773d708c0921b75dbf951142f14dfb80dd9eedac145043b6` |
| Native extension / compiled target library | `cb607aa0b3776b21882a6363a3cedf055caf6e41ae71ed913127fb467ba3cba9` |
| Canonical package-file hash map | `af581eddf511344e2cdeff49ce295153662350bf58bc9b84f287025d9a430963` |

Verify retained evidence without executing implementation code:

```sh
target/frontier-sdk-env/bin/python -I -B paper/artifacts/release/verify.py \
  paper/artifacts/release/2026-10-16-fresh-target-v2 \
  --replay paper/artifacts/release/2026-10-16-fresh-environment-v1
```

To also check retained scratch files, installed dependencies, current build
inputs and tools, append `--live --allow-catalog-drift`. This explicit option
reports changes to four shared conformance-root files (`README.md`, `index.json`,
`verify.py`, `audit-family-cards.py`) observed after the build while other work
continued. Every historical run-directory file must still match. The build's
own before/after snapshots matched all 287 then-existing conformance files;
this release worker did not edit any of those files. Strict `--live` without the
option rejects the changed root files. Exact differences are retained in
[`2026-10-16-verification-v1/live.log`](2026-10-16-verification-v1/live.log).

## Retained attempt history

- `2026-10-16-fresh-target-v1`: actual empty-target compilation completed with
  exit 0. The receipt driver then rejected its own incorrect assumption that
  the Rust library crate was named `pllm_python`; its configured name is
  `_native`. The attempt remains marked blocked; no fresh replay ran.
- `2026-10-16-fresh-target-v2`: actual empty-target compilation completed;
  68 crates and 91 rustc invocations are recorded, with no Cargo `Fresh` entries.
  Source, dependency and tool inventories were unchanged. An empty venv was
  created, but offline registry resolution could not find `annotated-doc==0.0.5`
  in its index cache. The install error and blocked receipt remain intact.
- `2026-10-16-python-cache-v1`: all 69 pinned dependencies were present as
  compatible extracted prebuilt wheel payloads in the local uv cache. Their
  3,901 payload files passed original RECORD validation. New wheel ZIP wrappers
  were generated without changing payloads or fetching/building dependencies.
  These wrappers are not claimed to be the original downloaded wheel ZIPs.
- `2026-10-16-fresh-target-v3`: another unique empty target was attempted after
  cache preparation. The outer harness's 120 s timeout interrupted receipt
  completion. Its log contains completed compilation and wheel creation, but
  no recorded subprocess exit/after-snapshots/replay. Partial files are retained;
  this attempt is not the completed evidence chain and is not counted as a pass.

## Reproduce a new run

The build interpreter, Cargo registry sources and compatible dependency payloads
must already exist locally. Output directory names must be unused.
Give the outer command runner at least 600 s; the build driver's internal build
timeout is 1,800 s, but a shorter harness timeout can interrupt its receipt.

```sh
target/frontier-sdk-env/bin/python -I -B scripts/build_paper_release.py \
  --output paper/artifacts/release/NEW-RUN \
  --dependency-cache paper/artifacts/release/2026-10-16-python-cache-v1
```

Each run creates a unique empty Cargo target and a separate new venv under
`.work/`. It performs a live headroom check, sets `CARGO_BUILD_JOBS=2`, disables
compiler wrappers/incremental compilation, and invokes locked offline Maturin.
Dependencies are installed offline with `--no-index --no-build` from the
RECORD-verified rewrapped wheelhouse. The cache receipt records that wheelhouse's
local location and exact hashes; it must still exist to reuse this command.

Recreate the dependency wheelhouse, if needed, with another new output name:

```sh
target/frontier-sdk-env/bin/python -I -B paper/artifacts/release/prepare_cached_wheels.py \
  --pins paper/artifacts/release/2026-10-16-fresh-target-v2/python-pins.json \
  --output paper/artifacts/release/NEW-DEPENDENCY-CACHE
```

For another fresh replay of the already recorded build, without recompilation:

```sh
target/frontier-sdk-env/bin/python -I -B paper/artifacts/release/replay_built_wheel.py \
  --build paper/artifacts/release/2026-10-16-fresh-target-v2 \
  --dependency-cache paper/artifacts/release/2026-10-16-python-cache-v1 \
  --output paper/artifacts/release/NEW-FRESH-ENVIRONMENT
```

## Scope

Replay checks generated Qwen2/Qwen3/Llama schedule parity, binding rejection,
and twelve frozen bounded algebra/native-boundary functions. It has no upstream
model runtime or independent numerical model oracle. It does not rerun the
historical serving, SDK/gateway, real-checkpoint, or performance cohorts.
Some boundary functions intentionally confirm counterfactual leaks or custody
limitations; their passing results are not cryptographic security proofs.

No prior conformance/archive index, shared implementation, manuscript or status
file is edited by this release workflow. All archive manifests are local
integrity inventories, not independent or signed build attestations.
