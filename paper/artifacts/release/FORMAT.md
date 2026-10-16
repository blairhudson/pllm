# Fresh-target local release evidence

`scripts/build_paper_release.py` creates a new, self-contained evidence archive.
Run it with `target/frontier-sdk-env/bin/python -I -B`. The requested output must
be a new direct child of `paper/artifacts/release/`. Compilation and installation
scratch directories live under the ignored `.work/` directory.

## Evidence chain

1. `target-before.json` records a newly created, nonsymlink, empty Cargo target.
   Cargo metadata must report that same target. `environment.json` records two
   Cargo jobs, disabled compiler wrappers and incremental compilation, and
   offline operation. `headroom-prebuild.json` records the live admission check.
2. `source.tar.gz` contains the exact inventoried repository build inputs.
   `sources-{before,after}.json` includes source/configuration hashes and explicit
   generated-file exclusions. External Cargo configuration files have archived
   copies. `cargo-dependencies-{before,after}.json` hashes resolved dependency
   source trees and available cached `.crate` archives.
3. `tools-{before,after}.json` hashes the specified tool binaries, Maturin files,
   Python executable/libpython, and Rust library tree. SDK identity, settings,
   and the libSystem link stub are recorded. This does not freeze the full OS,
   all SDK headers, or every compiler resource.
4. Every subprocess has exact arguments, working directory, UTC start, exit
   code, duration and log hash in `*.command.json`. `wheel-build.log` retains
   verbose Cargo compilation, including native-extension rustc invocations.
   `compilation-evidence.json` inventories resulting target artifacts.
   `wheel-build.monitor.json` records live available memory and child RSS.
5. `wheels/` retains the generated wheel. `wheel-files.json` hashes every payload
   file. `receipt.json` supplies wheel, native-extension and package digests.
   `historical-comparison.json` compares payloads without changing the identity
   or meaning of any historical execution.
6. A newly created venv has no installed distributions and cannot import PLLM
   before installation (`venv-empty.json`). Exact dependency versions come from
   the active base dependency closure of the existing installation plus pytest.
   `uv` installs prebuilt cached distributions and the new wheel offline with
   copying, without system site-packages or editable installation. Dependency
   constraints are checked by `uv pip check` and independently in replay.
   When registry cache metadata cannot resolve those versions, the optional
   `--dependency-cache` input supplies payload-preserving wheel ZIPs rewrapped
   from existing extracted wheel caches. Their original RECORDs are validated;
   original extracted files, new ZIPs, and installed payloads are cross-checked.
   These new ZIPs are explicitly not the original downloaded wheel archives.
7. Replay runs the fresh interpreter with `-I` from scratch space. It verifies
   the complete installed PLLM payload against the wheel and all installed
   distributions against their RECORD hashes. Before/after inventories check
   that replay did not alter installed distribution files. All observed PLLM
   module paths must resolve under that fresh installation.
8. `manifest.json` inventories every archived file except itself. The read-only
   verifier checks hashes and cross-record consistency. Its optional `--live`
   checks also read the retained scratch environment/target, original tool and
   dependency paths, repository inputs and pre-existing conformance files.

## Replay scope

`replay.py` generates new NumPy PCG64 fixtures (seed 23) for Qwen2, Qwen3 and
Llama schema paths, using two tiny layers and 258 vocabulary entries. For each,
it compares untiled execution against tile sizes 1, 7 and 16 over a 33-token
prefill and three prescribed one-token decodes. Comparisons cover final-position
logits and key/value arrays after every phase. Wrong-model binding and post-bind
normalization-weight mutation must reject.

The frozen copy of `tests/test_paper_boundary_security.py` supplies twelve
bounded algebra, coalition-view, mask-custody and native-boundary checks. Replay
calls its `CHECKS` functions, not its historical standalone evidence writer.
Some checks intentionally demonstrate limited or unsafe counterfactual behavior;
passing them does not establish cryptographic security or erase those limits.

These are new generated fixture observations. NumPy fixtures differ from the
historical Torch-generated fixtures. No Torch, Transformers, MLX or TenSEAL
distribution is installed or imported by replay. Schedule parity compares PLLM
with PLLM, not with an independent model oracle. No upstream implementation
runtime, real checkpoint, serving cohort, or performance benchmark is run.

## Honest cache and trust boundaries

The Cargo **compiled target is empty**; registry source/archive caches are
reused. The Python **environment is new**; prebuilt dependency caches and the
base interpreter are reused. The machine and OS are not fresh. Timing fields
describe the evidence-producing commands and are not performance measurements.
The archive is local agent-recorded evidence, not a signature, independent
attestation, bit-reproducibility proof across hosts, or third-party audit.

Existing conformance files are only read and hashed. Their original execution
identities and claims remain those in their own archives.
