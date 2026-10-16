# Local clean-build receipt method

`scripts/audit_paper_clean_build.py` reuses the source-tree, Cargo dependency,
wheel-payload and SHA-256 conventions of the archived
`paper/artifacts/conformance/build-source-receipt.py`. Each attempt receives a new
directory, an archived driver/helper, exclusive-create logs and a receipt. Failed
attempts remain intact. A successful receipt has status
`built_and_installed_execution_pending`.

## Build and installation boundary

- Resource preflight records available memory, disk, existing swap and inherited
  process limits. Minimums are 4 GiB available memory and 8 GiB free disk, checked
  again immediately before compilation. The driver does not change swap or limits.
- Cargo receives `CARGO_BUILD_JOBS=2`, `CARGO_NET_OFFLINE=true`,
  `CARGO_INCREMENTAL=0`, and a newly created `CARGO_TARGET_DIR`. Its directory
  listing must still be empty immediately before Maturin starts. Compiler wrapper
  environment variables are removed. Maturin uses `--release --locked --offline`
  and the newly provisioned environment's explicit interpreter.
- Source inventories cover `.cargo`, `crates`, `python`, `schemas` and the helper's
  enumerated root build/config files, including untracked inputs. Generated
  binaries, bytecode, target directories and Git metadata are excluded explicitly.
  Cargo configuration outside the repository is identified separately. Every
  resolved non-workspace Cargo package source tree is hashed before and after.
- The new standard-library venv starts without pip or system site packages and
  must contain zero installed distributions. Frozen dependency versions come
  from the original installed environment and must have matching entries in
  `uv.lock`. No editable PLLM installation is used.

## Offline dependency recovery

The original cache contains all 113 required compatible dependency wheel payloads
but lacks usable offline registry-resolution metadata. Attempts with UV 0.13.0
and cached UV 0.11.8 retain their failure logs. No network fallback is used.

The driver therefore constructs **new local dependency wheel archives** from the
existing unpacked cache. It verifies every included file against the cached
wheel's original RECORD hash and size, matches its WHEEL tags to its `uv.lock`
entry, and checks the reconstructed ZIP's complete payload hash inventory.
The new ZIP receives its own SHA-256. `python-dependency-wheel-inputs.json` retains
the source cache directory, original upstream URL/hash metadata, new local path
and hash, and payload digest. The original upstream ZIP is not present and its
archive hash is **not** claimed to match the reconstructed ZIP.

UV installs these local, hash-required inputs with `--offline --no-index
--no-build --no-deps --link-mode copy` using a new installer cache. No dependency
version is changed. The original cache is hashed before and after installation;
the installed dependency files are hashed before and after the PLLM build; and
dependency Python/native code hashes are compared against the original installed
environment. This recovers frozen local inputs, not independent publisher trust.

## Identity and evidence

The new PLLM release wheel is installed with a SHA-256 requirement. Its complete
`pllm/` payload must match the new installed package. The receipt retains wheel,
native, installed-package and environment identities and original v4 provenance.
All v4 comparisons are byte comparisons only. Different wheel/native bytes retain
distinct identities, and v4 execution results are never relabeled as a new run.
The receipt's `fresh_install_identity` is a receipt-content identity, not PLLM's
runtime `build_digest`.

`--verify RECEIPT` reads evidence, rehashes the release wheel and reconstructed
dependency wheels, checks every recorded installed distribution file, and checks
the recorded invariants. It does not import PLLM or execute models/operators.
Full command requests, completion records and logs are retained. Work directories
retain the Cargo target, venv, dependency ZIPs and installer cache locally; the
local `.gitignore` excludes those generated trees from Git.

## Limits

This is one local empty-target release build and a new local environment. Existing
Cargo registry sources, Python dependency payloads, Python/Rust toolchains and
macOS SDK are reused. It is not CI, independent third-party attestation, a
fully hermetic build, a clean operating-system image or a proof of bit-for-bit
reproducibility across hosts. Tool executables are hashed, but the entire SDK and
Rust sysroot are not. Offline tool flags are not an OS network sandbox.
Before/after inventories do not detect transient changes between observations.
Manuscripts and unrelated shared artifacts are outside the bounded build inputs.

No models, operator samples, SDK tests or benchmarks are executed by this driver.
The parent session must serialize any subsequent fresh-environment execution and
record its own new execution identity.
