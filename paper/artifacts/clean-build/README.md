# Clean local release build and fresh environment

**Complete: build, installation and hash validation. Fresh runtime execution is pending.**

Successful attempt: [`2026-10-16-local-v5/receipt.json`](2026-10-16-local-v5/receipt.json).
Validation: [`2026-10-16-local-v5.validation.json`](2026-10-16-local-v5.validation.json).
The four earlier setup attempts are preserved, each with a receipt, failure log
and sibling validation JSON. All four stopped before release compilation.

## Exact local paths

- Venv: `/Users/blairhudson/workspace/pllm/paper/artifacts/clean-build/2026-10-16-local-v5/work/venv`
- Interpreter: `/Users/blairhudson/workspace/pllm/paper/artifacts/clean-build/2026-10-16-local-v5/work/venv/bin/python`
- Wheel: `/Users/blairhudson/workspace/pllm/paper/artifacts/clean-build/2026-10-16-local-v5/wheels/pllm_run-0.1.0a2-cp311-abi3-macosx_11_0_arm64.whl`
- Cargo target: `/Users/blairhudson/workspace/pllm/paper/artifacts/clean-build/2026-10-16-local-v5/work/cargo-target`

## Verified result

The offline, locked Maturin release build completed in 100.02 seconds with
`CARGO_BUILD_JOBS=2` and `CARGO_INCREMENTAL=0`. Its new target was empty immediately
before the build; Cargo metadata used a separate target. The build log contains
68 `Compiling` lines and zero `Fresh` lines. Immediately before compilation,
available memory was 9.95 GiB and free disk was 28.37 GiB. Swap configuration and
resource limits were not changed.

The 632 bounded repository input files and 4,262 files across 81 resolved Cargo
dependency source trees were unchanged. Their inventory digests also match the
earlier cached-build receipt. These are bounded working-tree inputs at commit
`99e10e2c1e0e3dbeddc5f56dcc830e549cb1f2a5`, not a claim that the whole checkout was
clean or hermetic.

The new isolated venv initially contained zero distributions. It now contains
the new wheel and 113 frozen dependencies, whose versions match both the original
environment and `uv.lock`. All 12,836 dependency Python/native code files match
the original environment; the dependency files remained unchanged through the
build. UV's metadata-only dependency check reports all 114 installed packages
compatible. The original environment and original dependency-cache payloads
remained unchanged.

Because offline registry metadata was unavailable, dependency ZIPs were locally
reconstructed from unchanged, RECORD-verified cached payloads. They retain their
own new archive hashes and the original upstream URL/hash metadata. They are not
relabelled as the original upstream wheel archives. See [METHOD.md](METHOD.md).

The receipt audit verified 127 evidence artifacts, 26,231 installed distribution
files, release-wheel contents, and 113 reconstructed dependency archives. The
new target's `release/lib_native.dylib`, wheel native payload and installed native
payload have the same hash.

| Identity | SHA-256 |
| --- | --- |
| New release wheel | `1811d2d9d5c6cf1360091122a62fd9678031159acab1ab2a278b13b7872d2e8c` |
| New native payload | `826c2f41612912c38a2cd902a1988b50a069b4e278f1f05ffdb6980744e42c64` |
| Successful receipt | `4d274918781c78e3f0b4ebfdaef1501e2d19e741ee09e6847ba4e3ba665b243e` |
| Receipt-content installation identity | `49cd1bc85a673a033ddc5bbbbdc07827b678548da458e31a657de059e2f9d4bb` |

Relative to installed-v4's original wheel, all 354 non-native `pllm/` files match.
The native binary differs; full-wheel differences also include RECORD and the
generated SBOM. The original v4 wheel, native hash and runtime `build_digest`
remain in `v4-comparison.json` and copied provenance. **No v4 execution result is
transferred to this build.** The installation identity above is not a runtime
`build_digest`.

## Evidence files

All following paths are under `2026-10-16-local-v5/` unless stated otherwise:

- `receipt.json`: status, identities, command records, invariants, 127-file hash inventory.
- `driver.py`, `build-source-receipt.helpers.py`: exact executed driver and reused helper.
- `preflight.json`, `resources-before-build.json`, `resources-after.json`, resource logs.
- `target-before-build.json`, `build-environment.json`, `tools.json`, `installer-tool.json`.
- `sources-{before,after}.json`, `dependencies-{before,after}.json`, `cargo-metadata.json`.
- `python-cache-{before,after}.json`, `python-dependency-wheel-inputs.json`.
- `upstream-frozen-dependencies.txt`, `frozen-dependencies.txt`, `wheel-install.txt`.
- `environment-empty.json`, `venv-configuration.json`, `environment-before-build.json`, `environment-after.json`.
- `original-environment-{before,after}.json`, `installed-package-files.json`, `wheel-files.json`.
- `v4-provenance.json`, `v4-manifest.json`, `v4-wheel-files.json`, `v4-comparison.json`, `prior-cached-receipt.json`.
- `wheel-build`, `dependencies-install`, `wheel-install`, `dependency-check` and other commands each retain `.request.json`, `.command.json`, `.log`.
- `wheels/`: the release wheel. `work/`: retained local target, venv, dependency archives and installer cache.
- Parent directory: `2026-10-16-local-v{1,2,3,4,5}.validation.json`, `installer-tool-cache-check.json`, `METHOD.md`.

## Proposed parent commands

Run from `/Users/blairhudson/workspace/pllm`. The first command is a read-only
hash audit. Run the second only after the parent has serialized fresh-environment
execution and confirmed that its new conformance output directory does not exist.
The second command executes the existing generated conformance suite, including
its SDK, benchmark and bounded operator cases; none have been run in this work.

```sh
RUN=/Users/blairhudson/workspace/pllm/paper/artifacts/clean-build/2026-10-16-local-v5
PY="$RUN/work/venv/bin/python"
WHEEL="$RUN/wheels/pllm_run-0.1.0a2-cp311-abi3-macosx_11_0_arm64.whl"

"$PY" -I -B scripts/audit_paper_clean_build.py --verify "$RUN/receipt.json"

"$PY" -I -B scripts/audit_paper_conformance.py \
  --output paper/artifacts/conformance/2026-10-16-clean-local-v1 \
  --wheel "$WHEEL" --reference-python "$PY"
```

This remains a local empty-target build with reused registry sources, dependency
payloads, toolchain and OS SDK. It is not CI or independent third-party attestation;
the complete limits are in [METHOD.md](METHOD.md). No commit was made.
