# Provider-child provenance qualification and follow-up

Audit: 16 October 2026. Original executions are immutable. This follow-up runs
only Python subprocess probes and fixed-message codecs, with **zero provider,
model, cryptographic-material or operator executions**.

## Historical coverage

The named wheel/native/355-file package locks describe the inspected controller
installation. Python `-I` is a process flag, not inherited isolation. In the SDK
that matches v4's installed Python files, `runtime/servers.py::_commands` builds
`[sys.executable, "-m", ...]` without `-I`; `_environment` retains `PYTHONPATH`
(explicitly for worker/boundary roles), and `_spawn_role_process` passes that
environment to `subprocess.Popen` at the current working directory.

| Historical launch path | Exact enforcement in frozen source | Provider-child identity |
| --- | --- | --- |
| v4 generated direct SDK/gateway via `published_requests` | Controller invoked with `-I`; direct call changes thread/cache variables but does **not** scrub the parent import environment | **Unknown**; no per-child import/native receipt |
| v4 pytest subprocesses and unchanged prepared-SDK example | `command()` removes `PYTHONPATH`; Python child itself gets `-I`; pytest gets `-o pythonpath=`. Other `PYTHON*` variables were not removed by this helper | **Unknown** for provider grandchildren; they omit `-I` |
| v4 ordinary benchmark | Installed `metrics/benchmark_runner.py::_environment` removes **all** `PYTHON*`; inspector/worker use `-I`. This is stronger than the generic driver helper | **Unknown** for provider descendants; no per-child receipt |
| Six-family pytest/worker subprocesses and family follow-up | `run_command()` removes `PYTHONPATH`; Python child gets `-I`; pytest gets `-o pythonpath=` | **Unknown** where provider grandchildren are created, including verified-delegation/equalization SDK/gateway tests |

No frozen record states the complete inherited import environment for those
provider children. Absence of `PYTHONPATH` in a command line is not evidence that
the variable was unset. The successful outcomes and parent byte locks remain
recorded, but do not establish an all-provider execution of that exact wheel.
In-process numerical/oracle results are not provider-child identity receipts.
No new assumption about which package the historical children actually loaded
is introduced. Prior failed attempts also retain unknown child identity.

## Research-driver enforcement for the next execution

`python-child-isolation.py` is installed in-process by both current audit drivers
after wheel/package validation and before functional execution. It:

1. Removes `PYTHON*`, `DYLD_*`, and `LD_*` import/loader environment variables
   from the controller and each intercepted same-interpreter Python launch;
   records removed **names**, never environment values or role credentials.
2. Preserves the venv interpreter entry-point path (does not resolve its symlink),
   enforces child `-I`, and rejects ambiguous shell/executable overrides.
3. Uses a small `runpy` bootstrap to inspect the actual child before its original
   module/script/code entry point. The child imports `pllm` and `pllm._native`,
   checks every package file against the supplied wheel, rejects extra package
   files, checks the loaded native hash and origin, and records interpreter,
   flags, user-site setting, `sys.path`, PID/parent PID and loaded PLLM origins.
4. Installs the same process-local wrapper in that child, so nested launches and
   runtime environment allowlists cannot drop receipt configuration. The bootstrap
   carries receipt paths directly; no audit environment variable is required.
5. Captures start/end receipts and reconciles them with each launch/spawn record.
   Missing exits or mismatching bytes keep the identity gate failed. Provider
   selections require recorded `inference` and `preparation` launches.

The wrapper changes research launch instrumentation, not installed runtime files.
It covers same-interpreter `subprocess.Popen` launches used by the audited role
factory; a deliberately separate independent-reference interpreter retains its
own no-PLLM check. It is **local import/byte provenance**, not hermetic startup,
arbitrary-process instrumentation, a security proof or third-party attestation.
Identity hashing is diagnostic overhead and supplies no performance comparison.

Future family runs accept `--wheel PATH` and record their own identity plus
fieldwise comparison to original v4. A different new wheel/build does not inherit
v4's label. The cached-build supplement and v4 files are not rewritten.

## Focused validation, with no provider starts

Accepted command (exit 0):

```sh
target/frontier-sdk-env/bin/python -I paper/artifacts/conformance/audit-child-isolation.py \
  --output paper/artifacts/conformance/2026-10-16-child-isolation-v2 \
  --wheel target/certified-head-sdk-build-v3/wheels/pllm_run-0.1.0a2-cp311-abi3-macosx_11_0_arm64.whl
```

Seven checks pass, with three captured Python processes. The tests demonstrate:

- An unisolated control really imports the injected `PYTHONPATH` canary.
- The installed `_spawn_role_process` boundary launches a tiny import-only
  program with correct package/native bytes despite hostile Python variables.
- A grandchild launched after complete environment replacement stays isolated,
  even with a shadow package in its working directory.
- A deliberately wrong native digest rejects before the program body runs.
- Missing child-end receipt and ambiguous shell launch keep their gates closed.
- Fixed-message accounting accepts only timing-width variation while rejecting
  changed geometry, wire width, non-timing metadata length and truncated data.

`2026-10-16-child-isolation-v1` is retained as failed: six isolation checks pass,
but the first accounting parser incorrectly assumed a raw MessagePack map for
the prepared frame. The corrected parser handles the actual `PLLMPSR1/2` magic,
four-byte row header and five-field list, plus the distinct masked-stage map.
The v2 result does not turn the v1 attempt into a pass or rerun a provider test.

## Exact equalization failure and principled next test change

Failing node:

```text
tests/test_public_equalization.py::test_prepared_numeric_choices_have_matched_online_stage_body_sizes
```

The original assertion at line 192 compares `(stage_calls, upload_bytes,
download_bytes)` tuples: `(4,11580,16289)` versus `(4,11580,16291)`.
Original response frames were not retained. **That failure remains unresolved.**
The codec controls show a possible two-byte mechanism, not proof of its cause.

Minimal change for a future source-test fix:

1. Keep exact equality of stage calls and complete upload bytes.
2. Capture every raw response at `PreparedRemoteLinear._decode_reply`; retain
   the frames, per-stage public geometry and full-byte totals for both choices.
3. Require equal stage order, geometry, ring/wire width, arithmetic payload byte
   count and frame length with **only** `server_ns` normalized to zero. Do not
   compare masked arithmetic values across different quantizers.
4. Check each measured download total equals the sum of its retained full
   frames, and that the entire cross-choice byte difference equals precisely
   the difference in encoded `server_ns` value widths.

`equalization-wire-accounting.py` supplies executable bounded logic for steps
3–4. A test-only wrapper can collect its `account_response(payload,
stage_id=stage.id)` results before forwarding the original `_decode_reply`:

```python
assert bodies[0][:2] == bodies[1][:2]  # calls and full uploads unchanged
assert bodies[0][0] > 0
for body, responses in zip(bodies, retained, strict=True):
    assert len(responses) == body[0]
    assert sum(r["full_frame_bytes"] for r in responses) == body[2]
assert_equal_static_contract(retained[0], retained[1])
```

The helper limits frames to 1 MiB, dimensions to 256 and elements to 65,536,
validates the installed raw-frame codec and rejects unrecognized/noncanonical
framing. It does not silently remove arbitrary metadata or relax total-byte
accounting. No source test or runtime is changed in this follow-up. Capturing
future frames and applying this test fix is a separate gate; an occasional pass
of the original timing-sensitive assertion does not resolve the retained failure.

## Gate status

**Ready for parent-owned serialized fresh-environment replay.** New provider
SDK/gateway execution has not been attempted. Its receipt must contain actual
provider start/end identities and successful public-path outcomes; the import-only
probes cannot substitute for it. Original v4/build/family records and counts stay
unchanged. Clean-build and Listing-1 artifacts are outside this worker's scope.
