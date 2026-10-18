# Compact activation views and bounded tensor loops

This slice replaces expanded activation references with owner/offset/stride
spans. Slicing, broadcast, attention permutations and partial tiles preserve the
logical gather order. Six frozen private-weight execution identities remain
checked against the earlier installed wheel.

Public-weight lookup predicates, complete-sum Q18 rescaling and SiLU use bounded
128-lane loops. Each iteration creates a fresh consuming MPC executor; the
complete output and required source owners remain charged. Replay between
iterations retires partial outputs and the runtime. Every iteration contributes
to the whole-program work allowance. Public execution digests bind these new
loop instructions, main/tail programs and operand views.

The [contract](../../../../docs/research/puma-truncation-reference.md#compact-activation-views-and-tensor-loops)
retains the existing numeric, primitive, source, work and transport limits.
These changes qualify compiler/runtime ownership, not a serving-speed or
whole-client memory improvement.

## Full-Qwen gate

The source-only Qwen2.5 declaration retains four inputs and two outputs. Its
public mode now clears the compiler-view budget. Private lookup alone has a
minimum declared charge of **21,878,784** against **1,048,576** allowed units,
so admission rejects before checkpoint reads, publication or workers. These are
conservative compiler charges, not measured cryptographic operations or CPU.
Private mode retains its source-expansion rejection. Whole-request work
qualification, complete-model ranges, quality and measured resources stay open.

`qwen-metadata-replay.json` preserves both exact calls and rejection reasons.
The zero artifact pins are metadata placeholders; this check authenticates no
checkpoint values. Historical view-limit receipts remain in `local-v4/`.

## Evidence and replay

| Check | Result |
| --- | --- |
| Installed-wheel Python tests | 245 passed; seven optional skips |
| Native replicated-decoder tests | 20 passed |
| Installed package files | All 367 match before and after tests |
| Clippy | Passed with the two recorded lint allowances |
| Formatting and public coverage contract | Passed |
| Archived source/test inputs | 787 files, unchanged through validation |

`summary.json` records the accepted build, test counts and file checks.
`build/` preserves the guarded cached-local build receipts; `validation/` retains
the exact commands, installed package hashes, dependencies, JUnit report and
native/Clippy/coverage logs. This is not a clean release or revision-bound CI
attestation. Build swap observations have a separate scope from tests and serving.

`inputs.tar.gz` and `validation-inputs.json` preserve the tested source and test
inputs. Counts overlap earlier conformance slices and cannot be added. From this
directory, verify the archive with:

```sh
shasum -a 256 -c SHA256SUMS
```

Rebuild/install the preserved wheel sources and run the commands in
`validation/validation-commands.json`. Installed-wheel checks use `python -I`
with `-o pythonpath=.` only for repository test helpers; no root `pllm` package
shadows the installed wheel. The metadata rejection needs no checkpoint:

```sh
python -I - <<'PY'
import json
from pathlib import Path
from pllm import _native
r = json.loads(Path("qwen-metadata-replay.json").read_text())
for mode, case in r["modes"].items():
    try:
        _native.replicated_source_contract(
            case["experiment"].encode(), r["normalized_config"].encode(),
            r["rows"], r["outputs"], r["source_lock"], r["checkpoint"],
        )
    except ValueError as error:
        assert str(error) == case["reason"], (mode, str(error))
    else:
        raise AssertionError((mode, "historical gate unexpectedly admitted"))
PY
```
