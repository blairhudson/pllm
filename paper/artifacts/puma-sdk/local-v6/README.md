# Schedule-derived protected-work admission

This slice replaces heuristic replicated-decoder work charges with native
instruction-schedule accounting. It preserves the numeric schedule, fresh
party-local MPC state, source commitments and existing complete-program ceilings.
The [contract](../../../../docs/research/puma-truncation-reference.md#schedule-derived-work-admission)
defines the distinct counters and scope.

## Work and ownership

Sequence counts are consumed top-level ordinals, not network messages, Boolean
rounds or CPU instructions. Public and pairwise cursor counts are exact; private
draws and AES expansions are per-party upper bounds across public pivots. Every
fill rounds its own tail, including narrow Boolean backing widths and OT.

Each compiled tensor retains its public budget. Before allocation or randomness,
execution checks remaining sequence/AES allowance and cursor overflow. The decoder
adds every product, tensor invocation, full loop iteration and tail across every
D9 full-prefix selection. The independent maxima remain 1,048,576 ordinals and
3,224,371,200 AES blocks per complete program. The 256 KiB primitive payload
ceiling and source/range admission remain separate.

Tests compare budgets with consumed counters, exercise tight allowances,
continuation after consumed input, short budgets, cursor overflow, wide rows,
Boolean backing widths, tail fills and allocation/retirement. Frozen private
execution identities and generated public/private SDK requests remain covered.

## Qwen admission

One 128-lane lookup predicate uses 28 ordinals and at most 2,817 AES blocks per
party. At 151,936 vocabulary rows and four input/two output selections, lookup
alone accounts for 299,124 ordinals and 30,094,011 AES blocks per party. That
metadata-only gate admits. Sixteen inputs/two outputs need 1,096,788 lookup
ordinals and reject before source values. Private weights still reject source
expansion. These are schedule counts, not measured execution costs.

`qwen-metadata-replay.json` retains both modes and the larger rejection. Its zero
artifact pins are metadata placeholders and authenticate no checkpoint values.
The separate ordinary `benchmark_memory` retry uses the pinned source and rejects
shared Safetensors header admission: the reader still applies private-weight
tensor limits to public weights. `qwen-memory-preflight.json` retains that result.
There is no full-checkpoint execution, complete-model range qualification, quality
result or serving measurement in this slice.

## Evidence and replay

`summary.json` records the accepted build and validation. `build/` retains the
guarded cached-local build; `validation/` contains exact commands, JUnit output,
native checks, lint/coverage logs, installed-file hashes and dependencies.
`inputs.tar.gz` and `validation-inputs.json` preserve source/test inputs.
These counts overlap earlier slices and cannot be added. Build monitoring has
its own scope; it does not establish test or serving swap/RSS results.

| Check | Result |
| --- | --- |
| Unique installed-wheel Python cases | 245 passed; seven optional skips |
| Native core/compiler/integration checks | 139 passed |
| Ownership doctests | Five passed |
| Installed package files | All 367 match the wheel before and after validation |
| Clippy and public coverage inventory | Passed; two recorded Clippy allowances |

`attempts/` retains the earlier allocation-test failure, noncanonical-JSON
validation setup rejection and cancelled quality invocation. The unchanged
isolated quality retry passes. Remaining-suite validation excludes the original
22 passing cases; `validation/prior-passed-tests.json` binds their exact node IDs
and report. `validation/collection-audit.json` checks that all 252 collected
cases have final outcomes across these unchanged-source runs.
Original receipt paths retain their build-directory names. Their archived
counterparts are `build/receipt.json` and
`attempts/quality-cancelled/installed-tests.xml`.
The corrected allocation fixture checks earlier construction rejection and
independently preserves pre-allocation runtime
payload rejection. The metadata fixture now uses ordinary Experiment canonical
serialization. Neither change relaxes runtime admission.

Verify files from this directory:

```sh
shasum -a 256 -c SHA256SUMS
```

Rebuild/install the preserved sources, then run the commands in
`validation/validation-commands.json`, adjusting interpreter and output paths.
Omit the saved `--deselect` options to rerun all Python cases. Those options only
exclude already-passing cases from the historical retry. Python checks use the
installed wheel with `-I`; `-o pythonpath=.` supplies repository test helpers
only. Metadata replay requires no checkpoint:

```sh
python -I - <<'PY'
import json
from pathlib import Path
from pllm import _native
r = json.loads(Path("qwen-metadata-replay.json").read_text())
cases = [(case, r["rows"], r["outputs"]) for case in r["modes"].values()]
larger = r["larger_public"]
cases.append((larger, larger["rows"], larger["outputs"]))
for case, rows, outputs in cases:
    try:
        actual = _native.replicated_source_contract(
            case["experiment"].encode(), r["normalized_config"].encode(),
            rows, outputs, r["source_lock"], r["checkpoint"],
        )
    except ValueError as error:
        assert case["status"] == "rejected" and str(error) == case["reason"]
    else:
        assert case["status"] == "metadata-admitted"
        assert json.loads(actual) == case["contract"]
PY
```
