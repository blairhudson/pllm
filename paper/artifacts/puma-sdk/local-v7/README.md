# Native-derived source-header bounds

The shared Safetensors reader derives dimension and element bounds from native
source admission. It independently checks the source shapes against the semantic
plan before inspecting checkpoint headers. This removes the stale private-weight
size table that rejected native-admitted public PUMA checkpoints.

Exact tensor shapes, floating-point dtypes, contiguous ranges, shard ownership,
file identities and source locks still precede value access. Header/metadata
ceilings and bounded conversion remain enforced. GKM and both PUMA modes use
the same reader; no new numeric identity or execution capacity is introduced.

## Regression and source preflight

An 8,193-by-128 sparse token table crosses both former private-header limits.
Three F32/F16/BF16 tests check every converted word, including signed zero,
the frame boundary and final partial frame. Six malformed-header controls reject
wrong shapes, dimensions, dtypes, ranges and extra/missing tensors before source
hashing or value decoding. `regression/` retains the original old-wheel failure.

The ordinary `benchmark_memory` retry uses the pinned Qwen2.5-0.5B checkpoint
with four input and two output tokens. It now admits:

| Allocation estimate | Bytes |
| --- | ---: |
| Each role, including 25% margin | 587,202,560 |
| Complete client/three-worker topology | 2,348,810,240 |
| Required disk | 14,004,270,080 |
| Logical float32 source | 1,976,131,072 |

`qwen-memory-preflight.json` retains the exact Experiment, host observations and
unmodified SDK report. Source hashing checks the checkpoint bytes; tensor values
are not decoded and no workers launch. Full-model numeric/program admission and
checkpoint execution remain open. These estimates establish no measured RSS,
model quality or serving performance.

## Validation and replay

`summary.json` records the installed-wheel results. `build/` retains the guarded
cached-local build; `validation/` retains commands, JUnit output, package-file
hashes and dependency versions. `inputs.tar.gz` and `validation-inputs.json`
preserve the tested inputs. `package-delta.json` compares package contents with
the preceding work-budget wheel. Build monitoring has its own scope; it supplies
no test or serving swap/RSS measurement. Counts overlap previous slices.

| Check | Result |
| --- | --- |
| Installed-wheel Python cases | 317 passed; seven optional skips |
| New large-source/header cases | All nine pass; old reader fails the positive control |
| Installed package files | All 367 match the wheel before and after validation |
| Public coverage inventory | Passed |
| Native implementation | Same extension and all 311 source-file hashes as the preceding slice |

Native tests were not repeated for this Python-only change; the preceding slice
retains their original evidence. The seven skips require optional Metal or
explicitly configured real-checkpoint paths.

`attempts/initial-validation/` retains the first public CLI one-use request
failure. The host was normal before that run and reported memory-pressure
warning afterward. Those endpoint observations do not identify the failed
request's cause. The complete retry uses unchanged inputs and wheel bytes.
`attempts/launcher-timeout/` separately retains the interrupted 120-second
validation launcher. Its three surviving test workers were identified from
their generated source path and retired without forced termination; that
incomplete run supplies no passing-suite count.
The package delta changes only `pllm/runtime/delegated_source.py`; all other
366 installed files, including the native extension, match the previous wheel.

Verify this archive from its directory:

```sh
shasum -a 256 -c SHA256SUMS
```

Rebuild/install the preserved source, then run the commands in
`validation/validation-commands.json`, adjusting interpreter/output paths.
Python validation uses the installed wheel with `-I`; the repository root only
supplies test helpers. Repeat ordinary preflight from the repository root:

```sh
python -I - <<'PY'
import json
from pathlib import Path
from pllm import Experiment
from pllm.metrics import benchmark_memory
saved = json.loads(Path("paper/artifacts/puma-sdk/local-v7/qwen-memory-preflight.json").read_text())
report = benchmark_memory(Experiment.from_spec(saved["experiment"]), backend="native")
print(json.dumps(report, indent=2))
PY
```

Current host and disk headroom are rechecked; the historical admission does not
authorize a future request. The earlier private-header rejection remains in
`../local-v6/qwen-memory-preflight.json`.
