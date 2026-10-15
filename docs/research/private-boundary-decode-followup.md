# Independent private-boundary decode follow-up

This follow-up retains independent whole-response confirmation, longer-output
workloads, instrumented timing attribution and synthetic HTTP striping as separate
evidence. The original 150+8 screen remains in
`docs/research/private-boundary-decode-sdk-reference.md`.

## Preserved rejected confirmation

The first independent five-choice 150+8 confirmation completed all fifteen
clients. One serial/uniform-residue client recorded 80,084,992 bytes of host swap
growth. The unchanged zero-growth audit rejects the complete cohort. Its samples
do not establish confirmation or fill missing replicates in later cohorts.

Original results, plan and rejection are retained at
`docs/evidence/rejected/private-boundary-decode-confirmation-swap-rejected.json`,
`.plan.json` and `.progress.json`. The evidence registry lists the result under
historical exclusions. The guard did not terminate the request; the stricter
cohort audit rejected the recorded growth afterward. The rejected archive stays
outside the SDK inventory's flat evidence scan, so completed individual requests
cannot silently become accepted capability measurements.

## Independent 150+8 confirmation

The fresh three-choice cohort completes nine clients, with three balanced
replicates per choice on the same installed build and a new private salt.
All 78,842,880 checked logit/KV values, selections, RNG, positions and output
commitments agree. Every response reaches eight outputs; observed swap growth
is zero. Each worker includes the SDK, supervisor, dashboard, relay and state
audit. The idle launcher and providers remain outside client RSS.

| Metric, median unless noted | Maintained prepared | Serial/u32 boundary | Parallel/residue boundary |
| --- | ---: | ---: | ---: |
| Decode, tokens/s, N−1 outputs | 4.9205 | 2.1667 | 3.3124 |
| All-link TCP-stream bytes, decimal MB | 478.654 | 367.347 | 365.221 |
| Request, seconds | 98.955 | 76.153 | 75.025 |
| Startup through cleanup, seconds | 110.195 | 91.126 | 89.690 |
| Online first text, seconds | 38.780 | 42.027 | 41.883 |
| Client CPU, seconds | 14.543 | 12.465 | 12.464 |
| Client RSS, MiB | 230.89 | 214.77 | 220.38 |
| Largest complete-worker peak, MiB | 234.20 | 227.47 | 224.13 |

Parallel/residue decode improves **52.87%** against serial/u32, with **1.48%**
lower complete-request time and almost unchanged client CPU. Median client RSS
increases **2.61%**; all three choices qualify for observed Low and Medium tiers
using their largest peaks, rather than their medians. Conservative runtime
admission remains separate.

Against the maintained prepared control, traffic falls **23.70%** and requests
are **24.18%** faster. Decode remains **32.68%** slower and online first text is
**8.00%** later. The head codec removes exactly 2,125,920 application bytes versus
serial/u32. One replicate adds one lifecycle-control byte; raw body and TCP
reconciliation preserve that difference. No complete base/per-token attribution
is inferred from the residual.

This confirms the selected finalists within a source-present 150+8 workload.
The seven earlier incumbents are not rerun here. Historical and failed cohorts
remain separate. Checkpoint/tokenizer distribution, physical wire and Internet
RTT remain excluded, and the additional boundary workers retain their distinct
trust contract.

Original evidence is retained at
`docs/evidence/private-boundary-decode-confirmation-150x8.json`, `.plan.json`,
`.summary.json` and `.progress.json`. The initial auditor source is frozen at
`docs/evidence/private-boundary-confirmation-initial-auditor.py`. After measurement,
the current auditor gained observed-build validation and raw-duration/metric
contract checks; the docs reader also verifies every cached sample metric.
Both stronger readers reproduce the original summary without changing evidence.
The registry explicitly maps the original helper hash to its archived bytes;
future measurements use the hardened current helper.

Read-only summary replay:

```sh
target/frontier-sdk-env/bin/python - <<'PY'
import json
from pathlib import Path
from benchmarks.research.private_boundary_decode_confirmation import audit
p = Path('docs/evidence/private-boundary-decode-confirmation-150x8.json')
assert audit(json.loads(p.read_bytes())) == json.loads(p.with_suffix('.summary.json').read_bytes())
print('Independent 150+8 summary replays exactly.')
PY
```

## Separate 150+32 long-output screen

Nine fresh clients reach all 32 output selections and 31 decode intervals. All
**336,605,184** checked logit/KV values and complete state/output commitments agree;
observed swap growth is zero. All three replicates of each recipe claim their
16,929 issued rows, including 2,976 decode rows, with no unused rows. This first
longer-workload screen has its own private cohort and does not inherit the
150+8 confirmation status.

| Metric, median unless stated | Maintained prepared | Serial/u32 boundary | Parallel/residue boundary |
| --- | ---: | ---: | ---: |
| All-link TCP streams, MB | 536.30 | 454.46 | 445.96 |
| Full request, s | 106.50 | 89.56 | 84.32 |
| Startup-through-cleanup target, s | 117.42 | 104.49 | 99.21 |
| Online first text, s | 39.068 | 41.955 | 41.787 |
| N−1 text-event decode, tokens/s | 4.935 | 2.157 | 3.318 |
| Client cold CPU, s | 17.153 | 14.327 | 13.754 |
| Client process peak, MB | 229.00 | 240.68 | 234.29 |
| **Largest** complete-client peak, MiB | **218.81** | **229.66** | **230.69** |
| Preparation cold CPU, s | 8.521 | 8.581 | 8.415 |
| Preparation sampled peak, MB | 215.24 | 221.07 | 217.06 |

Parallel/residue decode improves **53.79%** against serial/u32, with exactly
**8,503,680** fewer head-reply bytes. Against maintained prepared execution,
traffic falls **16.84%**, full request latency **20.83%** and complete target time
**15.51%**. Decode remains **32.77% slower** and online first text is **6.96% later**.
Median client CPU falls **19.82%**, while median client RSS rises **2.31%**.
All three recipes fit observed Low/Medium ceilings using their largest complete
worker peaks. The same source-present, topology, resource-window and physical-wire
limits above apply.

Original raw report, plan, summary and progress are preserved byte-for-byte under
`docs/evidence/private-boundary-decode-confirmation-150x32.*`. The read-only Python
audit and registered JavaScript adapter reconstruct the metrics and costs. This
cohort used the strengthened helper directly; its frozen sources need no historical
snapshot mapping. Reproduce without executing a model:

```sh
target/frontier-sdk-env/bin/python - <<'PY'
import json
from pathlib import Path
from benchmarks.research.private_boundary_decode_confirmation import audit
p = Path('docs/evidence/private-boundary-decode-confirmation-150x32.json')
assert audit(json.loads(p.read_bytes()), max_output_tokens=32) == json.loads(
    p.with_suffix('.summary.json').read_bytes())
print('150+32 original summary reproduced')
PY
```

## Separate 150+64 long-output screen

Nine fresh clients reach all 64 output selections and 63 decode intervals. All
**729,833,472** checked logit/KV values and complete state/output commitments agree;
observed swap growth is zero. Each request claims all 20,001 issued rows, including
6,048 decode rows, with no unused rows. This first 64-output screen retains a
separate private cohort from both shorter workloads.

| Metric, median unless stated | Maintained prepared | Serial/u32 boundary | Parallel/residue boundary |
| --- | ---: | ---: | ---: |
| All-link TCP streams, MB | 613.15 | 570.62 | 553.61 |
| Full request, s | 124.07 | 115.77 | 105.28 |
| Startup-through-cleanup target, s | 135.14 | 131.99 | 120.19 |
| Online first text, s | 38.776 | 42.314 | 41.955 |
| N−1 text-event decode, tokens/s | 4.800 | 2.130 | 3.257 |
| Client cold CPU, s | 18.027 | 16.701 | 16.229 |
| Client process peak, MB | 232.37 | 223.28 | 234.23 |
| **Largest** complete-client peak, MiB | **235.11** | **220.72** | **230.56** |
| Preparation cold CPU, s | 9.659 | 10.154 | 9.642 |
| Preparation sampled peak, MB | 250.72 | 246.78 | 260.16 |

Parallel/residue decode improves **52.89%** against serial/u32; head replies remove
exactly **17,007,360** bytes. Its median client RSS rises **4.90%** against serial,
while client CPU falls **2.82%**. Against maintained prepared execution, traffic
falls **9.71%**, full request latency **15.14%** and complete target time **11.06%**.
Decode remains **32.14% slower** and online first text is **8.20% later**. Median
client CPU falls **9.97%**, while client RSS rises **0.80%**. All three recipes fit
observed Low/Medium ceilings using their largest complete worker peaks.

Across the separately retained 8/32/64-output workloads, source-present traffic
savings narrow as more decode work is charged. The cohorts are not pooled, their
controls remain workload-specific, and no unmeasured crossover is inferred.
Physical wire, Internet RTT and source/tokenizer distribution remain excluded.

Original report, plan, summary and progress are preserved byte-for-byte under
`docs/evidence/private-boundary-decode-confirmation-150x64.*`. Read-only replay:

```sh
target/frontier-sdk-env/bin/python - <<'PY'
import json
from pathlib import Path
from benchmarks.research.private_boundary_decode_confirmation import audit
p = Path('docs/evidence/private-boundary-decode-confirmation-150x64.json')
assert audit(json.loads(p.read_bytes()), max_output_tokens=64) == json.loads(
    p.with_suffix('.summary.json').read_bytes())
print('150+64 original summary reproduced')
PY
```

## Separate 150+8 timing-attribution cohort

Eighteen fresh clients compare three recipes, each with three ordinary controls
and three instrumented probes, under one separate private cohort and installed
build. All 157,685,760 checked logit/KV values and 144 selection/state commitments
agree; all guards pass with zero observed swap growth. Complete-client lifetime
peaks range from 206.33 to 235.78 MiB. These instrumented timings retain a separate
measurement identity and do not revise the confirmation or serving rankings.

The table takes each probe's exclusive after-first-output elapsed bins, divides
by seven inter-output intervals, then takes the median of three probes. Units
are **milliseconds per interval**. Every individual trace partitions exactly;
independently calculated medians need not sum to the median elapsed time.

| Exclusive elapsed owner | Maintained prepared | Serial/u32 boundary | Parallel/residue boundary |
| --- | ---: | ---: | ---: |
| Body excluding lookup/private head | 199.703 | 189.594 | 188.719 |
| Paired head exchange | 0 | 256.253 | 102.229 |
| Token lookup | 0.080 | 9.680 | 7.340 |
| Full-state audit and sampler | 1.597 | 2.901 | 1.892 |
| Head quantization | 0 | 0.016 | 0.016 |
| Head codec decode | 0 | 0 | 0.837 |
| Head dequantization | 0 | 0.125 | 0.086 |
| Unclassified head residual | 0 | 0.730 | 0.099 |
| Other unclassified elapsed | 1.398 | 2.302 | 1.192 |
| Complete interval | 202.779 | 461.600 | 302.409 |

Maintained execution's local head remains inside its body bin. The coupled body
bin includes local operators and prepared remote waits; it is not body CPU.
HTTP union coverage clipped to head exchanges is 256.141 ms serial and 102.058 ms
parallel/residue. Those diagnostic unions overlap the exchange owner and are not
additional elapsed costs. Parallel A/B request durations must not be summed.

Body execution is the largest remaining owner, followed by head exchange. The
small client codec and head residual do not identify remote kernel costs: the
exchange still combines remote computation, server encoding, network/pacing and
client waits. Share generation is not isolated. These observations support
testing bounded exchange streaming, but cannot rank stripes against SDOT or
establish a new serving speedup.

Control/probe median decode TPS is 4.938/4.931 for maintained execution,
2.159/2.166 for serial/u32 and 3.285/3.307 for parallel/residue. These differences
include instrumentation perturbation and run noise; no overhead subtraction is
used. Original timing spans and paired deltas are retained in
`docs/evidence/private-boundary-profile/150x8.json` and its plan, summary and
progress companions. `manifest.json` pins their original bytes. This separate
research archive stays outside automatic ordinary SDK evidence discovery.

Read-only replay and interval aggregation, using the installed SDK interpreter:

```sh
target/frontier-sdk-env/bin/python - <<'PY'
import hashlib, json, statistics
from pathlib import Path
from pllm._benchmark_contract import digest
from benchmarks.research.private_boundary_decode_profile import IDENTITY, audit
root = Path('docs/evidence/private-boundary-profile')
manifest = json.loads((root / 'manifest.json').read_bytes())
for name, expected in manifest['files'].items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected
p = root / '150x8.json'
document = json.loads(p.read_bytes())
plan = json.loads(p.with_suffix('.plan.json').read_bytes())
identity = digest({'schema': IDENTITY, 'sources': plan['frozen_sources']})
assert identity == plan['research_source_identity']
assert plan['suite'] == document['suite']
summary = audit(document, source_identity=identity)
assert summary == json.loads(p.with_suffix('.summary.json').read_bytes())
for name in sorted({row['case'] for row in summary['profiles']}):
    windows = [row['timing']['after_first_output'] for row in summary['profiles']
               if row['case'] == name]
    assert len(windows) == 3
    bins = {key: statistics.median(w['exclusive_wall_ns'][key] / 7e6 for w in windows)
            for key in windows[0]['exclusive_wall_ns']}
    print(name, bins, 'elapsed_ms', statistics.median(w['elapsed_ns'] / 7e6 for w in windows))
PY
```

## Separate exact-residue stripe fixture

Six fresh generated-source Qwen-geometry workers complete **108 head exchanges**
under shared native 100/40 caps, after the SDK cohorts finish. Three workers use
8,192-byte client reads and three use 1,024-byte reads. Each worker balances six
permutations of the installed parallel/residue baseline, sliced full outputs and
early-compute stripes backed by separate native snapshots. All exchanges retain
the same complete float32 logit digest, and every worker observes zero swap
growth. The fixture uses V=151,936, H=896 and 8,192 output rows per stripe; it
does not load checkpoint weights or execute a decoder.

Each table entry is the median of three worker medians, each over six exchanges.
Read-size cohorts stay separate; the archived summary also retains within-repeat
paired deltas and their two-level medians.

| Client read bytes | Arm | Exchange, ms | Aggregate fixture CPU, ms | Online TCP payload bytes |
| ---: | --- | ---: | ---: | ---: |
| 8,192 | Installed parallel/residue | 103.981 | 27.363 | 958,402 |
| 8,192 | Sliced full output | 107.438 | 35.432 | 966,570 |
| 8,192 | Early snapshot stripes | 98.000 | 37.045 | 966,570 |
| 1,024 | Installed parallel/residue | 104.009 | 27.343 | 958,402 |
| 1,024 | Sliced full output | 107.537 | 35.782 | 966,570 |
| 1,024 | Early snapshot stripes | 98.066 | 37.779 | 966,570 |

Relative to the matched installed baseline, snapshot stripes lower isolated
exchange medians **5.75%/5.71%**, while aggregate fixture CPU rises **35.38%/38.17%**.
Slicing complete native outputs is **3.32%/3.39% slower**. Both stripe arms add
**7,852 application-body bytes** and **8,168 TCP payload bytes** per exchange.
Per-exchange transport setup is separately measured at roughly 0.82–0.84 seconds;
fixture construction is also separate. Process startup/import and setup CPU are
not measured by these timing fields.

Snapshot stripes add **136,134,656 persistent native weight bytes**. The shared
fixture retains source, full native and striped native weight owners totaling
408,403,968 bytes before other owners. Logical current-plus-next output reservation
per role is 65,536 bytes, versus 607,744 bytes for a sliced full output; encoding,
conversion, framework and socket buffers are additional. No per-arm RSS or
whole-client memory tier is measured. The two logical workers share one process
and executor; aggregate fixture CPU is not separately attributed role CPU.

Candidate frames independently reconcile declared lengths and role body counters.
All 144 candidate owners retire, 144 replay attempts return HTTP 409 and no native
tasks remain. The baseline emits complete float32 logits but no direct integer
digest, native/encoding timings or retirement counters; its zero timing fields
are placeholders. Candidate source asserts complete integer-oracle equality before
success, but the archive cannot independently certify baseline integer equality.
Fault-injection tests remain separate from this sampled cohort. The read-only
summary therefore qualifies only exposed fixture timings and counters.

**Promotion remains blocked:** installed native execution lacks borrowed immutable
row-range evaluation, and no SDK stripe composition, complete-checkpoint parity or
whole-client resource result exists. Earlier bytes in this fixture do not isolate
streaming from async versus threaded HTTP, extra native calls, encoding and cache
effects. These results cannot rank stripes against the separate byte-dot API
screen or predict a whole-response gain.

Original reports, progress, JSONL rows, pre-run pins and summary are preserved
byte-for-byte in `docs/evidence/private-boundary-residue-stripes/`. The fixture
contract stays frozen at `docs/research/boundary-head-residue-stripes.md`.
Read-only stdlib replay from repository root:

```sh
python3 - <<'PY'
import hashlib, json
from pathlib import Path
from scripts.summarize_boundary_head_residue_stripes import summarize
root = Path('docs/evidence/private-boundary-residue-stripes')
raw = (root / 'manifest.json').read_bytes()
assert hashlib.sha256(raw).hexdigest() == 'a2aed02994ccabf99aa89b842acfeacdf1ae3ec084a6b64b279fa4dc27ea8f05'
for name, metadata in json.loads(raw)['files'].items():
    data = (root / name).read_bytes()
    assert len(data) == metadata['bytes']
    assert hashlib.sha256(data).hexdigest() == metadata['sha256']
reports = [root / f'residue-stripes-read{read}-worker{worker}-v1.json'
           for read in (8192, 1024) for worker in (1, 2, 3)]
pins = json.loads((root / 'residue-stripes-frozen-pins-v1.json').read_bytes())
assert summarize(reports, pins) == json.loads(
    (root / 'residue-stripes-summary-v1.json').read_bytes())
print('Six-worker stripe summary reproduced; qualification remains fixture-only.')
PY
```

## Verified measurement paths

Using the same installed SDK environment, the new generated-source smoke paths
complete nine confirmation clients and eighteen instrumented/control clients.
The separate three-arm HTTP stripe smoke preserves complete logits. These are
functionality checks, not Qwen performance or memory-tier results.

Contracts and exact execution commands:

- `docs/research/private-boundary-decode-confirmation.md`: independent maintained,
  serial/u32 and parallel/residue responses, natural EOS and actual executed work.
- `docs/research/private-boundary-decode-profile.md`: exclusive client elapsed
  partitions with separate matched instrumentation controls.
- `docs/research/boundary-head-residue-stripes.md`: installed parallel/residue
  delivery versus sliced full outputs and duplicate-snapshot early stripes.

All model and timing jobs run serially under the shared frontier lock. Original
memory admission, source/build checks, exact-state comparison and swap gates
remain enforced. Every workload and retry has fresh output paths and cohort
identity. Longer-output, timing-attribution and stripe measurements remain
separate from the completed 150+8 confirmation above.
