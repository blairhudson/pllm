# Full-geometry lookup/random-head fusion reference

**Three full-geometry native workers completed with exact independent oracles.**
Median isolated lookup/head CPU falls 20.22%, but the analytical two-worker
schedule delays body start and completion about 9.5 ms. SDK promotion remains
blocked by delivery scheduling and the missing admitted joint protocol.
Original 1,025/4,096-row results remain separate; no physical-memory saving or
client-memory tier follows from this mixed-fixture reference.

## Boundary and correctness

New `cfg(test)` grandchild `boundary_scan_fusion_full_probe.rs` accesses original
test-only `Fused`, original `Server::admit`, DPF primitives and immutable table
through child visibility. It neither copies private-pages runtime nor exports
masks. Original fixture semantics and 4 MiB cap remain unchanged. No production
API, SDK dispatch, registry, architecture or old evidence changes.

Full fixture is fixed at V=151936,H=896. Small correctness uses 129×19. Full run
requires explicit ignored-test admission and macOS native headroom support.
One immutable matrix backs two party tables; baseline, candidate, standalone
lookup and original `evaluate` oracle share those tables with independent ledgers.
The two tables own separate exact scale copies, not separate weight matrices.
Even fault controls share this same matrix. No raw/interleaved table oracle copy.

Every repeat checks first or last record, both parties, complete original reply
bytes against `evaluate_batch` **and** independent-ledger `evaluate`, and complete
decoded H+4 record including every original scale bit. Every V-coordinate fused
head matches unchanged SIMD-enabled `wrap32` and independent scalar signed-i8
wrapping arithmetic. Complementary `W(q-r)` reconstructs independently computed
full `Wq`. Fresh OS entropy supplies each repeat/party's r and head ticket,
independently of DPF issuance. Same r/query are deliberately shared only across
alternative comparison/oracle worlds with independent ledgers, never recycled
as deployment tokens or retries. Both roles are alternative paired-role cases;
this is not sending r to both deployed workers.

Outside samples, same-matrix controls check shape, role, context, malformed query,
pre-admission non-burn, partial seven-row failure, cancellation, full post-scan
response failure, query replay, old-ticket/fresh-query rejection and exhausted
ticket ledger. Both identities burn after admission. Original small tests retain
additional original ledger/table controls. Row count/order and timing-arm order
depend only on public repeat/party, never mask contents or selected token.

## Admission, owners, and bounded work

Before source data, matrix, keys or oracles allocate, Rust admits geometry,
2..16 even repetitions, aggregate allocation envelope, aggregate weight-work
envelope and host reserve. Launcher checks same envelope before dispatch.

Aggregate reserved payload/uncertainty bytes are

```
2VH + 3(4V) + 16(4V) + 4V + 32(4H) + 32 MiB
```

This includes retained source plus native matrix, retained caller plus two native
scale copies, **all** simultaneously live full outputs/reference owners (Wq,
standalone head, original-pair head, candidate head, complementary head, scalar
oracle and transient slack), batch masks, r/complement/entropy, lookup accumulators,
replies, expected records, packets/decoded correction words, table/ledger state,
DFS scratch, sampled JSON/progress and executor/allocator uncertainty. Sixteen
V-u32 slots is a conservative envelope, not a claim sixteen are always live.
32 MiB metadata slack is explicit, not an instrumented complete-RAM bound.
Peak native memory still needs observation; process/code/allocator overhead can
exceed payload estimates. No checkpoint/loader maps or deployment transport here.

Conservative weight-value work envelope is `VH * (2*repetitions*12 + 8)`; admission
caps it at the 16-repeat value. Twelve passes per repeat/party exceed timed arms
plus original/scalar verification; fixed eight cover source/native setup,
descriptor hashes, Wq and full/partial fault scans. Packet depth is fixed by
public geometry; at most 4095 packed tree nodes per traversal. No data-dependent
retry or unbounded task fanout. Worker timeout is 600 seconds.

Native guard uses `vm_stat` free+inactive pages and `sysctl vm.swapusage`, rejects
unknown units/platform/command failure, and requires **envelope + 2 GiB available**
before allocation, each repeat and after faults. Launcher checks before dispatch,
every approximately 100 ms, and after exit; any observed swap-use change rejects
and kills worker. This is sampled zero observed swap change, not proof against
unsampled or sub-display-resolution swap activity. Reserve is conservatively
retained even after allocation. Full run fails closed if guard unavailable.

## Samples and accounting

Every worker measures five arms: standalone packed lookup (`evaluate_batch`),
unchanged allocating `wrap32` random head, original sequential lookup+head pair,
original scalar fused pair, and complementary `wrap32` head. Each reports wall
and process CPU ns. macOS process CPU uses `CLOCK_PROCESS_CPUTIME_ID`; Linux CPU
clock is supported for small tests, but full native admission is macOS-only.
Query/r issuance is outside samples; fused admission/tree preparation is inside.
Setup wall time separately includes source, matrix, hashes, executor and Wq.
Checks/fault controls are outside arm times, but inside process lifetime/resources.

Orders rotate and reverse; pair-first order is exactly balanced across repeats
and parties. Default ten repeats spans five rotations and reversals. Both arms
coexist within each independent process for paired CPU. Three or more fresh
workers execute **serially**, not three concurrent workers. Ratios are per-worker
ratios of summed paired samples, then median across workers, not median sample
ratios. Per-arm totals and individual raw samples remain available.

Launcher retains flushed native per-sample progress, raw stdout/stderr, per-worker
reports and fsynced completed-worker JSONL. Failure stops cohort; partial evidence
is retained and cannot be resumed/reused under same prefix. Shared lock is
`~/.cache/pllm/frontier-probe.lock`, acquired nonblocking. All output paths use
exclusive creation; no existing prefix may be reused. Sources are snapshotted
before parent build, matched after build, frozen alongside successful cargo JSON
build log, toolchain strings, HEAD and exact binary hash, then rechecked around
every worker. Hashes establish reproducibility, not trusted-build attestation.
Parent must not mutate sources while compiling or measuring. Reader trusts
explicit frozen pins as its trust root; it does not execute tools or rewrite data.

Sampled process RSS maximum is **mixed-fixture**, including setup, both arms,
oracles, fault controls and checks. It is neither exact lifetime peak nor per-arm,
worker-deployment or Low-client RSS. Native logical role service is single-threaded;
two logical parties share process, immutable matrix and executor. No physical
two-worker throughput, cache isolation, HTTP/TLS or network observation follows.

## Two-worker/token work and analytical schedule

Across both workers, baseline has two lookups plus two heads:
`4VH + 8V` logical source bytes. Candidate has one fused pair, other lookup and
complementary head: `3VH + 8V`. Full-token optimistic source-visit reduction is
about 24.94%, not 50%; these are not measured DRAM bytes/physical scans. Source
values can be reloaded by compiler. Exact output payload still has two full head
shares plus two lookup replies; no codec saving is inferred.

Summary retains per-repeat two-worker CPU sums, for each alternative paired role:
baseline `original_pair_A + lookup_B + complementary_head_B`; candidate
`fused_pair_A + lookup_B + complementary_head_B`. These are service-cost sums from
serial samples, not observed concurrent token execution.
Complementary service uses the paired sample's own `q-r` measurement (the same r
whose `Wr` passed reconstruction), assigned analytically to logical worker B;
lookup B uses the opposite party's same-repeat packet. It never combines a head
with an unrelated sample's complement. Both logical services use the same matrix.

For sampled local services L_A,L_B,H_A,F_A,C_B and **user-supplied analytical** body
duration T (default arbitrary 200ms), reader computes:

* Installed ready-parallel: body starts `max(L_A,L_B)`; completion is
  `max(L_A,L_B) + T + max(H_A,C_B)`.
* Fused joint response: body starts `max(F_A,L_B)`; completion is
  `max(F_A,L_B) + T + C_B`. Lookup A cannot be delivered before F_A completes.
* Separate current-token runahead control: completion is
  `max(max(L_A,L_B)+T+C_B, L_A+H_A)`.

Body-start delay is explicitly `max(F_A,L_B)-max(L_A,L_B)`, including negative
values if fused happens to finish earlier. H_A and C_B can overlap only in this
analytical independent-worker model. Model/SDK/body timings are **not observed**.
No queueing, WAN/codec contention, other body traffic or output overlap is priced.
Full-head outputs and post-body complement remain charged regardless of fusion.

Promotion stays blocked even if scalar fusion wins: authenticated joint ticket
contract, full-checkpoint logits/KV/selection/RNG, unchanged SDK comparison,
schedule/non-delay validation, whole-response resource measurements and compiler
load review remain gates. `scalar_fusion_regresses` marks median wall **or** CPU
ratio >=1; preserve regression unchanged. Do not substitute additive byte-dot
candidate for unchanged `Matrix::wrap32` baseline or multiply unrelated gains.

## Exact parent commands

Run only after review, all other work idle, and sources stable. Choose new P every
attempt. Stdlib reader has no native/model imports. Build and native runs remain
parent-owned. Commands below use exclusive directory/files and shared lock.

```sh
PY=target/frontier-sdk-env/bin/python
P=target/fusion-full-review-v1
mkdir "$P"
set -C
"$PY" scripts/probe_boundary_scan_fusion_full.py sources --output "$P/sources.json"
"$PY" - "$P/build.jsonl" <<'PY'
import fcntl, pathlib, subprocess, sys
lock = pathlib.Path.home()/'.cache/pllm/frontier-probe.lock'
lock.parent.mkdir(parents=True, exist_ok=True)
with lock.open('a+') as held:
    fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
    with open(sys.argv[1], 'x') as out:
        subprocess.run(['cargo','test','--offline','--release','-p','pllm-garble',
                        '--lib','--no-run','--message-format=json'], stdout=out, check=True)
PY
BIN="$($PY - "$P/build.jsonl" <<'PY'
import json, sys
records = [json.loads(s) for s in open(sys.argv[1]) if s.startswith('{')]
paths = {r['executable'] for r in records if r.get('reason') == 'compiler-artifact'
         and r.get('target', {}).get('name') == 'pllm_garble'
         and r.get('profile', {}).get('test') and r.get('executable')}
assert len(paths) == 1
print(paths.pop())
PY
)"
# Small correctness only; ignored full fixture does not run.
"$PY" - "$BIN" <<'PY'
import fcntl, pathlib, subprocess, sys
with (pathlib.Path.home()/'.cache/pllm/frontier-probe.lock').open('a+') as held:
    fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
    subprocess.run([sys.argv[1], 'private_pages::boundary_scan_fusion_probe',
                    '--nocapture', '--test-threads=1'], check=True)
PY
"$PY" -m pytest -q tests/test_boundary_scan_fusion_full.py
"$PY" scripts/probe_boundary_scan_fusion_full.py pin --binary "$BIN" \
  --build-log "$P/build.jsonl" --source-manifest "$P/sources.json" \
  --workers 3 --repetitions 10 --output "$P/pins.json"
"$PY" scripts/probe_boundary_scan_fusion_full.py run \
  --pins "$P/pins.json" --prefix "$P/cohort"
# Strict read-only replay, shell alone creates new summary output.
"$PY" scripts/summarize_boundary_scan_fusion_full.py --pins "$P/pins.json" \
  --analytical-body-ms 200 "$P"/cohort.worker[0-9].json > "$P/summary.json"
```

Pure Python negative suite rejects changed geometry/build/source, insufficient
cohort, duplicate processes/samples, incomplete/reordered arms, bad parity/work,
headroom/swap failures, missing progress and forged cached summary/native fields.
Reader rederives all metrics from native stdout, reconciles progress, and never
trusts report-side summary caches. Synthetic audit timings in tests are not data.

## Full-size result and archived replay

Three serial fresh native workers execute ten balanced repeats per party at
151,936×896. Full lookup frames, decoded records/scales, every head coordinate,
complementary reconstruction and fault-retirement controls pass. Sampled host swap
does not change. Median worker ratio of summed paired CPU is **0.797818** and wall
is **0.797802**; all measured paired deltas favor fusion in this fixture.

| Arm, median across workers of per-worker sample means | Process CPU, ms | Wall, ms |
| --- | ---: | ---: |
| Standalone packed lookup | 5.338 | 5.337 |
| Existing allocating wrap32 head | 13.519 | 13.521 |
| Original sequential lookup + head | 18.904 | 18.904 |
| Fused pair | 15.035 | 15.034 |
| Complementary head | 13.636 | 13.635 |

These separate arm medians do not define the paired ratio. Charging the other
worker's lookup and complementary head gives median summed two-worker CPU ratio
**0.898778**, about **10.12%** less service CPU, from serial samples. No concurrent
two-worker or whole-token execution was measured.

With the explicitly arbitrary **200 ms analytical body**, worker mean body-start
delays are **9.426/9.543/10.288 ms**; finish delays versus installed ready-parallel
are **9.396/9.510/10.170 ms**. The median finish penalty is **9.510 ms**. Fusion
withholds the lookup until its random-head scan finishes, while the complementary
head still follows the decoder body. A scan-level CPU win therefore does not
justify this joint-response schedule. Separate runahead retains its own analytical
control; network/codec contention and real body timings are absent.

Each mixed-fixture sampled peak is **271.766 MiB**, including both comparison arms,
source/native weights, oracles and faults. It is neither a deployed worker peak,
per-arm saving nor a Low-client result. Native setup, all raw samples and logical
traffic/work counts remain in the worker reports. Runtime joint admission,
independently operated roles and whole-SDK state/resource validation remain open.

Original pins, source manifest, three worker reports and summary are retained as
`docs/evidence/boundary-scan-fusion-full-{pins,sources,worker0,worker1,worker2,summary}.json`.
Each worker embeds its original native output and progress. Read-only replay:

Final formatting reordered the core module declaration and reformatted three
test-only expressions after measurement. The exact measured versions are archived
as `docs/evidence/boundary-scan-fusion-full-original-core-lib.rs` and
`docs/evidence/boundary-scan-fusion-full-original-probe.rs`; their hashes match the
original source manifest. Original pins, binary identity and measurements remain
unchanged. A new execution must freeze its own current source and build.

```sh
target/frontier-sdk-env/bin/python scripts/summarize_boundary_scan_fusion_full.py \
  --pins docs/evidence/boundary-scan-fusion-full-pins.json --analytical-body-ms 200 \
  docs/evidence/boundary-scan-fusion-full-worker0.json \
  docs/evidence/boundary-scan-fusion-full-worker1.json \
  docs/evidence/boundary-scan-fusion-full-worker2.json
```

Its parsed output matches the retained summary exactly. This result stays separate
from the [whole-SDK byte-dot screen](private-boundary-byte-dot-sdk.md) and
[instrumented role-cost profile](private-boundary-role-profile.md).
