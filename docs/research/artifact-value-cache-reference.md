# Network-value public cache: bounded metadata screen

Status: **synthetic eviction screen only; whole-client Low unknown**. No production
policy change, SDK execution, checkpoints, providers, torch, performance timing,
builds or installs. H5 targets survival at fixed disk capacity, not first-cold
compression. Existing bounded LRU, authenticated repair, directory flock and
cross-process single-flight are controls, not discoveries.

## Frozen policy and exact replay

`scripts/probe_artifact_value_cache.py` uses only Python standard library. Each
fixture embeds source-locked public manifest **summaries**, ordered unique
first-use object identities and explicit request/model/placement/worker traces.
These are not exported SDK manifests. Synthetic object identities and source
locks are invented fixture commitments, not measured checkpoint commitments.
Recorded-size cohort uses original source lock, content digests and raw/encoded
sizes from `research-artifact-reuse-qwen3.json`; its access sequence and 2,048-byte
manifest / 256-byte control allowances remain synthetic. Its head label denotes
an 8 MiB embedding slice, not an actual complete head. No actual user workload
is asserted. Private prompt/KV fields are rejected.

Value policy is GreedyDual-style aging with score
`eviction_floor + min(past_access_count, frequency_cap) * encoded_bytes/raw_bytes`.
It can bypass newly downloaded objects. Q32 integer scores and lexical identity
ties make replay deterministic. Floor advances on eviction; count uses only past
public accesses, including misses. Candidate caps 1/4/16 are calibrated on fixed
training placements p0–p3, summed across 6/7/8/10/16 MiB capacities, then frozen
before held-out p4–p7, scan, phase change, model-switch, fault and worker traces.
Held-out cache starts empty; no training residency or future accesses leak.
Simple capped LFU with recency tie-break is separate comparison, not LRU.

```sh
target/frontier-sdk-env/bin/python -m scripts.probe_artifact_value_cache \
  --output docs/evidence/research-artifact-value-cache.json
target/frontier-sdk-env/bin/python -m scripts.probe_artifact_value_cache \
  --verify docs/evidence/research-artifact-value-cache.json
target/frontier-sdk-env/bin/python -m pytest -o pythonpath=. \
  tests/test_artifact_value_cache_probe.py
target/frontier-sdk-env/bin/ruff check scripts/probe_artifact_value_cache.py \
  tests/test_artifact_value_cache_probe.py
```

Evidence embeds complete inputs/results, calibration scores, independent tiny
oracle and source/own-code/reference hashes. Replay compares whole JSON exactly.
Do not hash evidence into itself. No clocks or process-memory sampling included.

## Accounting and boundaries

- Object network uses **encoded transfer bytes**; disk uses raw payload bytes.
  All per-import manifest/control bytes remain charged, warm or cold, all workers.
- LRU gets full disk ceiling. LFU/value reserve 8,256 bytes (128 fixed 64-byte
  public slots plus one floor/schema/owner header) inside same ceiling; insufficient ceilings reject. Object table
  bounded to 128, events to 4,096. This is packed-index design accounting, **not
  Python heap/RSS measurement**. Implementation Python dictionaries are not packed.
  Score rounds density downward to Q32; score/floor fit 128 bits under fixture
  event and 4 GiB length bounds. Slot layout allowance: 32-byte key, 16-byte score,
  8-byte recency, 1-byte count and 7 bytes flags/padding. Production indefinitely
  running counter/aging reset remains unimplemented; no unbounded history claim.
- Each policy gets additional 512-byte once-per-replay publication allowance on
  Preparation→Client, including ample-cache/cold cases. This is modeled policy
  distribution, not measured serialized production config. Index is local;
  64-byte logical index update per public access is charged separately. Atomic
  replacement/journal/filesystem amplification not measured; in-place fixed-slot
  production ownership must be designed and audited before using this allowance.
- Hits charge full authenticated raw local reads. Corrupt resident reads charge
  full raw read, fail closed, then encoded redelivery; missing reads redeliver.
  Writes charge raw admitted payload only; policy bypass avoids cache publication.
  Existing download/import verification remains required even when not retained.
- Owner is one shared public cache. Sequential events define an admitted fetch
  linearization consistent with existing download flock, not concurrency stress
  proof. Duplicate workers still download every manifest/control; ample cache
  avoids duplicate object download. Different valid lock schedules can change
  recency and winners. Batched lookahead/pending-copy lifetimes are not modeled.
- Private immutable native snapshots, tokenizer source/index, compiled checkpoint
  storage, publisher raw/encoded caches, OS file cache and download scratch remain
  **separate disk/RAM owners**. Public eviction never invalidates an admitted
  snapshot. Existing native-snapshot research supplies ownership contract, not
  memory qualification for this policy. No extra remote-body or numeric work is
  changed. All other directed links are held unchanged by hypothesis, not measured.
  TCP/TLS/kernel/retries/publication CPU and whole-response denominator unknown.

## Break-even and adverse cases

Net modeled win requires `avoided_encoded_redelivery > 512` bytes, **after** losses
from index capacity and ranking decisions. Equivalently, for fixed access stream,
`sum_i(encoded_i * (LRU_misses_i - value_misses_i)) > 512`. With equal-size encoded
objects, required miss-rate reduction is `512 / (accesses * encoded_size)`; unequal
objects require weighted byte misses, not aggregate hit rate. Additional actual
policy publication/controls must replace allowance before promotion. Projected
whole-client all-link percentage cannot use historical 25% bundle share as a new
measured denominator. No churn or first-cold stream cannot pay overhead.

Report includes ample-cache, exact-repeat, cache-boundary, phase-change and
exhaustive three-object six-access adversarial cases. Tiny oracle enumerates
retained subsets (including bypass), prices encoded misses at raw capacity and
uses future knowledge **only for lower bound**, never policy calibration. Tiny
search explicitly charges index overhead; it is not realistic MiB workload.
Arbitrary post-training workload can regress: retention of formerly hot expensive
objects and loss of 8,256 bytes capacity can force repeated misses. No universal savings
or competitive ratio established. One adversarial fixed-size trace is a lower
bound on worst regression, not global worst over all streams.

## Next gate

### Verdict and fixed-capacity byte results

Calibration selects cap **1** (density/aging, no extra frequency protection);
training covered-byte sums: 107,926,880 / 111,826,880 / 119,626,880 for caps 1/4/16.
Net savings below include 512-byte publication allowance and index capacity loss.
Positive means candidate saves; these are **not measured all-link SDK results**.

| Synthetic trace | 6 MiB | 7 MiB | 8 MiB | 10 MiB | 16 MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| Placement switching | 66,299,488 | -320,512 | -400,512 | -1,040,512 | -512 |
| Shared objects after scan | -11,400,512 | 5,699,488 | 5,699,488 | 5,699,488 | -512 |
| Model switching | 3,899,488 | -512 | -512 | 15,599,488 | -512 |
| Phase change | -75,720,512 | -880,512 | -1,760,512 | -512 | -512 |

First-cold all capacities: exactly same object bytes, **512-byte net loss**. Exact
repeat at 7 MiB loses **880,512 bytes** because 8,256-byte index prevents original
7 MiB working set from fitting. Phase-change worst among listed main traces loses
**75,720,512 bytes** at 6 MiB: original shared pair exactly fills LRU capacity,
but no longer fits candidate; subsequent small-object phase does not repay loss.

Recorded Qwen3 slice sizes, synthetic 24-import trace:

| Disk ceiling | LRU encoded object bytes | Value encoded object bytes | Net saved bytes |
| --- | ---: | ---: | ---: |
| 12 MiB | 121,848,468 | 38,033,385 | 83,814,571 |
| 19 / 20 / 24 MiB | 10,154,039 | 10,154,039 | -512 |

Tiny exhaustive search over 729 six-access traces: worst found **4,512-byte
regression** at 20,000-byte ceiling, sequence A B B B B B. LRU object bytes
10,000, value 14,000, exact full-capacity oracle 10,000. Every tiny object is
10,000 raw bytes; A costs 9,000 encoded and B/C 1,000. Index consumes almost one
slot, expensive A protection forces repeated B bypass. Neither main nor tiny
trace establishes global worst-case bound; repeat cheap misses can accumulate.

**Verdict: reject broad promotion of this frozen policy.** Conditional churn wins
exist, but training winner regresses severely on held-out boundaries/phase change.
No actual workload, complete-head measurement, exact numeric SDK execution or
<=256 MiB whole-client proof. Next gate must first demonstrate narrower robust
eviction/admission design on declared source-locked real manifest traces, without
retuning on these held-out results.

Parent only: opt-in `fresh_workers` **serial** experiment after overlapping screens
finish. First capture actual exported manifests, codec lengths and independently
declared public model/placement request traces; freeze recipe and source hashes.
Integrate packed index under existing flock/single-flight, audit authenticated
repair, crash recovery, count bound, batching, publication owners and multiple
lock schedules. Then fresh alternating admitted SDK clients, at least three
replicates, same source/numeric/private/remote-body contracts and original artifact
commitments, all directed bodies including policy distribution, failures and
imports; every client peak <=268,435,456 bytes plus admission/swap guard. No such
mode is launched or implemented by this metadata probe. Conditional survivor only
if churn replay wins; whole-client Low and all-link benefit remain unknown.
