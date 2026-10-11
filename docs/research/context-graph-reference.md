# Bounded authenticated exact-context graph

`ContextGraphExecution` subclasses the committed transactional tiered executor.
The numeric executor and `semantic_executor.py` are unchanged. Immutable encrypted
K/V pages already admit sequential branches: reopen one store with a caller-held
branch head, execute new rows, publish a new head. The improvement is **delta
manifest publication**, not another layer-paging comparison against replay.

## Change and contract

Every sixteenth revision publishes a full anchor. Other revisions publish an
authenticated parent commitment, new page records, new executed history, original
last logits, execution provenance and next greedy token. Prefix pages remain
unchanged and physically shared across branches. Each parent revision must be
exactly one less; restoration authenticates at most sixteen frames, reconstructs
the complete ledger and invokes the existing source/geometry/provenance checks.
Full logits remain in every revision: no approximate state or omitted numeric
lineage. The pending token is checked against saved logits and is not executed
history or KV. Explicit branch overrides remain ordinary caller-provided tokens.

The storage format has a separate AEAD domain. Pages and envelopes bind compiler
source, numeric composition, native continuation contract and public page geometry.
The key, branch identity and newest accepted head remain trusted caller owners;
storage cannot select a head. A legitimate older head can still be deliberately
opened as a fork. There is no external anti-rollback oracle or key-custody service.
One store has one writer. Branches execute sequentially, not concurrent writers.

Publication, durability, rollback and live-layer erasure use the existing atomic
generation boundary. A failure burns its attempted owner. Prior accepted heads
remain usable. Interruption at publication may leave charged orphan frames. Close
retires plaintext arrays, history, AEAD and writer lock; persisted immutable pages
remain owned by the caller's directory. No speculative reference-count GC deletes
pages still reachable by an unaccepted or separately held branch head.

Admission remains the original 128 MiB slab, 1 MiB tensor page, 2 MiB reconstructed
manifest, 2 GiB store, 32,768 files and 4,096 revisions. This does **not** make
history unbounded or increase the original horizon. Restoring a delta chain adds
bounded metadata ownership and authentication CPU. Current adapter still builds
the original complete in-memory ledger and serializes it before converting its
publication to a delta. Disk duplication falls; CPU or RSS need not improve.

## Correctness stages

`tests/test_context_graph.py` uses generated public numeric tensors without model
imports. Actual AES-GCM frames are written, synced and independently restored.
An independent whole-prefix attention oracle checks logits, divergent forks and
anchor rollover. Checks include authenticated-but-invalid pending/parent metadata,
wrong key/context/head, corrupted ancestors, publication failure, publication-point
interrupt, old-head rollback and owner cleanup. A matched full-manifest owner
checks that delta publication reduces bytes on this small fixture; this is not a
performance result. Fake compiler admission is explicit in these storage tests.
An additional generated-state gate exceeds a 256-byte resident immutable-cache
budget, verifies cache rejection and restores all persisted KV words exactly.
This establishes bounded mechanics beyond that selected budget, not a real-model
large-history performance claim. Shared-page corruption and writer arbitration
are checked separately.

`tests/test_context_graph_decoder.py` is a separate serial-only native compiled
generated Qwen2/Qwen3/Llama gate. It compares every logit and every active KV word
against uninterrupted canonical `prefix_f32` execution, across two divergent
branches, reopen after every step and the revision-16 anchor. All three generated
family gates passed in the authorized serial stage.

## Serial probe replay

From the assigned worktree, using the read-only installed native extension:

```sh
/Users/blairhudson/workspace/pllm/target/frontier-wheel-env/bin/python -c 'import sys, pathlib, pllm.runtime; pllm.runtime.__path__.insert(0, str(pathlib.Path("python/pllm/runtime").resolve())); import pytest; sys.exit(pytest.main(["-q", "tests/test_context_graph.py", "tests/test_context_graph_decoder.py"]))'
/Users/blairhudson/workspace/pllm/target/frontier-wheel-env/bin/python scripts/probe_context_graph.py --processes 3 --tokens 128 512 --steps 18 --branches 2 --output docs/evidence/research-context-graph.json
```

The probe uses the pinned existing Qwen2.5-0.5B W8A8 source and canonical numeric
contract. It runs 66 fresh workers under the exclusive frontier measurement lock,
4 GiB post-reserve admission, offline model resolution and swap-pressure guard.
Estimate: 1.1–1.6 GiB worker RSS, up to 4 GiB admitted workspace; roughly 5–15
minutes per repeat (15–45 minutes for three repeats), host dependent. Filesystem allocation can exceed ciphertext lengths.
No Cargo/maturin build, dependency installation or environment mutation is needed.
Source runtime modules are overlaid on the installed native wheel, and both source
hashes and installed build identity are recorded. Before any worker, the probe
freezes its own script, both probe helpers, all source-runtime Python modules and
installed PLLM Python/native files. It verifies those frozen bytes before and
after every worker and at cohort completion. Workers independently verify and
report actual imported module paths/hashes before model loading and after
execution, including semantic executor, model binding and immutable prefill cache.
Installed build stability remains checked by `fresh_workers`. Tiered/graph order
alternates across repeats and geometries: three starts each in the full replay.

Controls are existing **shared-store** full-manifest tiered continuation and
resident immutable eight-row `ExactPrefillCache` continuation. Both reuse the same
prefix rather than rebuilding it per branch. Candidate and tiered workers include
initial source/construction/prefill/publication, two divergent 18-step branches,
fresh-interpreter final continuation from each branch head, complete restored
attention, all reads/writes and authentication. Resident cache publication and
snapshot reconstruction/copies are charged. Every branch logit and final active
KV hash must match, not merely output token selection.

Reports distinguish ciphertext lengths, allocated filesystem blocks, full logical
manifest bytes, published manifest bytes, per-branch logical KV, physically shared
prefix-page counts, I/O/authentication combined CPU, phase CPU/wall and process RSS.
Authentication-only CPU is not separately isolated. Parity inspection reads are
separate from serving reads. Initial construction and full restored attention do
not disappear into a restore-only ratio. Keys, token IDs and prompts are never
persisted in evidence; trusted heads transit only through the coordinator process.
Parity scans have named CPU/wall/RSS phases, cumulative pre-parity cold totals,
and separate total CPU. Integrity audits also have named phases. Inclusive cold
CPU is explicitly labelled as including both; a separate non-parity/non-audit
lifecycle total prevents either from being attributed to serving work. Source
loading, initial prefill/publication, restore and complete attention remain
distinct phases. Storage read/write/authentication counters are split into restore,
initial publication and continued attention subsets; these nested costs must not
be added to their enclosing phase twice.

This cohort's resident cache fits. It does not measure histories exceeding a
resident cache, a persistent SDK conversation, provider privacy, complete serving
traffic or a whole-response speedup. Large-history/cache-eviction qualification
remains follow-up. Seven serving metrics are unknown, not zero.

## Authorized three-repeat result

Evidence: [`research-context-graph.json`](../evidence/research-context-graph.json).
All **66 fresh workers** completed: six resident workers, twelve initial issuers,
twenty-four continued branches and twenty-four fresh restarted branches. Across
six matched source/geometry/repeat groups, all twenty-four durable branch/restart
trajectories match the resident controls' full logits and final active KV hashes.
The generated compiled Qwen2/Qwen3/Llama gate additionally compares every KV word
after each of 216 branch steps across both storage formats. Focused storage,
provenance, generated decoder and existing tiered regressions: **45 tests passed**.
No failed cells occurred; no acceptance condition or process count was reduced.
Maximum observed swap growth: **0 bytes**.

The report freezes **457 source/installed/helper files** before execution and
verifies them before/after every worker and after the cohort. Actual imported
semantic executor, model binding and immutable-cache hashes are recorded in each
worker. Installed build digest:
`d4c5bc12436c5354c1974b2731c5ccd45ef6e68f35c22023c5dd057e78b0a04e`.
Source overlay identities remain separate from that installed build. Storage
order starts tiered three times and graph three times over the six groups.

Tables use separate medians, CPU seconds and decimal MB. Continuation/restart rows
pool six branch observations per geometry/format, not six independent sources.
Source loading is outside attention windows but remains charged in lifecycle
totals. Publication CPU is a nested subset of execution CPU; do not add it twice.

### Initial setup and complete attention

| CPU / allocation | 128 resident | 128 tiered | 128 graph | 512 resident | 512 tiered | 512 graph |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Initial source/construction CPU | 12.047 | 12.125 | 12.121 | 12.368 | 11.964 | 12.452 |
| Initial owner creation CPU | 1.102 | 2.435 | 2.470 | 1.199 | 2.610 | 2.565 |
| Initial full prefill CPU (durable formats include publication) | 8.377 | 8.563 | 8.532 | 38.109 | 38.431 | 38.159 |
| Separate resident cache publication CPU | 0.010 | — | — | 0.050 | — | — |
| Nested initial storage-write/authentication CPU | — | 0.051 | 0.053 | — | 0.174 | 0.170 |
| Initial published ciphertext/manifest MB | — | 3.770 | 3.770 | — | 13.255 | 13.255 |
| Initial issuer/process pre-parity peak RSS MB | 1,100.825 | 1,100.710 | 1,100.825 | 1,300.873 | 1,279.066 | 1,278.181 |
| Resident immutable cache payload MB | 3.753 | — | — | 13.191 | — | — |
| Resident branch restore CPU | 0.00214 | — | — | 0.00906 | — | — |
| Resident full 19-step branch attention CPU | 3.332 | — | — | 3.840 | — | — |

Resident peak includes cache publication and both branches in one process. Durable
initial issuer and branch processes have separately owned peaks and source loads;
these are not a matched whole-process memory-reduction claim.

| Eighteen-step durable branch | 128 tiered | 128 graph | 512 tiered | 512 graph |
| --- | ---: | ---: | ---: | ---: |
| Fresh source/construction CPU | 12.219 | 11.993 | 12.543 | 12.527 |
| Authenticated owner/restore CPU | 2.530 | 2.408 | 2.645 | 2.560 |
| Complete attention/publication CPU | 4.472 | 4.580 | 5.701 | 5.753 |
| Complete attention/publication wall seconds | 2.869 | 2.944 | 4.072 | 4.168 |
| Nested reads/authentication CPU | 0.774 | 0.790 | 1.409 | 1.420 |
| Nested writes/authentication CPU | 0.325 | 0.347 | 0.356 | 0.387 |
| Authenticated bytes read MB | 61.259 | 61.259 | 231.313 | 231.313 |
| New ciphertext/manifest bytes MB | 12.755 | 11.642 | 13.479 | 11.683 |
| Pre-parity peak RSS MB | 1,062.035 | 1,064.583 | 1,062.732 | 1,063.084 |
| Separate full-KV parity scan CPU | 0.077 | 0.083 | 0.120 | 0.116 |

Both formats retain the same initial **96/384 encrypted prefix pages** throughout
each branch. Neither republishes prefix K/V. Full attention reads remain unchanged;
the candidate changes only metadata publication. Complete branch CPU increases
**2.41%/0.92%**, and write/authentication CPU increases **6.50%/8.81%**. The adapter
still serializes the full ledger before constructing a delta, so disk savings do
not eliminate duplicate Python serialization work.

### Fresh restart and final store

| Fresh restarted one-step branch | 128 tiered | 128 graph | 512 tiered | 512 graph |
| --- | ---: | ---: | ---: | ---: |
| Fresh source/construction CPU | 12.098 | 11.906 | 11.992 | 12.487 |
| Authenticated owner/restore CPU | 2.597 | 2.524 | 2.539 | 2.629 |
| Complete resumed attention/publication CPU | 0.395 | 0.380 | 0.454 | 0.462 |
| Authenticated total bytes read MB | 4.353 | 5.569 | 13.838 | 15.055 |
| Manifest bytes authenticated MB | 0.738 | 1.954 | 0.778 | 1.995 |
| New ciphertext/manifest bytes MB | 0.770 | 0.641 | 0.810 | 0.641 |
| Pre-parity peak RSS MB | 1,060.487 | 1,062.715 | 1,061.143 | 1,062.339 |
| Separate full-KV parity scan CPU | 0.075 | 0.078 | 0.120 | 0.126 |
| Final branch logical KV MB | 3.613 | 3.613 | 13.050 | 13.050 |
| Final complete store ciphertext/file MB | 30.820 | 28.335 | 41.834 | 37.902 |
| Final complete store physical allocation MB | 37.782 | 35.348 | 49.963 | 46.088 |

Restart inherits **960/1,248 shared encrypted K/V pages**, reads all required
attention rows and appends only its new token. Restoring revision 18 authenticates
the revision-16 anchor plus two deltas rather than one full manifest: manifest
reads rise **164.94%/156.37%**, and total reads rise **27.95%/8.79%**. Full-prefix
attention, token provenance, trusted head and native/source lineage still match.

Final stores include initial construction, both complete nineteen-step branches,
every accepted old revision and immutable shared page. Ciphertext/file lengths fall
**8.06%/9.40%**; allocated filesystem blocks fall **6.44%/7.76%**. Both formats have
the same file counts (1,959/2,247). No GC or uncharged deletion creates this result.
Resident immutable block ownership remains stronger for already-resident reuse;
durable metadata savings do not improve its in-process continuation arithmetic.

Median complete non-parity/non-audit lifecycle CPU, charging each format's actual
owner/process pattern: resident single process **28.632/59.259 seconds**; five
tiered issuer/branch/restart processes **94.680/126.135 seconds**; corresponding
graph processes **93.121/126.571 seconds**. Source loading dominates this durable
lifecycle. The small mixed graph/tiered cold-CPU differences are not a demonstrated
serving speedup. Inclusive CPU, parity scans and integrity-audit phases remain
available separately in raw evidence; no parity scan is silently charged as
serving attention CPU.

**Decision:** bounded exact branch/restart graph passes; delta manifests deliver
an incremental durable-storage reduction against the strongest existing shared-page
tiered control. CPU/RSS and restored-read gates do not improve. Unbounded histories,
real model histories exceeding resident cache, direct delta construction avoiding
full-ledger serialization, safe caller-accepted-head GC, multi-writer branches,
SDK lifecycle/role separation and complete serving metrics remain open.
