# Low-tier hypothesis 1: disk-backed exact decode trie

**Decision: OPEN, not promoted.** Bounded synthetic state/storage screen passes;
native decoder parity, realistic cache hit rate, complete ALL-LINK traffic,
whole-client RSS <=268435456 bytes, and >=90% remote body work are unperformed.
No source/numeric production change. No model, checkpoint, provider, torch import,
build, installation, timing or RSS cohort occurs in this probe.

## Distinct opportunity and strongest control

`runtime/client.py` samples each step, leaves final max-output selection pending,
and executes earlier selections with `decode_step`. Initial prefix lookup happens
before this loop. Completed generated-prefix publication happens after provider
completion acknowledgement; only `evaluated_ids[:runtime.position]` enter its key.
`ExactPrefillCache` uses independent snapshots, immutable eight-row shared KV
blocks, a payload limit and entry ceiling. Generated reuse requires qualified
`canonical_incremental`, ownerless state under `prefix_f32`, not arbitrary response
state. Its payload budget is not an RSS guarantee.

Candidate differs by querying **inside every subsequent generation's decode loop**
and retaining branch nodes beyond resident eviction/restart. Root initial reuse
is already present in both arms. A live conversation already owns exact KV and
next logits; disk restoration adds no intrinsic traffic win there. Checkpoints
restore one caller-held response/history/pending head. Context graph already
shares immutable pages and supports exact divergent branches, but does not discover
cached generated children per sampled step. This hypothesis adds lookup policy,
not novel branch persistence. The strongest eventual control must enable current
generated-prefix qualification too; repeats that initial lookup already serves
earn no extra credit. Synthetic baseline here is current initial-prefix reuse
for identical short prompts, not conversation replay.

## Required production contract (not implemented by this toy)

1. Identity: domain/version, immutable source/bundle binding, numeric composition,
   compiler/native continuation contract, canonical causal arithmetic, full-KV
   geometry, executed token IDs/position and execution/verification provenance.
   Tokenization/chat rendering must produce identical IDs; token text alone is
   insufficient. Unsupported retained/windowed/tiled/Freivalds paths reject until
   independently admitted. Do not persist process-local state-basis seals as if
   portable; restore must authenticate then revalidate/reissue native provenance.
2. Store complete next-token logits and executed KV, not only sampled output.
   RNG belongs to each request. Run same sampler, temperature/top-p, filters,
   penalties and stop rules with same draw advancement on hit and miss. Those
   policies do not alter raw executed-state identity unless execution itself
   depends on them; policy/output cache is a separate contract. Never replay cached
   selection. EOS terminates without executing EOS; final max-output selection
   remains pending. Continuation must execute pending selection exactly once.
3. On child hit, authenticate complete numeric state before exposing it. Immutable
   prefix pages may be shared; mutation gets private append owners. On miss,
   restore parent KV (possibly lazily) and execute the selected child with **fresh**
   ordinary provider material. Do not reserve/burn the skipped rows unnecessarily;
   already reserved or failed rows remain charged. Never serialize credentials,
   tickets, masks or leases. Completed/acknowledged native execution is required
   before a response-independent node becomes discoverable.
4. AEAD domains bind source/numeric context, page geometry, executed path, revision
   and manifest/page commitments. Caller owns key and accepted index/head freshness;
   disk cannot nominate newest head. Bound index, reconstructed metadata, read
   chain, pages, files, history and disk before allocation. Plaintext indexes reveal
   token paths, so keep private and encrypted when durable. One writer or explicit
   arbitration; no unsupported concurrent publication. Old trusted heads may be
   deliberate forks, not silent rollback.
5. Sync pages/manifests/directory before publishing head. Failure publishes no
   partially accepted state, retires attempt, burns attempted material and preserves
   prior accepted heads. Charge orphan bytes. Close releases plaintext and key
   owners; persisted files remain caller-owned. GC needs all accepted/external head
   references, writer exclusion and authenticated reachability. No speculative
   deletion of shared pages. Private file permissions are not key custody or a
   multi-tenant security result.

## Implemented screen and exactness boundary

`scripts/probe_low_decode_trie.py` uses integer causal attention with three token
choices. Every executed row contains key/value; next scores use all preceding
rows. Whole-prefix oracle in `tests/test_low_decode_trie_probe.py` recomputes keys,
values and attention independently, algebraically distinct from incremental code.
Independent sampler uses cumulative boundaries. Sixteen seeds check complete KV,
logits, output, pending token and RNG state for divergent warmup, mixed and hot
branches. This is meaningful non-table attention correctness, **not decoder parity**:
no transformer layers, float32 reduction, native execution or provider protocol.
Production pending/EOS/tokenization behavior remains a native gate (toy has no EOS).

Probe actually writes AES-GCM frames with fresh random nonces, authenticates reads,
syncs disk and keeps caller-trusted heads. Same token-prefix row pages are physically
shared. Bounds: 128 indexed nodes, depth 32, 16 KiB encrypted frame, 2 MiB logical
writes. Tests cover corrupted shared page, wrong key/binding/revision, authenticated
invalid position, false pending KV, publication failure, owner close and bounds.
No durable index/reopen API, automatic GC, malicious valid-key producer validation,
portable native seal or security qualification is supplied. Test fixtures do not
assert that AEAD proves execution. Random ciphertext differs per replay; counters,
state counts, and frame lengths are deterministic. Filesystem allocation is host
dependent. Temporary-store teardown occurs after ledger collection, not as savings.

## Ownership and cost ledger

Evidence records logical encrypted reads/writes, retained file lengths, allocated
filesystem blocks, rows/manifests/nodes and orphan bytes. Serialized row and manifest
plaintext plus 28-byte nonce/tag per frame reconcile encrypted writes. Serialized
trusted-index bytes are an ownership proxy, explicitly not Python RSS. Initial root execution and
publication are included; each miss appends a row and manifest. Every toy hit reads
manifest plus all parent rows, so full restoration work is not silently free.
120 sampler calls per 24-generation workload are charged as counts in both arms.
Baseline performs 2 setup rows plus 96 decode rows (five selections, four executions
per generation); candidate setup remains two rows. Final pending selection is never
published. Local setup and publication are absent from baseline disk ledger because
baseline keeps root in memory. No inferred network-byte percentage from row savings.

Additional RSS owners: source/bundle/tokenizer/native imports, runtime execution,
restored KV/logits, old/new snapshot overlap, lookup index/trusted heads, Python
objects/history, AES-GCM key/ciphertext/plaintext buffers, sampler, metadata parsing,
resident prefix cache, mapped/file-backed resident pages and allocation overhead.
Toy disk bounds do **not** sum these owners or establish whole-client memory.
OS filesystem cache and physical I/O are separate unknown owners. Storage backup,
distribution or cross-device access would add links; client-private disk alone has
no provider traffic, but setup/control/reservation/acknowledgement traffic cannot be
assumed zero. CPU, authentication CPU, physical I/O, real remote fraction, ALL-LINK,
whole-client RSS and missing costs stay null in evidence.

## Workload break-even, including cold misses

For R generations and D executed decode rows each, baseline root reuse executes
RD rows. Candidate executes RD-H rows, where H counts authenticated cached child
hits, including zero-hit cold fill. Conditional total-link saving is
`sum(hit avoided complete-link bytes) - extra setup/control/retirement/distribution
link bytes`. Simplified constant-row form: `H*b > A`; hit fraction threshold
`H/(R*D) > A/(R*D*b)`. Both b and A are unknown here. Real per-row cost varies by
position and provider inventory; use full charged traffic in a matched experiment.
Disk I/O/authentication/sampler/index CPU and RSS are additional acceptance gates,
not units to subtract from network bytes. Miss execution, failed publications,
issued unused material, disk page faults and setup stay in lifecycle totals.

Actual bounded run: 24 generations per arm, 96 baseline decode executions. Greedy
repeat and repeated stochastic seed each execute 4 cold misses, reuse 92 rows,
retain 6 row pages and 5 nodes; 2,158 encrypted bytes written, 53,130 read. Distinct
seeds 0..23 execute 45 misses, reuse 51 rows, retain 47 pages and 46 nodes; 23,394
bytes written, 26,217 read. Observed filesystem allocation: 45,056 / 380,928 bytes.
Each workload checks 96 complete states containing 1,152 KV/logit integer words;
across three workloads: 288 states, 3,456 words. These are toy integer words, never
native float32 bits. Repeated-seed RNG equality does not establish independent
stochastic demand. These counts are not real-model traffic or cache-hit forecasts.

With one previously cached iid trajectory and per-depth token probabilities p,
prefix collision survival is `c^d`, `c=sum(p_i^2)`. Expected saved decode rows are
`sum(c^d, d=1..D)`. Uniform three-way sampling and D=4 give
`1/3+1/9+1/27+1/81=40/81` rows, versus four for a deterministic repeated path.
Context-dependent distributions require actual prefix-product probabilities;
multiple prior trajectories share ancestors, so do not count them as independent
copies. Novel prompts, high entropy, eviction, changed model/numerics, policy-
induced divergence and one-shot workloads suppress hits. Greedy or repeated seed
is favorable synthetic stress, not representative measured demand.

## Reproduce (lightweight concurrent-safe)

```sh
target/frontier-sdk-env/bin/python scripts/probe_low_decode_trie.py \
  --output docs/evidence/research-low-decode-trie.json
target/frontier-sdk-env/bin/python -m pytest -o pythonpath=. -q \
  tests/test_low_decode_trie_probe.py
target/frontier-sdk-env/bin/python -m ruff check \
  scripts/probe_low_decode_trie.py tests/test_low_decode_trie_probe.py
```

Evidence hashes probe, tests, this note, relevant implementation and three requested
reference documents. No parent-only native command is provided: integrating a
trustworthy trie with compiled decoder and fresh provider inventory is substantive
unimplemented work, not an existing runnable serial option. Parent next gate:
implement generated Qwen2/Qwen3/Llama native logit/KV-bit oracle across stochastic
forks, restart, pending continuation, EOS, failure and new material; run exclusively
using `scripts/frontier_probe_support.py:fresh_workers`, then matched ALL-LINK/RSS/
remote-body workload with installed prefix reuse and generated reuse controls.
Existing context-graph result reports roughly 1.06–1.30 GB whole-worker peaks;
it establishes neither this client's 256 MiB bound nor impossibility of a thinner
runtime. Existing conversation restore near RAM traffic floor cannot substitute
for decode-trie workload qualification.
