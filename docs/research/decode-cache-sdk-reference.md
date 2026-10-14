# Durable exact decode cache owner: integration contract

**Explicit SDK integration with separately measured whole-response tradeoffs.**
`pllm.state.DecodeCache` and `DecodeCacheHead` expose caller-owned encrypted
storage through `OpenAI(decode_cache=...)` and `AsyncOpenAI(decode_cache=...)`.
The Pipeline selects `ClientPrefixReuse(generated_prefixes=True, decode=True)`
with fixed one-selection demand/on-demand epochs and canonical `prefix_f32`
execution. Compiler admission binds an additional `decode_max_bytes` owner
envelope; storage and SDK failure-path validation precede matched benchmarks.
The [completed Qwen2.5 cohort](decode-frontier-sdk-results.md#exact-encrypted-decode-cache)
reduces three-request traffic 10.96% with repeated seeds and 4.73% with distinct
seeds, while increasing CPU, client RSS and complete-cell time. All observed
client peaks fit 256 MiB; workload-specific hit rates and runtime admission
retain their separate scopes.

## Exact API

```python
DecodeCache(
    directory: str | Path, key: bytes, *, head: DecodeCacheHead | None = None,
    max_nodes: int = 512, max_staged: int = 32, max_pages: int = 8192,
    max_files: int = 32768, max_disk_bytes: int = 256 << 20,
    max_index_bytes: int = 2 << 20, max_snapshot_bytes: int = 64 << 20,
    max_transient_bytes: int = 128 << 20, max_tokens: int = 4096,
)
cache.claim(client: object) -> None
cache.bind(compiled, runtime) -> None
cache.stage(token_ids: Sequence[int], snapshot: RuntimeSnapshot,
            logits: numpy.ndarray) -> str | None
cache.commit(tickets: Sequence[str], *, acknowledged: bool) -> DecodeCacheHead
cache.discard(tickets: Sequence[str] | None = None) -> None
cache.lookup(token_ids: Sequence[int]) -> tuple[RuntimeSnapshot, numpy.ndarray] | None
cache.lookup_child(parent_ids: Sequence[int], token_id: int) -> tuple[RuntimeSnapshot, numpy.ndarray] | None
cache.head -> DecodeCacheHead | None
cache.storage_metrics -> dict[str, int]
cache.close() -> None

DecodeCacheHead(revision: int, context: str, name: str, digest: str, bytes: int)
head.to_spec() -> dict
DecodeCacheHead.from_spec(spec: dict) -> DecodeCacheHead
```

Token and ticket sequences must concretely be lists or tuples; token IDs must be
plain integers in the bound vocabulary. No implicit text tokenization occurs.
Context manager supported; copying/pickling live owner is rejected. Head is frozen.

Constructor retains key but opens no directory until first bind. With no trusted
head, bind requires a new directory, never discovers a disk-selected latest index.
For restart, caller supplies the key **and** newest accepted head. Older accepted
heads remain explicit supported forks. Bounds are part of context identity;
reopening with changed bounds or compiled binding fails closed. Root directory and
all persisted ciphertext files remain caller-owned; caller must keep directory
ancestors and key private. This is not a hostile-directory, multi-tenant key-custody,
or anti-rollback service.

`claim(client)` enforces one SDK owner by object identity until close. SDK must call
it before attaching. Directory writer lock also excludes separate live owners,
including readers. Every call is serialized by owner's lock. Bind may replace the
borrowed runtime between requests, but only with the same validated compiled
source/numeric/native continuation context. Closing drops the borrowed runtime;
it does not reset or erase caller's runtime, input snapshots, or returned state.

## SDK ownership and publication

1. Explicit opt-in configuration accepts the caller-owned cache and calls
   `claim(client)`. SDK never implicitly constructs stores or chooses keys/heads.
2. Lazily loaded compiled model calls `bind(compiled, ordinary_runtime)`. Runtime
   must already have the native generated-prefix contract enabled. No checkpoint
   worker, local body weight loading, runtime construction, or provider callback
   occurs inside the cache.
3. Each completed native step can stage its executed IDs, independent qualified
   snapshot, and exact raw next logits. Completed owner-free prefill and native
   `canonical_incremental` snapshots qualify; response-owned/incremental-only,
   retained/windowed/hybrid/tiled/Freivalds/nonlinear/capsule states reject.
   IDs must have exactly snapshot position: exclude EOS and pending final selection.
4. Before issuing provider rows for a sampled next token, call `lookup_child` with
   the executed parent path and freshly sampled token. On hit, restore the returned
   independent snapshot into the ordinary runtime and use returned raw next logits.
   On miss, execute using fresh ordinary provider material; stage completed result.
   Keep RNG advancement, filters, penalties, sampling and stop rules unchanged.
   Never replay cached sampled tokens. Do not reserve rows merely to serve hits.
5. Only after authenticated provider completion/retirement, call
   `commit(tickets, acknowledged=True)` for a bounded batch. This boolean is an
   explicit **trusted integration assertion**, not a cryptographic receipt checker.
   The cache cannot observe SDK's retirement protocol. Before ACK, staged state is
   never discoverable. Failed/cancelled turns call `discard()`; attempted provider
   material remains burned by SDK, not managed by this owner.
6. Caller durably accepts returned head. No latest index file is written. If commit
   I/O fails, selected tickets retire and previous head remains accepted. Close on
   SDK retirement releases process-local plaintext indexes, AEAD/key and runtime
   references. Cancellation must drain synchronous work before close/discard.

Commit requires a nonempty unique valid ticket batch. A false ACK assertion rejects
before publication and leaves tickets private; SDK must discard failed turns.
Commit failure retires selected tickets. Stage failures expose no ticket and leave
charged ciphertext only. Nodes are not response IDs and never include pending
selections, response metadata, credentials, leases, masks, tickets or inventories.
An exact already-published or already-staged replay returns None after validation;
it is not a new successful state and writes no node manifest. SDK should not stage
root/child hits at all, and must skip commit when its ticket list is empty. Thus an
all-hit response needs no disk write or trusted-head publication.

Private SDK cancellation finalization uses
`cache._restore_head(previous: DecodeCacheHead | None) -> None`. Under the single
active response guard, SDK must first check current head equals this response's
just-published head, then pass its explicitly remembered prior trusted head. None
restores an empty published index; a prior head must match context and have an
earlier revision (or exactly equal current head). Its bounded encrypted index is
authenticated before adoption. Stages clear even on authentication failure;
failed restore does not adopt any partial/damaged index. SDK must close/retire
owner if rollback authentication fails. No newer/orphan file is deleted and no
attempted read/write or disk charge is refunded. This is not automatic arbitrary
on-disk head discovery or a public rollback service.

## Storage, authentication and limits

`runtime/decode_store.py` specializes the existing `TieredExecution` frame
commitment/AES-GCM protocol and uses existing `LayerCache`, `RuntimeSnapshot`,
full-snapshot validation, native continuation admission and snapshot-basis minting.
It does **not** instantiate `TieredExecution`/`ContextGraphExecution` execution.
Specialized frame I/O supplies configurable tighter bounds, private file modes,
no-follow bounded reads, failed-attempt charging and no speculative orphan deletion.

Immutable one-row key/value pages are indexed once by source/context, layer, offset,
geometry and exact float32 bytes. Format v2's compact encrypted index holds only
`tokens + node-record commitment` per node and one shared global page table.
Complete ordered page references, exact logits, basis, executed position and path
commitment live in independently authenticated **disk-backed node manifests**.
No full prefix reference list or vocabulary-sized logit vector remains in a staged
index entry. Staged page maps hold **newly introduced pages only**. One shared
private pool retains their records until the staging batch drains, including a
discarded creator's pages needed by a surviving child. Partial commit authenticates
selected node manifests against that pool, so children do not depend on publication
of a staged parent. Private pool metadata clears when no stages remain; persisted
files are never deleted. Divergent prefixes share the same immutable page objects.
Index-to-node-to-pages authentication binds paths to complete ordered state.
Context binds compiled source/bundle/numeric binding, native
continuation digest, exclusive full-KV geometry, vocabulary, version and bounds.
Index revision is in AEAD associated data; head commits full ciphertext.
Schemas are `pllm.decode_index.v2` / `pllm.decode_node.v2`, with AEAD domain
`pllm.decode_cache.v2`. Old prototype v1 directories/heads reject rather than being
silently reinterpreted; no automatic migration or deletion is provided.

Every lookup authenticates node and all referenced page ciphertexts, validates
numeric extent/finiteness and page content identity, re-admits native continuation,
then mints a fresh process-local execution basis. Old seals are never serialized or
deserialized as authority. AEAD authenticates trusted producer's state; it does not
independently prove model execution against a malicious valid-key producer.
No partial snapshot escapes; failed numeric reconstruction zeroes allocated arrays.
Successful snapshots/logits are independent caller-owned arrays.
Lookup results remain unpackable `(snapshot, logits)` tuples and also expose
`.snapshot` / `.logits` fields for SDK restoration.
The entire ordered node reference list, source/path/position, layer/start/kind and
global page-index membership validate **before allocating any snapshot/logit array**.
Binding borrows `runtime._generated_prefix_contract`, checks its immutable source,
composition and source-schedule digest against the validated compiled binding,
and never constructs a second continuation/native admission owner. Rebinding drops
the old borrowed contract reference in favor of the current runtime's same-context
contract. Existing runtime/source/numeric admission remains mandatory.

Bounds cover revisions (4096), nodes, stages, rows/pages, files, token extent,
index encoding and reconstruction, snapshot payload, logical transient payload,
and logical retained disk bytes. Page plaintext is capped at existing 1 MiB limit.
Metadata fanout is conservatively priced before disk writes; index encoding is
checked again. One-row layout deliberately has higher file/I/O overhead than
larger block paging and can reject long/deep models before snapshot limits.
No eviction/GC is implemented: reaching any limit rejects rather than silently
dropping externally trusted heads or orphan bytes. All pages/indexes and directory
are fsynced before the single trusted-head publication point.

Metrics: `disk_bytes`, `disk_files`, `read_bytes`, `written_bytes`, `node_count`,
`page_count`, `staged_count`, `hits`, `misses`, `publications`,
`peak_transient_bytes`, `pending_page_count`, `owner_metadata_bytes`,
`binding_owner_bytes`, `unpublished_bytes`, `unpublished_files`.
Attempted damaged reads and writes remain charged. Partial-write bytes are charged
conservatively for the live owner; reopening charges actual retained file lengths.
`unpublished_*` includes private stages, orphans **and older caller-trusted heads**
not reachable from the current index; it is not permission to delete them.
Close preserves final metrics/head while clearing plaintext metadata.

`max_transient_bytes` must cover `max_snapshot_bytes + 4*max_index_bytes + 4 MiB`.
That constructor minimum is **not** the complete operation admission: store also
prices actual metadata, current source binding and one largest transient manifest.
For position p and R=`2*layers*p`, page plaintext B, vocabulary V and index cap I:

- Snapshot payload S=`p*2*layers*B + 4*V`, constrained by `max_snapshot_bytes`.
- Node-manifest bound M=`4*V + 66*R + 4096`, constrained by I and 32768 references.
- Encoded-index fanout preflight: `1024 + 384*unique_pages + sum(512 + 9*path_length)`;
  actual encoded index is checked again against I. Full prefix refs are not counted
  once per resident node because they now live on disk, not because they are free.
- Python metadata envelope: 65536 base + **1536 per unique page object** + **96 per
  dictionary slot** (committed/private/staged-introduced/temporary maps explicitly
  charged) + `sum(1536 + 44*path_length)` per resident/staged node. Prospective new
  page objects/map slots/node tokens are priced before writes. Shared objects are
  counted once; duplicate map entries are still charged.
- Borrowed binding envelope: unique immutable continuation/source/composition byte
  lengths + `4*encoded_runtime_schedule_length + 65536`, including conservative
  temporary public-binding encoding allowance. No duplicate native handle is owned.
- Stage/lookup envelope: `S + metadata + binding + 4*I + 4*M + 128*R + 4*B`.
  This includes one complete transient reference list/logit manifest and its
  encoding/AEAD owners. Index restoration/publication separately prices incoming
  parsed index metadata + existing owner metadata + binding + `4*I`.
- Before MessagePack decoding allocates containers, a bounded streaming preflight
  rejects nesting beyond 32, containers beyond 32768 elements, non-ASCII metadata,
  oversized strings/binary payloads and total parsed-object cost beyond remaining
  transient budget. It charges every serialized object (including duplicate map
  keys): map `256 + 96*entries`, array `80 + 16*elements`, ASCII/binary `80 + bytes`,
  scalar 32. Thus small encodings with huge empty-container fanout cannot treat
  serialized bytes as Python memory. The parsed-index peak is additionally recorded.

All these envelopes must fit `max_transient_bytes`. They are explicit conservative
logical/Python-owner estimates, **not measured RSS**, native allocator proof,
physical filesystem blocks, OS page cache or a whole-client 256 MiB guarantee.
Older caller-held snapshots, sampling, compiled bundle/tokenizer, borrowed runtime
KV and SDK overlap owners remain separately charged. Stage still performs bounded
full-snapshot validation and row hashing/authentication; it does not claim O(1)
latency or free disk reads.

Concrete storage admission regression uses synthetic Qwen2.5 geometry: 24 layers,
2 KV heads, head_dim 64, vocabulary 151936; 54-row root plus 7 executed decode states
per request. Eight states stage in the first batch; three divergent requests retain
22 unique nodes and **3600 unique pages**, under 2 MiB index / 8 MiB per-snapshot /
24 MiB owner-transient bounds (and below a 32 MiB component cap). Every restored
float32 KV/logit bit is compared. Tests also check replay writes nothing, a 25th
unique node rejects, restart/old-head rollback retain orphan charges, and damaged,
replayed or authenticated-invalid node manifests fail before state-array allocation.
These are full-shaped **storage** tests with causal synthetic contents, not native
Qwen execution, provider traffic, body weights or measured RSS evidence.

## Tests and qualification boundary

Release-wheel validation covers generated Qwen2/Qwen3/Llama native and SDK
execution, fresh-interpreter restoration, exact logits/active KV, and failure,
cancellation and publication rollback. The generated Qwen3 example completes
cold, hot and caller-headed restart requests with identical numeric commitments;
hot and restarted requests issue and reserve zero provider rows. These checks
establish functional reuse, not a whole-response performance or memory result.

Storage-only adversarial units run without providers or checkpoint fixtures.
Generated native and SDK suites require the integrated installed wheel:

```sh
target/frontier-sdk-env/bin/python -I -m pytest -o pythonpath= -q tests/test_decode_store.py
target/frontier-sdk-env/bin/python -I -m pytest -o pythonpath= -q \
  tests/test_decode_cache_sdk.py tests/test_decode_cache_lifecycle.py
target/frontier-sdk-env/bin/ruff check python/pllm/state/decode.py \
  python/pllm/runtime/decode_store.py tests/test_decode_store.py
```

Tests include ACK/pending exclusion, divergent shared paths, bitwise arrays,
independent ownership, wrong key/context, writer/client exclusion, tamper/truncation,
authenticated malformed index and overbounds before state allocation/write,
failed publication preserving trusted head, changed-process-secret reminting,
old trusted fork, conflicting same-path rejection, orphan charging and close.
Native tests separately generate tiny Qwen2/Qwen3/Llama fixtures, compare every
float32 KV/logit bit, diverge, reopen into a fresh ordinary runtime, execute fresh
continuation against a whole-prefix oracle, then restore in a fresh interpreter
using **public bundle only**, with provider execution prohibited. Fresh-wheel
results must be recorded separately from the storage-only geometry controls.

Representative hit rates and independent security review remain separate gates.
The matched all-link/RSS/remote-linear-work cohort is linked above; its observed
RSS eligibility does not make logical storage bounds a whole-client guarantee.

`tests/test_decode_cache_sdk.py` adds public SDK-only integration regressions for
actual request-local RNG seeds, greedy and divergent stochastic paths, lookup-before-
material ordering, full hits with zero provider session/reservation, demand-one
miss inventories and exact ACK/stage-row accounting, pending/EOS exclusion,
store=False, failed retirement, stream close/cancel, async drain and final-commit
rollback races, corruption/source/head rejection, one active response/client owner,
and fresh-interpreter ordinary SDK restoration. These tests intentionally use public
component/native admission, not private bypasses for stale wheels. Lifecycle
regressions additionally cover terminal-event and response-conversion failure,
explicit cancellation at terminal readiness, admitted-executor failure, reserved
handshake failure, constructor rejection and shutdown after rollback failure.
Example:

```sh
target/frontier-sdk-env/bin/python examples/decode_cache_sdk.py \
  --root target/decode-sdk-example --family qwen3 --seed 17
```

Example requires a new root, generates a tiny model and compares cold/hot/reopened
numeric commitments while explicitly seeding each runtime RNG. It is validation,
not a hit-rate, traffic, remote-work or memory benchmark.
