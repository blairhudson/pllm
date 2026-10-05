# Pinned Qwen3-4B client storage

## Result and scope

Independent child processes imported the same authenticated **406,945,395-byte**
public bundle, bound its complete native semantic schedule, claimed all **71 rows
across 144 body stages**, and executed one 64-row plus seven one-row token lookups
and eight output-head operations. Publisher and client processes ran sequentially.
The source was cached `Qwen/Qwen3-4B` at
`1cfa9a7208912126459214e8b04321603b3df60c`.

| Isolated client scope | Eager masks + resident weights | Lazy masks + streamed paged weights |
| --- | ---: | ---: |
| Process peak RSS, decimal MB | 1,471.61 | 155.98 |
| Retained mask storage, MB | 507.765 | 0.224 |
| Import + inventory CPU, s | 5.219 | 4.598 |
| Boundary operations + mask claims CPU, s | 0.842 | 2.387 |
| Combined client CPU, s | 6.062 | 6.985 |
| Client elapsed time, s | 6.182 | 7.281 |
| Object download bodies, MB | 406.205 | 406.205 |
| Manifest body, MB | 0.304 | 0.304 |

Peak RSS fell **9.44× (89.4%)**. Combined CPU increased **15.2%**; paging trades
resident weights for repeated local reads and page-kernel work. Both branches
performed **3,111,649,280** head integer MACs. Bundle and semantic-plan identities,
mask bytes, one-use tickets and all token/head results matched exactly. Composition
and compiled binding digests intentionally differ. Observed swap growth was zero.

This is **not a complete decoder or response**: no attention, KV cache, nonlinear
body operation or provider session ran. Token IDs are fixed public fixture values;
the head consumes the final lookup row, not a decoder's hidden state. The eager
control invokes the retained independent mask expander; ordinary SDK execution
now uses lazy native masks. This comparison isolates those allocations plus the
real checkpoint's client import and boundary storage. It cannot establish 10×
whole-client RSS, generation quality, provider performance or Internet throughput.

RSS excludes OS filesystem cache and total host memory. Both clients retained the
same public object cache on disk. Paged import additionally uses bounded temporary
object files and independent private native snapshots. That is a disk trade, not
compression of the checkpoint. The measured publisher peaked at **1,358.17 MB**
and discards body weights after quantization and metadata validation; it is a
probe-only public-artifact publisher, not a serving optimization.

## Runtime integration

The ordinary opt-in SDK choice is
`ClientBundleTransport("artifacts", storage="paged")`, optionally combined with
zlib and object batching. Object and original bundle digests precede import;
native snapshots recheck shape, raw weight digest and signed bounds. Client token
rows never become remote object requests. A shared native executor avoids one
thread pool per page-backed matrix. Failed imports close provisional snapshots;
raw temporary files close on success and failure. Cache corruption is repaired,
and cache eviction or source mutation cannot change an admitted matrix.

Generated Qwen2/Qwen3 SDK and gateway controls cover prepared, Freivalds-verified,
two-offset, tied/untied heads, client-owned projections, repeated-prefix reuse and
raw/zlib artifact delivery. Full logits and every checked KV array agree exactly.
Tile/gather tests cover native per-call bounds without increasing them.
Client-owned paged body weights cannot simultaneously allocate Metal snapshots.
Existing resident defaults retain their composition identity.

## Reproduction

Run from the repository root with the pinned checkpoint already cached:

```bash
uv run --no-sync python scripts/probe_client_storage.py \
  --source "$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c" \
  --output /tmp/pllm-client-storage.json
```

Output must not exist. The probe preflights geometry without tensor values,
requires **3.92 GiB** additional physical headroom beyond the normal host reserve,
checks disk space, and aborts child work on pressure or swap growth. It was first
blocked when headroom was unavailable; the retained measurement ran only after
admission succeeded. It never starts the two 4B provider processes.

Machine-readable evidence: `client-storage-qwen3-4b-2026-10-05.json`.
Runtime configurations: `examples/benchmarks/client_memory.py`.
