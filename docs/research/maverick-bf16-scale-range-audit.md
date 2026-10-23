# Maverick source recovery: bounded BF16 scale-value inspection

**Observed on 23 October 2026: all 252 declared scale tensors / 221,760 BF16 words
are positive, finite and at most `2^45`.** Every scale word was inspected, not
sampled. This closes that observed-value premise under trusted publisher/pinned
source assumptions, **not** whole-shard integrity or checkpoint qualification.
No numerical bound, importer, runtime profile or execution permission changes.

## 1. Exact source and observations

Official source: `Qwen/Qwen3-4B-FP8`, revision
`96b30dc13593a244a5e59e84687309f53c375cfa`. Only the first shard,
`model-00001-of-00002.safetensors`, was accessed; expected size is 4,412,140,424
bytes. The configured/default HF cache had no matching pinned shard. All 252
selected names and BF16 scale shapes were present in the first-shard header,
whose SHA-256 matched the
[earlier audit](maverick-fp8-pinned-header-audit.md#1-exact-source-and-scope):
`9203d0277e3986e90216c106ebdba821ea34acef7fcea49e88b5e5660def7cca`.

| Observation | Exact result |
| --- | --- |
| Scale tensors / words / payload bytes | 252 / 221,760 / 443,520 |
| Positive, finite, at most `2^45` | All 221,760 words |
| NaN / infinity / either zero / negative finite / above bound | 0 each |
| Minimum | BF16 `0x376d`; `237/16777216 = 237*2^-24` |
| Maximum | BF16 `0x3b35`; `181/65536 = 181*2^-16` |

Little-endian unsigned words were classified by sign, exponent and zero magnitude.
For positive finite BF16, `word<=0x5600` is exactly the existing `S<=2^45` premise.
Extrema use exact rationals, not host-float estimates. The observed maximum does
not replace the approved bound or change
[D32's promotion rule](../../paper/methods/DECISIONS.md#d32-maverick-pinned-bf16-scale-storage-amendment).

## 2. All requested ranges and historical accounting

Header requests were inclusive `0–7` and `8–76551`. The 108 scale requests covered
only contiguous scale tensors; **no FP8 projection, embedding, head or norm
payload** was read. Define physical slot `j=0..35` by this lexicographic layer-name
order, not execution order:

```text
0,1,10,11,12,13,14,15,16,17,18,19,2,20,21,22,23,24,25,26,27,28,29,
3,30,31,32,33,34,35,4,5,6,7,8,9
```

For each slot the exact absolute byte ranges have the following starts and lengths
(inclusive end is `start+length-1`). All observed ranges matched this description.

| Scale-only group | Absolute start | Length |
| --- | --- | ---: |
| MLP down/gate/up | `777993992 + 23072*j` | 9,120 |
| Attention k/o | `778008488 + 23072*j` | 1,600 |
| Attention q/v | `778010344 + 23072*j` | 1,600 |

All 110 body-bearing responses passed exact 206/`Content-Range`/file-size and
`Content-Length` checks **before** reads. Automatic redirect draining and buffered
body read-ahead were disabled; individual reads were at most 16 KiB. HTTP 200,
encoded/chunked bodies or wrong ranges/lengths would close unread. The final public
official redirect URL was reused in RAM for the same object's scale ranges;
signed CDN queries were never logged. No unsafe response, retry or fallback occurred.

The historical ledger is **266,842** previous header bytes + **59,062** previous
history bytes + **76,552** necessary header reread bytes + **443,520** scale bytes
= **845,976 / 4,194,304 bytes**. There were **112 new requests** including redirects.
The original probe additionally reserved 105 requests for earlier operations,
giving a conservative total reservation of 217, below 300; 105 was not a measured
historical redirect count. These describe that run, **not completion gates or
requirements to replay a historical byte/request total**.

## 3. Durable probe, ordinary output and offline tests

The durable implementation is
[`benchmarks/research/maverick_scale_source.py`](../../benchmarks/research/maverick_scale_source.py).
It imports the existing pure `scripts.probe_gkm_checkpoint_bounds` parser and
`InputFile` owner, restores its scoped FP8 metadata-width entry, and prefers only
the exact pinned cached snapshot. Cached size/identity failures reject without
network fallback. It enforces the same 4 MiB/300-request limits, including supplied
prior consumption, rather than treating this run's historical debit as source policy.
The CLI requires explicit `--prior-body-bytes` and `--prior-requests`; future source
I/O still needs separate authorization and must not reset an ongoing budget.

The [existing ordinary output](../../target/maverick-scale-inspection.log) retains
all range responses and each named tensor's extent/counts. Its authoritative copy
remains `target/maverick-scale-inspection.log` in the original isolated
`pllm-maverick-scale-audit-7d8efd83` worktree; the code candidate links to that same
file, without copying raw source data. The live observation used the frozen
`target/inspect_maverick_scales.py` prototype there. **Packaging did not repeat any
network request or payload inspection, regenerate that output, or change its values.**

Normal [unit tests](../../tests/test_maverick_scale_source.py) use synthetic ranges,
tiny cache files and BF16 edge words only. They test the real metadata/file owners,
cap-before-read, no-200-body, redirect closure, no read-ahead, cache mutation and
incomplete-source rejection. Runtime imports and real connections are trapped.

```sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -I -B -m pytest --noconftest -o pythonpath= -q tests/test_maverick_scale_source.py
```

## 4. Narrow implication

The observed scale premise of the
[conditional product theorem](maverick-weight-product-exactness.md) holds for these
returned bytes. Trusted HTTPS or cache provenance and pinned headers **do not**
verify range payloads against whole-shard LFS hashes; no such integrity proof is
claimed. FP8 finite codes, embedding/gamma bounds, head/embedding equality, actual
whole-graph totality and honest no-wrap remain unproved. The index's **1,310,720-byte**
excess remains unexplained and unwaived. D31/D32, prior proofs and source hashes are
unchanged. This establishes no model implementation, trained quality, serving,
security, whole-method acceptance or publication qualification.
