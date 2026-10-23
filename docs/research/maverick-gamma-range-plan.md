# Maverick Gamma: metadata-only bounded source plan

**Chronology:** §§1–4 preserve the pre-inspection plan and its then-unread status.
§5 records the later, explicitly authorized bounded source observation.

**No Gamma value has been read.** The 145 required BF16 vectors occupy **392,192
bytes**, but strict Gamma-only requests do **not** fit the remaining request
reservation. A 37-range alternative fits both byte caps only by explicitly
rereading 115,200 known scale bytes. That overlap needs approval; this note
neither performs nor authorizes source I/O. No numeric/profile/implementation rule
changes, and Main/D31/D32/proof artifacts are untouched.

## 1. Metadata ownership without another fetch

The source is the existing official `Qwen/Qwen3-4B-FP8` revision
`96b30dc13593a244a5e59e84687309f53c375cfa`, not Maverick-author source. All required
vectors belong to `model-00001-of-00002.safetensors` (4,412,140,424 bytes).

Raw header bodies were not archived. Retained observations included their hash,
length, complete name/shape checks, global extents and all 252 scale ranges.
The [local metadata planner](../../benchmarks/research/maverick_gamma_range_plan.py)
reconstructs the first shard's directory in RAM from those selected shapes. The
compact JSON with `__metadata__={"format":"pt"}`, BF16-before-FP8/name order and
eight-byte space padding has **exactly** the retained 76,544-byte header SHA-256:

```text
9203d0277e3986e90216c106ebdba821ea34acef7fcea49e88b5e5660def7cca
```

The planner refuses offsets on a length/hash/extent mismatch, then reuses the
existing source owner's bounded header parser. Thus the ordering is not merely a
serialization guess: the reconstructed bytes match the recorded header identity.
This binds **metadata only**, under the existing trusted retained-hash assumption;
it is not tensor-value evidence or whole-shard/publisher authentication. No raw
header archive, tensor payload, new request, cache read or model/native import was
needed for this calculation.

## 2. Exact required names, shapes and absolute offsets

The selected [local schedule §3.3](maverick-numeric-source-recovery.md#33-full-local-schedule)
uses input/post-attention/final norms of dimension 2560 and shared Q/K norm
vectors of dimension 128. Gamma is indexed by head coordinate, not a separate
learned vector for every query/KV head or token. The
[totality premise §5.1](maverick-canonical-reply-totality.md#51-a-sufficient-public-source-envelope)
requires the maximum absolute coefficient across **all** of these vectors.

Let `l` be the actual layer number and `j` its position in this physical,
lexicographic layer-name order (not execution order):

```text
0,1,10,11,12,13,14,15,16,17,18,19,2,20,21,22,23,24,25,26,27,28,29,
3,30,31,32,33,34,35,4,5,6,7,8,9
```

All rows below are BF16; start offsets are absolute shard bytes, including the
76,552-byte length-prefix/header region. Inclusive end is `start+length-1`.

| Tensor name (`l=0..35`) | Shape | Absolute start | Length |
| --- | --- | --- | ---: |
| `model.layers.{l}.input_layernorm.weight` | `[2560]` | `777988872 + 23072*j` | 5,120 |
| `model.layers.{l}.post_attention_layernorm.weight` | `[2560]` | `778003112 + 23072*j` | 5,120 |
| `model.layers.{l}.self_attn.k_norm.weight` | `[128]` | `778008232 + 23072*j` | 256 |
| `model.layers.{l}.self_attn.q_norm.weight` | `[128]` | `778010088 + 23072*j` | 256 |
| `model.norm.weight` (one final vector) | `[2560]` | `778819464` | 5,120 |

There are `36*(2*2560+2*128)+2560 = 196096` coefficients in 145 vectors. The
planner emits every individual name/range and both grouped plans, without values.

From the repository root, this ordinary command reproduces only the local metadata
calculation; it opens no source/cache and performs no HTTP:

```sh
python -B -m benchmarks.research.maverick_gamma_range_plan
```

## 3. Grouping and cumulative resource arithmetic

Carry forward the original **845,976-byte debit** and conservative **217-request
reservation** from the [scale inspection](maverick-bf16-scale-range-audit.md#2-all-requested-ranges-and-historical-accounting).
The latter includes the original 105-request reservation for prior operations,
not a newly measured historical request count. Do not reclaim that uncertainty as
free request credit. Remaining caps are **3,348,328 bytes / 83 requests**, plus the
proposed **512 KiB incremental payload** ceiling for this next inspection.

**Strict Gamma-only:** merge contiguous post-attention/K-norm vectors, leaving
three groups per physical layer and one final group: **109 body requests**, maximum
group 5,376 bytes, total 392,192 bytes. Cumulative bytes would be 1,238,168, but the
minimum cumulative request reservation is **326**, already over 300 before any
redirect. No strict-only full inspection is executable under this reservation.

**Explicit scale-overlap proposal:** keep the initial input norm separate, then
use exactly these 37 ranges:

| Group | Absolute start | Length |
| --- | --- | ---: |
| First input norm only | `777988872` | 5,120 |
| For each `j=0..35` | `778003112 + 23072*j` | 13,952 |

Each repeated group covers post-attention norm, K norm, the 1,600-byte K/O scale
group, Q norm, the 1,600-byte Q/V scale group, then the next physical input norm
(or final norm for `j=35`). The planner checks every intersected tensor: **no FP8
projection, embedding, head or MLP scale bytes** enter these ranges. Only 145 Gamma
slices would be classified; the intervening known scales would be ignored, not
reinspected or re-reported as another scale observation.

This is nevertheless **115,200 repeated scale payload bytes**, which must be
explicitly approved and counted. Total payload is **507,392 bytes**, 16,896 below
512 KiB; cumulative inspection would be **1,353,368 bytes**. With retained metadata
there are 37 minimum new requests, a cumulative reservation of **254**, and room
for at most **46 additional redirects/requests**. Redirect behavior is not guaranteed;
the existing hard guard must abort before exceeding 300, without retries or
claiming a completed Gamma result. A fresh header reread is unnecessary with the
hash-matching retained directory; if one is required later, debit it too.

## 4. Narrow future extension and what could close

The existing [source transport owner](../../benchmarks/research/maverick_scale_source.py)
currently admits scale ranges and enforces a **positive** scale predicate. It
cannot be relabelled a Gamma inspector: learned Gamma coefficients may be negative
or either signed zero. A genuinely needed, still **unimplemented** extension would:

- Reuse that same HTTP/cache/parser path, exact 206/range/size checks, byte/request
  guards, close-before-body rule, no retries and query-secret suppression.
- Add the exact named Gamma allowlist (and only separately approved overlap spans),
  consuming the retained directory instead of fetching headers/scale slices again.
  Start the first permitted range at the canonical pinned URL; keep any official
  redirect only in RAM. Do not create another downloader or reuse an expired token.
- Classify only the 196,096 Gamma words: reject exponent-255 NaN/infinity, allow
  signed finite values/zeros, and compute `G=max(abs(gamma))` exactly. Charge all
  overlap bytes even though they contribute no Gamma words.

The existing sufficient theorem needs integers `h>=22`, `g>=0`, `h+g<=44`,
`E<=2^h`, `G<=2^g`. A future complete Gamma inspection can establish its **Gamma
subcondition** iff every word is finite and `G<=2^22`: choose the least
`g in 0..22` with `G<=2^g`. The corresponding positive BF16 endpoint is `0x4a80`;
this is derived from the existing envelope, **not a new clipping/admission policy**.
For `G=0`, choose `g=0`. A larger/nonfinite Gamma does not pass this sufficient
certificate; a tighter proof remains a separate possibility, not permission to
change the checkpoint or arithmetic.

Even a passing Gamma result would leave `E<=2^(44-g)` over the entire embedding
table unproved. It would prove neither whole-graph totality by itself, head tying,
honest no-wrap nor whole-shard integrity. **Actual Gamma is still unread/unproved.**
The metadata reconstruction and exact interval/budget arithmetic above used only
small stdlib calculations; no source I/O, framework/model execution, pytest suite or build
was performed for this plan. The index-size discrepancy and D32 no-waiver remain.

## 5. Authorized bounded observation, 23 October 2026

After the plan, the user explicitly approved its known-scale overlap and increased
the new HTTP allowance from 37 to **83 total attempts**, including redirects,
without raising the original 300-request/4-MiB caps or the **507,392-byte** new
body cap. One inspection completed using the
[reusable Gamma probe](../../benchmarks/research/maverick_gamma_source.py) and
existing bounded source transport. The owner gained only shared bounded-read
hooks; the scale probe's defaults and its 45 offline tests remain unchanged.
There is no parallel downloader or source/model importer.

| Observation | Exact result |
| --- | --- |
| Fully inspected vectors / coefficients | 145 / 196,096 |
| Finite positive / finite negative coefficients | 195,758 / 338 |
| NaN / infinity / either signed zero | 0 each |
| Maximum absolute Gamma | BF16 magnitude `0x4230`, exactly **44** |
| Least sufficient exponent | **`g=6`**, since `2^5 < 44 <= 2^6` |
| Named vector attaining the maximum | `model.layers.0.self_attn.k_norm.weight` |
| Completed ranges / unread or incompletely inspected vectors | 37 / 0 |

The maximum names a norm vector, not an assertion about the sign or coordinate
of the maximal coefficient. Every Gamma word was inspected, not sampled. Signed
finite coefficients were allowed; the scale positivity classifier was not reused.
The only retained payload-derived material is ordinary scalar statistics, not
tensor arrays or a checkpoint.

The [ordinary output](../../target/maverick-gamma-inspection.log) contains the 37
validated range records and all 145 named vector summaries. Actual new consumption
was **507,392 body bytes / 38 HTTP attempts**: 37 completed 206 responses plus one
followed redirect, with no retry, header/index refetch, cache fallback or unsafe
body response. **115,200 bytes** were the explicitly permitted already-observed
attention scales; they were counted as I/O but ignored by Gamma classification.
Cumulative accounting is **1,353,368 body bytes / 255 requests reserved** (the
original conservative 217 reservation plus 38 new attempts). Remaining original
headroom is 2,840,936 bytes / 45 requests; it is not new authorization to inspect
other source data or repeat this completed observation.

The historical command, run **once** with the then-current debit, was:

```sh
python -B -m benchmarks.research.maverick_gamma_source --inspect --prior-body-bytes 845976 --prior-requests 217
```

**Do not replay those stale debits.** Future source I/O needs explicit scope and
truthful current accounting. The reusable probe preflights the complete remaining
range budget, validates the retained metadata plan before I/O, follows only the
existing public HTTPS redirect policy, caps every body read, and never logs a
signed URL. Errors retain actual consumed bytes/attempts and the names of vectors
not fully inspected; a partially received failed group is not a completed vector.

The normal [Gamma tests](../../tests/test_maverick_gamma_source.py), together with
the existing scale tests, passed **64 offline cases** before live I/O. They use fake
responses and synthetic signed BF16 words, not these source observations, and
trap model/runtime imports and real connections. Cases cover ignored NaN-filled
scale overlap, multiple redirects, insufficient remaining budgets, partial reads,
non-206/oversized/wrong-range responses, metadata drift and no refresh/fallback.

```sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -I -B -m pytest --noconftest -o pythonpath= -q tests/test_maverick_scale_source.py tests/test_maverick_gamma_source.py
```

**Narrow closure:** under the immutable trusted pinned-publisher assumption, the
observed source satisfies the theorem's Gamma subcondition with `G=44`, `g=6`.
The joint envelope still needs the entire embedding table finite with
`E<=2^38` and the other source obligations. This is **not** whole-shard LFS
authentication, head tying, honest no-wrap, whole-graph totality, checkpoint
qualification, implementation approval, serving/SDK authority or a trained result.
The index-size discrepancy and D31/D32 no-waiver remain unchanged. No model,
framework, PLLM native, Cargo or heavy execution was used.
