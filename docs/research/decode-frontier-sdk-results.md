# Growing epochs and exact decode reuse: installed-SDK results

Two serial, separately salted Qwen2.5-0.5B-Instruct cohorts complete the first
matched whole-response measurement of the installed contracts. Each has three
alternating replicates, fresh client and provider processes, empty initial bundle
caches, and native shared 100/40 Mbps access caps without added RTT. The growing
cohort has 24 clients/responses; the decode-cache cohort has 18 clients and 54
responses. All checked final logits, active KV, executed history, pending selection
and outputs match within each workload. The stochastic cohort additionally matches
initial/final RNG state and per-selection sampling witnesses. Both memory guards
record zero observed swap growth.

**Growing epochs reduce early-stop traffic and first-text delay. Exact decode
reuse reduces repeated-request traffic but increases client CPU, memory and
complete-cell time against ordinary inventory.** Every recorded client peak is
below 256 MiB; this qualifies the observed-RSS threshold for these workloads,
not conservative runtime admission under a 256 MiB policy or other checkpoints.
The choices remain explicit and workload-dependent.

## Evidence and replay

| Cohort | Raw report | SHA-256 |
| --- | --- | --- |
| Growing epochs | [24-cell report](../evidence/growing-epochs-sdk-qwen25-2026-10-06.json) | `549c93873362e221094802bd530837d4691d736555d2b92ba24aa5872abe97cf` |
| Decode cache | [18-client report](../evidence/decode-cache-sdk-qwen25-2026-10-06.json) | `5a8b80919a81cf6d44d336fa8f4b2d8e97b86f989e82e433116b534d082d9eb1` |

Validated derived summaries retain every configuration, scope, median, maximum,
directed ledger, actual-work count and memory-admission record:
[growing](../evidence/growing-epochs-sdk-qwen25-2026-10-06-summary.json) and
[decode](../evidence/decode-cache-sdk-qwen25-2026-10-06-summary.json).
The [growing methodology](growing-epochs-sdk-reference.md) and
[decode methodology](decode-cache-benchmark-reference.md) remain byte-identical
to their premeasurement source locks. Their tiny smokes and earlier attempts are
separate from these complete checkpoint cohorts. Do not pool either private salt
or multiply gains across the two workloads.

Both raw reports bind installed build
`3037b0ea59ca74ab60c4a7361908bc7313f055c9e2c633721dd823580123804f`
and checkpoint source lock
`383f5ed6947ba56ceed6083a9503d6302f09559e1e57ee614409900b36084c29`.
All eight growing and nine decode measurement-source hashes still match. Actual
SDK/native file hashes are checked in each isolated `-I -S` child. State auditing,
identity checks and authenticated cache I/O remain charged.

Recompute each summary against the installed compatible wheel, with new output
paths:

```sh
target/frontier-sdk-env/bin/python -I scripts/summarize_decode_frontier_sdk.py \
  --input docs/evidence/growing-epochs-sdk-qwen25-2026-10-06.json \
  --output target/growing-epochs-published-summary.json
target/frontier-sdk-env/bin/python -I scripts/summarize_decode_frontier_sdk.py \
  --input docs/evidence/decode-cache-sdk-qwen25-2026-10-06.json \
  --output target/decode-cache-published-summary.json
```

The summarizer rechecks complete coverage, exact state/sampling commitments,
canonical policies, source/body geometry, WAN identities, issued/reserved/burned
rows, application-body conservation and disk windows. Real execution replays use
the frozen methodology commands and obtain new private cohorts.

## Growing `(4, 60)` epochs

Public `PreparedInventory(decode_epoch_schedule=(4, 60))` is compared with full
inventory and fixed four/eight-output epochs. The early workload has 36 inputs
and naturally produces three outputs; the full workload has 54 inputs and reaches
all 64 allowed outputs. Both retain the same 128-position state bound, paged
client/Preparation weights, indexed tokenizer, exact W8A8 `prefix_f32` arithmetic
and 32 MiB prefix cache.

The seven priority metrics below are independent per-cell medians. Client RSS is
process lifetime high-water in **MiB**; Preparation RSS is 50 ms sampled peak in
decimal **MB**. CPU and first-text delay are seconds. TCP traffic is decimal MB.
Decode TPS includes every epoch pause and uses the N−1 denominator.

| Workload / policy | Client MiB | Client CPU s | Prep CPU s | Prep MB | First text s | Decode tok/s | All-link TCP MB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Early / full inventory | 131.16 | 8.98 | 12.76 | 171.66 | 30.07 | 2.98 | 266.10 |
| Early / fixed 4 | 131.48 | 9.22 | 10.42 | 144.08 | 19.66 | 3.98 | 213.40 |
| Early / fixed 8 | 133.09 | 9.19 | 10.59 | 149.72 | 20.35 | 3.92 | 216.91 |
| Early / growing | 131.48 | 9.23 | 10.39 | 143.39 | 19.62 | 3.96 | 213.40 |
| Full / full inventory | 133.64 | 14.68 | 13.45 | 190.86 | 37.60 | 3.49 | 397.92 |
| Full / fixed 4 | 144.33 | 18.76 | 15.33 | 158.66 | 26.97 | 1.97 | 401.53 |
| Full / fixed 8 | 144.02 | 16.76 | 14.10 | 160.69 | 27.82 | 2.22 | 399.55 |
| Full / growing | 138.00 | 15.33 | 13.45 | 165.89 | 27.22 | 2.19 | 398.08 |

Against full inventory, early growing epochs reduce TCP traffic **19.81%** and
request time **34.36%**, from 31.066 to 20.391 seconds. Complete-cell time,
including role startup and cleanup, falls **13.57%**, from 79.797 to 68.968 seconds.
Preparation birth CPU falls **18.61%** and aggregate cleanup-inclusive CPU
**9.99%**; client lifetime CPU rises **2.81%**. Growing and fixed-four both end
inside their first epoch and issue exactly 3,639 stage rows. Full inventory issues
9,399 and burns 5,760 unused rows. This early cell does not distinguish the later
growing schedule from fixed-four.

At full output use, every policy issues and consumes 11,073 stage rows. Growing
adds one 11.265-second median idle preparation pause, compared with 15 pauses for
fixed-four and seven for fixed-eight. Its first-text delay falls **27.60%**, but
charged decode TPS falls **37.25%**, from 3.486 to 2.187. Request time rises
**0.76%**, from 55.622 to 56.042 seconds; TCP traffic rises **0.04%**. Complete-cell
time is nearly unchanged (104.612 versus 104.488 seconds). Client CPU rises
**4.42%** and median peak RSS **3.26%**. Moving work behind the first output does
not eliminate it.

## Exact encrypted decode cache

Public `ClientPrefixReuse(generated_prefixes=True, decode=True,
decode_max_bytes=32 << 20)` and caller-owned `DecodeCache` use fixed one-output
epochs. Each client makes three 54+8-token requests at temperature 0.7/top-p 0.9.
The harness seeds the actual sampler with `[7,7,7]` or `[7,11,19]`; this is a
declared benchmark fixture, not a new SDK seed API. Ordinary and fixed-one controls
already have the same completed-prefix reuse, public delivery and numeric choices.

### Complete three-request lifecycle

| Scenario / policy | Client MiB | Client CPU s | Prep CPU s | Prep MB | Complete cell s | All-link TCP MB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Same seeds / ordinary | 146.22 | 16.68 | 11.88 | 160.69 | 89.58 | 297.98 |
| Same seeds / fixed 1 | 150.77 | 22.38 | 14.55 | 151.83 | 98.14 | 304.66 |
| Same seeds / decode cache | 164.19 | 25.63 | 11.95 | 154.39 | 95.10 | 265.33 |
| Distinct seeds / ordinary | 148.78 | 16.76 | 11.87 | 163.74 | 89.67 | 297.98 |
| Distinct seeds / fixed 1 | 148.55 | 22.29 | 14.59 | 153.78 | 97.80 | 304.66 |
| Distinct seeds / decode cache | 166.59 | 28.30 | 13.11 | 155.47 | 101.26 | 283.87 |

Versus ordinary inventory, cache traffic falls **10.96%** for same seeds and
**4.73%** for distinct seeds. Complete-cell time rises **6.16% / 12.93%**, client
lifetime CPU **53.63% / 68.79%**, aggregate cleanup-inclusive CPU
**17.62% / 29.22%**, and median client peak RSS **12.29% / 11.97%**. The cold cache
request takes about 36.10 seconds against ordinary's 29.75; its one-token epochs
and encrypted publication stay charged.

Versus the separate fixed-one control, cache traffic falls **12.91% / 6.82%**.
Complete-cell time improves **3.09%** with same seeds but regresses **3.54%** with
distinct seeds. These comparisons cannot substitute for the faster ordinary
control.

### Cold and later request windows

Each cell is **first-text seconds / charged N−1 tokens per second / total request
seconds**. These windows exclude earlier cold client setup, which remains charged
in the complete lifecycle above.

| Scenario / policy | Request 1, cold | Request 2 | Request 3 |
| --- | ---: | ---: | ---: |
| Same seeds / ordinary | 27.84 / 3.71 / 29.75 | 3.85 / 3.70 / 5.77 | 3.83 / 3.70 / 5.75 |
| Same seeds / fixed 1 | 26.77 / 1.22 / 32.56 | 2.89 / 1.23 / 8.62 | 2.93 / 1.22 / 8.68 |
| Same seeds / decode cache | 28.26 / 0.91 / 36.10 | 3.61 / 4.47 / 5.19 | 3.63 / 4.51 / 5.20 |
| Distinct seeds / ordinary | 27.86 / 3.71 / 29.77 | 3.86 / 3.71 / 5.76 | 3.85 / 3.69 / 5.79 |
| Distinct seeds / fixed 1 | 26.72 / 1.23 / 32.44 | 2.88 / 1.23 / 8.61 | 2.91 / 1.24 / 8.57 |
| Distinct seeds / decode cache | 28.24 / 0.91 / 36.06 | 3.61 / 0.91 / 11.42 | 3.62 / 4.48 / 5.19 |

Same-seed requests two and three restore one root and seven executed children,
issue/reserve no provider rows, record zero covered request bodies, and perform
no cache writes or new publication. Sampling still executes. Each warm request
reads **18.245 MB** of authenticated local files and takes about 5.2 seconds.
Those zero request bodies do not erase the cold lifecycle's 265.33 MB of TCP
traffic.

The three distinct seeds produce **two**, not three, distinct output commitments
in every policy/replicate. Request two reuses only the root and executes seven
fresh decode paths; request three has a full path hit. This narrow fixture does
not estimate hit rates for unrelated prompts or arbitrary sampling workloads.

| Cache scenario | Issued/claimed stage rows | Encrypted writes MB | Encrypted reads MB | Retained files | Nodes / pages | Publications |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Same seeds | 5,697 | 8.551 | 53.155 | 2,937 | 8 / 2,928 | 1 |
| Distinct seeds | 6,369 | 14.999 | 52.973 | 3,281 | 15 / 3,264 | 2 |

Ordinary controls issue/claim 7,041 rows per client; fixed-one issues 7,233,
claims 7,041 and burns 192. Cache scenarios burn/discard zero rows. Distinct-seed
retained disk includes the older accepted index (647,311 bytes); logical I/O
does not include filesystem metadata, physical block amplification or OS cache.
The cache has no implicit eviction. Whole-checkpoint fresh-interpreter restore
is not measured here; generated SDK restart parity remains a separate check.

## Low-RSS and remote-work gates

The observed Low threshold uses the **largest** process lifetime peak across
every replicate and workload, rather than the medians above:

| Cohort / policy | Largest client MiB | At most 256 MiB observed |
| --- | ---: | --- |
| Growing / full inventory | 135.14 | Yes |
| Growing / fixed 4 | 144.34 | Yes |
| Growing / fixed 8 | 144.48 | Yes |
| Growing / growing | 140.86 | Yes |
| Decode / ordinary | 151.73 | Yes |
| Decode / fixed 1 | 158.86 | Yes |
| Decode / decode cache | 169.59 | Yes |

All executed scheduled transformer-body dense linear MACs remain remote; cache
hits eliminate work rather than move it to the client. Fully eliminated requests
have zero MACs and a null remote fraction. Head, attention, normalization, masking
and Preparation are outside that denominator. Whole-body FLOP/CPU fractions are
unknown. Neither study establishes a tenfold client-memory improvement.

## Accounting and scope

TCP totals cover native relay startup through provider/network shutdown. Covered
v3 application bodies independently charge client construction through close,
including cold bundles, corrections, controls and ACKs. They retain partial base,
output-dependent and unattributed partitions; incomplete attribution cannot
supply complete base or per-output headline rates. Independent medians need not
sum, while the archived cohort-sum partitions conserve exactly.

Client CPU/RSS include process startup, state hashing and owner close. Provider
CPU starts at process birth and ends at client exit; provider shutdown is counted
only in aggregate cleanup CPU. Provider RSS samples begin after startup, so their
startup high-water is unknown. Full-cell time includes setup/cleanup but excludes
the launcher's preceding source resolution/preflight. All full-cap state and
cache bounds remain admitted before work, without an assumed hit discount.

The pinned checkpoint and trusted public tokenizer artifact are pre-positioned;
their distribution and compilation are outside these response counters. Physical
wire, Internet RTT, filesystem-cache/device memory, representative generation
quality, independent operators and cryptographic review remain outside the
measurements. These results support explicit workload-specific choices, not
automatic selection or a universal traffic/latency gain.
