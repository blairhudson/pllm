# Transactional layer-paged decoder state

This internal executor runs an existing compiler-bound dense decoder while owning
at most one layer's full float32 K/V slab. `TieredExecution` authenticates bounded
append-only pages with AES-GCM and publishes a complete phase manifest only after
all layers and the original logits have completed. It uses the original attention
arithmetic, not selected rows, compressed state or an approximate attention rule.

## Ownership and admission

Native continuation admission binds the source/state contract. The reference
additionally requires batch-one, canonical `prefix_f32`, exclusive dense full-KV
and contiguous layer execution. Windowed/recurrent/retained state, verification,
protected nonlinear execution and MLP tiling do not compose with this layout.

Public bounds are one 128 MiB layer slab, 1 MiB per tensor page, a 2 MiB manifest,
2 GiB of ciphertext files, 32,768 files and 4,096 revisions. Manifest, file and
disk growth are preflighted before each phase. These bounds do not account for
all decoder tensors or imply a process-RSS ceiling. Page decryption, attention
scratch, immutable plaintext bytes and caller copies are additional owners.

Every ciphertext binds its source/continuation context and exact layer, range
and K/V role. The manifest authenticates executed token history, logits, state
provenance and every page commitment. Pages authenticate before their values
enter attention. Live layer arrays are erased on retirement or failure; Python
bytes and cryptographic library internals have their own allocation lifetimes.

The directory has one writer. A single in-memory generation publication follows
file/directory synchronization. Interrupted publication may retain charged orphan
ciphertext, but the prior trusted head remains usable. Older manifests remain
immutable because the caller might not yet have durably accepted a newer head.
The caller retains the key and newest head: there is no external rollback oracle,
key-custody service, reusable preparation material or persistent SDK conversation.

Generated Qwen2/Qwen3/Llama checks compare full logits and active KV bits across
append, close and restore. Corruption, wrong keys, changed heads, concurrent
writers, remote failures, publication failures and a publication-point interrupt
exercise failure ownership. Ordinary snapshot/continuation entry points reject
this owner.

The earlier [`research-tiered-decoder.json`](../evidence/research-tiered-decoder.json)
is an initial cohort from before the single-generation publication correction.
Its original build and script identities remain intact; it is not pooled with
the corrected implementation's cohort. A subsequent interrupted run produced no
completed evidence record.

## Replay

```sh
uv run --no-sync pytest -q tests/test_tiered_execution.py
uv run --no-sync python scripts/probe_tiered_decoder.py \
  --output target/tiered-decoder.json --processes 3
```

The probe loads the pinned Qwen2.5-0.5B checkpoint in each fresh process and uses
public numeric token fixtures at 128 and 512 input rows, followed by three decode
steps. Resident, paged, publishing and fresh-interpreter restored workers have
separate timings and memory samples. All checkpoint weights remain resident;
source loading, authentication, disk traffic and publication stay charged. This
is a complete local-clear decoder gate, not an SDK/provider response benchmark.

## Pinned checkpoint result

[`research-tiered-decoder-r3.json`](../evidence/research-tiered-decoder-r3.json)
contains 24 fresh processes: three resident/paged/publisher/restarted groups at
each input length. All four logit vectors and final active KV hashes match in
every group, with zero observed swap growth. Values below are separate
three-process medians in decimal MB and seconds.

| Measurement | 128 resident | 128 paged | 512 resident | 512 paged |
| --- | ---: | ---: | ---: | ---: |
| Final dense KV / maximum live paged slab, MB | 3.219 | 0.134 | 12.657 | 0.527 |
| Process lifetime peak RSS, MB | 1,090.19 | 1,098.55 | 1,275.23 | 1,277.80 |
| Execution CPU, s | 10.261 | 11.443 | 38.177 | 39.666 |
| Execution elapsed, s | 6.298 | 7.278 | 22.778 | 24.385 |
| Cold process CPU, s | 22.031 | 23.154 | 49.821 | 50.821 |
| Authenticated reads, MB | — | 9.523 | — | 37.859 |
| Published ciphertext/manifest bytes, MB | — | 5.750 | — | 15.356 |

Layer ownership reduces the live KV slab 24×, but measured process peak RSS
rises 0.77%/0.20%. Paging increases execution CPU 11.52%/3.90% and elapsed time
15.56%/7.06%. Full model weights and ordinary decoder temporaries remain live;
the slab reduction is not a whole-client memory improvement.

The fresh restarted workers take 3.101/3.111 execution CPU seconds and
14.117/13.605 cold process CPU seconds, including fresh source loading in the
latter. Original publishing costs another 22.137/50.489 cold process CPU seconds
and 3.770/13.255 MB of stored ciphertext/metadata. Restart then reads
10.144/38.520 MB and writes 1.981/2.101 MB while continuing. Those separately
charged costs establish exact bounded recovery, not free initialization or a
persistent SDK conversation.
