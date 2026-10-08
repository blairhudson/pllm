# Prepared serving: client allocations, pacing and artifact reuse

These follow-ups retain controlled offline Preparation and one online Inference
provider. Client memory/CPU, Preparation resources, capped throughput and all-link
traffic must be assessed together. The Research chart now makes those priorities
its default axes and explains their rationale. Quality and privacy contracts
remain independent requirements; resource measurements are not prices or energy.

## Combined-stack client allocation profile

The Qwen3-4B 32+8 bound rejects before launch: the native estimate needs **7.61
GiB**, against **7.12 GiB** post-reserve headroom. The rejected admission is kept
in `docs/evidence/research-prepared-serving-qwen3-admission.json`; no 4B result
is inferred from the smaller control.
After the smaller matched SDK pair retires, a second preflight still rejects at
**7.05 GiB** headroom. That retry is retained in
`docs/evidence/research-prepared-serving-qwen3-admission-r2.json`.

The admitted Qwen2.5-0.5B W8A8 150+8-token control uses the ordinary serial
prepared factory with paged client/Preparation weights, demand issuance, indexed
tokenization, packed/residue frames, fused client boundaries and uncompressed
masked WebSockets. Its allocation-instrumented report is
`docs/evidence/research-prepared-serving-client-profile-qwen25.json`.
All runtime checks pass, and the profiled source hashes remain unchanged.

| Phase | Elapsed s | RSS before / largest after, MB |
| --- | ---: | ---: |
| Public artifact bundle load | 37.00 | 161.56 / 272.11 |
| Runtime construction | 6.33 | 285.90 / 301.81 |
| Prefill | 66.12 | 301.81 / 350.24 |

These intervals nest and must not be summed. The measured lifetime peak is
355.19 MB **with tracing enabled**, including 36.02 MB of tracing metadata at
exit. Python-traced live/peak allocations are 59.96/97.50 MB; import machinery
accounts for roughly 30 MB of the retained traced objects. Native, allocator,
shared-library and tracing owners prevent treating those figures as a complete
RSS decomposition or subtracting tracing bytes to manufacture an uninstrumented
peak. The profile localizes growth to loading and prefill, but does not identify
a removable 100 MB bundle array. It establishes neither a new memory reduction
nor the 10× target. A matched Qwen3 profile remains admission-blocked.

## Calibrating the native rate emulator

The prior relay requested an entire 16 KiB token-bucket burst on each read.
When a timer woke late, refill beyond that ceiling was discarded before the
whole burst was consumed. On this host, a saturated nominal 40 Mbps access link
therefore delivered about 32 Mbps. The relay now reads **8 KiB** while retaining
the same **16 KiB shared burst**, rates, per-party sharing, connection bounds
and socket-buffer limits. Snapshots record the read size.

`scripts/probe_native_pacing.py` transfers public bytes through actual TCP
relays in eighteen fresh processes. Source-to-sink SHA-256, exact directed-byte
conservation, configured-rate validation and resource retirement all pass for
**251,658,240 payload bytes**. Evidence:
`docs/evidence/research-native-pacing-calibration.json`.

| Geometry | 16 KiB read Mbps | 8 KiB read Mbps | Old / calibrated process CPU s |
| --- | ---: | ---: | ---: |
| Preparation upload | 31.91 | 40.03 | 0.155 / 0.191 |
| Two peers sharing one upload | 31.90 | 40.01 | 0.348 / 0.416 |
| Independent simultaneous directions, combined | 64.18 | 80.01 | 0.251 / 0.315 |

The tiny excess over 40 Mbps is permitted by the unchanged 16 KiB burst allowance.
Deterministic late-timer controls additionally check every observed time window
against the original rate-plus-burst bound. More relay wakeups raise local CPU.
This corrects emulator utilization; it is not a faster decoder, smaller protocol
or measured Internet result. Historical native-capped cohorts retain their
original implementation and scope. Compare matched emulator implementations,
not historical and calibrated throughput as if they were one cohort.

### Matched installed-SDK calibration pair

The separate Qwen2.5 150+8-token cold/repeat pair uses fresh installed clients,
isolated artifact caches and one private cohort salt. Native libraries and
dependencies match; `runtime/native_wan.py` is the only changed Python module.
All four responses and the canonical numeric/workload/output comparison pass,
with zero observed new swap. Evidence:
`docs/evidence/research-prepared-pacing-qwen25.json`.

| Metric | Historical pacing cold / repeat | Calibrated pacing cold / repeat |
| --- | ---: | ---: |
| Complete request s | 138.071 / 105.113 | 113.980 / 86.693 |
| Request tokens/s | 0.05794 / 0.07611 | 0.07019 / 0.09228 |
| N−1 decode tokens/s | 3.335 / 3.396 | 3.593 / 3.221 |
| Preparation elapsed s | 40.873 / 40.888 | 33.901 / 33.710 |
| Client run-window CPU s | 22.360 / 17.642 | 21.652 / 16.334 |
| Preparation run-window CPU s | 8.001 / 7.667 | 7.528 / 7.468 |
| Covered application bodies MB | 476.953 / 352.048 | 476.953 / 352.048 |

Correcting emulator utilization lowers request latency **17.45% / 17.52%** and
Preparation elapsed time **17.06% / 17.55%**. The roughly 132.83 MB correction
stream has a **26.57-second** 40 Mbps floor. The remaining elapsed gap is about
7.1–7.3 seconds, which still includes computation, serialization and stage
acknowledgements; CPU seconds alone do not prove its critical-path decomposition.
Decode throughput changes **+7.75% / −5.17%**, so this pair does not establish a
consistent later-token speedup.

Client process-lifetime peak RSS **rises 229.82 to 251.18 MB (9.30%)**. Preparation
peak rises 202.15 to 204.10 MB (0.96%); sampled simultaneous topology RSS falls
1,554.96 to 1,473.92 MB (5.21%). Cold-first aggregate CPU falls 75.28 to 72.14
seconds (4.17%). This is one sequence per build, without an allocation-causal
explanation for the opposing memory changes or a client-memory improvement.

Outputs and application bodies match exactly. Two-response TCP-stream totals
are **830,938,177 / 830,938,132 bytes**; the 45-byte difference is transport/control
variation, not an arithmetic-body reduction. The pair isolates a measurement-tool
change, not a new protocol optimization. Checkpoint and public-index distribution,
physical wire, Internet RTT and independently operated roles remain outside scope.

## Public artifact reuse across fresh clients

`scripts/probe_artifact_reuse.py` extends the earlier file-only screen with a
separate HTTP provider process and fresh client processes. It uses ordinary
bounded zlib framing and verified private `FilePayload` imports. Every client
fetches both public objects; the experiment does not use a warm client cache.
Three repetitions alternate provider order, for **24 fresh clients** total.

The pinned Qwen3 source provides an 8.39 MB W8 embedding slice and its original
11.42 MB tokenizer. Both variants retain the same public source payload. Raw
and encoded digests match on every import, byte counts reconcile, and private
client files and provider resources retire. Evidence:
`docs/evidence/research-artifact-reuse-qwen3.json`.

| Fresh clients served | On-request provider CPU s | Pre-encoded provider CPU including publication s | On-request / pre-encoded client CPU s |
| --- | ---: | ---: | ---: |
| 1 | 0.321 | 0.354 | 0.105 / 0.105 |
| 2 | 0.641 | 0.366 | 0.209 / 0.209 |
| 4 | 1.281 | 0.387 | 0.419 / 0.415 |

Pre-encoding pays back by the **second fresh client** in this bounded screen.
At four clients it reduces publication-inclusive provider CPU **69.77%**.
Publication alone takes 0.343 CPU seconds; the first client costs 10.24% more
provider CPU including that publication. Client CPU is nearly unchanged.

Source objects occupy **19.81 MB**, and pre-encoded frames add **10.15 MB** of
provider disk. Every client still downloads **10.15 MB** of encoded bodies and
writes **19.81 MB** of decoded snapshots over time, with at most **11.42 MB**
live snapshot disk because objects retire serially. Publication and serving are
timed separately; common source extraction/quantization and provider loading/
validation, interpreter startup, filesystem cache and physical wire are outside
the table. Client process-birth CPU and process peaks are retained separately in
the raw report.

This is a bounded local HTTP prototype over two objects, not a full SDK bundle,
provider cache, WAN latency or complete-client memory result. A production
pre-encoded cache still needs bounded storage, admission, identity validation,
eviction and lifecycle integration. A cold/repeat client that downloads once
and then reuses local artifacts cannot claim the multiple-client payoff.

## Replay

Output paths must be new. The pinned public checkpoint and tokenizer index must
already be available; the ordinary allocation and host-pressure gates remain.

```sh
PLLM_RESEARCH_ARTIFACTS=target/prepared-frontier-artifacts \
  .venv/bin/python -m scripts.profile_client_frontier \
  --experiment benchmarks.research.prepared_frontier:serial \
  --prompt-file target/prepared-frontier-artifacts/prompt.txt \
  --allocations --output target/evidence/serving-client-profile.json

.venv/bin/python -m scripts.probe_native_pacing \
  --output target/evidence/serving-pacing.json

.venv/bin/python -m scripts.probe_artifact_reuse \
  --output target/evidence/serving-artifact-reuse.json

PLLM_RESEARCH_ARTIFACTS=target/prepared-frontier-artifacts \
  .venv/bin/python -m scripts.probe_prepared_path \
  --probe responses --change-scope native-pacing \
  --baseline-python target/venvs/prepared-refinements/bin/python \
  --candidate-python target/venvs/prepared-pacing/bin/python \
  --experiment benchmarks.research.prepared_frontier:serial \
  --prompt-file target/prepared-frontier-artifacts/prompt.txt \
  --output target/evidence/serving-pacing-responses.json
```

The client profile above used the historical 16 KiB relay read; replaying on the
calibrated implementation is a new instrumented sample, not the same timing
cohort. Reports retain exact implementation hashes.
The installed-build pair requires the two historical wheel variants and matching
dependencies: its source check admits only the recorded native-relay change.
