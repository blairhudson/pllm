# GKM full/seeded sharing: bounded Qwen response screen

Status: **six fresh clients complete; initial short-workload screen**. This is
not the frozen ten-method comparison, an independent confirmation or broad
source-model quality qualification.

## Experiment and evidence

[`examples/benchmarks/gkm_sharing.py`](../../examples/benchmarks/gkm_sharing.py)
uses ordinary `BenchmarkTarget.response` and `benchmark_suite`, with three fresh
clients per sharing choice in one balanced private cohort. The installed streamed
single-window implementation preserves the original Q11 numeric contract.

- Pinned Qwen2.5-0.5B-Instruct, revision
  `7ae557604adf67be50417f59c2c2f167def9a775`.
- Four actual input tokens, two fixed-count greedy outputs; compiled input bound
  eight. There is no sampler RNG or streaming decode-throughput measurement.
- Two source-present local worker processes; authenticated loopback is not
  independent operators or a secure remote deployment.
- Shared native 100/40 Mbps access caps, no added RTT. Stream counters include
  relay framing but exclude kernel headers, ACKs, retransmissions and checkpoint
  distribution.
- SDK, supervisor, dashboard, relay and state audit belong to each measured
  client worker. Source loading, publication, final verification and cleanup
  remain charged in their respective windows.
- The default 25% memory safety margin remains enforced. The 16 GiB suite
  memory budget is whole-topology admission, not a client tier.

The [complete ordinary report](../evidence/gkm-sharing-qwen-streamed-v1.json)
retains all six cases, metric windows, source/build identities and exact replay
configuration. The [prior partial report](../evidence/gkm-sharing-qwen-streamed-partial-v1.json)
retains one initial full-share host-pressure rejection and five completed cells.
Its cells are not pooled with this fresh cohort. The earlier
[six blocked cells](../evidence/gkm-sharing-qwen-blocked.json) remain separate.

## Correctness and resources

All six requests pass final native verification and ordinary target checks.
Accepted-only per-selection commitments agree across full/seeded execution for
complete logits, active KV, history, positions/cache lengths and selected/pending
tokens. They cover **2,155,008 logit/KV values** across twelve selections; this
counts overlapping values across selections, not distinct source elements.
Cryptographic randomness stays fresh. Every monitored request observes zero
host swap growth and no pressure-guard trip.

The separate [checkpoint oracle](gkm-sdk-reference.md#measurement-and-remaining-gates)
compares full and seeded accepted state independently. These benchmark commitments
establish within-cohort agreement, not a new FP32 quality comparison.

## Results

Medians of three fresh processes per choice; MB are decimal, MiB binary.
Percent changes are seeded relative to full. This explicit ablation changes the
privacy assumption: seeded sharing adds the AES PRG assumption. It is not a
same-privacy global ranking.

| Metric and window | Full | Seeded | Change |
| --- | ---: | ---: | ---: |
| All-link TCP streams, covered lifecycle | 87.729 MB | 80.369 MB | −8.39% |
| Covered application bodies, setup through response | 87.228 MB | 79.869 MB | −8.44% |
| Complete request | 88.240 s | 87.398 s | −0.95% |
| Target startup through cleanup | 100.765 s | 99.958 s | −0.80% |
| Client CPU, startup through first response | 51.818 s | 51.706 s | −0.22% |
| Sum of client/worker cold CPU | 86.122 s | 85.962 s | −0.19% |
| Complete-client lifetime peak RSS | 1,832.94 MiB | 1,832.47 MiB | −0.03% |

The exact application-body reduction is 7,359,180 bytes per request; these are
covered bodies, not physical wire. Request times range 88.238–89.354 seconds for
full and 87.126–87.425 for seeded. CPU ranges overlap. Paired RSS changes have
mixed signs; the tiny median difference establishes no client-memory improvement.
Largest client peaks are 1,853.56/1,833.55 MiB for full/seeded. They fit observed
Medium, not Low, and do not certify conservative admission under an explicit
`max_client_memory="medium"` policy or whole-device memory.

## Replay and remaining gates

```sh
uv run --no-sync python -m examples.benchmarks.gkm_sharing \
  --output target/gkm-sharing-qwen.json
```

The exact invocation completed all six cells in the retained fresh cohort.
Source-present short-request traffic savings do not establish longer-output,
Internet-RTT or source-missing performance. Broader held-out and original-model
quality, source-style batched prefill, independent confirmation and the frozen
publication comparison remain open. Final-only release cannot supply model
decode throughput. No claim of thin-client memory savings or independent
cryptographic review follows.
