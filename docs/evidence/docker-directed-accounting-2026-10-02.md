# Directed Docker service accounting

Ordinary Docker benchmarks retain application-body ledgers and separately scoped
IPv4/TCP kernel-hook counters. A namespace-local nftables helper classifies
client traffic versus preparation peer traffic without reading payloads. Peer
traffic is counted only at Inference, avoiding two-endpoint duplication. Helpers
have independent cgroup CPU/memory samples. Provider roles keep their minimal
image and capabilities. Installed wheels retain the measurement Dockerfile.

| Cohort | Application bodies | Directed service IP hooks | Difference |
| --- | ---: | ---: | ---: |
| Generated tiny, 10 ms provider-egress delay, 100 Mbps, seeded 0.1% loss | 217,091 | 311,246 | 94,155 |
| Pinned Qwen2.5, unshaped, two outputs | 269,702,636 | 270,791,237 | 1,088,601 |

Both conserved application bodies; all six prepared directions were observed.
The tiny shaped response passed masked runtime checks. Prepared, verified-prepared
and two-offset Docker SDK requests additionally pass directed counters, helper
resource samples and owned-container cleanup. Profile/backend mismatches are
excluded from matched benchmark ranking.

Reproduce:

```bash
pllm benchmark run --docker --tiny --docker-network-profile examples/benchmarks/docker_link.json --max-output-tokens 2 --temperature 0 --output docs/evidence/docker-directed-shaped-tiny-2026-10-02.json
pllm benchmark run --docker --max-output-tokens 2 --temperature 0 --output docs/evidence/docker-directed-qwen25-2026-10-02.json
python scripts/probe_directed_accounting.py docs/evidence/docker-directed-shaped-tiny-2026-10-02.json docs/evidence/docker-directed-qwen25-2026-10-02.json --output docs/evidence/docker-directed-conservation-2026-10-02.json
```

This is **not complete wire accounting**: kernel hooks see skbs, offload can
aggregate headers, and egress counters precede qdisc drops. Telemetry/DNS/IPv6,
host application API and upstream checkpoint distribution are excluded. Loss and
delay apply to provider egress, not a symmetric WAN. These co-located containers
do not prove independence. Measurement CPU is separate overhead, not free
inference work. One response is not representative compute-cap evidence.
