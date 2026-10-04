# Enforced party access rates and measured token throughput

`pllm benchmark run --wan` now enforces the selected upload/download rates
between local parties. The default is **100 Mbps down / 40 Mbps up per party**,
including the client. Each direction is shared across every peer and connection.

## Bounded Qwen result

One fresh 39-input/8-output request per condition, pinned
`Qwen/Qwen2.5-0.5B-Instruct` revision
`7ae557604adf67be50417f59c2c2f167def9a775`. All three use the same compiled W8A8
`prefix_f32` prepared-residue Experiment, on-demand inventory, compressed
artifacts and greedy selection. No warmup. All eight generated outputs match
across the three conditions; output digest
`1cac53d4c8028517dd1f6e29301031eb3dd7a0f4abc81a4c6169ae2ad478d0fb`.
The common body fingerprint is
`5d631be30158b3ea2a72cae355ce09a3b304bbfddde48b5758c339c346a34974`.

| Per-party access | Full request, s | Online, s | Online TTFT, s | End-to-end tokens/s | Online tokens/s | Decode tokens/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Uncapped Docker control | 27.843 | 11.434 | 8.224 | 0.287 | 0.700 | 2.203 |
| 100 down / 40 up Mbps | 67.482 | 21.938 | 18.534 | 0.119 | 0.365 | 2.081 |
| 20 down / 8 up Mbps | 265.295 | 78.632 | 69.521 | 0.030 | 0.102 | 0.775 |

End-to-end uses eight outputs divided by full request time, including demand
preparation and client bundle delivery. Provider startup and warmups are outside
that request clock. Online excludes preparation. Decode uses **seven** outputs
over the first-to-last output interval: 3.178, 3.363 and 9.033 seconds respectively.
First-token latency is not hidden in a decode-rate figure.

All runs report zero plaintext prompt/token-ID bytes sent. Communication is
unchanged: approximately 71.193 MB online and 236.601 MB setup-inclusive covered
application bodies per response. Those counters are separate from measured
kernel IP-hook counters. The 100/40 analytical online capacity floor is 8.086 s;
the **measured online time is 21.938 s**, demonstrating why the capacity floor
cannot stand in for actual throughput.

Raw canonical reports:

- [Uncapped](wan-emulation-qwen25-uncapped-2026-10-04.json)
- [100/40 Mbps](wan-emulation-qwen25-100-40-2026-10-04.json)
- [20/8 Mbps](wan-emulation-qwen25-20-8-2026-10-04.json)

These are three single-sample, co-located Docker runs with **zero added RTT/loss**.
They are local rate-emulation measurements, not measured public-Internet service,
representative quality, complete physical wire, or independent operators.
Checkpoint files are mounted locally. Helper CPU/memory is reported separately;
the capped path adds routing helpers and a client TCP bridge relative to the
uncapped control. Its latency difference is not attributed to bandwidth alone.

## Enforcement and validation

The existing role supervisor owns one endpoint namespace and one routed access
namespace per party. WAN egress has the shared upload TBF queue; private-LAN
egress has the shared download queue. Routes keep both directions on those
queues. This works without IFB support on Docker Desktop. Logical roles grouped
on a party share the same queues; intra-party traffic stays local.

Service control, TCP ACKs and retransmissions use the capped path. Telemetry and
benchmark management use a separate management bridge. Kernel queue rates are
read back at startup and around every measured response. Missing/mismatched
samples cannot produce a measured-throughput claim. Queue burst/limit parameters
and helper resource samples are retained in each report.

Two simultaneous 4 MiB echo flows validate shared caps rather than per-connection
allowances. At an 8 Mbps aggregate bottleneck, the checked upload, client-download
and grouped-provider-upload cases took 9.527, 9.129 and 9.217 seconds; the payload
serialization lower bound is 8.389 seconds before framing/ACKs and bounded burst.
Exact returned bytes, queue readbacks and owned-resource cleanup pass. Cancellation
also removes the owned namespaces. Prepared, verified-prepared and two-offset
SDK responses pass under enforced profiles.

## Reproduce

From an editable checkout with a running Linux Docker engine:

```bash
uv run --no-sync python -m pllm benchmark run \
  --experiment examples/benchmarks/prepared_residues.py:compact --trust-python \
  --wan --warmups 0 --repetitions 1 --max-output-tokens 8 --temperature 0 \
  --capture-output-digest --timeout 300 --output /tmp/wan-qwen-100-40.json
```

For the slower condition, add `--wan-download-mbps 20 --wan-upload-mbps 8`.
For the uncapped control, replace `--wan` with `--docker`. The retained runs
explicitly used runtime image
`sha256:e1164a11b5b3d1bfc196e688bf5f5b8f9f0ba04262f7eef643e274537dcb280b`;
omitting `--docker-image` builds the current source image. The ordinary report
retains the selected runtime/helper image identities.

`--wan-party preparation:100:10` can override one party; immutable JSON profiles
also support shared role-to-party assignments. `--wan-estimate` computes capacity
floors without throttling. Reforecasting a report under a different profile does
not relabel its original timings as measured under the new rates.
