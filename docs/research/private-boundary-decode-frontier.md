# Coupled boundary decode: five bounded follow-ups

This screen follows the source-present Qwen2.5 150+8 confirmation in
[`private-boundary-tiers-reference.md`](private-boundary-tiers-reference.md).
It investigates exact full-head delivery and execution without restricting the
vocabulary, replacing complete logits with top-k, or changing the numeric profile.
Five separate Sol 6.1 investigations supplied the initial probes; parent review
added direct parallel controls, actual bounded pipelining, a uniform certificate,
test-only native fusion, counterbalancing and build provenance.

## Baseline motivation

The retained complete-response cohort records 9,726,176 head-download bytes out
of 10,185,414 boundary-operation bytes for eight output selections. Thus head
replies account for about 95.49% of those additional operation bodies. This is a
partition of the additional boundary traffic, not 95% of the complete response.

At the declared 40 Mbps upload capacity, two sequential 607,744-byte head
payloads have a 243.10 ms serialization floor. Independent uploads can overlap,
subject to the client's shared 100 Mbps download capacity. This arithmetic
motivates an experiment; it does not attribute the historical decode delay.

## Candidate contracts

| Candidate | Executed bounded reference | Decision constraint |
| --- | --- | --- |
| Parallel and pipelined delivery | Original native ring32 heads; serial, parallel, striped and one-stripe-ahead TCP arms | Complete reconstruction, failure drainage and shared-party pacing; compare pipelining against parallel, not only serial |
| Random-head-share runahead | Fresh `r`, early `Wr`, later `W(q-r)`; ready-serial and ready-parallel controls | Equal independent worker service has zero ideal gain over parallel; benefits require asymmetry or contention |
| Exact residue replies | Source-bound public widths, native packing, independent integer reconstruction, exact float32 outputs | Thin-client uniform bound needs no weight scan; tighter L1 certificates have additional derivation and delivery costs |
| Lookup/random-head fusion | Test-only original packed DPF traversal plus exact wrapping dot product | Charges the complementary head; source-level scan counts do not establish DRAM savings or latency improvement |
| Byte-dot head kernel | Runtime-selected ARM SDOT with balanced signed-byte limbs and wrapping reconstruction | Same immutable weights; packing-inclusive comparison with existing native `Matrix::wrap32` |

Every reference retains complete output coordinates. Generated fixture equality
does not establish real-checkpoint logits/KV, sampling or whole-response costs.
The existing SDK remains the independent control for a later integration gate.

### Runahead is conditional

For body time `T` and independent complete worker service times `A` and `B`,
ready-parallel completion is `T + max(A, B)`; current-step B runahead is
`max(T + A, B)`. If `A == B`, these are identical. A network model must additionally
charge shared client download and body traffic rather than sum isolated savings.

Runahead for the already admitted current token needs no unused next-token head
at natural EOS or the output cap. Cancellation after issuance can waste that
head; the probe retains an explicit unused-tail ablation. The current shared
lookup/head sequence needs an admitted paired reservation; separate stage lanes
are one possible design, not a requirement of the algebra.

For Qwen geometry, one buffered `Wr` is 607,744 bytes and its mask 3,584 bytes.
Framing, native/Python copies and full logits remain additional owners. Mutable
fixture copies are wiped; immutable native replies do not support a secure
backing-erasure claim.

### A thin-client certificate can use public geometry

With signed-i8 weights and symmetric A8 inputs, the conservative bound is
`128 * 127 * 896 = 14,565,376`. Signed 25-bit reconstruction is exact. The uniform
profile transmits one width byte under source/configuration commitments, then
expands widths locally for the existing native codec. It does not ask a thin
client to download or scan its removed vocabulary matrix.

Each 151,936-row reply shrinks from 607,744 to 474,800 payload bytes: 21.875%.
The concrete proposed uniform framing and three-recipient certificate delivery
leave 2,122,510 net application bytes saved across eight heads against headerless
u32 reply payloads. This is a conservative serializer budget, not measured SDK
traffic. Certificate hashes bind identity; trusted external admission must bind
the actual quantizer, source, geometry, scales and codec. They establish neither
certificate authenticity nor malicious-worker correctness. Actual checkpoint row
bounds and a reviewed admission path remain separate evidence.

### Fusion retains both work and scheduling costs

The test-only fused visitor uses the original packed128 DPF domain and one
immutable native matrix. Its 16-byte terminal mask feeds unchanged XOR lookup
and exact `Wr`; query and head tickets retire together. Partial/full faults and
cancellation burn both identities. No raw selection mask crosses the native
boundary.

Across two workers, separate lookup plus both heads logically reads
`4VH + 8V` source bytes; the optimistic fused layout reads `3VH + 8V`.
The full-token ceiling is about 24.94%, not 50%. This counts source visits, not
physical memory traffic. Joint lookup/head completion can also delay the start
of the decoder body. The fixture remains capped at 4 MiB of generated weights.

## Completed serial measurements

Original reports, exact commands, build/binary hashes and host-monitor observations
are preserved in
[`boundary-decode-frontier-2026-10-15.json`](../evidence/boundary-decode-frontier-2026-10-15.json).
The read-only
[`summary`](../evidence/boundary-decode-frontier-2026-10-15.summary.json)
checks report hashes, complete groups, exact output commitments, build/geometry
identity, owner retirement and byte conservation before calculating these tables.
All thirteen monitored invocations succeeded with zero observed swap growth.
Compilation preceded measurement. This is a generated-source operator screen;
its resource observations cannot inherit the earlier complete SDK client's Low tier.

### Full-geometry delivery

Four fresh processes each run four counterbalanced repetitions of all four arms
at 151,936×896 under shared native 100/40 Mbps party capacities. Each table value
is the median of four process medians. Every reconstructed integer and full
float32 logit matches its independent generated-source oracle.

| Delivery | Head exchange | Aggregate fixture CPU | Application stream bytes |
| --- | ---: | ---: | ---: |
| Serial | 293.233 ms | 70.964 ms | 1,223,276 |
| Parallel full replies | 148.515 ms | 63.582 ms | 1,223,276 |
| 8,192-row stripes | 128.961 ms | 91.686 ms | 1,229,176 |
| Stripes with one native call ahead | 129.425 ms | 96.592 ms | 1,229,176 |

Parallel replies lower this isolated exchange 49.35%. Stripes lower it another
13.17% against parallel, but raise fixture CPU 44.20% and add 5,900 framed bytes.
Explicit native-call pipelining provides no additional measured gain here. Socket
buffering already permits some overlap in the striped control; the distinct
pipeline arm bounds an actual next native call to one per worker.

The single fixture process owns generated weights, full/stripe native matrices,
logical workers, client and oracles; full and striped matrices share one native
executor. It does not measure two independent worker processes or a whole thin
client. TCP connection, framed exchange, native calls, reconstruction, checks and
cleanup are timed; fixture construction is separate. HTTP/TLS, physical wire,
Internet RTT and checkpoint distribution are absent.

### Native and codec follow-ups

| Probe | Measured result | Scope |
| --- | --- | --- |
| Byte-dot API | 6.7535→4.23975 ms CPU per head; 37.22% lower | Six fresh Apple M5 native workers, four alternating samples each; full 151,936×896 generated matrix |
| Fusion, 1,025×896 | 0.8091 fused/separate elapsed ratio | Three fresh workers; 32 repetitions per worker |
| Fusion, 4,096×896 | 0.8213 fused/separate elapsed ratio | Three fresh workers; 32 repetitions per worker |
| Uniform 25-bit residues | 474,800 bytes per party; 21.875% smaller payload | One full-geometry generated fixture, complete integer/float32 parity |
| Runahead | Zero ideal gain over equal-service ready-parallel | Actual owner controls plus separately labeled body/service models |

The byte-dot figure includes input packing and compares a preallocated candidate
against the allocating `Matrix::wrap32` API. It establishes an API-level result,
not isolated SDOT instruction speed: an allocation-matched control remains needed.
No extra persistent weight owner is added. Both paths and the oracle coexist in
each measured native process, so its peak is not a candidate/control memory result.

Fusion ratios are medians of **worker-level ratios of aggregate elapsed times**,
not medians of individual paired-sample ratios. The small matrices fit a different
cache regime from Qwen's complete head. Original DPF lookup and full head values
match; neither full-checkpoint timing nor physical DRAM savings is measured.

In the uniform codec sample, native packing costs 0.277/0.280 ms CPU per worker;
client frame validation costs 1.146 ms, checked reconstruction 0.292 ms and common
dequantization 0.061 ms. The raw client centering control costs 0.281 ms. These
ten-repeat, one-process measurements keep conversion overhead visible; they are
not a request-speed comparison. Tighter row-L1 derivation plus independent
recomputation costs 8.31 seconds on this generated matrix and would require
136.74 MB of weights/scales for the implemented client verification. The uniform
profile avoids that scan; its metadata construction/verification costs 55 μs.

## Next implementation gate

1. **Parallel full replies first.** Preserve complete numeric behavior and bytes
   while binding both outstanding requests to one admitted attempt. Drain both
   workers on cancellation, partial failure or malformed replies.
2. **Uniform residues as a separate exact codec.** Bind the public 25-bit proof,
   source, quantizer and layout before issuing masks; include fresh framing,
   certificate distribution and client conversions in the response bill.
3. **Byte-dot native follow-up.** Match output-allocation contracts, then retain
   exact runtime feature dispatch and scalar fallback before SDK selection.
4. Keep stripes as an explicit CPU/latency tradeoff. Fusion needs full-geometry
   evidence and a schedule that does not delay lookup/body execution; runahead
   needs measured worker asymmetry or contention beyond the parallel control.

A promoted combination must pass real-checkpoint full logits, active KV,
selections, history and RNG checks, then fresh complete-response CPU/RSS/traffic
measurement against the current coupled boundary and tier incumbents. Isolated
gains cannot be multiplied. No new SOTA or Low/Medium qualification follows from
this screen.

## Reproduction and narrower contracts

- [`boundary-head-duplex-probe.md`](boundary-head-duplex-probe.md)
- [`boundary-head-runahead-probe.md`](boundary-head-runahead-probe.md)
- [`boundary-head-residues-probe.md`](boundary-head-residues-probe.md)
- [`boundary-scan-fusion-probe.md`](boundary-scan-fusion-probe.md)
- [`boundary-head-dotprod-probe.md`](boundary-head-dotprod-probe.md)

Each note links its executable fixture, independent correctness controls,
allocation bounds, omissions and exact command. Timing probes run serially after
agent work finishes. Baseline and candidate samples share each fixture's numeric
contract; results from different fixtures are not multiplied into a serving claim.

Recompute the archived summary without rerunning timings:

```sh
target/frontier-sdk-env/bin/python scripts/summarize_boundary_decode_frontier.py \
  docs/evidence/boundary-decode-frontier-2026-10-15.json \
  --output target/boundary-decode-frontier-summary.json
```

Validation: 128 probe tests plus 22 subtests, three Rust fusion tests and eleven
read-only summary rejection/reconciliation tests. The explicitly ignored Rust
timing fixture is exercised by the separate serial measurement commands.
