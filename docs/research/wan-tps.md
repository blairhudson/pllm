# WAN-constrained throughput: next five experiments

The objective is higher measured end-to-end and decode tokens/second under
shared per-party access caps, preserving the admitted numeric and privacy
contracts and keeping client compute and memory bounded. Defaults remain
100 Mbps download and 40 Mbps upload. Add the existing `LinkConditions` delay
profile to measure dependency costs as well as bandwidth; a 20 ms egress delay
at each party adds 40 ms to a two-party request/reply path before other costs.

Controls use the same checkpoint, W8A8 body, input/output cohort, sampling,
cache state and one-use inventory policy. Count preparation and public bundle
delivery in cold request latency, and use N−1 outputs for decode throughput.
Neither lower body bytes nor a synthetic link calculation is a TPS result.

## Hypotheses

| Method | Hypothesis | First executable gate | Promotion gate |
| --- | --- | --- | --- |
| Seed-first worker rendezvous | A seeded worker can compute and return its share while the client constructs/uploads the other share; sequential worker RPCs unnecessarily serialize independent work | Two authenticated committed sessions, exact prefill/decode logits and KV, either-side failure/cancellation burns both | Matched WAN improvement over seeded/residue offset; unchanged bodies, bounded extra client thread/buffers |
| Entropy-coded public artifacts | Small signed quantized values have exploitable symbol entropy that zlib's byte-string model misses | Native bounded lossless codec with raw fallback, strict frame/header/size/hash checks, independently checked output | Transfer plus client decoding beats zlib under selected caps; no private tensor compression or extra client model ownership |
| Coalesced public-object fetches | An authenticated batch of missing objects amortizes many serial request round trips without discarding individual cache identities | Bounded ordered manifest-bound object stream, complete per-object verification, partial/corrupt delivery cannot activate a bundle | Lower cold latency at matched cache state; bounded transient memory and exact existing raw bundle |
| Windowed preparation | Independent stage jobs can overlap computation, upload and acknowledgement without moving preparation online | Bounded outstanding jobs/bytes; immutable authorization, atomic per-stage consumption and cancellation; seal only after every acknowledged push | Higher cold/request throughput without excessive aggregate CPU or retained material; all issued/consumed/burned rows reconcile |
| Duplex prefill chunks | Streaming public-sized row chunks permits response download while later masked rows upload, reducing sum-of-directions stalls | Exact compiled batch result, fixed public chunking, bounded window, one-use row accounting and authenticated ordered results | Prefill/TTFT improvement beats added framing, RTT and smaller-kernel CPU; decode remains unchanged |

These are engineering hypotheses, not cryptographic novelty or established
speedups. Rust owns entropy coding, ring arithmetic, seed expansion and matrix
work. Python owns bounded I/O scheduling, immutable SDK choices and benchmark
orchestration. Existing source/plan/bundle admission must precede material
reservation. Every role keeps its original share and trust boundary.

## Measurement sequence

1. Confirm rate **and delay/loss** kernel readbacks and a small actual RTT probe.
2. Screen exact codecs and scheduling on bounded fixtures; reject losers early.
3. Expose passing choices through existing components, compiler admission and
   ordinary SDK/gateway/benchmark paths, preserving defaults and old digests.
4. Run bounded pinned Qwen cohorts at 100/40 Mbps with 40 ms added RTT;
   keep earlier zero-delay controls separate and use slower access only when it
   distinguishes a live hypothesis.
5. Retain output/logit/KV parity, all-link bodies, setup/online/decode timing,
   client/aggregate CPU and measured or explicitly unknown memory. Commit each
   completed method, including vetoed screens.

Tenfold fresh-response improvement remains a research goal. Combinations must
be measured together; independent percentage gains cannot be added.

## Outcome

The [matched cohort](../evidence/wan-tps-qwen25-2026-10-04.md) and
[SDK guide](../content/docs/sdk/evaluate/wan-tps.mdx) record all five screens.
Seed-first improves offset decode TPS 1.94×; public-object coalescing plus bounded
preparation overlap improves prepared request TPS 17.3% at 100/40 Mbps and 40 ms
added RTT. Online bodies do not shrink. Native entropy coding loses the byte and
client-CPU gate; duplex prefill's incremental runtime benefit is inconclusive.
Four live choices are explicit and defaults retain their previous identities.
The next decisive bottleneck is the number of dependent client/provider cuts per
decoded token; these overlap/control changes do not remove those cuts or meet 10×.
