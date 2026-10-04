# WAN readiness and five network-reduction gates

## Result

One method reached the ordinary SDK: exact public-bound residue coding for prepared
corrections and online outputs. A pinned Qwen2.5-0.5B W8A8 cohort reduces the existing
lean stack's covered setup-inclusive bodies **507.46 → 490.41 MB (3.36%)** and online
bodies **243.28 → 234.43 MB (3.64%)**. All six responses match outputs and usage.
This is an incremental win, **not a tenfold result**.

The cohort has three ordered requests: 45 input tokens, an identical repeat, then
51 input tokens; each generates 32 tokens. Both configurations use canonical
`prefix_f32`, exact generated-prefix reuse, request-sized/on-demand inventory and
compressed content-addressed artifacts. Client bundle caches are isolated per
configuration. All numbers are decimal MB of covered application bodies; providers
are two co-located children. Source distribution, full wire and client peak memory
remain unmeasured.

Final report: `prepared-residues-final-qwen25-2026-10-04.json`.

| Matched measure | Lean raw control | Prepared row residues |
| --- | ---: | ---: |
| Online bodies, all three requests | 243.276 MB | 234.432 MB |
| Setup-inclusive bodies, all three requests | 507.464 MB | 490.409 MB |
| Online MB/generated token | 2.534126 | 2.441997 |
| Setup-inclusive MB/generated token | 5.286085 | 5.108426 |
| Decode-only MB/generated token after first output | 1.654720 | 1.593734 |
| Setup through first response | 319.028 MB | 310.394 MB |
| Cold first-response client CPU | 16.727 s | 16.638 s |
| Cold first-response aggregate CPU | 65.088 s | 66.574 s |
| Hypothetical online access-capacity floor | 28.111 s | 26.342 s |
| Hypothetical setup-inclusive access-capacity floor | 53.086 s | 51.331 s |

CPU is one sample per configuration, not a demonstrated speedup or compute-cap
admission. Client body weights remain unchanged; public decoded width metadata
adds 304,128 bytes. That is a logical storage count, not a peak-memory sample.
The three-request tenfold target is 50.746 MB. The new first response alone costs
310.394 MB, so eliminating both later responses still cannot meet it.

## Protocol and client boundary

For public quantized row `W[j]` and the committed activation bound `A`, select
`b[j] = max(1, ceil(log2(2 A sum(abs(W[j])) + 1)))`. The existing full-width input
mask is unchanged. Reduce both `W r - s` corrections and `W (x-r) + correction`
outputs modulo `2**b[j]`. The client adds its own mask and center-decodes; the
public bound makes this the original exact integer output. Bias, dequantization
and model arithmetic retain their existing contracts.

Widths depend only on public weights and numeric configuration. Truncating an
independent uniform output pad to its public modulus leaves it uniform; correction
and online masked-input views retain the existing semi-honest/non-collusion
assumption. There is no prompt-dependent width, sparse index or refinement count.
One-use reservations and cancellation burns are unchanged. Native verification
still checks restored full integer outputs before dequantization.

The immutable `MaskedLinear(output_encoding="row_residues")` choice reaches native
admission, role construction, source-bound bundles, preparation, authenticated
inventory and online frames. The layout digest binds each stage, weight digest,
shape, activation precision and width vector. Inference independently derives and
checks correction layouts. Raw and offset frames cannot masquerade as prepared
residue frames. Reuse, compressed artifacts, placement and Freivalds combinations
pass tiny end-to-end checks. Frames are bounded to four million output cells.

Rust owns residue coding, mask arithmetic and reconstruction; Python owns immutable
selection, protocol orchestration and independent oracles. Native mask/unmask also
replaces the existing prepared path's temporary NumPy arithmetic arrays. Raw is
still the default and old default composition identities are preserved.

## Five falsifiable screens

These are new PLLM engineering experiments, not claims of literature-first novelty.
All use pinned public source/shape evidence and explicit privacy/resource gates.

| Method | Checked result | Decision |
| --- | --- | --- |
| Project before degree-reduction resharing | Exact field outputs; 117,012 → 21,780 peer-frame bytes for the real 4,864→896 projection, **5.37×**. The 39+32 projection still costs **36.59 MB** before missing decoder work. | Keep research-only. Changes to three semi-honest parties; three public projections, missing numeric conversion/rescaling, SiLU, attention, normalization and private feedback. It exceeds the historical matched 17.90 MB tenfold body budget on known costs alone. |
| Public-bound prepared residues | Exact integer reference; 39+32 arithmetic-only projection 176.98 → 169.02 MB. Complete SDK result is above, including metadata and control. | Promote opt-in through the existing protocol and benchmark. |
| Signed bit-plane public-weight delivery | Exact raw hash; zlib 124.90 MB versus bit-plane/zlib **131.74 MB**. Decode CPU 0.517 → 0.416 s in the isolated codec. | Reject as a network optimization; it grows bodies 5.47%. No delivery component enabled. |
| Fixed-batch private row retrieval | Eight hidden 896-byte rows match exactly. One logical table traversal replaces eight; replies plus queries remain **21,584 bytes**. Both-worker CPU remains about **6.25 s**. | Keep research-only. Logical traversal counts are not DRAM measurements; independent DPF expansion remains. Private head use still lacks a public candidate-capacity certificate. |
| Public block-orthogonal activation spreading | No training or client body weights. At A6, rotation improves checked prefill matches 4/8→7/8 and same-token decode 21/24→22/24; adds 0.128 s client transform CPU. | Reject numeric promotion: it still fails the locked W8A8 decision gate. The transformed source is dequantized W8 re-quantized after public rotation, not a newly validated checkpoint numeric contract. |

Raw screens are `wan-{resharing,residues,artifacts,retrieval,orthogonal}-qwen25-2026-10-04.json`.
The isolated residue screen predates runtime promotion; its original scope text is
retained as historical evidence. The other four remain SDK research probes, not
executable Pipeline components or cryptographic review claims.

## WAN accounting

`WanConditions` defaults each party to **100 Mbps down, 40 Mbps up**: decimal
12,500,000 and 5,000,000 bytes/second. For a covered window, the capacity floor is
the maximum across parties of total outgoing bytes/upload capacity and total
incoming bytes/download capacity. Simultaneous peers share these totals. Declared
roles on one party share access; intra-party bodies are retained separately.
Authenticated live placement maps roles to actual declared party IDs.

Reports retain the immutable conditions digest, directed bodies, bottleneck,
online/setup-inclusive floors, token-throughput ceilings, scope and missing
measurements. Startup/warmups are charged once. Unknown ledgers remain unknown.
The final cohort bottleneck is Inference's hypothetical 40 Mbps upload. A retained
10 Mbps Preparation-upload sensitivity instead makes Preparation the bottleneck.
These are **analytic capacity floors**, not predicted end-to-end latency, physical
wire, shaped measurements, independent-provider evidence or a readiness certificate.

These historical method cohorts used analytical profiles. `benchmark run --wan`
now enforces shared party rates, including the client, through routed Docker
namespaces. Explicit download/upload, party or profile options also enable it;
`--wan-estimate` selects analytical-only operation. Its separately retained
[39+8 Qwen cohort](wan-emulation-2026-10-04.md) measures actual token throughput
under 100/40 and 20/8 Mbps caps. `WanConditions.link_conditions()` still represents
only a single-flow controlled link, not a shared multi-peer access limiter.

## Implementation diagnostic and limits

The first row layout used one artifact object per stage. Public compressed layouts
now remain bounded inline records in the authenticated manifest; preparation uses
bounded binary layout data. A real-sized fixture caught the existing 512-byte
identifier limit being incorrectly applied to a base64 layout string; binary
encoding preserves that identifier bound. Streamed HTTP rejections now retain
their HTTP status without reading an unbounded error body.

Client profiling also found repeated `ModelPlan.prefill` reconstruction inside the
stage-layout loop: nine checks consumed 6.21 process CPU-seconds, versus 0.116 s for
2,208 native output decodes. Reading the immutable prefill bound once removes that
repeat work while retaining every layout check. The diagnostic instrumented,
overlapping windows in `prepared-residues-client-profile-qwen25-2026-10-04.json`
are not independent timing totals. Earlier 8-output and 32-output layout reports
are retained; the final uninstrumented report above is authoritative for promotion.

## Reproduce

From a built editable checkout:

```bash
.venv/bin/python scripts/probe_wan_methods.py --real --method resharing --output /tmp/resharing.json
.venv/bin/python scripts/probe_wan_methods.py --real --method residues --output /tmp/residues.json
.venv/bin/python scripts/probe_wan_methods.py --real --method artifacts --output /tmp/artifacts.json
.venv/bin/python scripts/probe_wan_methods.py --real --method retrieval --output /tmp/retrieval.json
.venv/bin/python scripts/probe_wan_methods.py --real --method orthogonal --output /tmp/orthogonal.json
.venv/bin/python scripts/probe_prepared_residues.py --real --tokens 32 --requests 3 --output /tmp/prepared-residues.json
```

SDK experiment targets: `examples/benchmarks/prepared_residues.py:control` and
`:compact`. The reference probes use Rust hot paths and independent Python integer
oracles. Further tenfold work must attack both initial client-boundary delivery
and fresh-response protected computation; this round authorizes neither a smaller
private head nor a complete resident-share decoder.
