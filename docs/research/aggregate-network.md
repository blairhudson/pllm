# Fresh aggregate-network hypotheses

The objective is fewer **all-link bytes per generated token**, with bounded
client CPU and memory. Mbps measures a rate, not a per-token cost. Both offline
traffic and cold delivery count; neither faster overlap nor moving bytes to a
peer is automatically a saving.

This round investigates five new constructions beyond the previous overlap,
artifact, mask, nonlinear and boundary-offload screens. These are project
research hypotheses, not claims of priority in the literature.

## Five exact-source candidates

| Hypothesis | Independent gate | Pinned Qwen2.5 result | Disposition |
| --- | --- | --- | --- |
| Terminal-row demand slicing | Backward graph liveness starts at logits **and every persistent state output**; only whole row-independent stage groups can shrink | At 39+8 tokens, the isolated oracle removes 114 remote rows, 527.34 million single-worker integer MACs and 3.64 MB of arithmetic bodies (3.13%); every checked logit and KV bit agrees across three prompts | Compiler-owned `MaskedLinear(prefill_pruning="terminal")`; live issuance still reserves and burns the full row budget, so the oracle's offline saving is not inherited |
| Public-prefix state capsules | Compile an explicitly public prefix once, deliver exact KV, then continue a private suffix under `prefix_f32` | A 256-token prefix produces a 6.29 MB capsule. Three suffixes plus eight outputs preserve all checked logits/KV; arithmetic plus capsule delivery falls 17.37–21.30× | Research-only: trusted in-process producer; needs a portable authenticated source/numeric/state contract |
| Exact integer anchor splitting | Factor quantized `W = R + A S`; evaluate cheap public block sums locally, preserve integer accumulators and quantization boundaries | 32-coordinate blocks save 33.63 kB of correction/output bodies but add 22.36 MB of client anchors and 514.38 million client integer MACs. 128-coordinate blocks save 8.31 kB but add 5.59 MB | Reject on this cold/network/client-cost gate; no client weights added to SDK |
| Lossless output-basis lifting | Public unit-triangular output transform, exact inverse additions, unchanged full-width input masks | 455 selected edges save 5,630 bytes against existing row residues; public forest costs 3,276 bytes, with 20,930 extra client additions. Twelve stages exceed the native signed-i8 residual domain | Reject promotion; tiny residual benefit and incompatible coefficients |
| Lease-relative protocol frames | Recover redundant commitments from admitted inventory and authenticated outer session; retain fresh nonce and one-use ticket | Isolated request encoding saves 277,088 decode upload bytes over seven decode rows; no new client weights/state | `MaskedLinear(request_encoding="compact")`; distinct one-row namespace, explicit session acknowledgement and existing burn rules |

The oracle charges emitted arithmetic rows only. The ordinary SDK benchmark
charges actual preparation, control, artifact and online bodies; use its separate
report for live claims. The anchor/lifting comparison is against existing
per-output row residues, not a padded raw-output strawman.

### Ordinary SDK cohort

One 39-input/8-output cold response per candidate uses the same pinned source,
W8A8 body, greedy sampling and cohort salt. All start with fresh client artifact
caches and retain paged/compressed delivery, row residues, request-sized inventory
and four-stage issuance overlap. Every generated output fingerprint agrees.

| Candidate | Covered setup-inclusive MB | MB/output | Online MB | Cold client CPU s | Cold aggregate CPU s | Request s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Control | 236.559 | 29.570 | 71.193 | 8.672 | 37.221 | 13.274 |
| Compact requests | 236.279 | 29.535 | 70.913 | 8.493 | 36.753 | 13.267 |
| Terminal pruning | 234.349 | 29.294 | 68.983 | 8.542 | 36.740 | 13.214 |
| Combined | 234.068 | 29.259 | 68.702 | 8.529 | 37.118 | 13.456 |

The combined choice saves **2.491 MB (1.05%) covered** and **3.50% online**.
Client CPU is slightly lower and total CPU nearly equal in this single cohort;
request latency is 1.37% higher. This supports an optional byte-saving choice,
not automatic selection or a speed claim. Compact requests alone reduce the
measured N−1 decode rate from 1.594 to 1.554 MB/output.

Pruning reserves all 4,416 stage rows, claims 4,302 and burns 114. The native
material ledger conserves every row, and offline bytes are unchanged. Every
candidate adds zero client body weights, sends zero plaintext prompt/token-ID
bytes, passes runtime/privacy checks and observes zero new swap. The client's
process-lifetime RSS spans sequential candidates, so no independent peak-memory
comparison follows. Cold CPU covers client/dashboard startup and provider
process birth through the response; request time excludes provider startup.
Source checkpoint distribution, full wire, independent operators, representative
latency and a matched two-offset full-compute comparison remain unmeasured.

### Public-prefix cold costs

The large capsule ratio applies to an unusually prefix-heavy, explicitly public
workload. Add the 145.98 MB raw client bundle to both sides and the ratio falls
to **4.46–4.61×**. Add two 988.10 MB checkpoint deliveries as well and it falls
to **about 1.30×**. These are exact artifact lengths plus projected arithmetic,
not HTTP or full-wire measurements. Public capsule computation takes 11.35 CPU
seconds once in this sample. The remaining request's local algorithm CPU is
0.71–0.77 seconds versus 4.14–4.20 seconds; mask expansion, transport and public
artifact authentication are outside those local clear-kernel timings.

All private suffixes stay client-local. A prefix must be declared public before
private input exists; discovering and exporting a prefix from a private prompt
would violate this contract. The snapshot's current seal establishes trusted
client provenance, not authentication of bytes from an arbitrary producer.
Whole-client RSS and authenticated portable import remain open.

## Five potential 100× constructions

The retained 39+32 prepared comparator uses 178.97 MB of covered bodies, leaving
**1.790 MB per response, or 55.93 kB per output**, for a 100× candidate. At 70
executed rows and width 896, one two-party u32 full-width opening costs 501,760
bytes. At most three fit if every other cost were free. This necessary bound
rules out a full-width client opening in every one of 24 layers.

1. **Finite-task private response capsules.** Compile every response of a small
   public task grammar offline, and retrieve a fixed-length continuation through
   two-server DPF lookup. A 16-task, eight-output Qwen table contains 512 bytes;
   native private lookup uses 494 bytes per query and exact response bytes. Both
   worker tables plus one query/reply use 1,518 bytes against at least 126.41 MB
   of arithmetic in the matching decoder controls. However, **downloading the
   512-byte public table and querying locally is better** at this domain size.
   Public compilation costs 67.98 CPU seconds, about 30.3 median clear executions
   including the probe's overhead. This is a finite-task specialization, not
   arbitrary-prompt inference; checkpoint distribution, authenticated distributed
   lookup, full wire and independent cryptographic review remain open.
2. **Secret-basis conjugate decoder.** Attempt to retain a whole decoder behind
   a secret orthogonal basis instead of opening every layer. Rejected: the
   original public linear operator and its transformed operator expose the
   basis through a linear solve. A bounded leakage witness reconstructs the
   private input to within `3.45e-15`. A transform alone is not input privacy.
3. **Layer-span rounding-debt certificates.** Carry exact residual debts across
   several linear/quantized stages and settle them privately at rare cuts.
   The minimal witness already gives `round_even(1/2)*3 = 0` but
   `round_even(3/2) = 2`. Naive delayed rounding fails. A useful construction
   needs private, capacity-bounded debt settlement and a numeric certificate;
   no such executable decoder is established.
4. **Cross-request threshold-SIMD resident decoder.** Put requests in ciphertext
   lanes under a threshold key, retain hidden state, and amortize key switching
   and nonlinear settlements across lanes. Lane packing alone does not remove
   per-client secret-shared work. Complete activation, attention, selection and
   key-switch costs plus latency/privacy under mixed request arrivals are needed.
   No measured improvement is claimed.
5. **One-island distilled decoder.** Jointly train a public-affine recurrent
   trunk with one narrow private nonlinear settlement per output, rather than
   repeating a cut in each original decoder layer. This changes the trained
   model; it cannot be obtained by simply skipping Qwen nonlinearities. A trained
   checkpoint, held-out language quality, exact private rescaling/selection and
   whole-response cost admission are missing. No measured improvement is claimed.

Thus this round establishes neither a general 10× nor a general 100× result.
The constrained capsule results identify workload specialization as a promising
direction, while the two privacy/numeric counterexamples reject shortcuts.

## Reproduce

```bash
HF_HUB_OFFLINE=1 uv run --no-sync python scripts/probe_aggregate_network.py \
  --outputs 8 --output aggregate-network.json

HF_HUB_OFFLINE=1 uv run --no-sync pllm benchmark run \
  --experiment scripts/aggregate_network_experiments.py:control \
  --experiment scripts/aggregate_network_experiments.py:compact \
  --experiment scripts/aggregate_network_experiments.py:terminal \
  --experiment scripts/aggregate_network_experiments.py:combined \
  --factory --trust-python --backend native --warmups 0 --repetitions 1 \
  --max-output-tokens 8 --temperature 0 --capture-output-digest \
  --timeout 300 --output aggregate-network-sdk.json
```

The first command uses bounded real-checkpoint clear kernels and independent
algebra, leakage and liveness oracles. It records source/body/plan locks,
output/logit/KV digests and zero observed new swap, without recording private
payloads. Its whole-process RSS includes the research fixture and is not client
RSS. The second command runs four ordinary Experiments through separately
launched local prepared roles and one cohort salt, with fresh client bundle
caches. Co-located roles do not establish operator independence.

Evidence: [isolated probes](../evidence/aggregate-network-qwen25-2026-10-06.json),
[ordinary SDK cohort](../evidence/aggregate-network-sdk-qwen25-2026-10-06.json).

The next [exact-ring algebra round](network-algebra.md) adds optional stage-packed
input, measures its incremental SDK cost, and tests five further whole-decoder
100× hypotheses.
