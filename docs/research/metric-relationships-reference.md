# Low/Medium frontier and metric relationships

This is a read-only analysis of existing, hash-checked scorecard cohorts. The
interactive matrix and scatter plots live on **Research → Metric relationships**
(`/research/relationships/`) and inside **Research → SOTA**. The dedicated page
opens the seven-prepared-configuration tier confirmation with Low RAM selected;
model, cohort, topology and RAM controls reuse the same checked scorecard.
No new serving samples are collected by this analysis.

## Current frontier

Low is **256 MiB** whole-client RSS; Medium is **2 GiB**. These are nested maxima,
not disjoint bins. Eligibility uses the **largest recorded client peak across
replicates**, while table and correlation axes retain the original metric
summaries. Observed eligibility and conservative runtime admission are separate.

For **Qwen2.5-0.5B, 150 input + 8 output tokens, native shared 100/40 Mbps**, the
principal measured choices fit Low and therefore also fit Medium. There is no
separately demonstrated Medium-only winner in these two matched cohorts:

| Objective | Measured choice | Evidence |
| --- | --- | --- |
| Smallest client footprint in the eight-recipe tier confirmation | MLP tiling + digit8 | 185.56 MiB largest client peak; 185.25 MiB median |
| Client CPU in the eight-recipe tier confirmation | Coupled boundary | 19.13 CPU s; 2.42% below the best prepared median, with overlapping replicate ranges |
| Cold traffic and complete request latency in the later three-choice confirmation | Parallel/uniform-residue private boundary | 365.22 MB; 75.03 s; 224.13 MiB largest client peak |
| Decode in that later confirmation | Maintained prepared | 4.92 tokens/s versus boundary 3.31 |
| Prepared decode in the eight-recipe tier confirmation | MLP tiling + digit8 + selective placement | 4.39 tokens/s; at least 90% body MACs remain remote |
| Online first text in the eight-recipe tier confirmation | Digit8 + selective placement | 37.38 s |
| Preparation CPU in the eight-recipe tier confirmation | MLP tiling + digit8 + selective placement | 10.65 CPU s in the reported cold-first window |
| Preparation peak in the eight-recipe tier confirmation | Serial digit8 issuance | 186.41 MiB median provider peak |

These rows keep their original cohort comparisons. In particular, 4.92 and 4.39
tokens/s above were measured in different cohorts and cannot rank the two
prepared recipes against one another. The later confirmation compares boundary
against its fresh maintained control: **23.70% less traffic, 24.18% shorter
requests, 32.68% slower decode and 8.00% later online first text**. All checked
state/output commitments match, with zero observed swap growth. Boundary
execution retains two additional workers and its additional non-collusion
contract. Source-present measurements exclude checkpoint/tokenizer distribution
and physical wire.

The subsequent byte-dot screen did not improve serving decode. The installed
parallel/residue choice remains the confirmed boundary delivery choice; the
byte-dot kernel is an explicit non-winning screen, not a replacement champion.

Medium matters for other workloads. In the separate **Qwen3-4B 16+8** resident
versus paged pair, resident client weights use **973.19 MiB** and take **142.67 s**;
paging uses **248.95 MiB** and takes **155.46 s**. Medium admits both observed
peaks, whereas Low admits only paging. This is a single-pair short-workload
result, not a correlation sample or a quality ranking against Qwen2.5.

## What currently moves together

The broad same-build tier confirmation has seven prepared configurations and
one coupled-boundary configuration, each summarized from three clients. All
eight fit Low. For the **seven prepared configurations alone**:

| Pair | Pearson r | Spearman ρ | Interpretation within this sample |
| --- | ---: | ---: | --- |
| All-link traffic / decode throughput | −0.999 | −0.893 | Selective placement lowers traffic and raises decode speed; the configurations form placement groups |
| All-link traffic / online TTFT | +0.879 | +0.857 | Lower traffic accompanies earlier online first text |
| All-link traffic / client CPU | −0.817 | −0.429 | Moving selected linear work to the client exchanges bytes for local CPU |
| Client peak / online TTFT | −0.436 | −0.464 | Tiling's smaller footprint accompanies longer first-text delay; other changes weaken the relationship |
| Preparation CPU / aggregate CPU | +0.996 | +1.000 | Preparation drives much of the CPU variation; it is also a summand of aggregate CPU |
| All-link traffic / complete request latency | +0.102 | −0.036 | Total bytes alone explain little of request variation among these prepared recipes |
| Client peak / all-link traffic | +0.021 | −0.679 | Rank order exaggerates sub-kilobyte differences among several recipes; inspect magnitudes, not ranks alone |

These are descriptive correlations of deliberately chosen configurations, not
random independent treatment assignments. They identify patterns to investigate,
not causal effect sizes or statistical significance. An apparently large
coefficient may describe one placement split rather than a continuous response.

### Mixing topology changes the answer

Adding the single original serial/u32 coupled-boundary choice changes traffic
versus decode's Pearson coefficient from **−0.999 to +0.825**, and traffic versus
complete latency from **+0.102 to +0.962**. The boundary removes cold client head
delivery but adds online head exchanges. Thus it improves cold completion while
slowing decode. Dropping this one configuration restores the prepared-only
coefficients; the cross-topology result is highly sensitive to the population.
The scatter plot colors topologies separately and the matrix defaults to one
topology. Cross-topology comparison requires an explicit selection.

The strongest mechanism evidence comes from the separate pacing trace:
**109.74 ms** paired head exchange includes **92.68 ms** of overlapping pacing
waits. A worker's roughly 475 kB head reply needs about **95 ms** at its 40 Mbps
upload cap. Faster native head arithmetic therefore has limited leverage unless
it also changes bytes or their overlap with execution. Those role-local and
paired durations are distinct scopes and cannot simply be subtracted or summed.

### Useful relationships, not interchangeable objectives

- **CPU is work; latency is elapsed time.** More parallel issuance can raise
  peak memory while reducing waiting. Faster Preparation arithmetic can improve
  cold completion without changing online decode or its bytes.
- **RAM is a live maximum.** Tiling, paging and compressed delivery change
  allocation overlap, CPU, I/O and exchange count differently. A smaller tensor
  payload does not ensure a smaller process peak.
- **Cold bytes and per-token bytes have different leverage.** For one fixed
  workload/lifecycle, all-link traffic contains public delivery, prefill, actual
  decode, prepaid-unused material and controls. Complete base/per-token
  attribution remains unknown; dividing by output count does not recover it.
- **TTFT and decode price different phases.** With aligned event windows,
  request elapsed consists of pre-online setup, online first-text delay,
  post-first-text decode and finalization. Decode duration is `(N−1)/decode_tps`
  for the measured text-event convention. Medians of phases need not add to the
  median of the complete request.
- **Traffic is directed.** Relevant bandwidth floors use each party's shared
  upload/download load, not aggregate all-link bytes divided by one bandwidth.
  Dependent round trips, compute and queueing remain additional constraints.

## Analysis contract

`docs/lib/research-relationships.mjs` is shared by the UI and read-only CLI:

1. Select exactly one checked scorecard cohort: source, numeric identity,
   workload, lifecycle, build and original metric contracts remain intact.
2. Apply the existing largest-replicate RAM gate and selected topology. Do not
   merge historical cohorts, cold/repeat windows, workloads or private salts.
3. Use one **existing configuration summary** per point. Repeated clients are
   not independent configuration points. This describes differences between
   recipes, not within-recipe run-to-run covariance.
4. Exclude missing/nonfinite observations **pairwise**. Preserve zeros. Show
   each pair's count and both observed ranges; never impute unavailable metrics.
5. Calculate Pearson on raw values and Spearman with average ranks for exact
   ties. Constant axes have no coefficient. Below five paired configurations,
   show raw points but no coefficient. Five is a display floor, not evidence
   of significance.
6. Recompute after omitting each configuration. Show the defined min/max and
   constant omissions. This sensitivity range is **not a confidence interval**.
7. Preserve physical signs: positive means raw metrics rise together. Decode
   speed has the opposite optimization direction to costs. The supplementary
   direction-alignment label accounts for that difference.

The nine axes cover the seven priority metrics plus complete request latency
and aggregate CPU. Algebraically derived request throughput and whole-request
MB/token are omitted as redundant axes. Aggregate CPU includes role CPU, so
that relationship is partly mathematical. CPU, traffic, process-lifetime RSS
and online timing can have distinct measurement windows; the cohort description
and scoped labels retain them. A RAM filter can itself alter the correlation.

## Replay

From the repository root, with the existing docs Node dependencies:

```sh
node docs/scripts/research-relationships.mjs --cohort qwen25-private-boundary-tiers-150-8 --memory low --topology Prepared
node docs/scripts/research-relationships.mjs --cohort qwen25-private-boundary-tiers-150-8 --memory medium --topology Prepared
node docs/scripts/research-relationships.mjs --cohort qwen25-private-boundary-tiers-150-8 --memory low --topology Prepared --include-topologies
node --test docs/tests/research-relationships.test.mjs docs/tests/research-memory.test.mjs
```

The command validates the original registry/evidence via `buildScorecard()` and
prints deterministic JSON including report hash, metric labels, original row
summaries, raw points and all 36 pairwise relationships. It does not rewrite
evidence or collect timing. Low and Medium have identical seven-point populations
in this cohort. Three-choice finalist confirmations remain useful paired
comparisons but cannot supply the five-point display floor.

Original evidence and contracts:

- [Tier-incumbent confirmation](private-boundary-tiers-reference.md), cohort
  `qwen25-private-boundary-tiers-150-8`, report SHA-256
  `0bc6061e24f41fdd12ae60c67c9e313ad8551fbefdb36e92bceadfd88e12e1ca`.
- [Independent boundary confirmation and longer-output screens](private-boundary-decode-followup.md),
  cohort `qwen25-private-boundary-decode-confirmation-150-8`, report SHA-256
  `333b652c4f09aff213012ecff083f40471c8f6a2418ab053eb29fb968d724c34`.
- [Byte-dot non-winning SDK screen](private-boundary-byte-dot-sdk.md).
- [Worker-upload pacing attribution](private-boundary-pacing-profile.md).
- [Qwen3 resident/paged client pair](qwen3-client-memory.md), cohort
  `qwen3-client-memory-16-8`, report SHA-256
  `3ebe715a7482b5e5d91f26b1895e1d9a9004bb638a144a8f8204b44c12809654`.
