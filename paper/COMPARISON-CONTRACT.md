# PLLM publication comparison contract, version 1

Protocol date: 16 October 2026. Source audit: repository revision
`99e10e2c1e0e3dbeddc5f56dcc830e549cb1f2a5`.

This document fixes the normative part of P1-A for both papers. “Must” identifies
a qualification requirement. A campaign must freeze its remaining workload,
security evidence, configurations and resource manifest before collecting results.
Defining a requirement does not mean that the SDK enforces it or that an existing
cohort satisfies it. Section 10 identifies those gaps. The accompanying
`paper/comparison-contract.v1.json` records constants and the historical sensitivity
checks; it is publication metadata, not an SDK configuration or JSON Schema.

## 1. Unit of comparison and claim

A comparison cell fixes the execution track, reference computation, security
profile, source, prompt, sampling and stopping rules, public workload envelope,
lifecycle, client ceiling, outsourcing requirement, provider budget, network,
instrumentation, hardware and installed build. Replicates repeat that cell in
fresh processes. Changing one of these identities creates another cell, even when
its displayed model name is unchanged.

Paper 1 reports separate objectives and their tradeoffs. Paper 2 minimizes
all-link TCP-stream bytes in the declared request lifecycle, subject to the
qualification gates below; complete-request latency is secondary. Application-body
bytes are a separate supporting metric. A claim of lowest communication must name
its measurement plane and eligible population. No weighted score substitutes for
the seven metrics. A Pareto result is not a minimum on every axis.

All eligible contemporary methods remain in the comparator inventory. Missing
code, failed reproduction, absent source access and an unsupported PLLM adapter
are unresolved comparisons, not evidence that another method loses. Component
measurements cannot fill a complete-decoder cell. Client-only execution is a
computational reference and cannot win the predominantly remote track.

### 1.1 Exact-execution track

Freeze the model configuration, tokenizer and chat/input formatting, checkpoint
revision and artifact digests, weight/activation quantization, scales, rounding,
overflow behavior, nonlinear expressions, attention/state layout and sampler.
W8A8 is a family name, not a sufficient numerical identity. The control is the
unprotected implementation of precisely that numerical computation, with an
independent numerical reference evaluated separately.

Qualification requires equality of complete sampled logits, active keys and
values, positions and cache lengths, executed token history, pending selected
tokens, selections, output and sampler/RNG state at declared boundaries. Require
greedy and predeclared stochastic trajectories in conformance validation, plus
continuation, cancellation and failure-retirement controls. Do not infer equality
from selected tokens or decoded text alone. Sampled-logit equality does not claim
equality of intermediate tensors that were not checked. A changed approximation
belongs in the quality-matched track unless exact equivalence is established.

The installed `BenchmarkTarget.response(..., state_audit=True)` hashes complete
logits and active KV at each selection, position/cache lengths, selected tokens
and RNG states. It uses greedy temperature zero and sampler seed 42, bounds the
audit to 64 selections, requires a cold native serial response, and charges its
work in the measured client. It does **not** publish executed-history or pending-
token commitments. Those require separate conformance evidence; the diagnostic
`scripts/benchmark_private_boundary.py` has a broader stored-state witness but a
different historical client-memory scope. Neither witness may be silently
substituted for the other. A common private cohort salt permits matched commitment
checks; cryptographic protocol randomness must remain fresh. Public readers cannot
recover private prompts or authenticate execution from those commitments.

### 1.2 Quality-matched track

This track permits numerical changes, not weaker privacy or an easier task.
Before evaluating candidates, freeze the task datasets and versions, development,
calibration and held-out splits, reference model, formatting, scoring code,
decoding policy, per-task sample sizes and a numerical non-inferiority margin for
each primary task. Each margin must have a task/application rationale. An absent
margin leaves this track unqualified; there is no retrospective “similar quality”
exception and no universal accuracy tolerance in this contract.

Let degradation be reference score minus candidate score for a higher-is-better
metric, with the sign reversed for a loss. Each task qualifies only if its
predeclared one-sided confidence upper bound is at most its margin. Use matched
examples and a fixed analysis seed; specify the estimator, handling of failed
answers and dependence among examples before execution. For multiple primary
tasks, use 95% familywise coverage, for example Bonferroni-adjusted per-task bounds.
Failure on one primary task cannot be hidden by averaging gains on another.
Report sample sizes, raw scores, intervals and actual output lengths. Performance
measurements must use the same qualified candidate configuration and stopping
semantics. Shorter answers are not an equal-work throughput improvement.

Full-state equality to the candidate's own clear implementation remains a useful
correctness check, but it does not establish equality to the exact-track control.
BF16/FP32 fidelity, held-out task quality and exact quantized execution are three
different claims. The ordinary response target provides no task-quality evaluator
and fixes temperature zero; a separate frozen quality evaluator is required.

## 2. Security, leakage and additional workers

The common functionality is inference with public weights and private client
inputs, intermediate values and outputs. State explicitly whether model privacy,
output integrity, availability or side-channel resistance is claimed. Public
weights/configuration, stage geometry, public capacity bounds, executed lengths,
message sizes, timing, public stage-level access patterns and termination are
permitted leakage in the baseline profile. This does not permit sending token IDs
or secret lookup indices under the label “access pattern.” Microarchitectural
side-channel protection is not established by the software transcript contract.
List any additional cache-hit, equality, prefix, certificate, publisher or reuse
leakage. A public prefix is not a private prompt. An application requiring less
leakage needs another profile and corresponding measurements.

Name the trusted client and every helper, publisher/dealer, provider and operator;
record credential and preprocessing custody, erasure, setup assumptions,
cryptographic parameters and session/replay limits. Adversary sets concern all
transcripts and retained state over the entire measured lifecycle. A party cannot
be treated as independently corrupted in two different roles at different times
when the adversary retains both views. Authentication and a source digest do not
establish independent operators or confidentiality.

| Profile | Permitted adversary and qualification rule |
| --- | --- |
| `clear-reference` | Trusted local computation; no remote confidentiality claim. |
| `sh1-public` | Static semi-honest corruption of at most one provider role, with honest client/setup and the stated Preparation erasure obligation. Allowed coalitions are the empty set and each provider singleton. Every coalition of two or more providers is outside this profile. Each protocol/composition still needs its own argument for this guarantee. |
| `sh-pairwise-composed` | A stronger, separately reviewed collusion profile. For prepared body plus boundary workers, allowed maximal coalitions are `{Preparation, A}`, `{Preparation, B}`, `{Inference, A}`, `{Inference, B}` and their subsets. Any coalition containing `{Preparation, Inference}` or `{A, B}` is forbidden; all triples and the quadruple are therefore forbidden. This profile remains unqualified for the coupled method until the complete composition argument covers those four cross-pair views. |
| `verified-linear` | Declared confidentiality profile plus a bounded integrity guarantee against malicious Inference for specified delegated linear stages, with honest Preparation. Record failure bits, total attempts, reuse horizon and uncovered operations. This is not a malicious-secure decoder. |
| `malicious-complete` / `hardware-trusted` | Separate profiles with explicit corrupted coalitions, abort/security guarantees or hardware/attestation assumptions. They are not supplied by a verification flag or by framework membership. |

For two-offset workers, `sh1-public` permits `{worker_a}` or `{worker_b}`, never
their union. For prepared execution it permits `{Preparation}` or `{Inference}`
under the stated honest-erasing setup model. For the coupled topology it permits
each of the four singletons. This is a conservative common comparison class,
not a new proof. The installed role graph encodes only the two separation pairs
`Preparation/Inference` and `A/B`; it does not certify the four cross-pair
coalitions above. Paper 2's composition analysis remains necessary even for the
weaker single-corruption claim.

Cross-topology cost tables may compare eligible `sh1-public` implementations only
with the changed number of operators, setup assumptions, and provider budget
visible. Keep topology-specific leaders as well. Mapping four roles to two
operators would allow a corrupted operator to observe cross-pair coalitions and
requires the stronger analysis. Local co-location is an execution experiment,
not evidence of any independent-operator deployment. Do not modify privacy
digests to make the ordinary comparison API rank different topologies. Use a
reviewed, explicitly labeled cross-topology analysis with the common conditions
stated; otherwise mark the comparison unranked. A stronger-security method can be
reported in the common class at its actually measured costs without erasing its
additional guarantees or assumptions. The coupled installed path currently
rejects verification, so it has no measured `verified-linear` arm.

## 3. Client-memory qualification and rationale

| Nested tier | Ceiling in bytes | Display |
| --- | ---: | ---: |
| Low | 268,435,456 | 256 MiB |
| Medium | 2,147,483,648 | 2 GiB |
| High | 17,179,869,184 | 16 GiB |

These are explicit powers-of-two resource conventions. The eightfold spacing
provides a reproducible, coarse logarithmic budget grid rather than many nearly
identical labels. Low tests whether inference ownership can be constrained to a
small fraction of a GiB; successive tiers relax that allowance by a fixed factor.
This is a design-space convention, not an industry standard, a survey of installed
RAM, a demographic classification, or evidence that a device/application is
suitable. Device suitability also depends on unmeasured memory owners, other
applications, latency and energy. Report exact byte peaks so the grid does not
hide continuous tradeoffs.

For one configuration and declared workload scope, qualification is

`max(complete_client_peak_bytes for every accepted fresh-process replicate) <= ceiling`.

For a claim spanning several prompts/caps, take the maximum across all of its
claimed cells, while retaining cell-level results. A missing peak makes
qualification unknown; a measured overrun fails the relevant tier. A median or
percentile cannot hide an overrun. Use inclusive comparison in bytes before
rounding for display. Lower tiers are subsets of higher ones.

The client owner includes SDK, tokenizer, gateway if used, supervisor, dashboard,
relay, state audit, private caches and all client-side helpers. In the ordinary
native response target these serving owners share one fresh worker; use
`process.peak_rss` through target finalization for tier qualification, and retain
`client.peak_rss` through the scenario as the original SDK metric. Check that the
complete peak is at least the scenario peak. The final observation precedes
serialization of the result envelope; it is not an OS observation after worker
exit. The idle suite launcher may be
excluded only while it performs no serving work. If a helper moves outside the
worker, a simultaneous complete-client process-group measurement is required.
Do not add unrelated lifetime peaks. Disclose RSS shared-page double counting and
sampling intervals when using a sampled sum; do not call a sampled maximum a hard
upper bound. Provider memory, kernel filesystem cache, private disk files, GPU/
device allocations and total device RAM have separate scopes. Resident mapped
pages already charged to process RSS must not be subtracted as “file cache.”

`Sampling.memory_bytes` is a conservative target/whole-topology admission input,
not a client-tier setting or native per-process hard limit. The installed response
path runs `benchmark_memory` and adds 32 MiB of audit headroom; it also has target
and host-pressure checks. `TargetBudget("process.peak_rss", maximum=...)` assesses
an observation after execution. Record all three: the requested/estimated
admission, runtime pressure outcome, and observed tier. A Low observation does
not certify that a topology would be admitted under a 256 MiB host budget.

Reject publication timing/RSS samples with a pressure trip or any observed host
swap growth; preserve the raw attempts. This is stricter than the native watchdog,
whose swap-abort threshold is 256 MiB. Zero observed growth means neither zero
pre-existing swap nor absence of unobserved paging. Predeclare host reserve,
baseline pressure telemetry and the rerun policy in Section 8.

### 3.1 Required threshold sensitivity

For every ceiling `C`, report the eligible set and per-objective frontier at
`3C/4`, `7C/8`, `C`, `9C/8`, and `5C/4`, using the same unrounded peaks. The Low
grid is 192/224/256/288/320 MiB; Medium is 1.5/1.75/2/2.25/2.5 GiB; High is
12/14/16/18/20 GiB. These are analysis thresholds, not new SDK admission settings.
Also publish each configuration's exact observed crossing point and distance from
the nominal ceiling. A frontier crossing is not a discontinuity in the algorithm.
If all observations lie far below a tier, report that there is no information
about behavior near that ceiling rather than claiming its value is empirically
optimal. Historical sensitivity in Section 11 supplies context, not a new campaign.

## 4. Source and lifecycle identity

Freeze the full `Experiment`/`BenchmarkCase` and `BenchmarkSuite` specs, repository
revision and dirty-state/source hashes, installed `BuildRef` digest, Python/native
extension/dependency identities, hardware/OS, thread and accelerator settings,
source-lock and actual artifact identities, tokenizer/index/certificate versions,
request envelope and schedule binding. `BuildRef` hashes installed SDK and native
code; it does not replace hashes of external drivers, prompt manifests, task
scorers or publisher tools. A local path or source-config alias is not a checkpoint
digest. Archive legitimate source-acquisition instructions without redistributing
restricted weights. Freeze compressed artifact sizes and source-certificate
publication separately from response delivery.

| Cell | Initial condition and charged work | Current driver mapping |
| --- | --- | --- |
| `source-present-cold-client` | Checkpoint/tokenizer sources already available to workers; fresh client-owned public cache, fresh private material, fresh processes. Price delivered bundles and new request setup. Retain source provisioning/publication as separate costs. | `Scenario.cold`; fresh temporary role root. Shared checkpoint/compiled caches are not cleared. Provider startup is in target time, not request time. |
| `warm-public-fresh-private` | Public artifacts retained, new private prompt/request and one-use material. List every retained cache and disable private response/prefix/KV reuse. Price warm request and initial acquisition separately. | No dedicated public-cache-only scenario. A `Scenario.sequence` can keep a runtime alive, but requires verified cache/state isolation and separate auditing before receiving this label. |
| `repeat-or-continuation` | Same input, prefix or persistent conversation state; declare what is retained, refreshed and leaked. Rank separately. | `Scenario.repeat` repeats a prompt; `Scenario.sequence` executes multiple requests. Sequence alone does not request continuation. Component support and private-input handling must be checked. Full-state response audit rejects both kinds. |
| `source-cold` | Include checkpoint/tokenizer distribution, worker provisioning, public indexing/certification and startup before first response. | Not supplied by the ordinary cached-source response measurement. Retain unknown/blocked costs until a separately metered distribution workflow is supplied. |

Never label source-present workers “source-cold.” Public filesystem/compiled-cache
warmth is an explicit initial condition even when client-owned caches are cold.
Record worker birth/reuse, setup/dealer lifetime and artifact reuse horizon `H`.
Report first-use acquisition and marginal request costs before any
`(acquisition + H requests)/H` average. One-use private material cannot be reused
because public artifacts are warm. Include renewal, invalidation, fallback,
retirement and unused prepared material; unavailable charges remain unknown.

## 5. Metrics and complete ownership

Preserve the seven priority columns and their original measurement contracts.
Use seconds for wall/CPU time, exact bytes for stored observations, binary units
for memory ceilings and decimal MB/Mbps for traffic/rates.

| Metric | Installed field and interpretation |
| --- | --- |
| Client RSS | `client.peak_rss`: worker lifetime high-water through the scenario. Also show the larger finalization-inclusive `process.peak_rss`, whose largest replicate gates tiers. |
| Client CPU | `client.cold_cpu`: benchmark/dashboard start through first cold response, including in-process relay/dashboard. It is not process-birth-through-cleanup CPU. |
| Preparation CPU | `preparation.cold_cpu`: role birth through first cold response. |
| Preparation RSS | `preparation.peak_rss`: sampled request-window maximum, not a provider startup/lifetime peak. |
| Online TTFT | `response.ttft` / `request.i.ttft`: online start to first text event. Report request-to-first-text separately where both boundaries exist. |
| Decode throughput | `response.decode_tps` / `request.i.decode_tps`: `(N-1)/(last_text_event-first_text_event)` for actual emitted output count `N>1`. Include dependent work and pauses inside that interval. For `N<=1`, this metric is unavailable, not zero or infinite. |
| All-link traffic | `network.all_link_stream_bytes`: conserved directed TCP-stream payload bytes from relay start through the scenario, with every protocol role included. |

Also report `request.i.full` (request setup through completion), `target.latency`
(the timed response executor, including ordinary startup and cleanup), and
`process.cpu`/`target.cpu` with their distinct client windows. Target time excludes
the suite controller, build inspection and worker interpreter work before the
timed action; it is not an entire deployment-provisioning timer. Event-window
decomposition uses per-run timestamps, not differences between unrelated medians.
EOS selection/finalization after the last emitted text is outside N−1 decode time
and remains charged in full-request time.

Report `report.process_cpu_accounting.cold_first_response_cpu_seconds_by_role`
and its aggregate for **all** providers, including both boundary workers. Aggregate
CPU is a sum of non-overlapping role owners in the same stated window, not wall
time and not a sum of child CPU already included in a parent's reaped-child total.
The current ordinary report excludes CPU after the first response; complete
startup-through-cleanup aggregate CPU requires additional lifecycle accounting.
Do not rename the available cold-first aggregate “complete-case CPU.” Likewise
retain per-provider RSS and simultaneous topology RSS when available; neither the
largest provider peak nor a sum of separate peaks is a simultaneous topology peak.

Separate publication/build CPU, public delivery, private and public disk high-water
ownership, filesystem cache and accelerator allocations. Report model/source file
replication at every owner. A topology without Preparation has `not_applicable`
for the role metric; any preprocessing performed at another owner is still charged
there. Missing telemetry is `unknown`, never measured zero preparation.

### 5.1 Three communication planes

1. **Application bodies:** serialized bodies with a versioned counter-coverage
   identity and directed edge ledger. `network.covered_body_bytes` is the sum of
   covered startup and request windows. “Covered” is substantive: framing and
   omitted controls cannot be claimed included. Different topology coverage
   versions require an audited mapping of all relevant edges.
2. **TCP streams:** payload carried on each directed service connection, including
   framing that traverses the metered relay. Sum each directed edge exactly once.
   Check sender-upload, receiver-download and total-stream conservation. Excluded
   telemetry/control channels must be named. Relay stop and provider shutdown
   traffic after the final snapshot are not automatically included.
3. **Physical wire:** captured link-layer/IP/TCP/TLS traffic with retransmission,
   ACK and interface/capture scope specified. Native stream counters and Docker
   service counters do not provide this measurement. It remains null here.

Compare like planes and like windows. Check covered bodies against their enclosing
directed stream windows; do not require equality. Include provider-to-provider
preprocessing, boundary traffic, delivery and lifecycle controls in their measured
scope. For a claim covering later cleanup traffic, add a metered closing window
or leave that complete-lifecycle total unavailable. Never infer physical bytes
from a constant percentage overhead.

Publish base/output-dependent traffic only if all covered bytes are partitioned
and reconcile. Prepaid output material, padding, unused reservations and mixed
envelopes must be classified or retained as unattributed. Dividing whole-request
traffic by `N`, or taking a difference between independent cohorts, does not
recover an incremental per-token cost. Preserve `network.unattributed_body_bytes`
and null base/token fields when attribution is incomplete.

## 6. Predominantly remote body execution

The default outsourcing requirement is at least **90% logical body-linear MACs
off-client separately in prefill and decode**. It operationalizes a declared
design requirement that the client perform no more than one tenth of that work
in either phase; it is not a measured device threshold or a claim about 90% of
total inference CPU. Publish 80/85/90/95/100% sensitivity on the same candidate
population so this convention cannot conceal competitive local-work tradeoffs.
Do not choose the threshold after observing which configuration wins.

For phase `p`, sum `executed_rows * input_width * output_width` over the admitted
semantic body-linear stages. The offloaded fraction is the remote-owned logical
sum divided by the total logical sum. Count each logical product once, regardless
of the two share products, preprocessing products or verification work required
to implement it. Exclude lookup and output head, attention matrix products,
nonlinear work, encoding, hashing and transport from this **body-linear**
denominator; report those costs separately. Excluding the head also prevents a
large vocabulary boundary from masking mostly local transformer execution.

Bind stage ownership, phase and row counts to the executed schedule. Charge
terminal-row pruning and actual decode/EOS work explicitly; reused prefill belongs
in a separate lifecycle cell and cannot manufacture a 100% ratio from a zero
denominator. If a phase performs no work, report `not_applicable` with the reason,
not a qualifying fraction. Phase-dependent placement requires two separate sums.
For a uniform all-remote body both nonempty phases are 100% by construction.

The current dashboard's `client_body_placement` records per-row shape sums,
`declared_body_linear_macs_per_row_client` and `_remote`. The planner provides a
combined schedule/workload estimate. Neither is a general measured, phase-split
MAC counter. Use them only with a verified schedule/row-count derivation; otherwise
label the fraction “plan/per-row” and leave the stricter phase qualification
unknown. An unverified per-row value cannot close this gate for selective
placement. An unavailable ratio is not zero remote work.

## 7. Provider and network budgets

Every campaign records processor/accelerator model, available cores, thread pools,
worker count, role assignment, CPU quota/affinity if enforced, memory caps and
observed peaks, disk/storage, access limits, RTT/loss and operator mapping.
`Sampling.threads` sets BLAS/OpenMP environment limits; native kernel threads are
also selected by components. Neither alone is a hard aggregate provider CPU cap.
CPU seconds measure consumed work, not reserved capacity.

Use two separately labeled resource profiles:

- **Per-party profile:** native shared full-duplex 100 Mbps download / 40 Mbps
  upload per role, with distinct party identities. Both a two-provider and a
  four-provider system keep that per-party limit. The four-provider system
  consequently receives twice the aggregate provider access allowance. Record
  that fact alongside its latency.
- **Aggregate-normalized profile:** hold client access at 100/40 and the complete
  provider pool at 200 Mbps download, 80 Mbps upload, four CPU-core equivalents
  on the same hardware class, and 8 GiB of allowed provider RSS. Use fixed equal
  partitions: for two providers, 100/40, two cores and 4 GiB each; for four,
  50/20, one core and 2 GiB each. These are synthetic comparison budgets, not
  claims about market deployments. Other capacities are additional frozen cells.
  Do not lend an idle Preparation budget to another role unless a separate,
  identically applied work-conserving policy was predeclared. Report both the
  reservations and actual usage, including the trusted/offline role.

The normalized link profile is directly expressible today:

```python
from pllm.deployment import PartyAccess, WanConditions

def normalized_links(provider_roles):
    count = len(provider_roles)
    return WanConditions(
        download_mbps=100, upload_mbps=40,
        parties=tuple(PartyAccess(role, download_mbps=200 / count,
                                 upload_mbps=80 / count)
                      for role in provider_roles),
    )

# Prepared: ("preparation", "inference").
# Coupled: ("preparation", "inference", "boundary_a", "boundary_b").
```

Keep distinct role/party identities: mapping all providers to one party suppresses
intra-party WAN charges and changes the experiment. Explicit per-party partitions
give equal summed access budgets but do not simulate one shared provider uplink.
`WanConditions` has no RTT/loss field; native stream pacing is local bandwidth
emulation. Its `link_conditions(..., latency_ms=...)` helper does not install a
shared-access RTT emulator. Internet/RTT claims need a separately controlled
network experiment. Add a 25/10 Mbps client-access sensitivity cell before making
claims beyond the original access setting; its provider normalization rule and
every role's rates must remain explicit.

The current native driver does not enforce the aggregate CPU/RSS pool or obtain
complete provider lifetime peaks. Provider quotas require an independently
validated supervisor/OS deployment; the coupled path currently rejects Docker.
An OS/cgroup memory ceiling must retain its actual accounting scope, including
any charged page cache; it cannot simply be relabeled an RSS ceiling.
Without that enforcement/evidence, the executable experiment is **link-normalized
only**, not resource-normalized. Setting an 8 GiB `Sampling.memory_bytes` budget
does not implement the 8 GiB provider pool, because it admits the whole local
topology plus other owners. Record this full-normalization cell as blocked until
the missing enforcement and accounting exist. Unnormalized latency can still be
reported with its actual capacity and narrowed claim.

## 8. Workload, serial sampling and failures

### 8.1 Freeze and ordering

The initial publication grid is 150/288 authoritative input tokens and 8/32/64
output caps, with cold and explicitly qualified warm/reuse cells separate. Use at
least ten distinct held-out prompt fixtures per input length as the starting
performance design. Freeze provenance, prompt identifiers, formatting and exact
tokenization before inspecting their candidate outputs. Do not pad/truncate a
task prompt silently to obtain its advertised length. If exact-length fixtures
are synthetic, label them performance fixtures and evaluate task quality on its
own representative held-out set. Repeated executions of one fixture measure
system variability, not workload breadth. A single checkpoint supports only a
checkpoint-specific empirical claim.

Use five fresh-process replicates per primary method/prompt/cell initially,
`repetitions=1`, `warmups=0`, serial case execution and balanced order. Record a
predeclared power/precision rationale; five replicates and ten prompts are starting
allocations, not evidence of adequate uncertainty. Separate calibration/tuning,
performance holdout and independent-confirmation inputs. Privately salted prompt
bindings permit within-cohort checks; preserve exact public fixtures or legitimate
controlled access for independent replay without publishing salts/credentials.

The installed balanced order rotates by `(replicate // 2) % case_count` and reverses
odd replicates. It is deterministic, not randomized or a complete Latin square.
Freeze the initial case order (a recorded seeded shuffle is acceptable) and retain
actual execution order/timestamps. Run no overlapping benchmark campaign on the
host. Host load, power/thermal mode and competing processes belong in the manifest;
the historical driver lock prevents overlapping cooperating drivers, not arbitrary
external load.

### 8.2 Stopping semantics

`max_output_tokens` is a cap. The runtime naturally stops on its configured EOS
before emitting that token; there is no `ignore_eos` or forced-length option in
`BenchmarkTarget.response`. Distinguish:

- **At-cap performance stratum:** the frozen reference reaches the cap. Every
  exact-track arm must reach the same count. Candidate divergence is a correctness
  failure, not permission to discard the prompt.
- **Natural-EOS stratum:** the frozen prompt suite is run without suppressing EOS,
  with the stated cap as censoring. Retain early terminations and cap-censored
  requests; report the length distribution and both rates. Do not rank a candidate
  as faster for stopping earlier on different output.

Publish these strata after running the fixed workload list; do not silently
replace early-EOS prompts with long-output ones. A genuine forced-length experiment
would require a separately identified execution contract and adapter.

For `N` emitted tokens at the cap, there are `N` selections and `N-1` executed
decode steps; the last token is pending. For EOS before the cap there is one extra
unemitted EOS selection, with its decode work, head traffic and unused prepared
material charged. The confirmation auditor implements this distinction for
positive-output cold requests. Its zero-output case is explicitly unsupported:
retain that attempt as a telemetry gap, not a zero-cost response. A one-output
case has unknown N−1 throughput. Natural-EOS requests can be valid despite that
one metric being inapplicable; the full-request/traffic objectives remain usable
when their own accounting is complete.

### 8.3 Statistics and missing cells

Retain every replicate and report medians, ranges and the maximum client RSS.
For paired method effects, pair by prompt and replicate index; keep absolute
differences and ratios with their direction explicit. Do not average per-prompt
rates as if they were one pooled rate. For a multi-prompt headline, predeclare the
estimator (default: equal-weight mean of per-prompt median paired log-ratios) and
obtain a 95% paired hierarchical bootstrap interval, resampling prompts and then
replicate pairs within prompts, using 10,000 draws and a published analysis seed.
For cost metrics use candidate/control ratios; for throughput use control/candidate,
so ratios below one consistently indicate improvement. Exponentiate the aggregate
log-ratio and its interval for reporting. Zero-valued denominators have no defined
ratio: retain absolute differences and exclude that metric from a ratio headline
unless a separate estimator was predeclared. No arbitrary pseudocount is added.
With one prompt, report only its repeated-run effect and uncertainty scope. Memory
eligibility always uses observed maxima, never a bootstrap estimate. Numerical
differences smaller than variability or framing changes are not compelling wins.

Preserve the denominator of planned cells and a status for each: completed,
unsupported, blocked/admission, timeout, failed, cancelled, missing, or excluded by
a named predeclared environmental rule. Keep `not_applicable`, `unknown` and measured
zero distinct at metric level. `BenchmarkRun.complete` means all expected records
exist; it does not mean every trial succeeded. A failed/missing required replicate
prevents a complete primary ranking; partial results may be shown as incomplete.
Do not assign infinity/zero, drop a slow successful run, or import a historical
sample to complete a cell.

The default environmental exclusion is a host-pressure trip or observed swap
growth, applied equally to every arm. Preserve the original cohort and failure
reason. If retrying, start a new complete matched cohort under a new output path
and salt; do not splice accepted records across attempts. Configuration/build,
timeout, workload or exclusion changes start a new campaign version. An early
benchmark stop is not a win. Preflight rejection describes the tested admission
model/environment, not a theorem that the algorithm cannot meet the client tier.

## 9. Executing a publication cell with existing interfaces

The following factory uses current APIs. It accepts already frozen `Experiment`
objects and caller-owned prompts. It does not load a checkpoint, install a build
or run a benchmark by itself. Each experiment must already have matching input/
output budgets, source identity and intended component selections. Use a separate
suite per source/input-length/output-cap/lifecycle/network/security profile, so
prompts and methods stay paired without pooling incompatible cells.

```python
from pllm.metrics import (
    BenchmarkCase, BenchmarkSuite, BenchmarkTarget, Sampling, Scenario, TargetBudget,
)

def publication_suite(experiments, prompts, *, build, output_cap, wan,
                      admission_bytes, client_ceiling=256 << 20):
    cases, inputs = [], {}
    for prompt_id, prompt in prompts.items():
        for method, experiment in experiments.items():
            name = f"{prompt_id}/{method}"
            cases.append(BenchmarkCase(
                name,
                BenchmarkTarget.response(
                    experiment, scenario=Scenario.cold(max_output_tokens=output_cap),
                    backend="native", wan=wan, state_audit=True,
                ),
                build,
                budgets=(TargetBudget("process.peak_rss", maximum=client_ceiling),),
            ))
            inputs[name] = {"prompt": prompt}
    suite = BenchmarkSuite(
        tuple(cases),
        Sampling(processes=5, repetitions=1, warmups=0, order="balanced",
                 threads=1, timeout_seconds=3600, memory_bytes=admission_bytes),
        "publication-cold-v1",
    )
    return suite, inputs
```

Capture an installed `BuildRef` before execution, archive the suite/source/input
manifest, run whole-topology plus audit admission for every case, and preserve
failed admissions. Execute with `benchmark_suite(suite, inputs=inputs,
output=fresh_path)` only in the separately authorized measurement campaign. The
runner generates a fresh salt, serially launches isolated workers, atomically
checkpoints results and rejects build drift. The API permits 1–32 processes,
1–256 cases and at most 1,024 isolated trials per suite; split a larger grid into
predeclared matched blocks. At five replicates, no suite may exceed 204 cases.
The example's ten prompts times eight recipes gives 80 cases and 400 trials.
`Sampling.memory_bytes` currently accepts 16 MiB through 16 GiB. A topology needing
an admission allowance above 16 GiB cannot be executed through this target
unchanged, even though the High sensitivity analysis includes larger cutoffs.

Use a common `WanConditions` containing overrides for every provider role in the
suite, or separate targets per topology when normalization needs different rates.
The factory above accepts one shared network definition; it does not choose those
per-topology rates automatically. Do not override the existing fixed historical
auditors to accept new cells: their assertions deliberately bind original prompts,
recipes, output caps and three-replicate cohorts. A new publication audit must
check the new plan, all planned cells, matching states within each prompt, distinct
salts between cohorts, complete metric contracts, role/stream conservation and
largest complete-client peaks. `BenchmarkRun.compare` intentionally refuses
different privacy/workload/measurement contracts; it is not a cross-topology or
cross-prompt statistical aggregator.

## 10. Confirmation and current capability gates

After screening, freeze the finalist, comparator set, executable configs, primary
estimand, claimed cells, exclusion policy and analysis code. Then collect a new
serial matched cohort with a fresh salt, retained controls and reserved prompts.
Keep both repeated published fixtures (reproducibility) and disjoint confirmation
fixtures (breadth) labeled. Do not tune the finalist or choose only favorable
cells from confirmation. A regression returns the design to screening and requires
another independent confirmation after any revision.

An independent reviewer must reconstruct tables from raw records and perform a
clean replay of the frozen source/build/configurations. Record who selected the
method, ran confirmation and audited it. A fresh cohort run by the same author is
independent sampling, not independent implementation or external replication.
Historical finalist confirmations remain in their original scope.

| Requirement | Available now | Gate still required for the papers |
| --- | --- | --- |
| Cold native response, fresh processes, balanced serial order | Generic SDK target/suite and complete typed failure records | Frozen publication workload and matched campaign; historical drivers are not the new campaign. |
| Complete-client observed RSS | Single-worker serving owner and finalization high-water | Multi-process client ownership if introduced; conservative admission remains separate. |
| Full-state greedy cold audit | Complete selected logits/active KV/positions/RNG, maximum 64 selections | Explicit history/pending state, stochastic and failure/continuation conformance at the publication build. |
| Exactness versus quality | Separate numerical contracts and scoped reference controls exist | Held-out task evaluator, margins and independently checked model fidelity. |
| Natural EOS | Runtime stops naturally; positive-output confirmation auditor charges EOS/unused inventory | Zero-output telemetry; a new prompt-general auditor. No forced-length response option. |
| Warm public, fresh private material | Generic sequence/repeat requests | Public-only cache isolation and warm full-state audit; boundary reuse compositions are restricted. |
| Provider capacity normalization | `PartyAccess` supports equal summed bandwidth partitions | Hard provider CPU/RSS budgets and complete provider lifetime accounting. |
| Complete aggregate CPU/traffic lifecycle | Cold-first role CPU, scenario stream/body ledgers, target wall time | Provider CPU after first response and any traffic after final snapshot. Source-cold/publication costs require separate records. |
| Phase-split remote-work gate | Per-row placement snapshot and schedule estimates | Executed phase/row derivation for selective/pruned/reused paths; no universal telemetry counter. |
| Security profile | Role graph, source binding and execution controls | Composition proof, allowed operator coalitions and external trust/deployment evidence. |
| Source-cold, Internet RTT, physical wire | No corresponding complete ordinary response record | Separately instrumented deployment/network work; keep unknown/blocked until measured. |

Thus normative P1-A definitions are fixed here. The new campaign, near-Medium/High
empirical coverage, phase-split selective-placement proof and full resource-
normalized deployment remain evidence gates. This document does not close P1-D,
P2-B, P2-C, P2-D or P2-E.

## 11. Read-only historical sensitivity, separate from publication evidence

On the protocol date, `buildScorecard()` successfully revalidated the registered
historical report hashes and their original audited scorecard adapters. No model
was loaded and no timing, build, prompt, source or earlier evidence file changed.
All calculations below filter each cohort independently, using its recorded
largest client peak. These are empirical threshold illustrations, not new
replicates or proof that a tier is appropriate for a device population.
This contract postdates those runs; it is not a retrospective preregistration.

### 11.1 Eight-recipe Qwen2.5 150+8 tier cohort

Report: `docs/evidence/private-boundary-tiers-qwen25-2026-10-06.json`.
SHA-256: `0bc6061e24f41fdd12ae60c67c9e313ad8551fbefdb36e92bceadfd88e12e1ca`.
Each recipe has three clients. The original source-present, local native 100/40,
greedy W8A8, measured-window and additional-boundary-worker assumptions remain.

| Ceiling | Eligible recipes | Historical traffic / full-request / decode leaders |
| --- | --- | --- |
| 192 MiB | `tiling`, `tiling-digit8`, `tiling-digit8-placement` | `tiling-digit8-placement` / `tiling-digit8` / `tiling-digit8-placement` |
| 224 MiB | Above, plus `maintained`, `coupled-boundary` | `coupled-boundary` / `coupled-boundary` / `tiling-digit8-placement` |
| 256, 288, 320 MiB | All eight recipes | Same leaders as 224 MiB |

This table deliberately exposes cross-topology costs; it does not collapse their
privacy profiles. At 192 MiB, the coupled recipe is ineligible; it crosses at
220,430,336 bytes (210.21875 MiB). The minimum observed peak is `tiling-digit8` at
194,576,384 bytes (185.5625 MiB). The 90% per-row body-offload rule admits all eight;
95% and 100% remove both selective-placement recipes, whose recorded per-row
offload is 90.44871794871796%. The other six record 100%. This changes the eligible
prepared decode frontier; it does not establish the new phase-split gate for the
90.45% recipes. The selected threshold demonstrably matters and is therefore
reported rather than hidden.

### 11.2 Three-finalist Qwen2.5 150+8 confirmation

Report: `docs/evidence/private-boundary-decode-confirmation-150x8.json`.
SHA-256: `333b652c4f09aff213012ecff083f40471c8f6a2418ab053eb29fb968d724c34`.
Three fresh clients per finalist, in its own build/cohort. No finalist qualifies
at 192 or 224 MiB. All three qualify at 256/288/320 MiB. The parallel/uniform-residue
candidate crosses at 235,012,096 bytes (224.125 MiB), only 128 KiB above the 224 MiB
sensitivity cutoff; this is not a categorical algorithmic change. It leads its
own traffic/full-request columns while maintained prepared leads decode. All three
record 100% body offload. These observations cannot rank a recipe absent from
this confirmation against the eight-recipe cohort above.

### 11.3 Separate Qwen3 16+8 resident/paged pair

Report: `docs/evidence/research-qwen3-client-memory.json`.
SHA-256: `3ebe715a7482b5e5d91f26b1895e1d9a9004bb638a144a8f8204b44c12809654`.
This is one request per arm and an older client/dashboard lifetime measurement,
not the finalization-inclusive v2 worker metric. Keep that original owner/window;
do not upgrade it to the new publication audit or full-state exactness gate.
Neither observed peak fits 192/224 MiB. Only paging fits 256/288/320 MiB:
261,046,272 bytes (248.953125 MiB). Resident crosses at 1,020,461,056 bytes
(973.1875 MiB); both fit every tested Medium sensitivity threshold, with resident
faster in its own full-request comparison. The 199-byte TCP difference is framing/
control variation, not a meaningful architectural traffic win.

All rows in each of these three cohorts separately qualify at every Medium and
High sensitivity threshold. Consequently they provide no observations near
2 GiB or 16 GiB and do not validate those exact cutoffs. New workloads are needed
for that part of the empirical rationale. Counts, exact peaks, per-row offload,
report hashes and replay parameters are recorded in the companion JSON.

### 11.4 Read-only replay

Run from the repository root with the existing Node environment, passing the
following block to `node --input-type=module` on standard input. This verifies
the frozen historical sensitivity against the hash-checked scorecard; it does
not run inference or write evidence.

```javascript
import fs from 'node:fs';
import assert from 'node:assert/strict';
import { buildScorecard } from './docs/scripts/research-scorecard.mjs';
import { memoryBudgetRows } from './docs/lib/research-rankings.mjs';

const contract = JSON.parse(fs.readFileSync('paper/comparison-contract.v1.json'));
const card = buildScorecard();
for (const frozen of contract.historical_sensitivity.cohorts) {
  const cohort = card.cohorts.find(item => item.id === frozen.id);
  assert.ok(cohort);
  assert.equal(cohort.report, frozen.report);
  assert.equal(cohort.reportSha256, frozen.sha256);
  assert.deepEqual(cohort.rows.map(row => ({
    id: row.id, maximum_client_peak_bytes: row.clientMemory.maxPeakBytes,
    plan_body_offload_percent: row.metrics.bodyOffload,
  })), frozen.rows);
  for (const [tier, limits] of Object.entries(contract.memory.sensitivity_bytes)) {
    assert.deepEqual(limits.map(limit => memoryBudgetRows(cohort.rows, limit).rows.length),
                     frozen.eligible_counts[tier]);
  }
  assert.deepEqual(contract.remote_body.sensitivity_percent.map(minimum =>
    cohort.rows.filter(row => Number.isFinite(row.metrics.bodyOffload)
      && row.metrics.bodyOffload >= minimum).length), frozen.plan_offload_eligible_counts);
}
console.log('Historical sensitivity verified; cohorts remain separate.');
```

## 12. Checked implementation sources

The following paths were inspected for this contract, rather than inferred from
interface names. Their revision is pinned at the start of this document.

- `python/pllm/_benchmark_contract.py`: target, suite, sampling, scenario and
  metric-budget API; audit/composition bounds.
- `python/pllm/metrics/benchmark_runner.py`, `python/pllm/_benchmark_worker.py`,
  `python/pllm/_benchmark_build.py`, `python/pllm/_benchmark_evidence.py`,
  `python/pllm/_benchmark_workload.py`:
  serial order, process ownership, salt/build binding, completion and comparisons.
- `python/pllm/runtime/benchmark_response_target.py`,
  `benchmark_response_audit.py`, `benchmark_cli.py`, `benchmark_resources.py`,
  `benchmark_memory.py`: metric windows, audit contents, cache reset, admission,
  pressure sampling and response-driver limitations.
- `python/pllm/runtime/dashboard.py`, `client.py`, `topology_accounting.py`:
  event timing, EOS/pending behavior, body placement, role CPU and source-delivery
  accounting.
- `python/pllm/roles/topology.py`, `python/pllm/deployment/wan.py`,
  `python/pllm/search/placement.py`: separation pairs, access partitions and
  planned logical-work scope.
- `benchmarks/research/private_boundary_tiers.py`, `private_boundary_decode.py`,
  `private_boundary_decode_confirmation.py`, their `scripts/benchmark_*.py`
  entry points and `scripts/benchmark_private_boundary.py`: frozen historical
  cohorts, stronger read-only audits and diagnostic-state scope.
- `python/pllm/search/client_memory_tiers.json`,
  `docs/scripts/research-scorecard.mjs`, `research-memory.mjs`,
  `docs/lib/research-rankings.mjs`, `research-relationships.mjs`, and
  `docs/research/metric-relationships-reference.md`,
  `private-boundary-tiers-reference.md`, `private-boundary-sdk-reference.md`,
  `qwen3-client-memory.md`: tier constants and historical measurement provenance.
