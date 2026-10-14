# Private boundary against cold-request tier incumbents

This comparison uses one installed build and the ordinary fresh-process SDK
response benchmark. It measures the coupled private lookup/full-head composition
against seven existing prepared recipes on the same pinned Qwen2.5-0.5B W8A8
150-input/eight-output request. Each recipe receives three balanced, serial fresh
clients under shared native 100/40 Mbps access caps and one private cohort salt.

## Completed tier confirmation

All 24 clients completed. **210,247,680 logit/KV values**, positions, selections,
greedy RNG commitments and output digests agree; maximum observed swap growth is
zero. Every recipe's largest complete-worker peak fits Low (256 MiB), so the same
eligible set also fits Medium (2 GiB) and High (16 GiB). The coupled boundary's
largest peak is **210.22 MiB**, including the SDK, dashboard and relay.

Against the **best eligible measured incumbent for each metric**, the coupled
boundary has the following results in both Low and Medium. RAM and traffic use
decimal MB here; tier thresholds use binary MiB/GiB.

| Metric | Best incumbent | Incumbent median | Coupled median | Coupled change |
| --- | --- | ---: | ---: | ---: |
| Client lifetime peak RSS | tiling + digit8 | 194.25 MB | 217.63 MB | +12.04% |
| Client cold CPU | digit8 | 19.607 s | 19.133 s | −2.42% |
| Preparation cold CPU | tiling + digit8 + placement | 10.650 s | 14.846 s | +39.40% |
| Preparation sampled RSS | serial digit8 | 195.46 MB | 215.79 MB | +10.40% |
| Online TTFT | digit8 + placement | 37.384 s | 46.822 s | +25.25% |
| N−1 decode throughput | tiling + digit8 + placement | 4.393 tokens/s | 1.924 tokens/s | −56.22% |
| All-link TCP streams | digit8 + placement | 458.775 MB | 367.347 MB | **−19.93%** |
| Full request latency | digit8 | 104.026 s | 82.951 s | **−20.26%** |
| Target startup-through-cleanup time | digit8 | 123.413 s | 109.984 s | **−10.88%** |

This establishes a Low/Medium-eligible traffic and cold-latency improvement over
the tested prepared frontier, under the additional boundary-worker trust contract.
The small client-CPU median improvement has overlapping observed ranges:
19.120–19.713 seconds coupled versus 19.592–21.032 seconds for digit8.
Aggregate cold-first CPU is 61.869 seconds coupled versus 53.812 seconds for the
best incumbent; client CPU alone does not describe the topology's compute cost.
The best client-memory result remains tiling + digit8. Online TTFT excludes cold
delivery, explaining why it worsens while full request latency improves.

### Every recipe

All values except maximum client RSS are three-client medians. Maximum client RSS
is the tier qualification statistic; CPU and timing are seconds.

| Recipe | Max client MiB | Client CPU | Prep CPU | Prep MB | Online TTFT | Decode tokens/s | TCP MB | Full request |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| maintained | 213.44 | 19.692 | 14.899 | 213.60 | 40.907 | 3.794 | 478.654 | 106.642 |
| tiling | 189.59 | 20.123 | 14.992 | 215.48 | 42.644 | 3.788 | 478.654 | 108.490 |
| digit8 | 224.30 | 19.607 | 11.187 | 215.24 | 40.912 | 3.782 | 478.654 | 104.026 |
| tiling-digit8 | 185.56 | 20.025 | 11.248 | 226.85 | 42.540 | 3.762 | 478.654 | 106.436 |
| tiling-digit8-placement | 189.56 | 20.626 | 10.650 | 214.79 | 39.082 | 4.393 | 458.775 | 107.099 |
| digit8-placement | 226.50 | 20.268 | 10.700 | 220.77 | 37.384 | 4.389 | 458.775 | 105.971 |
| serial-digit8 | 226.89 | 19.753 | 11.141 | 195.46 | 40.981 | 3.791 | 478.654 | 108.752 |
| coupled-boundary | 210.22 | 19.133 | 14.846 | 215.79 | 46.822 | 1.924 | 367.347 | 82.951 |

The independent cohort is registered as
`qwen25-private-boundary-tiers-150-8` in the SOTA/evidence tables. The seven
Prepared rows and the five-role boundary row retain separate topology-specific
leaders. Their explicit cross-topology comparison is the table above. This cold
150-token confirmation does not complete the older 150/288-token cold/repeat
combination study's independent confirmation.

## Frozen choices

`benchmarks/research/private_boundary_tiers.py` replays existing component choices:

| Case | Selection |
| --- | --- |
| `maintained` | Maintained paged, demand-issued prepared stack |
| `tiling` | Installed 64-row MLP prefill tiling |
| `digit8` | Native digit8 Preparation |
| `tiling-digit8` | Both choices |
| `tiling-digit8-placement` | Both plus frozen selective client placement |
| `digit8-placement` | Independently confirmed selective-placement recipe |
| `serial-digit8` | Independently confirmed serial digit8 issuance recipe |
| `coupled-boundary` | Private lookup/full head with two additional boundary workers |

Every request and fixed-prefix-cache envelope is rebound to the same public
150+8 bound. This is a new matched cohort; it does not reuse historical timings
from the wider 384-token reservation. First requests have empty client caches.
There is no public-prefix capsule. The indexed tokenizer and checkpoint already
exist locally; their distribution is unmeasured. Public-prefix reuse and
mostly-client body execution have different assumptions and are outside this
comparison.

The coupled composition currently rejects MLP tiling, prefix caching and selective
body placement. Those recipes remain separate competitors. Adding two online
boundary workers changes the role and non-collusion contract. Cross-topology cost
comparisons here are explicit; ordinary benchmark privacy commitments and
topology-specific SOTA tables remain separate.

## Complete client ownership and exactness

`BenchmarkTarget.response(..., state_audit=True)` records private-salt commitments
to every complete logit vector, active full KV, position, selected token and greedy
sampler state. Generated Qwen2/Qwen3/Llama controls detect a one-bit KV change and
check sampler restoration and failure cleanup. The audit hashes bounded chunks
without retaining tensor copies, and prices 32 MiB of scratch/metadata headroom.
Its work is charged in every measured arm. Only the sampler uses the public seed;
one-use cryptographic randomness remains fresh.

The SDK, role supervisor, dashboard, native WAN relay and audit share one fresh
worker process. `client.peak_rss` ends after the scenario; `process.peak_rss`
also includes target finalization. Tier eligibility uses the **largest complete
worker peak across every replicate**, never the median or a sum of unrelated
peaks. The idle suite launcher is outside the serving client; provider processes
have their own ownership. Observed 256 MiB/2 GiB/16 GiB eligibility does not certify
the more conservative SDK admission estimate or whole-device memory.

The earlier 48+8 and 150+8 private-boundary cohorts excluded their parent/relay
from client RSS. Their tier status remains unknown; this measurement cannot
retroactively qualify those reports.

## Measurement windows

- Client CPU: benchmark start through first response, including the in-process
  dashboard and relay.
- Preparation CPU: process birth through first response; its RSS is sampled in
  the request window.
- Client RSS: process birth through target finalization.
- TTFT: online start to first token. Full request latency includes request setup
  but excludes provider process startup. `target.latency` separately charges
  ordinary benchmark startup and cleanup.
- Decode: seven later outputs over the N−1 generation window.
- Traffic: conserved all-link TCP-stream payloads from relay startup through the
  scenario. Physical wire, checkpoint/tokenizer distribution and Internet RTT
  remain unmeasured. Different application-body coverage identities remain intact.

Results are compared only within this build, source, workload and private cohort.
Each tier comparison uses the best eligible prepared recipe for that metric.
Winning such a comparison is a bounded frontier result, not a global optimum,
independent-provider privacy result or representative generation-quality claim.

## Replay

The driver is `scripts/benchmark_private_boundary_tiers.py`. It freezes its source,
helper and public artifact hashes, checks input tokenization, and preflights every
recipe before running the ordinary `benchmark_suite`. It writes the exact suite,
admissions, raw typed results and a read-only summary. Original failed attempts
remain separate; failed or missing cells cannot supply a complete comparison.

The generated-checkpoint smoke selects maintained, tiled and coupled execution
without downloading a checkpoint. Real measurements require the pinned source
and `target/prepared-frontier-artifacts/{prompt.txt,indexed.json,tokenizer}` from
the maintained prepared study. Outputs must use a fresh path; the cohort lock
prevents overlapping repository benchmark runs.

The executed real-checkpoint command was:

```sh
target/frontier-sdk-env/bin/python scripts/benchmark_private_boundary_tiers.py \
  --processes 3 --output target/private-boundary-tiers-qwen-v2.json
```

Choose a new output path for another run. Archived immutable files:

- [Raw ordinary SDK results](../evidence/private-boundary-tiers-qwen25-2026-10-06.json)
  — SHA-256 `0bc6061e24f41fdd12ae60c67c9e313ad8551fbefdb36e92bceadfd88e12e1ca`.
- [Frozen suite, source hashes and admissions](../evidence/private-boundary-tiers-qwen25-2026-10-06.plan.json)
  — SHA-256 `76a0816cbe5bae10df141072a54312949ca682883fb131e168a4652a073b3ea4`.
- [Recomputed tier summary](../evidence/private-boundary-tiers-qwen25-2026-10-06.summary.json)
  — SHA-256 `bd93e6bba34f96039791b17eb8bac09fa878b5eb221295a3dc60aa553bf2163c`.
- [Earlier rejected preflight](../evidence/private-boundary-tiers-qwen25-preflight-rejected.plan.json)
  and [retained failure status](../evidence/private-boundary-tiers-qwen25-preflight-rejected.progress.json).
  The five-role graph rejected on host headroom before any measurements. It
  contributes no clients or samples to the successful rerun.

Read-only summary replay, without executing a model:

```python
import json
from pathlib import Path
from benchmarks.research.private_boundary_tiers import audit

root = Path("docs/evidence/private-boundary-tiers-qwen25-2026-10-06")
raw = json.loads(root.with_suffix(".json").read_bytes())
saved = json.loads(root.with_suffix(".summary.json").read_bytes())
assert audit(raw) == saved
```

Each SOTA row also retains its complete `recordedCase`, accepted by
`pllm.metrics.BenchmarkCase.from_spec`, including the original BuildRef and exact
target. A later build needs new measurements and cannot inherit these timings.
