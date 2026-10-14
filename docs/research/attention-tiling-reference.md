# Compiler-bound attention lifetime tiling

This implements the strongest bounded candidate from the
[five orthogonal resource screens](orthogonal-frontier-reference.md).
The initial synthetic operator measurements remain in
[exact attention streaming](exact-attention-streaming-reference.md).
Their isolated memory reductions are not complete-client measurements.

## Native contract and execution

The model-neutral Rust pass selects an exclusive, contiguous initial-prefill
region: attention scores, scaling, causal masking, softmax and attention values.
It checks semantic operators, dependencies, shapes, layer identity and numeric
attributes. External consumers and state roots of interior tensors reject.
Every dense attention operation must belong to a complete eligible region.

Source-plan, composition and original schedule digests bind the region alongside
its public query-row and KV-head tile geometry. Admission allows 1–256 query
rows and 1–64 KV heads, bounded full outputs and score domains. The runtime
rechecks its compiled binding, original schedule, input geometry and positions.
An independent execution-basis digest records the selected region.

Each tile invokes the existing numeric operators over borrowed strided Q/K/V
views. Complete key capacity, including masked unused slots, stays intact.
`prefix_f32` keeps the original query-relative causal positions and reduction
domain. Dense mode retains its own numeric contract. Neither operand copying nor
replacing the selected reduction with another profile is permitted. Complete
attention outputs and ordinary full KV remain live; tile intermediates retire
before the next tile.

CPU client-only and unverified prepared compositions are admitted, including the
installed MLP-prefill scheduler. Metal, Freivalds, protected nonlinear regions,
retained/windowed state, epochs, in-flight sharing and decode reuse reject.
Decode and restored-prefix continuation execute the original schedule. Failure
retires active state and burns the response owner; ordinary SDK cleanup retires
reserved one-use material.

This remains an internal research executor, selectable through
`BenchmarkTarget.response(client_attention={"enabled": True,
"query_rows": 16, "kv_heads": 2})`. It is not a Pipeline scheduler or automatic
planning choice.

## Correctness gates

Generated Qwen2, Qwen3 and Llama checkpoints compare all prefill logits and active
KV bits, three decode steps, and restored-state continuation. Both dense and
`prefix_f32` profiles, partial final tiles, Q/K normalization and MLP tiling are
covered. Weak-reference controls check intermediate retirement. Separate prepared
role children and gateway requests preserve complete checked state and outputs;
injected tile failure burns reservations and rejects runtime reuse.

Two native contract tests pass. The installed-wheel attention, benchmark,
historical streaming and target regression suite passes 142 tests. The final
30-test evidence/coverage suite also passes, including the additional
control-byte reconciliation regression and reviewed capability mappings.
The generated tiny complete-response pair checks 21,452 logit/KV values across
four cold/repeat responses, with zero observed swap growth. Its one-byte
lifecycle-control difference remains charged; non-control application bodies
match exactly. Tiny-checkpoint process peaks do not qualify a real model.

## Matched measurement contract

`scripts/benchmark_attention_tiling.py` constructs fresh matched full/tiled arms
over three frozen recipes from the completed combination screen:

- MLP tiling with integer Preparation;
- digit8 Preparation with full-stage MLP execution;
- MLP tiling, digit8 Preparation and selective client placement.

The checkpoint is pinned Qwen2.5-0.5B W8A8 `prefix_f32`. Workloads are 150/288
input tokens plus up to eight outputs under a 384-token compiled input bound.
Each fresh client performs a cold request and an exact repeat. Three balanced
serial replicates share a private cohort salt and installed build identity.
Native shared 100/40 Mbps pacing includes all instrumented service streams.
The placement recipe independently preserves at least 90% remote body MACs in
prefill and decode; attention tiling adds no body weights or local projections.

Both arms compile region eligibility and pay for bounded hashing of **all
executed logits and active KV**, including decode. Hashing streams strided arrays
in small buffers and exports only salted commitments and checked-value counts.
It does not take full-state snapshots. Hashing changes the measurement identity,
so older uninstrumented timing cohorts cannot supply a matched control.

Admission retains conservative full-stage topology pricing and adds 160 MiB of
client region/audit overlap allowance, including margin. Actual whole-client
RSS remains a separate measurement. Low eligibility uses the largest observed
client peak across all required replicates, at most 268,435,456 bytes; Medium
uses 2,147,483,648 bytes. These observed thresholds do not replace conservative
runtime admission or account for filesystem/device memory.

Every request's non-control application bodies must match. Variable lifecycle
metadata remains in covered totals and TCP counters, with its observed range
reported. Cold/repeat latency, online TTFT and N−1 decode metrics retain separate
windows from client process-lifetime RSS. All seven priority metrics remain
visible. Missing or failed cells cannot borrow another cohort's replicates.

The first complete-screen preflight rejected the 150-token digit8 case: its
6.74 GiB topology estimate exceeded 6.38 GiB post-reserve headroom, before the
additional research allowance. No tensors or providers were loaded for that
rejected screen. Separately admitted workload cells retain distinct cohorts.
The subsequent 150-token retry rejected digit8 at 6.90 GiB including research
overlap against 6.75 GiB headroom. A separately selected tiling/placement screen
rejected placement at 6.46 GiB against 6.37 GiB, before research overlap. These
header-only rejections retain their [initial](../evidence/research-attention-tiling-initial-rejection.plan.json),
[digit8](../evidence/research-attention-tiling-150-digit8-rejection.plan.json) and
[placement](../evidence/research-attention-tiling-150-placement-rejection.plan.json)
plans; their corresponding progress records retain public failure codes. No
measured clients ran in these attempts. The lower-cost 150-token MLP-tiling pair
is a separate cohort; its result cannot fill either blocked recipe.

## Completed 288-token screen

**Whole-client benefit is small and recipe-dependent; no Low/Medium SOTA gain is
established.** Eighteen fresh clients complete 36 cold/repeat responses, each with
eight outputs. All **524,973,312 checked logit/KV values** and output commitments
match within their paired recipes. All runtime checks pass and observed swap
growth is zero. Placement retains 90.06% of prefill and 90.45% of decode body MACs
remotely. The remaining recipes keep all body projections remote.

Original evidence is preserved byte-for-byte:

- [Report](../evidence/research-attention-tiling-qwen25-288.json), SHA-256
  `d2dfaa490db3d81b8cf2183426979a88498a1088a8e74e64197bdea37abf7c93`.
- [Plan and admission](../evidence/research-attention-tiling-qwen25-288.plan.json),
  SHA-256 `3427478bf77b5fb9b27895eebe20ff4ee51c9fee2f63dce826e9b6fd9bd77cde`.
- [Replayed summary](../evidence/research-attention-tiling-qwen25-288.summary.json),
  SHA-256 `fb233f8dab9573013a6af8d26d63cb82f59305cb731b6737c67b76549f37a78b`.

The installed build is
`aa8aa10f44aec143421bf90ae1dc93fd7a50189d6d034aaf14c720b0f6efd233`;
checkpoint source lock is
`383f5ed6947ba56ceed6083a9503d6302f09559e1e57ee614409900b36084c29`.
The frozen driver/factory/artifact hashes match, and re-running `audit` reproduces
the saved summary exactly. All six choices have three fresh-client replicates.

### Seven priority metrics

All entries are medians of three process samples. RAM and traffic use decimal
MB; CPU/latency use seconds. Client RAM is process-lifetime cold-plus-repeat high
water, including the dashboard and native relay. Client CPU runs from benchmark
startup through the first response; Preparation CPU starts at role birth.
Preparation RAM is the maximum sampled peak across the two request windows.
TTFT and N−1 decode throughput below use the **cold online window**. TCP bytes
cover the complete cold-plus-repeat relay lifecycle. Full-state hashing remains
charged in both arms.

| Recipe / attention | Client RAM MB | Client CPU s | Prep CPU s | Prep RAM MB | Cold TTFT s | Cold decode tok/s | Pair TCP MB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MLP tiling / full | 236.91 | 27.03 | 20.10 | 261.13 | 79.902 | 3.680 | 805.29 |
| MLP tiling / tiled | 231.19 | 27.02 | 20.17 | 263.55 | 79.856 | 3.684 | 805.29 |
| Digit8 / full | 266.32 | 26.35 | 13.15 | 259.49 | 76.473 | 3.690 | 805.29 |
| Digit8 / tiled | 279.45 | 26.39 | 13.14 | 260.13 | 76.501 | 3.684 | 805.29 |
| MLP tiling + digit8 + placement / full | 235.73 | 27.19 | 12.30 | 259.31 | 73.297 | 4.279 | 738.80 |
| MLP tiling + digit8 + placement / tiled | 233.23 | 27.18 | 12.33 | 259.52 | 73.259 | 4.268 | 738.80 |

### Complete requests and repeat windows

| Recipe / attention | Cold request s | Repeat request s | Repeat online TTFT s | Repeat N−1 tok/s |
| --- | ---: | ---: | ---: | ---: |
| MLP tiling / full | 181.938 | 11.327 | 2.694 | 3.699 |
| MLP tiling / tiled | 181.932 | 11.298 | 2.678 | 3.710 |
| Digit8 / full | 171.513 | 11.058 | 2.513 | 3.710 |
| Digit8 / tiled | 171.575 | 11.000 | 2.515 | 3.727 |
| MLP tiling + digit8 + placement / full | 168.100 | 10.858 | 2.683 | 4.295 |
| MLP tiling + digit8 + placement / tiled | 168.173 | 10.887 | 2.677 | 4.285 |

Exact repeats hit the completed-prefill cache and do not execute tiled attention.
Complete requests include client setup and Preparation, but exclude earlier
provider startup. Repeat CPU and independent repeat RSS remain unmeasured.

### Tier and decision audit

| Recipe | Largest full-attention client peak, MiB | Largest tiled-attention client peak, MiB | Observed Low eligibility, full / tiled |
| --- | ---: | ---: | --- |
| MLP tiling | 228.02 | 221.38 | Yes / yes |
| Digit8 | 254.28 | 269.63 | Yes / no |
| MLP tiling + digit8 + placement | 226.22 | 228.23 | Yes / yes |

All arms fit the observed Medium threshold. Attention tiling lowers median
client RSS **2.41%** atop MLP tiling, with reductions in all three paired samples.
The combined-placement median falls **1.06%**, but one paired sample regresses
and its largest peak increases. Digit8 alone regresses **4.93%** in median client
RSS and loses observed Low eligibility. The isolated 16.30% attention-process
reduction therefore does not predict the whole-client result.

Cold latency changes by less than 0.05% in each recipe; neither latency nor CPU
shows a convincing improvement. Non-control application bodies match exactly.
Variable lifecycle controls produce only 2–3-byte covered-total ranges within
recipes, and remain charged. Complete base/per-output traffic headline metrics
stay unknown. No fresh-request traffic saving, universal memory improvement,
tenfold result or automatic selection follows. Independent finalist confirmation
and both missing 150-token recipe comparisons remain open.

## Completed 150-token MLP-tiling pair

The separately admitted integer-Preparation/MLP-tiling recipe completes **six
fresh clients and twelve cold/repeat responses**, each with eight outputs.
All **98,682,624 checked logit/KV values** and output commitments match, all
runtime checks pass, and observed swap growth is zero. All body projections
remain remote. This cohort has its own private salt and does not fill the
150-token digit8 or combined-placement admission gaps.

Original evidence is preserved byte-for-byte:

- [Report](../evidence/research-attention-tiling-qwen25-150-tiling.json), SHA-256
  `47a5ddc25f8682b2a3354deadd49c3436d641063f976ae82f0726ec5658a3a9e`.
- [Plan and admission](../evidence/research-attention-tiling-qwen25-150-tiling.plan.json),
  SHA-256 `c3bc5ec76286078ac9b2155ce0808ebc8d37af46ab8219a498e26cfbb5751308`.
- [Replayed summary](../evidence/research-attention-tiling-qwen25-150-tiling.summary.json),
  SHA-256 `878c7ffad5a6d04e469e01921ada0030a7bc55dbae3e64b5453d8d7743023733`.

Source lock and installed build match the 288-token screen. Frozen source hashes
pass and `audit` reproduces the summary exactly. The same metric units and
windows apply; each arm has three process samples.

| Attention | Client RAM MB | Client CPU s | Prep CPU s | Prep RAM MB | Cold TTFT s | Cold decode tok/s | Pair TCP MB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Full | 199.82 | 21.25 | 16.36 | 215.96 | 42.877 | 3.738 | 496.06 |
| Tiled | 198.31 | 21.01 | 15.77 | 216.14 | 42.880 | 3.755 | 496.06 |

| Attention | Cold request s | Repeat request s | Repeat online TTFT s | Repeat N−1 tok/s |
| --- | ---: | ---: | ---: | ---: |
| Full | 108.986 | 10.960 | 2.690 | 3.676 |
| Tiled | 108.898 | 10.849 | 2.659 | 3.786 |

Median client RSS falls **0.75%**, but two of three paired samples regress and
the largest peak rises from **191.72 to 192.38 MiB**. Both arms remain within
the observed Low and Medium thresholds. Cold request latency changes only
0.08%; cold TTFT is effectively unchanged. Preparation execution is unchanged,
and repeat cache hits do not execute tiled attention, so their timing changes
do not establish a causal benefit from attention tiling. Non-control application
bodies match; the two-byte covered-total range remains charged.

This result does not justify extending the executor or selecting it
automatically. No new Low/Medium SOTA, reliable whole-client memory improvement
or fresh-request traffic saving is established. The 288-token result retains
its own cohort, and the two blocked 150-token recipes remain unmeasured.

## Replay

Use the installed wheel containing this native pass and the pre-positioned
artifacts built by `examples/prepared_sdk.py`. The benchmark uses the existing
frontier lock, topology admission, fresh-worker build validation and host-pressure
monitor. It never retries by reducing admission margins or altering a recipe.

```sh
target/frontier-sdk-env/bin/python -m pytest -o pythonpath= \
  tests/test_attention_tiling.py tests/test_attention_tiling_benchmark.py \
  tests/test_exact_attention_streaming_probe.py tests/test_benchmark_targets.py -q

target/frontier-sdk-env/bin/python scripts/benchmark_attention_tiling.py \
  --preflight-only --workload 288 \
  --output target/attention-tiling-qwen288-preflight.json

target/frontier-sdk-env/bin/python scripts/benchmark_attention_tiling.py \
  --workload 288 --output target/attention-tiling-qwen288.json

target/frontier-sdk-env/bin/python scripts/benchmark_attention_tiling.py \
  --workload 150 --recipe tiling \
  --output target/attention-tiling-qwen150-tiling.json
```

Recheck both archived cohorts independently without running a model:

```sh
target/frontier-sdk-env/bin/python - <<'PY'
import json
from pathlib import Path
from pllm.metrics import BenchmarkRun
from benchmarks.research.attention_tiling import audit
from benchmarks.research.sota_combinations import placement_pairs

_, placement = placement_pairs()
for name in ("288", "150-tiling"):
    path = Path(f"docs/evidence/research-attention-tiling-qwen25-{name}.json")
    summary = audit(BenchmarkRun.load(path), placement_record=placement)
    assert summary == json.loads(path.with_suffix(".summary.json").read_bytes())
    print(name, summary["completed_clients"], summary["checked_values"])
PY
```

Omitting `--workload` requests both workloads. `--recipe` restricts the explicit
recipe set; selection never merges private cohorts. Output, plan, progress and
summary files retain separate records. Pre-positioned checkpoint/tokenizer
distribution, physical wire and independently operated providers remain outside
these local response measurements.
