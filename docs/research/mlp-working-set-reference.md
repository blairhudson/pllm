# Compiler-bound MLP working sets

This research slice follows the rejected whole-client memory result for
[native stage ownership](client-stage-ownership.md). It changes the lifetime of
complete MLP intermediates: a small row tile passes through gate/up projections,
SiLU, multiplication and the down projection before the next tile starts.
Transport-only chunking still reconstructs a full stage output; this experiment
bounds the intervening MLP tensors as well.

## Contract and owners

The Rust compiler derives four-step regions from the admitted semantic schedule.
Eligibility requires grouped linear gate/up inputs, client-local float32 SiLU and
multiply, a matching down projection, common layer identity and exclusive interior
consumers. State roots, terminal-row pruning and externally consumed intermediates
cannot be silently folded into a region. Source-plan, composition, source-schedule,
row bound and exact operation identities bind the derived contract. Tile sizes
are public, from 1 to 256 rows, under the existing bounded matrix codecs.

The original linear and local numeric executors perform every operation. Tiling
does not split the feature reduction or change per-row quantization. Each tile's
intermediates retire at last use and before the next tile; a complete hidden-width
output remains for the residual consumer. The original one-use inventory supplies
disjoint row claims. Exceptions propagate through ordinary SDK cancellation and
burn unused reserved rows. Preparation remains offline and body weights remain
provider-owned in the measured prepared composition.

Only eligible initial-prefill regions are tiled. Decode, continuation, terminal
pruning, tokenizer, attention and full KV retain their ordinary execution. The
internal override has its own execution identity and is not a selectable Pipeline
component. Freivalds, protected nonlinear regions and retained-state compositions
reject. Whole-topology memory admission retains its conservative existing bound;
tiling cannot bypass full-stage residue-layout admission.

## Validation and measurement

Generated Qwen2, Qwen3 and Llama controls compare complete 33-row prefill logits
and three same-token decode steps, including every active KV bit. Tile sizes
1, 7 and 16 exercise partial final tiles. Weak-reference checks reject retaining
the previous tile's product into the next remote exchange. Separate prepared-role
children execute the same SDK requests and compare logits, state and output text.
Injected failure after a completed tile verifies reservation retirement and
rejected reuse.

The response benchmark's explicit `client_mlp={"tile_rows": ...}` selector uses
zero as its full-tensor control. Both arms compile eligibility and count the same
public metadata owners. Instrumentation records eligible regions, executed tiles
and maximum tile rows; these are ownership counts, not RSS. Unique Python-object
bytes for graph, schedule and binding metadata are reported separately from
process memory and native allocations.

The existing `FrontierStudy` runner supplies fresh clients, exact installed-build
locks, a shared private cohort salt, source/numeric/output comparison gates,
memory admission and native shared 100/40 Mbps pacing. Each workload retains all
seven serving metrics. A cold request and exact-repeat request share each client;
public checkpoint and tokenizer artifacts are pre-positioned. Added exchanges,
CPU, latency and traffic remain charged.

Replay with the public artifacts from the maintained prepared study:

```sh
python -m benchmarks.research.client_stage_ownership --working-set --processes 1 \
  --workload medium --workload long --input-bound 384 --serial-issuance \
  --output target/mlp-working-set-screen.json
```

The explicit 384-token bound fits existing residue admission for Qwen2.5's grouped
gate/up stage. It is not a change to the codec ceiling. Confirm a nominated cell
using the same study arguments plus `--confirm SCREEN --nominate medium/tile64`;
confirmation creates at least three fresh clients per arm. Physical wire,
checkpoint distribution and independent provider operation remain outside these
local measurements.

## Initial checkpoint screen

Six fresh clients execute cold/repeat Qwen2.5-0.5B requests at **150+8** and
**288+8** tokens. The public input bound is 384 and Preparation is explicitly
serial. All twelve requests pass ordinary checks, preserve within-workload output
commitments and observe zero host swap growth. The installed build is
`b94df5b1343957e180d4470b45e406bcdaa3f5926ca998e9bd946d99491ebdff`.

CPU and TTFT use seconds; RSS and stream traffic use decimal MB. Client RSS is
process-lifetime high-water. Preparation RSS is sampled in request windows. CPU
covers cold-first execution, TTFT/decode summarize both requests, and traffic
covers setup plus cold/repeat. The seven priority axes remain separate.

| Workload / rows | Client RSS | Client CPU | Prep CPU | Prep RSS | TTFT | Decode TPS | Stream MB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 150+8 / full | 233.46 | 21.248 | 15.877 | 188.89 | 21.927 | 3.657 | 496.046 |
| 150+8 / 16 | 189.04 | 21.260 | 15.772 | 193.05 | 25.161 | 3.653 | 496.046 |
| 150+8 / 64 | 197.67 | 20.986 | 15.955 | 192.02 | 22.674 | 3.807 | 496.046 |
| 288+8 / full | 267.55 | 27.847 | 21.162 | 257.15 | 39.640 | 3.483 | 805.269 |
| 288+8 / 16 | 235.57 | 28.829 | 21.615 | 264.44 | 46.707 | 3.677 | 805.269 |
| 288+8 / 64 | 236.77 | 28.174 | 21.226 | 259.57 | 41.375 | 3.655 | 805.269 |

The 64-row candidate lowers client RSS **15.33% / 11.51%**, while median online
TTFT rises **3.41% / 4.38%**. Complete cold request time rises
**113.834 → 114.702 seconds** and **179.487 → 182.884 seconds**. Sixteen-row tiles
save slightly more memory but increase cold latency and TTFT further. Small
stream-count differences are framing/control bytes; there is no arithmetic-byte
saving. Both arms count **3,955,137 bytes** of unique public Python metadata per
runtime; this is not construction-peak attribution.

These single-process cells nominate 64-row tiles for a separate three-client-per-arm
confirmation. They do not establish repeatable savings or a tenfold improvement.
Evidence: `docs/evidence/research-mlp-tiling-qwen25-screen.json`, with its
`.plan.json` and `.frontier.json` companions.
