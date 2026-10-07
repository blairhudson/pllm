# Experimental masked-wire decoder

`MaskedWireGatedMultiplyQ12(layout="unfused" | "fused")` executes complete
SiLU × up regions through `TwoOnlineOffsetCpu` Experiments. The compiler binds
each semantic pair and its input/output conversion. The trusted client floors
float32 inputs to signed Q24, enforces public gate/up ranges, issues fresh keys
in chunks of at most 256 lanes, relays worker peer frames and reconstructs Q12
outputs. Attention and other local operations keep their existing ownership.

Each authenticated worker holds one party's keys. Sessions bind plan, region,
incarnation, phase and monotonic element budgets; malformed input, replay and
cancellation burn the pending material. Traffic includes both key packets and
both relay legs. Issuance occurs during the response and is charged to client
CPU. Memory admission prices client chunk buffers and every worker session slot.
This in-process dealer/co-located transport is experimental and unreviewed.

Generated Qwen2/Qwen3 prefill/decode matches an independent integer oracle,
including all checked logits and KV. Separate child workers pass SDK and gateway
requests. An installed wheel, tested outside the checkout without Cargo, also
completes the ordinary benchmark with nonzero key, peer and element counters.

## Fresh checkpoint quality

The pinned Qwen2.5-0.5B W8A8 cohort freezes twelve new public prompts and compares
clear execution of the installed numeric mapping with the exact W8A8 control.
Each teacher-forced prompt checks prefill and three same-token decode positions;
each free-running generation feeds back its own greedy selections, capped at
eight outputs. Both layouts have identical numerics.

| Measurement | Exact control | Masked-wire Q12 |
| --- | ---: | ---: |
| Prefill selections | 12/12 | 10/12 |
| Same-token decode selections | 36/36 | 36/36 |
| Bit-exact logits/KV positions | 48/48 | 0/48 |
| Worst absolute logit error | 0 | 3.46194 |
| Exact eight-token trajectories | 12/12 | 9/12 |
| Matching generation positions | 96/96 | 76/96 |

No prompt exceeds the public ranges. Native numeric outputs agree with an
independent integer oracle on 132,145,152 elements. This is a narrow fidelity
diagnostic, not a representative answer-quality evaluation or a comparison with
upstream floating-point generation. The prompts are now spent as held-out data.
Protected execution costs are measured independently through the ordinary SDK.

```sh
uv run python -m benchmarks.research.nonlinear_decoder_quality \
  --output nonlinear-decoder-quality-reproduction.json
```

Evidence: `docs/evidence/research-nonlinear-decoder-quality-qwen25.json`.
Replayable protected controls and exact seeded/residue transport combinations:
`benchmarks/research/nonlinear_decoders.py`. Fused/unfused may rank within their
numeric cohort; clear-SiLU has a different numeric identity even if selections
happen to match. No cost or throughput conclusion follows from quality replay.

## Bounded cost attempt

The five-configuration Qwen2.5 sweep (clear SiLU, both masked-wire layouts,
then both exact transport compositions) used a frozen installed wheel, isolated
clients, shared native 100/40 Mbps caps and two requested outputs. It stopped
at the **3,600-second per-request timeout**. The last live request had produced
no first token. The CLI did not persist its final comparison or earlier candidate
reports, so no completed response-cost or layout ranking is available.

`docs/evidence/research-nonlinear-decoders-qwen25-timeout.json` preserves the
command, factory and wheel hashes, observed failure and unknown measurements.
Future long sweeps need per-candidate persistence before another cost attempt.

## Qwen3-4B allocation gate

The same composed two-worker factory was inspected with the pinned Qwen3-4B
checkpoint, a 64-token input bound, two output tokens and native WAN pacing.
Both layouts require an estimated **14.26 GiB** of host memory, including
**0.97 GiB** for the client and **6.63 GiB per worker**, plus relay memory.
Admission rejected them against **7.08–7.09 GiB** of safe headroom at inspection.
This is current-host admission while the Qwen2.5 cohort runs, not a measured
peak or an assertion about every 32 GiB host. No 4B weights or provider roles
were loaded by this check.

`docs/evidence/research-nonlinear-decoders-qwen3-preflight.json` retains both
complete Experiment specifications, allocation owners and host observations.
Recheck either saved specification with the ordinary SDK:

```python
import json
from pathlib import Path
from pllm import Experiment
from pllm.metrics import benchmark_memory

document = json.loads(Path(
    "docs/evidence/research-nonlinear-decoders-qwen3-preflight.json"
).read_text())
experiment = Experiment.from_spec(document["cases"][0]["experiment"])
print(benchmark_memory(experiment, backend="native", enforced_wan=True))
```
