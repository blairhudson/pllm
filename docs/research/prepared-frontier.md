# Prepared-stack latency and memory frontier

## Measurement contract

The retained frontier cohort's two public-prefix variants spent approximately
25.7 seconds online despite very different request durations. Bundle setup and
the point at which offline preparation was charged therefore needed a matched
test before attributing that difference to paging kernels.

`benchmarks/research/prepared_frontier.py` selects four ordinary prepared SDK
Experiments with the same pinned Qwen2.5-0.5B W8A8 body, full-causal `prefix_f32`
numeric contract, packed requests, output residues, compressed artifact delivery,
indexed tokenizer and compiler-demand, on-demand correction issuance:

1. Paged client and Preparation weights, serial stage preparation.
2. The same composition with four concurrent stages under a 64 MiB work window.
3. The windowed composition with explicitly enabled client prefix reuse.
4. The windowed composition with resident client weights and paged Preparation.

All body linear stages remain provider-owned. The first three retain paged client
snapshots; the resident candidate isolates client storage. Preparation stays paged
in all four. Each fresh client executes the same 150-input/eight-output-token
request twice through an explicit ordered sequence and a two-request execution
budget (`benchmarks/research/prepared_frontier_sequence.py`). One parent
invocation owns the private cohort salt. Native relays share each party's
100 Mbps download and 40 Mbps upload access capacity. The first response and
repeat are reported separately; their setup and reuse costs are not interchangeable.

The public tokenizer index is pre-positioned. Its 9,285,336 bytes and the public
artifact builder's CPU are additional costs. The builder also creates a public
prefix capsule, but these four Experiments do not select it. Cached checkpoint
distribution, physical wire, Internet RTT and independent operator deployment
are outside the measurement.

## Bounded preparation window

`PreparedInventory(stage_window=4, stage_window_bytes=64 << 20)` explicitly
admits larger concurrent correction jobs. The default remains 16 MiB. Python and
native compilation reject budgets outside 1–64 MiB, booleans and nondefault
budgets without concurrency. The Pipeline binds the selected budget.

Runtime preflights every stage before submitting any job. Outstanding declared
work must fit both the byte budget and stage count. Acknowledgements precede
inventory sealing. Failure or cancellation burns the attempt before joining
workers and discards completed unused results. Whole-topology memory admission
conservatively prices all concurrent owners in addition to retained weights.

The window affects offline work only. It neither changes output numerics nor
introduces preparation during an active response. Exact generated-checkpoint
SDK/transport controls cover demand allocation, paging, codecs, prefix reuse and
the separately admitted resident-Preparation Freivalds composition.

## Matched ordered-sequence result

The canonical report is
`docs/evidence/research-prepared-frontier-sequence-qwen25.json`.
All eight requests and the comparison pass. Each request produces eight tokens
with the same output digest; all four clients observe zero new swap. These are
one cold and one exact-repeat request per isolated client, rather than repeated
samples of each lifecycle state. Decimal MB are used below.

| Composition | Cold request s | Repeat request s | Cold / repeat preparation s | Client lifetime peak MB | Two-request TCP streams MB | Cold-first aggregate CPU s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Paged, serial | 140.02 | 107.40 | 42.04 / 41.98 | 297.99 | 830.67 | 78.92 |
| Paged, four stages / 64 MiB | 136.29 | 103.45 | 38.18 / 38.09 | 281.85 | 830.67 | 80.51 |
| Paged, windowed, prefix reuse | 139.11 | 11.67 | 40.92 / 5.89 | 284.39 | 495.07 | 83.57 |
| Resident client, windowed | 135.72 | 100.49 | 38.10 / 34.92 | 564.77 | 830.67 | 80.49 |

The larger window lowers measured cold preparation by **9.18%**, but complete
cold/repeat request times improve only **2.67%/3.67%**. Covered application bodies
are identical, and cold-first aggregate CPU rises **2.02%**. The bounded option
therefore remains explicit.

Exact-prefix reuse lowers the second request **88.72%** against the windowed
control and cuts the two-request TCP-stream total **40.40%**. Covered bodies
fall from 829.00 to 493.44 MB, with setup counted once. Its first response still
costs 139.11 seconds and the same 476.95 MB covered bodies as every candidate;
the saving depends on this repeated prompt. Per-response decode bodies also
remain unchanged. The pre-positioned 9.29 MB tokenizer index is additional in
every case.

Windowed paging halves client lifetime peak RSS against resident client weights:
564.77 to 281.85 MB (**50.09%**). Online times stay near 65 seconds in both
non-reuse layouts. Their repeat-request gap is dominated by Preparation timing,
so this one sequence does not isolate a paging-kernel latency regression.
Sampled simultaneous total RSS is 1,649.07 MB paged and 1,938.93 MB resident;
it is not a sum of independent lifetime peaks. Client peaks cover both requests,
while the aggregate CPU column covers startup through only the first response.

The source/replay hashes are in
`docs/evidence/research-prepared-frontier-sequence-configurations.json`.
The separate publisher report records 16.19 CPU seconds including checkpoint
loading and the unused capsule build; it does not isolate tokenizer compiler
CPU. This cohort establishes neither representative generation quality nor a
10× client-memory improvement. Full physical wire, artifact distribution and
independent operators remain unmeasured.

## Reproduction

Run from the checkout after building the release extension. Python module
invocation makes the checkout's benchmark factories importable. Output paths
must be new.

```sh
uv run --no-sync python scripts/build_incremental_sdk_experiments.py \
  target/prepared-frontier-artifacts
uv run --no-sync python - <<'PY'
import json
from pathlib import Path
root = Path("target/prepared-frontier-artifacts")
prompt = (root / "prompt.txt").read_text().strip()
(root / "repeat-sequence.json").write_text(json.dumps([prompt, prompt]) + "\n")
PY
PLLM_RESEARCH_ARTIFACTS=target/prepared-frontier-artifacts \
uv run --no-sync python -m pllm benchmark run \
  --experiment benchmarks.research.prepared_frontier_sequence:serial \
  --experiment benchmarks.research.prepared_frontier_sequence:windowed \
  --experiment benchmarks.research.prepared_frontier_sequence:reused \
  --experiment benchmarks.research.prepared_frontier_sequence:resident \
  --factory --trust-python --backend native --wan --isolate-candidates \
  --prompt-sequence-file target/prepared-frontier-artifacts/repeat-sequence.json \
  --max-output-tokens 8 --warmups 0 --repetitions 1 --temperature 0 \
  --capture-output-digest --timeout 900 \
  --output target/evidence/prepared-frontier-reproduction.json
```

The unchanged artifact builder also records public compilation time and observed
swap growth. Its unrelated generated experiments and capsule are not inputs to
this cohort.

The initial repetitions-based screen is retained separately in
`docs/evidence/research-prepared-frontier-screen-qwen25.json`. All eight individual
requests passed their runtime/audit checks and matched output digests, but the
comparison rejected mixed cold/warm repetitions. Its failed comparison is not a
ranked cohort; the ordered-sequence contract explicitly represents that lifecycle.

The subsequent [prepared-kernel implementation ablation](prepared-kernel-reference.md)
keeps this topology and numeric contract while measuring NEON correction/online
kernels and bounded compressed-frame decoding. Its synthetic kernel screen and
matched installed-build response measurements have separate scopes.

The later [fused client-boundary ablation](prepared-boundary-reference.md)
preserves encoded inputs and output bits while removing intermediate tensors.
Its matched cold/repeat pair measures a 2.51% client peak-RSS reduction with
inconclusive complete-request latency benefit.

The subsequent [correction, decode and artifact screen](prepared-delivery-reference.md)
narrows exact correction subtraction and declines WebSocket deflate on prepared
arithmetic frames. A matched SDK pair improves decode throughput 19–24% and
reduces Preparation peak RSS 8.35%, but cold latency and client peak RSS rise.
Application bodies match while TCP-stream traffic rises slightly. Prebuilt
public-artifact replay is cheaper locally, but first-use publication plus replay
costs more CPU; ordinary warm clients already reuse their own object cache.

A separate [frozen precision screen](prepared-precision-reference.md) compares
four numeric choices against streamed float32 prefill and same-token decode.
Equalization improves this cohort's prefill selections but regresses decode;
neither it nor the lower-bit choices earns a default change.
