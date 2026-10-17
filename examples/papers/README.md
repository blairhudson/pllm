# Paper experiments

Each implemented method is a small Python `Experiment` factory. Model and budget
come from `common.py`; public inputs come from `prompts.json`. Change those shared
inputs to repeat the study on another supported checkpoint. Factories also accept
`model` and `budget` in ordinary Python.

From the repository root, with PLLM and its `quality` extra installed:

```sh
python -m pllm benchmark run \
  --experiment examples.papers.gkm:experiment \
  --experiment examples.papers.prepared:experiment \
  --factory --trust-python --backend native --temperature 0 \
  --prompt-sequence-file examples/papers/prompts.json \
  --max-output-tokens 8 --warmups 0 --repetitions 3 \
  --isolate-candidates --output results.json

python -m pllm benchmark quality \
  --experiment examples.papers.gkm:experiment \
  --experiment examples.papers.prepared:experiment \
  --factory --trust-python --prompts-file examples/papers/prompts.json \
  --reference-storage streamed --decode-steps 3 --output quality.json
```

The default source is an offline, revision-pinned Hub snapshot. Set `MODEL` to a
local `Model.path(...)`, or set `local_files_only=False` in `common.py` for the
initial Hub download. Normal SDK source and memory admission apply. Add
`--preflight-only` to `benchmark run` to check available host memory first.

`benchmark run` measures the selected runtime and reports its usual resource,
traffic and latency metrics. GKM buffers output until verification, so streaming
decode throughput is unavailable. Its Q11 numeric profile differs from W8A8;
the ordinary comparison rules retain that distinction.

`benchmark quality` uses the same source and token cohort for all candidates.
The first candidate supplies the shared decode trajectory. GKM runs its actual
two-worker protocol and releases logits only after final verification; prepared
candidates use their existing clear-kernel numeric diagnostic. Quality reports
contain aggregate agreement and error, not serving costs or private payloads.

## Implementation sequence

GKM has the first executable paper factory. Maverick, SIGMA, FuseFSS, BumbleBee,
PUMA, NEXUS, THOR, CacheMir and ROSETTA still need complete SDK compositions.
Their existing primitive references do not substitute for those decoders.
As each composition lands, add its factory here and use these same commands.
Current gaps are tracked in [the ten-method coverage table](../../paper/TEN-METHOD-COVERAGE.md).

Implement missing algorithms in reusable native/compiler/runtime components.
Extend shared benchmark metrics when needed. Paper files contain configuration;
the SDK owns execution, admission, measurement and report output.
