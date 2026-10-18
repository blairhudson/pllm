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

### Completed quality smoke

The [ordinary CLI report](../../paper/artifacts/gkm-sdk/quality-smoke.json) records
the three shared prompts, with 5/4/11 input tokens and three same-token decode
steps each. GKM matches all 3 prefill and 9 decode reference selections; prepared
W8A8 matches 2/3 and 9/9. Worst absolute prefill/decode logit errors are
1.187/1.865 for GKM and 3.467/6.121 for W8A8. This small diagnostic does not
qualify task accuracy, complete state equality or serving performance.

The completed invocation used the shared factories and requested zero additional
memory margin:

```sh
python -m pllm benchmark quality \
  --experiment examples.papers.gkm:experiment \
  --experiment examples.papers.prepared:experiment \
  --factory --trust-python --prompts-file examples/papers/prompts.json \
  --reference-storage streamed --decode-steps 3 \
  --memory-safety-margin-percent 0 --output quality.json --format json
```

## Implementation sequence

GKM and PUMA have executable paper factories. `puma.py` uses the same `MODEL`
and `BUDGET`; its current bounded admission rejects the common full-size Qwen
checkpoint rather than substituting a generated model. To exercise an admitted
source, call `puma.experiment(model=..., budget=...)`, save its ordinary Experiment,
and use the same gateway and `benchmark run` entry points. Generated-model SDK,
gateway and benchmark controls live in `tests/test_replicated_sdk.py` and
`tests/test_replicated_interfaces.py`.

PUMA keeps activations and token feedback shared across three workers. D9 requires
full-prefix recomputation for each fixed-count selection; only final token IDs
reach the client. Streaming first-token/decode rates and full-logit quality
metrics therefore remain unavailable. Peer-to-peer bodies are included in its
ordinary benchmark ledger. Its Q18 profile has its own numeric identity and
does not inherit GKM or W8A8 quality evidence. See the
[PUMA contract](../../docs/research/puma-truncation-reference.md#bounded-sdk-composition).

Maverick, SIGMA, FuseFSS, BumbleBee, NEXUS, THOR, CacheMir and ROSETTA still need
complete SDK compositions under their approved source contracts.
Their existing primitive references do not substitute for those decoders.
As each composition lands, add its factory here and use these same commands.
Current gaps are tracked in [the ten-method coverage table](../../paper/TEN-METHOD-COVERAGE.md).

Implement missing algorithms in reusable native/compiler/runtime components.
Extend shared benchmark metrics when needed. Paper files contain configuration;
the SDK owns execution, admission, measurement and report output.
