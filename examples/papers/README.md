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

## Broader GKM source-model quality

`gkm_quality.py` reuses the approved full/seeded GKM factories, pinned offline
Qwen checkpoint, one-thread CPU, two workers, final verification, and unchanged
32-request/64-input/8-output budget. It uses the existing twelve public prompts
below, not a downloaded dataset. From the repository root in the matching
installed environment with the `quality` extra and checkpoint already cached:

```sh
python -m pllm benchmark quality \
  --experiment examples.papers.gkm_quality:experiment \
  --experiment examples.papers.gkm_quality:seeded \
  --factory --trust-python \
  --prompts-file benchmarks/research/data/paper-quality-prompts.json \
  --reference-storage streamed --decode-steps 3 \
  --memory-safety-margin-percent 25 \
  --output docs/evidence/gkm-quality-public-v1.json --format json
```

This is one invocation: full sharing supplies the common greedy decode tokens,
and both candidates and the independent reference consume those same token IDs.
The harness checks actual tokenization/source locks and enforces admission without
truncation or fallback. Each GKM prompt requests four selections and accepts only
after request-end verification. The existing adapter can then observe full logits
for agreement/error metrics. It does not measure KV parity or streaming decode.
An ID-only candidate supports top-1 agreement only, with logit metrics unknown.

GKM uses nearest-even scale-2048 i16/Q11 linears, ring64 products, original float32
lookup, and the declared client-float32 profile. The independent reference uses
the same checkpoint with eager Transformers FP32 operators and module-streamed
weights. This measures source-model fidelity, not exact Q11 equality. W8A8 is
neither the source reference nor a substitute GKM execution. Full/seeded sharing
keeps its distinct randomness assumptions. The ordinary report records actual
source locks, numerical profiles, token-cohort identities and supported metrics.
The command remains unrun here; this public diagnostic is not a held-out task
suite, serving-cost result, or publication qualification. Failures abort rather
than omit positions or create partial-cohort scores. Keep the output write-once.

## Implementation sequence

GKM and PUMA have executable paper factories. `puma.py` uses the same `MODEL`
and `BUDGET`; its current bounded admission rejects the common full-size Qwen
checkpoint rather than substituting a generated model. To exercise an admitted
source, call `puma.experiment(model=..., budget=...)`, save its ordinary Experiment,
and use the same gateway, `benchmark run` and `benchmark quality` entry points.
Generated-model SDK, gateway and benchmark controls live in
`tests/test_replicated_sdk.py`, `tests/test_replicated_interfaces.py` and
`tests/test_replicated_quality.py`.

PUMA keeps activations and token feedback shared across three workers. D9 requires
full-prefix recomputation for each fixed-count selection; only final token IDs
reach the client. Streaming first-token/decode rates and full-logit quality
metrics therefore remain unavailable. Peer-to-peer bodies are included in its
ordinary benchmark ledger. Its Q18 profile has its own numeric identity and
does not inherit GKM or W8A8 quality evidence. See the
[PUMA contract](../../docs/research/puma-truncation-reference.md#bounded-sdk-composition).
Top-1 quality uses fresh one-selection requests on the shared full prefixes.
Budget every measured position as a request and include decode tokens in the
input bound. Top-k recall and logit error stay unknown; this diagnostic does not
measure a single multi-output serving response.

Maverick, SIGMA, FuseFSS, BumbleBee, NEXUS, THOR, CacheMir and ROSETTA still need
complete SDK compositions under their approved source contracts.
Their existing primitive references do not substitute for those decoders.
As each composition lands, add its factory here and use these same commands.
Current gaps are tracked in [the ten-method coverage table](../../paper/TEN-METHOD-COVERAGE.md).

Implement missing algorithms in reusable native/compiler/runtime components.
Extend shared benchmark metrics when needed. Paper files contain configuration;
the SDK owns execution, admission, measurement and report output.

## Matched Qwen3 client-memory campaign

`low_memory_qwen3.py` configures maintained prepared versus paged/prewarmed
execution of the same offline Qwen3-4B revision, explicit W8A8/prefix-f32 body,
full client KV, greedy sampler and state audit. Both arms commit all 71 uniform
one-use rows before readiness; refill is idle-only. Missing rows reject rather
than generate material during a response. Paging already committed public
artifacts is allowed. Source/compiler/privacy/host admission is unchanged.

`low_memory_qwen3_prompt.txt` is public text, not token-ID padding. The opt-in
cached-tokenizer test checks this exact asset with the production SDK normalizer,
chat renderer and tokenizer descriptor at revision
`1cfa9a7208912126459214e8b04321603b3df60c`, including stable config/tokenizer
hashes before/after. This is **tokenizer-only 64**, not accepted SDK usage or a
guaranteed eight-token completion. No instructions, tools, history, extra BOS or
thinking-mode override is supplied. Early EOS, unknown usage or actual counts
other than 64 input / 8 output fail ordinary strict runtime checks; no EOS
suppression or forced minimum length is used.

After the generic strict-count/Suite CLI and opt-in client-memory API changes
are installed, use the ordinary trusted zero-argument wrapper. Supply all four
public choices explicitly; nothing captures a build, chooses host memory, or
downloads a checkpoint on import. `BuildRef.capture()` can identify a previously
approved installed build; it does not build one. Select the allowance within the
unchanged ordinary 16 GiB ceiling and actual host/whole-topology admission.

```sh
export PLLM_LOW_MEMORY_BUILD_DIGEST='<approved installed BuildRef.expected_digest>'
export PLLM_LOW_MEMORY_BYTES='<explicit whole-topology byte allowance>'
export PLLM_LOW_MEMORY_PROCESSES='<explicit independent replicate count, 1..32>'
export PLLM_LOW_MEMORY_TIMEOUT_SECONDS='<explicit per-attempt timeout, at most 3600>'
python -m pllm benchmark run \
  --experiment examples.papers.low_memory_qwen3:cli_suite \
  --factory --trust-python --no-input \
  --prompt-file examples/papers/low_memory_qwen3_prompt.txt \
  --output target/low-memory-qwen3.json --format json
```

These placeholders must be replaced, not executed literally. One invocation
supplies one private salt/cohort and fresh isolated workers per arm/replicate;
do not combine separate CLI invocations. Suite configuration supplies count,
state, sampling and resource contracts; conflicting CLI overrides are rejected.
Output uses public `pllm.benchmark_run.v1`, **not** the legacy loopback/WAN schema.
Completion means every attempt has a record, not that any measurement qualifies.

Inspect owner-aware comparisons through the ordinary report, without substituting
the root process for SDK/client helpers:

```python
from pllm.metrics import BenchmarkRun
from examples.papers.low_memory_qwen3 import CLIENT_MEMORY_METRICS

run = BenchmarkRun.load("target/low-memory-qwen3.json")
for metric in CLIENT_MEMORY_METRICS:
    print(run.compare("maintained", "paged-prepared", metric, minimum_reduction=10))
```

Sampled owner sums are non-atomic estimates; maximum individual observations
are lower bounds, not group peaks. Only a qualified isolated client-group
birth-through-complete-retirement interval can establish a scoped reduction
bound. Missing helper/source-setup ownership or final per-member HWM remains
unranked; a live root snapshot is insufficient. Scalar availability budgets
remain unknown when telemetry is absent. Providers stay separately charged.
Keep source loading, client helpers, retained inventory and retirement in the
declared lifetime; never silently narrow it to the online/root-only window.
The historical 8.25 GiB estimate and 0.83 GiB target are not measured RSS.
**No accepted 64/8 campaign or tenfold memory reduction has been measured here.**
