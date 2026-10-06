# Promoted Qwen3-4B SDK tokenizer measurement

This fresh-process pair uses ordinary compiled SDK Experiments with the pinned
`Qwen/Qwen3-4B` checkpoint at revision
`1cfa9a7208912126459214e8b04321603b3df60c`. Both candidates use CPU W8A8,
paged client and Preparation weights, demand-aware issuance, packed requests,
row-residue outputs and compressed artifact delivery. The candidate adds only
`IndexedTokenizer`; private tokenizer queries remain local.

## Matched response

One cold **16-input / 8-output-token** response per candidate ran through separate
native Inference and Preparation children, with fresh isolated client processes
and shared per-party **100 Mbps download / 40 Mbps upload** TCP-stream caps.
Checkpoint artifacts were already cached. Runtime checks pass, outputs and
covered application bodies match, and no new swap was observed.

| Metric | Original tokenizer | Indexed tokenizer |
| --- | ---: | ---: |
| Client/dashboard lifetime peak RSS | 360.89 MB | 262.73 MB |
| Simultaneously sampled aggregate peak RSS | 5,574.97 MB | 5,815.50 MB |
| Full request latency | 166.341 s | 166.604 s |
| Cold client CPU, including setup | 43.949 s | 43.729 s |
| Cold aggregate CPU, including role startup | 235.032 s | 230.448 s |
| Covered setup-inclusive application bodies | 556.31 MB | 556.31 MB |
| Cold TCP-stream bodies | 557.81 MB | 557.81 MB |

The client peak falls **27.20%**. This is a client-memory improvement in one
response, not an aggregate-memory or throughput win. Aggregate sampling covers
live client/provider RSS every 50 ms rather than summed independent peaks;
short peaks can be missed and shared pages can be counted more than once.
The ordinary report retains directed TCP-stream counters separately from
application bodies; kernel headers, retransmissions and full wire are absent.

## Additional public setup

The trusted offline tokenizer compiler produces **9,285,852 bytes** of public
SQLite index and contract artifacts, using **0.733 CPU seconds** in the separate
publisher sample. These artifacts were pre-positioned for the SDK pair; their
distribution is additional to response traffic. Eight public multilingual,
special-token and whitespace cases match original encode/decode results.

A separate warm microprobe performs 400 encode calls and 400 decode calls per
implementation. Original/indexed encode CPU is 0.00513/0.06459 seconds; decode
CPU is 0.000600/0.005366 seconds. The index trades tokenizer CPU and additional
artifact storage for smaller client memory. These local microprobe totals do
not establish representative tokenization latency.

The retained native 4B measurements do not include a matched resident-weight
control, generation-quality comparison, independent provider operators or cold
checkpoint transfer. Earlier probe-only memory ratios are separate cohorts and
are not multiplied into this result. No 10× whole-client claim follows.

## Reproduce

Cache the exact checkpoint first. Build artifacts into a new directory:

```sh
uv run python -m benchmarks.research.qwen3_artifacts \
  .artifacts/qwen3-sdk-reproduction
export PLLM_RESEARCH_ARTIFACTS="$PWD/.artifacts/qwen3-sdk-reproduction"
uv run python -m pllm benchmark run \
  --experiment benchmarks.research.qwen3_sdk_control:experiment \
  --experiment benchmarks.research.qwen3_sdk_indexed:experiment \
  --factory --trust-python --backend native --wan --isolate-candidates \
  --prompt-file "$PLLM_RESEARCH_ARTIFACTS/prompt.txt" \
  --temperature 0 --capture-output-digest --max-output-tokens 8 \
  --repetitions 1 --timeout 900 \
  --output docs/evidence/research-qwen3-sdk-artifacts-reproduction.json
```

Add `--preflight-only --format json` to inspect the same memory admission before
launching providers. The runtime enforces admission and aborts owned work on
reserve loss or new swap growth.

Canonical evidence:

- `docs/evidence/research-qwen3-sdk-artifacts.json`
- `docs/evidence/research-qwen3-sdk-artifacts-publisher.json`
- Replay factories: `benchmarks/research/qwen3_sdk_control.py` and
  `benchmarks/research/qwen3_sdk_indexed.py`

The public-artifact publisher records its tokenizer source, contract and compiler
digests. The scorecard checks those commitments and retains this model as a
separate cohort from Qwen2.5.
