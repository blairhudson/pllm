# Request tokenizer ownership and client memory

## Matched result

Pinned Qwen2.5-0.5B W8A8, 37 input tokens and eight greedy outputs, with lazy
native masks and paged artifact storage in both candidates:

| Fresh client process | Eager decoder tokenizer control | Shared request tokenizer |
| --- | ---: | ---: |
| Process peak RSS, decimal MB | 308.003 | 290.963 |
| Cold client CPU, s | 11.119 | 10.889 |
| Request time, s | 19.344 | 19.032 |
| Tokenizer constructions | 4 | 3 |

Peak RSS fell **5.53%** in this single matched pair. Outputs and runtime/privacy
checks pass. Observed host swap growth is zero. This is a complete response through
two native prepared-provider children, using the existing benchmark driver and
fresh isolated clients. RSS includes the in-process dashboard and excludes the
providers, OS file cache and total host memory. CPU and latency are single-sample
observations, not a general speed claim.

The SDK already creates a request tokenizer and encodes the full prompt before
inventory admission. It now reuses those exact token IDs, and uses the same
tokenizer for continuation suffixes and output decoding. The decoder creates its
own tokenizer only if a direct string helper needs it. No persistent tokenizer
cache or new private-token retention was added. Direct runtime string helpers
retain their behavior; token-ID prefill/decode does not construct a vocabulary.

The control reinstates eager decoder construction while retaining the current
shared execution path. It is an allocation ablation, not a complete historical
SDK binary. Tiny Qwen2/Qwen3 tests cover prepared, verified, two-offset, prefix
reuse, client placement and the gateway, comparing every checked logit and KV
array. They additionally fail if SDK execution accesses a decoder-owned tokenizer.

## Diagnostic sequence

Deferring construction alone did not reduce memory: the SDK still called
`runtime.encode_prompt()` and decoded through `runtime.tokenizer`, causing the
vocabulary to be rebuilt later. Both initial diagnostic pairs still constructed
four tokenizers. Retained call-site traces identified that path; the final change
reuses the already-admitted token cohort and request-owned tokenizer.

Initial controls: `tokenizer-memory-qwen25-2026-10-05.json` and
`tokenizer-memory-confirmed-qwen25-2026-10-05.json`.
Final matched evidence: `tokenizer-memory-shared-qwen25-2026-10-05.json`.
Call-site traces contain source locations, not prompts, token IDs or payloads.

## Reproduction and limits

```bash
uv run --no-sync python scripts/probe_client_runtime_memory.py \
  --tokenizer-ablation --output /tmp/pllm-tokenizer-memory.json
```

The output must not exist and ordinary whole-topology admission must pass. A
subsequent complete three-candidate rerun was rejected before model inspection
when host headroom fell below its reserve; it supplies no result. The earlier
2.17× complete-response cohort and this 5.53% pair have different salts and must
not be multiplied into a new matched reduction factor.

**The 10× whole-client target remains unmet.** The 9.44× Qwen3-4B storage result
still excludes body/KV execution, and a complete 4B run remains subject to the
whole-topology gate.
