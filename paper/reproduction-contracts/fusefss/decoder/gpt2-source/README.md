# Original GPT-2 public metadata bytes only

Captured 20 October 2026 from `openai-community/gpt2` at immutable revision
`607a30d783dfa663caf39e06633721c8d4cfcd7e`. `source-lock.json` records exact byte
counts/SHA-256, requested/resolved URLs, HTTP status and explicit exclusions.
Source files are inert JSON evidence, never executed.

Actual bytes acquired:

| File | Bytes | Observation |
| --- | ---: | --- |
| `config.json` | 665 | GPT2LMHeadModel; hidden 768, heads 12, layers 12, positions/context 1024, vocabulary 50257 |
| `generation_config.json` | 124 | BOS/EOS 50256; declared transformers version is metadata, not an evaluated frontend pin |
| `tokenizer_config.json` | 26 | `model_max_length=1024`; not complete tokenizer identity |
| `repository-metadata.json` | 4967 | Revision/sibling listing snapshot, including `model.safetensors` filename only |

Top-level `special_tokens_map.json` and `model.safetensors.index.json` return
**HTTP 404** at this revision and are not listed in repository metadata. No
alternative ONNX map is silently substituted. Repository lists unsharded
`model.safetensors`, not a sharded safetensors index. No weight file, safetensors
header range, tokenizer vocabulary/merge payload, pickle, or author code downloaded.
Closed fetch allowlist and 256-KiB per-response ceiling prevent accidental weight
downloads. API snapshot is a timestamped source observation; its own public counters
may change if explicitly fetched again, despite immutable checkpoint revision.

Source declarations differ intentionally from D13: `activation_function=gelu_new`
and `layer_norm_epsilon=1e-5`, while approved integer decoder uses D8 table GELU
and sum-squares-plus-integer-one guard. Source `task_specific_params` sampling/max
length declarations do not override D9 fixed-T private greedy transaction.

This is **header/metadata-only source evidence**, not a full checkpoint/tokenizer
lock, safetensors tensor-shape inventory, native run, historical FuseFSS evaluated
build identity or original-model quality result. Config alone does not freeze
learned weights, tokenizer bytes, Conv1D/head storage aliasing or frontend behavior.

Offline verification:

```sh
python3 -B paper/reproduction-contracts/fusefss/decoder/fetch_gpt2_source.py --check
```

Explicit network refresh (not invoked by tests/checkers): same command with
`--fetch`. Review metadata and source-lock diffs, then regenerate decoder freeze.
