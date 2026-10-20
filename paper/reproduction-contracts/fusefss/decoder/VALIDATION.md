# Bounded Python validation evidence

Initial independent oracle validation: **25 tests pass** using standard-library
Python only.

```sh
python3 -B -m unittest discover -s paper/reproduction-contracts/fusefss/decoder -p 'test_decoder.py' -v
python3 -B paper/reproduction-contracts/fusefss/decoder/freeze.py --check
```

Freeze check reports **5 bounded D13 cases, 5,125 captured tensors**.
Fifteen selections total, complete recomputation lengths 2/3/4 per case.
Generated IDs: GPT-2 tied `[4,4,4]`; untied all-score tie/EOS control `[0,0,0]`;
Qwen2/Qwen3/Llama dense-profile controls each `[4,4,4]`. Prefix `[2,4]` retained in
final client array. Zero EOS never short-circuits control. Vocabulary only seven.

Canonical JSON sizes remain approximately 79 KiB inputs and 1 MiB expected.
`source-lock.json` records SHA-256 of every input/source/implementation/schema/test
file and expected bytes; checker validates those hashes before comparing results.

Coverage includes:

* Exhaustive 63,488 finite F16 patterns versus exact independent Fraction rounding.
* All 65,280 finite BF16 patterns: exact Fraction result or signed50 rejection.
* Deterministic 4,096-pattern F32 bit sweep plus signed50 endpoints; positive and
  negative exact half ties, negative zero, subnormals, NaN/infinity rejection.
* Independently composed activation, softmax and norm compared against existing
  scalar reference, including negative floor and width-768 reciprocal mean.
* Contraction accumulate-then-floor, unrenormalized softmax `[3013,1107]`, masked
  future maximum exclusion, zero-row norm guard, overflow-before-cancellation,
  per-product RoPE floors versus numerically different combined-product floor.
* Nontrivial physical GQA `[0,0,1,1]` and actual dot product against correct KV
  head, with distinct incorrect modulo-grouping result.
* Full embedding/learned positions/two-layer graph/final norm/full seven-column
  head/private tie selection/three feedback steps. Qwen3 Q/K norm before RoPE.
* Seventeen major-stage value mutations, private-state/release/KV mutations,
  malformed public shapes/weights/encoding/graph declarations, unknown fields,
  missing tensors, duplicate JSON keys, NaN, float-as-integer and bool-as-integer.
* Checker self-test accepts frozen **Python** expected copy, rejects mutated
  producer-shaped output. JSON object key order does not change tensor order.
* D13 family/profile binding, pinned attention scale and F32-only RoPE coefficient
  declarations reject mismatches; actual-byte GPT-2 metadata hashes/config/404
  observations verify offline without any network or weight access.

No native producer ran in this initial validation. No native replay, security, complete source checkpoint, original
evaluated numerical identity, full-checkpoint quality, SDK or performance evidence
is claimed. D13 approves arithmetic under separate GPT-2/dense-RoPE profile IDs;
all whole graphs carry `d13-approved-arithmetic`. Approval is not execution or
checkpoint qualification. Details are listed in `NUMERIC_DECISIONS.md`.
The subsequent semantic native replay is recorded in `SEMANTIC_VALIDATION.md`.

Separate metadata-only evidence: `gpt2-source/README.md` and its byte-level lock.
Three public config/tokenizer-config JSON files plus pinned repository listing
were captured; special-token map and safetensors index absent (HTTP 404). No
weight or model header downloaded. This does not qualify tokenizer/checkpoint
identity or model quality.
