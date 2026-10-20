# Strict generated decoder exchange v1

Input bundle: `inputs.json`, schema `pllm.fusefss.decoder-inputs.v1`, exactly
`schema,cases`. Five ordered generated cases. Per-case exact fields and bounds
are executable in `oracle.validate_contract`; unknown or omitted fields reject.
Each external tensor is exactly `{dtype,shape,words}`; dtype is `f32`, `f16`, or
`bf16`. Words are fixed-width lowercase hexadecimal IEEE payloads, most significant
digit first, not host-native byte strings. Shape is explicit, row-major. Canonical
matrix orientation is `[out,in]`; embedding/head is `[vocab,hidden]`.

Output bundle: exactly `{schema,results}` with schema
`pllm.fusefss.decoder-batch.v1`. One result per case, same order. Each result:

* `schema`: `pllm.fusefss.decoder-exchange.v1`.
* `profile`: `fusefss.gpt2-complete-ring50-q12.v1` for GPT-2;
  `fusefss.dense-rope-ring50-q12.v1` for Qwen2/Qwen3/Llama controls. Input case
  includes the same `profile`; family/profile mismatch rejects.
* `graph_status`: `d13-approved-arithmetic` (arithmetic approval only).
* `fixture_id`: matching input `id`.
* `input_sha256`: SHA-256 of canonical per-case JSON (sorted object keys, compact
  separators, ASCII escapes, one trailing LF), not whole bundle bytes.
* `tensors`: ordered complete operator-intermediate tensor list from `expected.json`.
  Each exactly `{name,shape,axes,fractional,bits,signed,values}`. `bits=50`,
  `signed=true`; values are mathematical **signed decimal integers**, not ring
  payloads, floats, booleans, hashes, or tolerances. Q24 raw contractions precede
  Q12 floor outputs. `fractional=0` for IDs/masks/indices. `axes` names and tensor
  order are normative. Batch-one head tensors use `[batch,heads,sequence,feature]`;
  score tensors use `[query_sequence,key_sequence]` per query head. Query head
  `h` maps to physical KV head `h//(heads/kv_heads)`. No materialized repeated KV
  axis is allowed to masquerade as original physical layout.
* `states`: one ordered record per selection, exactly `step,prefix_length_in,
  prefix_length_out,positions_recomputed,persistent_kv,selected_id_private,
  prefix_ids_private,released`. The ID-bearing values are diagnostic clear oracle
  observations, **not authorized wire disclosure by protected workers**.
  `persistent_kv=false`, `released=false`; recomputed positions start at zero.
* `final`: exactly `recipient,token_ids,generated_ids,selection_count,persistent_kv`.
  Recipient `client-only`. `token_ids` includes input prefix plus all T generated
  IDs. `generated_ids` is suffix only. T does not stop at EOS. No logits output.

All dimensions/values/counts must have exact JSON types. Duplicate keys, NaN,
Infinity, missing/unknown/reordered tensors, wrong axes/scales, different tie or
feedback state, integer-as-float and boolean-as-integer all reject. Parser limits
whole input file to 8 MiB. Frozen fixture bounds: hidden<=32, intermediate<=64,
layers<=3, heads/KV heads<=4, head_dim<=8, vocabulary<=32, positions<=16, T<=4.
Contractions check every product and ordered partial sum for signed50 admission,
even if later cancellation or rescale would fit. Not a model-wide range proof.

Future native producer must export this exact bounded diagnostic schema, then:

```sh
python3 -B paper/reproduction-contracts/fusefss/decoder/check_outputs.py native-output.json
```

Checker verifies source/input/output hashes, regenerates Python expected tensors,
checks frozen expected, then compares every producer value with exact types and
fields. It launches no producer. Passing an oracle copy is only checker self-test,
never native evidence. Native receipt/build identity must be recorded separately.

Mutation controls in tests alter every major stage, states, types, shapes, fields,
causal/GQA declarations and source inputs. Expected outputs cannot update during
checking; explicit `freeze.py --write` is sole regeneration action. Review changed
source/input/expected hashes before accepting a new freeze.

## Producer parsing and layout recipe

Read `inputs.json` directly as JSON; do not import this Python oracle in native
execution. Case order is normative. Input dictionaries are unordered: iterate
public tensor names in **lexicographic order** for `public.*` output traces.
Other trace names/order follow `expected.json`. Each case includes `family`,
`profile`, `graph_status`, `dimensions`, `prefix`, `generation_count`, `eos_id`,
`tied_head`, `attention_scale`, `norm_contract`, `rope_contract`, `tensors`, `id`.
All source tensor shapes/hex words are in each case: no source checkpoint parser,
numpy, external dependencies, implicit weight filenames or RNG is needed.

`attention_scale` uses the same `{dtype,shape,words}` format with shape `[1]`.
Bounded head dimensions 2/4/6/8 bind F32 inverse-root bit words
`3f3504f3`/`3f000000`/`3ed105ec`/`3eb504f3`, respectively. Generated fixtures all
use head_dim 4. A different coefficient or non-F32 RoPE declaration rejects.
Norm identity is `d8-layer-sumsq-plus-one` or `d13-rms-sumsq-plus-one`; RoPE is
`none` or `d13-split-half-per-product-floor`. Embedding and tied head share exactly
`tensors.embedding` bytes. Untied head is `tensors.head`. All projection weights
are `[out,in]`; bias tensors exist only where explicitly declared. Never fill
missing unexpected weights with zeros: only graph's absent bias defaults to zero.
GPT-2 Conv1D storage conversion is already reflected in canonical generated
matrices. Tensor values are flattened row-major.

For a first native export, preserve all captured tensors, including public Q12
weights, Q24 accumulators and separately floored RoPE terms. Array counts vary
by family. Approval labels must not be replaced with native-execution claims.

External IEEE decode needs no host floating-point rounding: extract sign,
exponent and fraction; add hidden leading one for normal values; form magnitude
`mantissa * 2^(unbiased_exponent - fraction_bits + 12)`. For negative powers,
divide nonnegative mantissa by that power of two, increment quotient iff remainder
is above half or is exactly half with odd quotient, then apply sign. Reject
all-ones exponents and out-of-signed50 results. Positive signed50 endpoint rejects;
negative endpoint is permitted. Zero/subnormal handling is explicit in `ieee_q12`.
Internal floors are different: signed division must round toward negative
infinity, not Rust/C truncation toward zero. Every raw product and ordered partial
sum is range-checked before division; later cancellation cannot repair overflow.
