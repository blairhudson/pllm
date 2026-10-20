# Strict semantic-execution replay v1 (distinct from original EXCHANGE)

This is a new, explicitly narrower **evidence interface**, not a waiver or passing
subset of the original 5,125-tensor exchange. `inputs.json`, `expected.json`,
`oracle.py`, original checker and historical native witness/rejection artifacts
remain unchanged. D13's clarification preserves existing oracle arithmetic:
`floor(l*c/4096)+floor((-r)*s/4096)`; negation is before product/floor. The old
evidence README's contrary interpretation is historical, not normative.

## Fixed coverage before any semantic native replay

`semantic-graphs.json` freezes **every** operation, dependency, declared shape,
attribute and ID/index from the previously captured public source-model lowering
snapshot, bound by its SHA-256. It contains no private native results. Independently
prescribed `blueprint()` validates architecture-specific complete node order:
58 GPT-2, 62 Qwen2/Llama, 66 Qwen3 nodes per selection, three selections each,
**918 observations total**. A producer cannot submit a different graph or choose
which tensors pass. Graph modification is a source-lock failure. This is a frozen
reviewed public semantic graph, not a claim that two unrelated model lowerings
independently generated identical IR.

`semantic_reference.py` interprets that graph independently in Python integer
arithmetic, using the existing independent oracle's frozen scalar tables and
equations. No Rust/native calls, model code, checkpoint, producer values or native
clear recomputation supplies expected values. All public IEEE encodings are exact.
The source values/shapes/encodings and every node are required even if redundant.

## Required strict batch

Root fields: `schema="pllm.fusefss.semantic-execution-batch.v1"`, ordered `results`
containing all five cases in `inputs.json` order. Each result has exactly:

* `schema="pllm.fusefss.semantic-execution.v1"`, `fixture_id`, `profile`;
* `input_sha256`: canonical input case; `semantic_graph_sha256`: canonical fixed
  graph case. Canonical JSON is sorted ASCII keys, compact separators, newline;
* `source_encodings`: lexicographically sorted **all** fixture public tensors plus
  `attention_scale`. Exact fields: `name,dtype,shape,ieee_words,q12_values`.
  Native exporter must run `Tensor::from_ieee` on actual source encodings, not copy
  expected Q12 values. Original source hex words are public input metadata;
* `nodes`: selection-major, then exact frozen graph order, every node once.
  Exact fields: `selection,semantic_index,id,shape,fractional,values`;
* `states`: one per selection, exact fields `selection,active_prefix_ids_private,
  input_token_buffer_private,active_positions,physical_positions_computed,
  selected_id_private,logical_prefix_after_selection_private,persistent_kv,released`;
* `final`: exact fields `recipient="client-only",token_ids,generated_ids,
  selection_count,persistent_kv=false`.

`values` are signed ring50 integers in complete row-major native physical shape,
never selected coordinates. Fractional bits 12 except selection/feedback IDs (0).
Physical width is `S=len(initial_prefix)+T-1=4`. All actually computed padding
rows are included, **not just active rows**. Three-dimensional hidden tensors use
`[1,S,width]`; physical Q/K/V use `[1,heads,S,head_dim]`; attention uses
`[1,query_heads,S,S]`, causal GQA KV head `query_head//group_size`.
Both cache append/view nodes expose their full fresh arrays each selection.
No persistent KV substitution. Source head split/merge and fused GPT-2 QKV slice
nodes are required, not silently removed as metadata.

At each selection token-buffer tail is ID zero, position indices span `0..S-1`,
and active prefix is initial IDs plus **actual native** earlier selections.
Padded query rows are computed; softmax admits keys `k<=query && k<active_prefix`.
`CausalMask` records the actual score identity view; masking occurs inside
softmax, not invented negative infinity. Public inputs are metadata, not missing
semantic operators. All source/public encoding outputs are independently checked.

`LastToken` exposes only row `active_prefix-1`, then `OutputHead` exposes the
**complete vocabulary** `[1,vocab]` for that last row. Lowest-index argmax and
private feedback both appear as native observations; they must agree with actual
final reconstructed `TokenShares`. State token buffers and logical prefix lists
are assembled only from original private inputs and reconstructed native feedback
IDs; no expected selected IDs. The final logical prefix includes the last token,
but does not assert a nonexistent final write beyond the native fixed buffer.
Only the test coordinator reconstructs checkpoints. Production remains final-only.

## Deliberate exclusions, no original-EXCHANGE pass

Earlier-position vocabulary head rows are **not executed**, therefore absent from
this schema. Nonlinear internal registers (rsqrt indices/inverses, raw multiply
registers, softmax internal tables, one-hot bit registers) are not exposed by
`Driver::observe`, therefore excluded from **this schema only**. Their separate
275-case primitive-composition oracle is supporting evidence, not whole-decoder
node replay or proof that original EXCHANGE passed. No clear arithmetic may be
invented as native internal output. Future internal instrumentation may satisfy
the original schema independently; its existing checker remains strict.

## Checker, mutation controls and claim ceiling

`python3 -B check_semantic.py PATH` regenerates all expected node/state/source
values independently, verifies both frozen locks, then performs exact recursive
type/field/list-order/value equality. Missing, extra, reordered, wrong-family,
changed source words, integer-vs-bool/float, wrong padding, changed selections,
changed graph, incomplete vocabulary and wrong final IDs reject. No producer
subset or allowlist is accepted. Primitive traces supplied as extras also reject.

`freeze_semantic.py --check` verifies frozen artifacts; `--write` writes only new
semantic artifacts. Root `freeze.py --refresh-source-lock` verifies original
payloads and writes only their source lock. Neither rewrites
original inputs/expected/oracle. The lock covers independent oracle inputs and
equations; native implementation history belongs in Git. Admission, resource/cap proofs,
fresh preprocessing, frames, source/build identity, transport/SDK qualification,
whole-method status and checkpoint quality remain separate evidence. No full
native run or cap changes are authorized by this contract.

Once the parent confirms source freeze/admission and authorizes replay, run only
the new exact test (parent must have registered the cfg(test) module):

```sh
PLLM_FSS_DECODER_EXPORT_DIR=target/fss-decoder-semantic-parent-v2 CARGO_BUILD_JOBS=1 cargo test -p pllm-compiler --release --locked --offline --lib fss_decoder::conformance::conformance_tests::attempt_strict_semantic_execution_replay -- --ignored --exact --nocapture --test-threads=1
python3 -B paper/reproduction-contracts/fusefss/decoder/check_semantic.py target/fss-decoder-semantic-parent-v2/semantic-v1-native-output.json
```

The test writes distinct `semantic-v1-native-output.json` and
`semantic-v1-receipt.json`; it never overwrites original rejection receipts.
Original-EXCHANGE reruns now use `original-exchange-rerun-v2-` artifact prefixes.
The corrected signed-product kernel regression writes
`rope-signed-product-correction-v2.json`, never the preserved historical witness.
