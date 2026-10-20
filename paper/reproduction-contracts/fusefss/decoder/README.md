# FuseFSS whole-decoder integer oracle (D13 arithmetic)

Independent pure Python arithmetic composition; public generated weights only.
No author code, Rust/native arithmetic, checkpoint-weight download, SDK/core changes,
protected execution, native replay, or model quality/performance claim.

Read [numeric decisions](NUMERIC_DECISIONS.md) first: D8/D9 settle scalar and
transaction rules; D13 approves whole linear/attention-scale and dense RMSNorm/
RoPE arithmetic under two explicit decoder profile IDs. Five generated controls:
two GPT-2 learned-position/LayerNorm/GELU graphs (tied head and untied all-score
tie/EOS control), plus Qwen2, Qwen3 per-head Q/K norm, and Llama structural
controls using dense SiLU/RMSNorm/split-half RoPE and causal GQA. Arithmetic
approval does not qualify native execution, checkpoints or model quality.

Each graph: two layers, head_dim 4, vocabulary 7, prefix `[2,4]`, three actual
selections with full-prefix lengths 2,3,4. GPT-2 has hidden 8 and two query/KV
heads. Qwen2 has hidden 16, four query heads, two physical KV heads, exercising
nontrivial contiguous grouping `[0,0,1,1]` rather than incorrect modulo grouping.
Qwen3/Llama controls have hidden 8, two query heads and one KV head. Every selection computes
complete embedding, attention/MLP, final norm and all seven generated vocabulary
scores. This deliberately avoids huge all-checkpoint/all-vocabulary fixtures.
No persistent KV cache or EOS early stop. Public weights rotate exact F32/F16/BF16
dyadic payloads; external exhaustive encoding controls cover half ties, signed
zero, subnormals, nonfinite and signed50 rejection.

```sh
python3 -B paper/reproduction-contracts/fusefss/decoder/freeze.py --check
python3 -B -m unittest discover -s paper/reproduction-contracts/fusefss/decoder -p 'test_decoder.py' -v
```

`inputs.json` freezes all source bits and public graph choices. `expected.json`
freezes every captured intermediate tensor, selection and state. `source-lock.json`
hashes sources, oracle, tests, equations, schema and both bundles. Existing frozen
scalar LUT bytes are reused with independently checked hashes; norm, activation,
softmax, contractions, rotations, GQA and feedback arithmetic are composed locally,
without calling gate-set functions. Tests alone compare scalar stages to existing
gate-set reference. [Exchange schema](EXCHANGE.md) is ready for later native
producer; no native output has been supplied or checked.

Explicit regeneration (not part of validation):

```sh
python3 -B paper/reproduction-contracts/fusefss/decoder/freeze.py --write
```

All traces are **clear test diagnostics**. A protected implementation must never
send these secret tensors/IDs to provider or coordinator as diagnostic output;
native conformance collection requires an isolated generated-test boundary.

Original GPT-2 small public metadata bytes are separately pinned under
[`gpt2-source/`](gpt2-source/README.md), with actual SHA-256 and HTTP 404 evidence
for absent special-token map/safetensors index. This is not a complete checkpoint
or tokenizer lock. `fetch_gpt2_source.py --check` verifies offline; normal decoder
tests never fetch network sources.
