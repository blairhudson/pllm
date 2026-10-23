# Maverick numeric candidate: pinned official FP8 header audit

**Chronology, 2026-10-23:** the header audit first contradicted D31's F32-storage
requirement. The user subsequently approved only the exact BF16-to-F32 scale
promotion recorded in [D32](../../paper/methods/DECISIONS.md#d32-maverick-pinned-bf16-scale-storage-amendment).
The candidate remains unimplemented and unqualified; source payload values and
all remaining gates are unproved. No further network fetches or payload reads were
performed for this documentation amendment.

**Original finding, before D32 (preserved):**

> **Metadata-only observation, 2026-10-23. BLOCKED against the selected importer:**
> all 252 stored block-scale tensors are **BF16**, not the F32 required by the
> [selected §3.1 import contract](maverick-numeric-source-recovery.md#31-import-and-workload).
> That clause rejects dtype conversion or a replacement/repair importer. D31's
> mathematical-map selection does not remove this contradiction or authorize an
> implementation, checkpoint run, or new source/numeric adoption.

**Subsequent limited reconciliation:** D32 explicitly amends that one storage
requirement. Accept the same pinned BF16 scales, promote their stored values
exactly to F32, then retain the original multiplication, Q8 and all other rules.
This does not retroactively make the original F32-storage assumption true. The
[self-contained proof](#6-self-contained-exact-promotion-proof) establishes exact
representation, not scale-value validity or recovery of earlier F32 precision.

The source is official **Qwen**, not an identified Maverick-author artifact.
This audit establishes neither author provenance nor equivalence to stock FP8
execution. The document-only name remains `M-N/Qwen3-4B-8prompt-candidate-1`, with
**D31 plus D32** identifying the amended source rule, not a new runtime capability.

## 1. Exact source and scope

Repository: `Qwen/Qwen3-4B-FP8`, revision
`96b30dc13593a244a5e59e84687309f53c375cfa`. Public canonical inputs:

- [config.json](https://huggingface.co/Qwen/Qwen3-4B-FP8/resolve/96b30dc13593a244a5e59e84687309f53c375cfa/config.json)
- [model.safetensors.index.json](https://huggingface.co/Qwen/Qwen3-4B-FP8/resolve/96b30dc13593a244a5e59e84687309f53c375cfa/model.safetensors.index.json)
- [shard 1](https://huggingface.co/Qwen/Qwen3-4B-FP8/resolve/96b30dc13593a244a5e59e84687309f53c375cfa/model-00001-of-00002.safetensors)
- [shard 2](https://huggingface.co/Qwen/Qwen3-4B-FP8/resolve/96b30dc13593a244a5e59e84687309f53c375cfa/model-00002-of-00002.safetensors)

The standard Hugging Face cache had no matching FP8 model directory; no alternate
checkpoint was substituted. Only config, index, eight-byte shard prefixes and
bounded JSON tensor headers were fetched. No tensor payload, tokenizer, model
library, native extension, source resolver, model download or inference was used.
The concurrent GKM cache/environment/model slot was not touched.

| Metadata | Bytes per pass | Observed SHA-256 |
| --- | ---: | --- |
| Config | 894 | `68fff9378aacf280608bbe1fbf594432a7e1eaeee12d738d53ed55024da1fc29` |
| Weight index | 55,847 | `be5d78efd19058e26d5363a322f148b16ade6dcd85a6f335139595df0e4eb64b` |
| Shard 1 JSON header, excluding prefix | 76,544 | `9203d0277e3986e90216c106ebdba821ea34acef7fcea49e88b5e5660def7cca` |
| Shard 2 JSON header, excluding prefix | 120 | `129e90eb84a96166217a4f5e538a1500b18bed4eec1c4c8fb963e09442bf81fb` |

Config/index hashes match the earlier [§2 source pins](maverick-numeric-source-recovery.md#2-recovered-qwen-artifacts-and-conflicts).
Header hashes identify the small metadata bodies read here, **not** verified
whole-shard LFS hashes or a cryptographic source/authorship proof. The publisher
and its pinned HTTPS inputs remain trusted.

## 2. Config and complete header inventory

Config confirms 36 layers, hidden size 2560, intermediate size 9728, 32 query
heads, 8 KV heads, head width 128, vocabulary 151936 and maximum positions 40960.
It declares `rms_norm_eps=1e-6`, RoPE theta 1000000, no RoPE scaling, no sliding
window, zero attention dropout, `attention_bias=false`, SiLU and tied embeddings.
The default dtype is `bfloat16`, Transformers version is `4.51.0`. Quantization
metadata is `quant_method=fp8`, `fmt=e4m3`, `weight_block_size=[128,128]`,
`activation_scheme=dynamic`. The source's default `use_cache=true` and dynamic
activation quantization are not executed by this audit; the selected one-shot
fixed-Q8 map is already a disclosed adaptation, not stock-FP8 parity.

The existing bounded parser
[`parse_header`](../../scripts/probe_gkm_checkpoint_bounds.py) was reused without
editing it or invoking its checkpoint/host admission, tensor decoder or main
program. Its float-only size table received an **in-memory metadata-width entry**
`F8_E4M3: 1`; this is header extent recognition, not an FP8 value decoder or a new
numeric profile. The literal projection dtype tag observed is `F8_E4M3`, consistent
with the selected E4M3 serialization family; no payload bit-pattern interpretation,
NaN exclusion or numerical decoder parity is established here.

Across both shards the parser found exactly **651** unique named tensors, with
ownership exactly equal to the index. Names, every shape and all non-scale dtypes
match the selected inventory. Against **D31's original F32-storage requirement**,
there were **zero shape mismatches** and **252 dtype mismatches**, all scales.
That historical mismatch motivated D32; the observations have not changed. The
index contains no extra bias tensors. Dtype totals are `F8_E4M3: 252`, `BF16: 399`,
**no F32** tensors.

All 36 layers have the following stored shapes. The table uses the selected
output-by-input weight and output-block-by-input-block scale interpretation:

| Role | `F8_E4M3` weight shape | **BF16** scale shape |
| --- | --- | --- |
| q | 4096 × 2560 | 32 × 20 |
| k, v, each | 1024 × 2560 | 8 × 20 |
| o | 2560 × 4096 | 20 × 32 |
| gate, up, each | 9728 × 2560 | 76 × 20 |
| down | 2560 × 9728 | 20 × 76 |

Block dimensions and declared matrix/scale-axis shapes fit the selected 128×128
contract. Stored precision failed the original F32 requirement and is now the
explicit D32 exception. Shape agreement alone does not prove how a numerical
loader indexes blocks; that interpretation still belongs to the selected
graph/import rules and future source validation. There are 252
`.weight_scale_inv` tensors containing 221,760 declared scalars.
Input/post-attention norms are BF16 vectors of length 2560; Q/K norm weights are
BF16 vectors of length 128 in every layer, not separate learned vectors per head.
The final norm is BF16 length 2560. This accounts for 145 norm tensors; the
remaining two BF16 non-scale tensors are the embedding and head.

## 3. Head/embedding tying: compatible metadata, unknown bytes

Both `model.embed_tokens.weight` and `lm_head.weight` are separately serialized
BF16 arrays of shape `[151936,2560]`, each with 388,956,160 elements and
777,912,320 declared payload bytes:

| Tensor | Shard | Offset relative to payload | Absolute byte start |
| --- | --- | --- | ---: |
| Embedding | 1 | `[0,777912320)` | 76,552 |
| Head | 2 | `[0,777912320)` | 128 |

They are **not a shared on-disk extent**. Config's `tie_word_embeddings=true`
does not establish equality of these two arrays. The selected requirement of
byte-for-byte equal BF16 copies remains unproved; no equality or inequality of
their values was inferred from names, shapes, lengths or tying metadata. A future
observed inequality would independently reject this candidate, not permit a loader
overwrite or silently choose one copy. No such comparison was performed here.

## 4. Bounded reads and structural checks

Each metadata pass read **133,421 network body bytes**. Two passes (the second
separating shape and dtype mismatch totals) consumed **266,842 bytes in total**,
below the task-wide **4,194,304-byte** cap. The prior pass was debited before the
second began. No raw header/source snapshot archive was retained.

Config/index were length-bounded HTTP 200 bodies. For each shard the only body
spans, on each pass, were:

| Shard | Required HTTP 206 byte ranges | Validated total file length |
| --- | --- | ---: |
| 1 | `0–7`, `8–76551` | 4,412,140,424 |
| 2 | `0–7`, `8–127` | 777,912,448 |

The HTTP adapter disabled automatic redirect-body draining and buffered HTTP
read-ahead. It required matching 206 status, exact `Content-Range`, exact bounded
`Content-Length`, identity encoding and no chunked transfer **before any shard
body read**. A missing/mismatched/oversized length or ignored range would close the
response without reading its body. Individual body reads were capped at 64 KiB,
the requested remainder and the remaining global budget. Header length was
separately capped at the existing parser's 1 MiB. Redirects were followed only from
the canonical pinned URLs to public HTTPS Hugging Face/CDN hosts; no signed CDN
query credentials were logged. All actual shard requests returned valid 206.

The parser checked duplicate JSON names, positive dimensions, byte-width/shape
agreement and contiguous, non-overlapping extents ending at the validated file
length. Shard 1 contains 650 tensors and 4,412,063,872 declared payload bytes;
shard 2 contains the head only, 777,912,320 bytes. Their payload-byte sum is
**5,189,976,192**, also the sum of validated file lengths minus prefixes/headers.
The index instead declares `total_size=5,191,286,912`, an **excess of 1,310,720
bytes**. Its cause is unestablished. This is an additional bookkeeping discrepancy,
not evidence of missing tensor names or payload corruption: the weight map and
all header extents passed their structural checks. Neither value is silently
repaired or treated as measured memory. These are serialization metadata, **not**
measured RAM/RSS, numerical values or whole-file hash verification.

## 5. Original contradiction, D32 reconciliation and remaining gates

1. **Original source-import contract was contradicted:** D31's selected §3.1
   required stored F32 scales; actual headers declare BF16 for every scale. §3.2
   also named stored F32 scales. D31 alone did not authorize their conversion.
   **Subsequently, D32 explicitly approved only these pinned BF16 scales with exact
   F32 promotion before the unchanged multiplication.** This is a recorded narrow
   amendment, not a silent repair, general dtype fallback or recovered author rule.
2. The index's overstated `total_size` still needs explanation during source
   validation; no index rewrite or payload-corruption conclusion is made here.
3. Actual tensor/shard bytes still need integrity, FP8 finite-code,
   finite/positive scale, source embedding/gamma range, and head/embedding equality
   checks. Headers and exact representability establish none of those values.
4. The [conditional totality](maverick-canonical-reply-totality.md#5-conditional-totality-of-the-complete-client-numeric-graph)
   and [honest no-wrap](maverick-honest-no-wrap-conditions.md#6-exact-remaining-inductive-certificate)
   source inequalities remain unproved. Exact supplied-number checker tests are
   not certificates about this checkpoint. Neither the historical contradiction
   nor D32 refutes or discharges the conditional mathematical theorems.
5. Author-model identity, full method security/storage/batch/state gates and all
   trained quality/publication qualification remain separate. D32 authorizes no
   implementation, payload acquisition, native/model import, inference or heavy
   execution. The original D31 quality criteria are unchanged.

## 6. Self-contained exact-promotion proof

**Claim (representation only):** every finite IEEE BF16 bit pattern has an exact
binary32 representation whose logical bits are the unsigned 16-bit BF16 word
`b` shifted left by 16. This is a mathematical bit-encoding statement, not a
production decoder, tensor scan or evidence about the pinned payload.

Write BF16's sign as `s`, its eight-bit exponent as `a`, and its seven-bit fraction
as `f`, with `0<=f<128`. Binary32 has the same eight-bit exponent and bias 127,
but 23 fraction bits. The word `b<<16` puts sign and exponent in the corresponding
binary32 positions and fraction `F=f*2^16` in those 23 fraction bits. Logical word
promotion does not change the selected little-endian source interpretation.

- **Normal, `1<=a<=254`:** BF16's exact value is
  `(-1)^s*(1+f/2^7)*2^(a-127)`. The constructed binary32 value is
  `(-1)^s*(1+F/2^23)*2^(a-127)`, identical since `F/2^23=f/2^7`.
- **Subnormal, `a=0,f>0`:** BF16's exact value is `(-1)^s*f*2^-133`.
  Binary32's is `(-1)^s*F*2^-149`, the same rational. Gradual underflow is retained;
  no flush-to-zero or arithmetic rounding is introduced.
- **Zero, `a=f=0`:** sign bit `s` is preserved, so both signed zeros are represented
  exactly. Either is still forbidden by the separate strictly-positive scale gate.

Thus a finite exponent never becomes a binary32 nonfinite exponent; no promotion
overflow, underflow loss, tie or rounding choice arises. Patterns with `a=255`
are nonfinite and outside the claim; source finiteness must reject them rather
than treating promotion as validation. The payload has not been read, so this
argument does **not** assert that any scale passes the positive-finite gate.

The multiplication remains `Weff=R32(v*S)` using that exactly represented stored
scale, followed by the unchanged exact `I(256*Weff)`. Multiplication can still
overflow for finite operands, and Q8 can still exceed i64; all original checks
remain. Filling the low 16 binary32 fraction bits with zeros **preserves BF16's
stored value**. It neither guesses nor recovers any lower bits of a hypothetical
F32 scale that may have existed before BF16 serialization, and proves no equality
to such a hypothetical model, stock FP8 execution or the authors' artifact.
