# Independent D22 oracle definitions

`src/masked_wire_reference/typed/linear_activation/tests.rs` imports `fixture.rs`
and `oracle.rs` as sibling test modules. Existing `sha2` and `serde_json`
dependencies suffice. No source-certified production facade is assumed or imported.

## Fixture and scope

Existing local immutable trained source: `Qwen/Qwen2.5-0.5B-Instruct` revision
`7ae557604adf67be50417f59c2c2f167def9a775`, `model.safetensors`, existing HF
content-addressed blob SHA-256
`fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe`.
Full blob hash was not recomputed during selection. Selected bytes independently
hashed; pin in `fixture.rs`. No model execution, download, build or installation.

No fitting complete original layer established. Qwen gate projection is bias-free
but full `[4864,896]` does not fit. This fixture is explicitly a
**checkpoint-derived submatrix**, original OI row `[0,1)`, columns `[0,256)`.
Source BF16 widens exactly; each finite value independently floor-encodes to Q12.
Semantic operation is a new bias-free submatrix dot then SiLU, not an original
layer, complete gated MLP or decoder. GELU tests are additional operator checks,
not Qwen source-activation claims. Raw source header has no matching gate bias;
source config declares `hidden_act="silu"`.

Fixture geometry: input `[1,256]`, right-hand IO weights `[256,1]`, output `[1,1]`.
Every fixture dot uses all 256 coordinates. Original 896-wide dot is NOT executed.
OI-to-IO layout mapping selects original `(output=0,input=k)` for IO `(k,0)`.
Single output means physical order is unchanged, not that layouts are interchangeable.

Public synthetic input domain: Q12 integers `[-4096,4096]` inclusive (real
`[-1,1]`), independent of generated sample extrema. Immutable selected weights
exhaustively floor-encode into `[-280,324]`. Endpoint-products × full inner 256
prove complete Q24 sum `[-339738624,339738624]`, strictly inside quarter ring.
One signed floor /4096 gives `[-82944,82944]`, inside unchanged D20 domain.
These are fixture-only bounds, not trained-checkpoint activation-range policy.

## Native integration and remaining certified bridge

The local native tests consume matrix owners through GapARS12 and unchanged D20
and compare final decoded output to `activate(rescale(dot(...)), frozen_table)`.
Native source labels and intervals remain developer declarations. The compiler's
source-certified facade stays closed pending aggregate overlap admission.

Required certified-facade rejection tests, pending enabled facade:

- Repository/revision/content/tensor substitution and changed coordinate selection.
- Original OI `[4864,896]` mislabeled IO, swapped axes, changed source shape or
  fixture coordinates; square-layout coincidences cannot prove correct binding.
- Claiming original inner 896 while supplying only 256 terms, or dropping final
  fixture coordinate. Pure helper already rejects incomplete declared inner length.
- Biased semantic op, including bias encoded zero: reject, never omit/relocate bias.
- Encoding/profile/table substitution (nearest-even versus floor, wrong activation).
- Source-bound certificate mismatch, out-of-domain private roots, oversized geometry.
- Actual seeded/mask parity, zero-draw preflight rejection, exact/one-byte-short
  reservation, original-owner continuity, consuming reuse, fault/cancel retirement,
  retained frames and matrix singleton. Native local tests now cover these native
  owners, but do not establish certified-facade admission or source authority;
  pure oracle tests alone do not establish either class of ownership controls.

Table bytes are existing frozen independent paper-equation fixtures with pinned
u64LE hashes, not production arithmetic or author-libm identity. No new evidence
tree, receipt harness, benchmark result or checkpoint-quality claim.
