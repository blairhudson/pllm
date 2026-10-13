# Context-conditioned lossless public-weight coding

**Decision: reject this frozen contextual model on both charged held-out gates.**
The larger-frame gate removes the tiny-zlib-reset ambiguity; tensor-only ablation
also beats causal conditioning. Complete-object behavior remains unmeasured, not
an integration candidate. This is H4 of the
[Low-tier plan](../plans/low-tier-all-link-hypotheses.md), called ranked hypothesis
5 in the worker assignment. No production files or execution placement changed.

## Frozen mechanism and identities

`scripts/probe_conditional_weight_codec.py` implements byte-renormalized rANS,
12-bit probability totals, positive frequency for every original byte. Context is
previous decoded signed-i8 magnitude divided by 16, capped at seven, resetting to
context zero at each row. Three public tensor classes have eight tables each.
There is no signed folding, residual, dictionary, delta or zlib transform.
Reverse encoding uses causal contexts computed from source bytes; forward decoding
uses only already decoded bytes and public row width/class.

Source is cached `Qwen/Qwen2.5-0.5B-Instruct`, revision
`7ae557604adf67be50417f59c2c2f167def9a775`. Fixed tensors: embedding, layer-zero Q
projection, layer-zero gate projection. Calibration uses 32-row ranges starting
at 0, 64, 128; held-out starts are 256, 384, 512. All calibration finishes before
held-out reads; model never adapts or refits. Frequency normalization uses positive
pseudocounts and deterministic largest-remainder ties by original byte id.

[Evidence](../evidence/research-conditional-weight-codec.json) contains every
source BF16, exact i8, original little-endian f32 scale and decoded digest, tensor
shape/range, public header hash, model hash, script/test, installed quantizer,
native extension and interpreter hashes. Bounded NumPy quantization is checked
against the installed quantizer independently. No complete checkpoint scan ran:
the published expected checkpoint digest and resolved content-addressed cache
path are recorded, **not independently reverified whole-source integrity**.
Sample digests and frozen header make the limited replay auditable. Full original
object identities remain a serial gate; sample hashes are not whole-object hashes.

## Original 32-row byte screen (unchanged)

18 BF16 slices read **1,032,192 bytes** total. Nine held-out blocks contain
258,048 original i8 bytes and 1,152 unchanged scale bytes. Comparison uses zlib
level one on identical 32-row samples with eight-byte framing, **not** the
production 1 MiB-frame control.

| Held-out tensor class | zlib1 frames | Conditional frames |
| --- | ---: | ---: |
| Embedding | 79,140 | 78,677 |
| Layer-zero Q | 76,814 | 76,247 |
| Layer-zero gate | 78,780 | 78,331 |
| Total | 234,734 | 233,255 |

All nine blocks decode byte-for-byte, with matching trusted decoded hashes.
Conditional framing is 83 bytes per block, including original decoded/model
digests, dimensions, class and encoded length; rANS also carries four state bytes.
Tables cost **12,288 bytes**, explicit context/tensor descriptor **212 bytes**.
Including unchanged scales, control costs **235,886 bytes**, candidate
**246,907 bytes**: **11,021-byte regression, 4.67%**. Saving before shared tables
is only 1,479 bytes. This original cohort/result remains separately recorded in
the evidence's top-level fields and `held_out` list.

No sample extrapolation or all-link
percentage is promoted. Q/gate weights count only where matching selective
placement actually delivers them; embedding alone is the always-delivered public
boundary object. Historical all-link sensitivity multiplies a verified public
delivery reduction by **only its matching delivery fraction**, never whole
traffic. Uniform privately masked tensors get **zero credited saving**. Warm
artifact hits get no cold-delivery saving. Body MAC placement remains unchanged.

## Fresh contiguous, production-sized frame gate

Parent predeclared embedding and gate rows **1024–2047**, and Q rows **544–895**.
These ranges are disjoint from every original calibration/held-out row. Header,
all range shapes, disjointness and total source-read budget are checked before
new range payload reads. Original contextual model is frozen first and explicitly
required to retain SHA256
`52d375e5900191d22892d2d773a0acde17e7eba7175630a14066682e2f6b65a0`.
No tuning follows new held-out reads.

A separately frozen tensor-only ablation uses **only original calibration**,
identical frequency normalization and exact byte rANS. Its eight context tables
per tensor are identical. Wire format/decoder remain unchanged, so all repeated
tables are still charged: **12,288 bytes**, not hypothetical deduplicated tables.
Its descriptor costs 198 bytes versus contextual model's unchanged 212 bytes.
Both models freeze before new range reads.

Each contiguous slice is one bounded level-one zlib frame, with the production
eight-byte big-endian raw/compressed-length header. Embedding and gate contain
917,504 raw i8 bytes each; Q contains 315,392 bytes. These fit production's
**1 MiB byte** limit. That limit is not a 1024-row cap. A streaming zlib ledger
matches independent `zlib.compress` and installed `encode_bundle_frames` bytes,
including tested splitting at exactly 1 MiB. rANS keeps its existing 32-row
blocks: 32 embedding, 11 Q, 32 gate blocks, all original headers/state bytes
included. Whole serialized bundles and cross-object metadata remain outside this
matched raw-weight slice gate.

| Fresh held-out slice | zlib1 frame bytes | Contextual frames | Tensor-only frames |
| --- | ---: | ---: | ---: |
| Embedding | 842,988 | 837,995 | 836,356 |
| Q | 277,705 | 275,509 | 275,180 |
| Gate | 840,438 | 835,192 | 833,719 |

Common exact scales cost 9,600 bytes. Charging shared full tables once and actual
descriptor, all three slices cost:

| Recipe | Charged bytes | Saving vs matched zlib1 |
| --- | ---: | ---: |
| zlib1 | 1,970,731 | — |
| Frozen contextual model | 1,970,796 | **−65 bytes (−0.00330%)** |
| Frozen tensor-only ablation | 1,967,341 | **3,390 bytes (0.1720%)** |

Contextual frames save 12,435 bytes before model/descriptor, but still lose after
the 12,500-byte shared charge. Tensor-only frames beat contextual frames by
3,441 bytes; causal neighbor conditioning provides **no byte advantage here**.
The tiny ablation win covers this exact three-object selection only, including
Q/gate delivery; it is not an always-delivered public-bundle saving.

For independent one-object deliveries, current immutable wire model still pays
all 12,288 table bytes, including 8,192 unused class bytes, and full descriptor.
Contextual standalone embedding/Q/gate regress by **7,507 / 10,304 / 7,254 bytes**;
tensor-only regress by **5,854 / 9,961 / 5,767 bytes**. No unimplemented class
pruning or table deduplication saving is credited. Complete-object amortization
could change results but cannot be inferred from these contiguous slices.

New BF16 payload reads total **4,300,800 bytes**. Including original calibration
and original held-out cohort, each complete lightweight replay reads
**5,332,992 bytes**, below 8 MiB. All 75 new blocks and three concatenated slices
decode exactly for both frozen models, with original i8/scale/source-slice and
decoded hashes retained in `larger_frame_follow_up`. Whole checkpoint hash stays
unverified. Compression is streamed from bounded row blocks without holding a
917 KiB float or whole source/decoded slice buffer; zlib state, hashes and bounded
block buffers remain additional owners. These are size/function observations,
not measured CPU/RSS or Low qualification.

## Bounds and Low-tier status

Decoder rejects excessive dimensions/classes, model length/frequencies, model or
trusted decoded hash mismatch, wrong metadata, truncation, overlong declared
payload, trailing bytes and invalid initial/final rANS states. Output size is
validated before allocation and capped at 65,536 bytes. Renormalization cannot
read beyond bounded frame. Model digest checks against trusted caller tables;
decoded digest must match trusted caller identity, not merely an in-frame value.
This research frame does not add a new authentication protocol.

Packed model is 12,288 bytes; projected u16 cumulative tables 12,336 bytes; maximum
frame 131,159 bytes; decoded block at most 65,536 bytes; arithmetic state four
bytes. Encoder/corruption checks and Python object/integer overhead are additional
owners. These figures are **not measured RSS**. Existing client floor, SDK/native
state, relay, page buffers, loaded code, model/table objects, frame and output
owners must all fit the same whole-client 268,435,456-byte maximum. Whole-client
RSS, CPU, latency, wire traffic and Low qualification remain unknown; no timing
or RSS cohort ran during overlapping workers. Filesystem cache has separate owner.

## Replay and parent-only next gate

```sh
target/frontier-sdk-env/bin/python -m pytest -o pythonpath=. tests/test_conditional_weight_codec_probe.py -q
target/frontier-sdk-env/bin/ruff check scripts/probe_conditional_weight_codec.py tests/test_conditional_weight_codec_probe.py
target/frontier-sdk-env/bin/python scripts/probe_conditional_weight_codec.py --output target/conditional-weight-screen-new.json
```

Fifteen lightweight checks cover deterministic bytes, all original symbols, random
maximum-size blocks, row resets/classes, malformed tables, trusted hashes,
truncated renormalization, overlong framing with matching declared size, corrupt
state/payload and oversized declarations before allocation. New checks independently
compare streaming control with actual production frame bytes, validate fresh-range
shape/budget admission, verify published original/new aggregates, and exercise a
table-amortization regression whose sign changes without changing codec payloads.

Optional **parent only**, not run here:

```sh
target/frontier-sdk-env/bin/python scripts/probe_conditional_weight_codec.py --serial-full --output target/conditional-weight-full-new.json
```

This opt-in worker uses `frontier_probe_support.fresh_workers`, exclusive frontier
lock, fresh process, 512 MiB post-reserve admission, swap guard and four-hour
timeout. It fully verifies the published checkpoint hash, freezes identical
calibration, streams complete three source objects, independently checks exact
integer/scales, hashes reconstructed objects and compares candidate frames with
the existing zlib1 1 MiB-byte control. It retains bounded row/block memory but
necessarily scans the cached checkpoint; coordinate before running. No downloads.
It is a byte/identity gate, not a serving timing or Low qualification cohort.
Only if separately authorized charged complete-object delivery wins should parent
price native bounded decoder ownership and fresh whole-client SDK RSS/CPU,
unchanged exact logits/KV and matching all-link traffic. Current evidence earns
no integration promotion.
