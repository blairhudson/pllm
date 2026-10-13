# Cyclic masked-output codec: bounded reference, not a protocol

**Feasible arithmetic and serialization:** a sender knowing only canonical
`y=(h+x) mod q` can send a public color, and a receiver knowing `h` recovers
bounded `x` exactly. This permits narrow replies even when a separate coded
delegation argument requires a prime greater than `2^31`. No SDK selection,
malicious-protocol proof, full checkpoint parity, latency or peak-memory claim
is made. The codec does not certify the original value's range.

## Geometry and proof

Let a public integer interval be `[a,b]`, with `M=b-a+1 <= q`, `q>=2`.
For symmetric bounds use `a=-B,b=B`. Define

```
N = floor(q/M)
L = floor(q/N)
r = q mod N
start(j) = j*L + min(j,r)                 (0 <= j < N)
length(j) = L + [j < r]
K = L + [r != 0] = ceil(q/N)
color(y) = y - start(block(y))
```

First `r` blocks have length `L+1`, remaining blocks length `L`. They partition
the cycle exactly: `N*L+r=q`. Since `N<=q/M`, `L>=M`.
Color `c<L` occurs once in every block; successive cyclic gaps equal the
preceding block's length, including the wrap, hence are at least `M`.
If `c=L`, only long blocks contain it. Gaps inside that prefix are `L+1`;
the wrap includes all skipped short blocks and is also at least `M`.
A color occurring once has no competing point. An interval of `M` consecutive
residues has endpoint distance `M-1`, so contains at most one point of any color.
The actual bounded `x` supplies existence, proving exact recovery.

This alphabet size is optimal for deterministic zero-error encoding for all
masks and all values in this interval. Any same-color cyclic gap smaller than
`M` allows two candidates inside some receiver interval. Consequently each
color class has at most `floor(q/M)=N` elements (singleton included), and
covering `q` residues needs at least `ceil(q/N)=K` colors. The construction
meets that bound. Fixed-width binary serialization still rounds up to whole bits.

For asymmetric bounds, set `anchor=(h+a) mod q`; consider color positions in
`block(anchor)` and its two cyclic neighbors. Keep a candidate iff
`t=(candidate-anchor) mod q < M`, returning `a+t`.
An interval of `M` positions cannot cross more than one next block boundary
because every block has length at least `M`; three blocks therefore suffice.
The reference deduplicates neighbors for `N=1,2`. For symmetric bounds one
may instead use neighbors of `block(h)` and centered differences. Asymmetric
intervals need the shifted anchor; naively retaining only neighbors of `h`
fails for intervals far from zero.

Primality is irrelevant to this proof: even and composite moduli work too.
`B=0` gives `N=q,K=1` and zero payload bits. `N=1` gives `K=q`: no compression.
For odd `q`, maximal symmetric `B=(q-1)/2` covers the entire cycle; for even
`q`, maximal symmetric `B=q/2-1` covers `q-1` residues. A general interval
of size exactly `q` still reconstructs uniquely. Intervals larger than `q`
alias and are rejected. Even-modulus centered tie conventions do not affect
the interval-based reference; the symmetric admissible range excludes the tie.

## Counterexamples and security boundary

For `q=17,B=1,h=0`, block starts are `0,4,8,11,14`. An out-of-domain
`x=4` sends color `0`, identical to bounded `x=0`, and decodes as `0`.
Conversely out-of-domain `x=3` sends color `3` and produces no bounded candidate.
Thus neither successful decoding nor rejecting some malformed outputs proves
that the sender's original value was bounded. If `M>q`, even full field values
cannot distinguish integers separated by `q`.

Encoding is deterministic public post-processing of `y` and requires no `h`.
If the original masked view already has a privacy proof, honest encoding adds
no information to that view. This statement is **not** a malicious interactive
protocol proof. A forged color's decodability depends on the secret mask.
An accept/reject signal, distinct error, retry, early termination, timing, or
whether later verification runs can become a chosen-response mask oracle.
Concretely, canonical color `3` at `q=17,B=1` rejects with `h=0` but succeeds
with `h=3`, decoding `x=0`; this secret-dependent behavior has a dedicated test.
Python reference is not constant-time and exposes errors; it is not a safe
network implementation. Public framing/color checks occur before decoding,
but this does not resolve secret-dependent decode behavior.

For post-response verification, receive and freeze the entire canonical frame
before sampling fresh secret challenges. Decode the whole vector locally,
reconstruct the claimed field reply as `(h+decoded_x) mod q`, and verify this
reconstructed reply, not a presumed original sender value. Invalid positions
need a reviewed fixed-work substitute path, uniform externally visible failure,
consumed one-use state on every finish path, and no retries on the same mask.
Both secret checks should run before a combined decision. These are requirements
for review, **not implemented or proved here**. Reconstructed forged replies
depend on `h`, so existing soundness arguments for a sender-fixed raw field
vector do not automatically apply. Compression changes the adversarial response
interface and needs a new soundness/privacy treatment, even if challenges are
chosen after receiving colors.

## Actual serializer and resource contract

`scripts/probe_cyclic_mask_codec.py` implements numeric encoding/decoding,
an independent bounded-domain oracle, LSB-first heterogeneous bit packing,
and canonical serialization/deserialization. Each color consumes
`ceil(log2 K)=(K-1).bit_length()` bits; singleton colors consume zero.
Layout order is element order; repeated row widths are explicitly supplied.
Final byte's unused high bits must be zero. Invalid colors, trailing bytes,
truncation, wrong magic/count/length and nonzero padding reject.

Frame is a 12-byte little-endian `<4sII>` header: magic `CMC1`, element count,
payload byte count, followed by `ceil(sum(widths)/8)` payload bytes. It is not
authenticated and does not bind layout/session/model. Public layout must be
agreed and authenticated externally; two same-sized wrong layouts can parse.
The actual layout storage serializer uses `<Qqq>` per element: 24 bytes for
`q,a,b`, round-tripped independently. Evidence charges this uncompressed
layout separately, not as free metadata. Deduplicated row tables could reduce
storage but are not implemented or priced. Retained secret masks cost
`ceil(log2(q)/8)` fixed-storage bytes per element in a native representation,
apart from ownership/authentication machinery and Python object overhead.
Reply framing is not a replacement for existing transport framing.

Admission: `2<=q<=2^64-1`, signed64 interval endpoints, `M<=q`,
1..65,536 elements per frame. Thus payload is at most 524,288 bytes plus
12-byte header; layout at most 1,572,864 bytes. Numeric block lookup and
candidate enumeration allocate no `N`-sized table. Python arbitrary precision
prevents integer overflow; native ports need checked/widened arithmetic for
`b-a+1`, `h+a`, ceiling divisions, bit reservoirs (up to 71 bits before flush),
frame sizes and signed conversions. Python heap size and peak memory are not
inferred from serialized payload bounds. Oracle limits interval scans to 4096.

## Source-derived pricing, distinct from checkpoint values

Existing `crates/pllm-core/src/offset_transport.rs::row_residue_bits` uses
`B=max_input*sum(abs(i8 weight))`, allowing public `max_input=1..127`;
width is `max(1, B.bit_length()+1)`, with widths capped at 32. Native residue
packing is already LSB-first, terminal-padding checked, and covers corrections
as well as outputs. This probe uses a separate conservative full signed-i8
bound `B=n*128*128`, covering inputs including `-128`; synthetic row-bound
prices use public absolute input bound 128 and do not pretend that the current
API accepts that parameter. Real checkpoint row norms were not loaded.

Pinned prior evidence identifies Qwen2.5-0.5B-Instruct revision
`7ae557604adf67be50417f59c2c2f167def9a775`, config SHA-256
`18e18afcaccafade98daf13a54092927904649e1dd4eba8299ab717d5d94ff45`.
Dimensions and stage multiplicities come from that evidence, not a new
checkpoint execution. Source-derived shapes each occur 24 times:

| Output × input | Public B | Cyclic bits, q=4,294,967,291 | Existing residue bits | Cyclic payload / new frame | Wide u32 reply |
| --- | ---: | ---: | ---: | ---: | ---: |
| 896 × 896 | 14,680,064 | 25 | 25 | 2,800 / 2,812 B | 3,584 B |
| 1152 × 896 | 14,680,064 | 25 | 25 | 3,600 / 3,612 B | 4,608 B |
| 896 × 4864 | 79,691,776 | 28 | 28 | 3,136 / 3,148 B | 3,584 B |
| 9728 × 896 | 14,680,064 | 25 | 25 | 30,400 / 30,412 B | 38,912 B |

Raw reply payload savings versus fixed u32 are 21.875% for input width 896
and 12.5% for width 4864. In these geometries codec **matches** existing
power-of-two residue payloads, not an extra improvement over them. Frame adds
12 bytes; real transport/layout framing is additional. The prime `2,147,483,659`
also preserves these widths. Prime `2^61-1` still preserves these bounded
reply widths, while requests remain 8-byte fixed-storage fields (or 61 bits
if separately packed). Input expansion, P/Q storage, setup, corrections and
verification traffic are not eliminated by this output codec. When `N` is
small or bounds lie near bit thresholds, `K` can increase width above
`ceil(log2 M)`; matching row-residue width is not universal. For example prime
`q=17,B=3` gives `M=7,N=2,K=9`: cyclic codec requires four bits while an
existing power-of-two ring residue needs three. The alphabet lower bound above
shows this field-side-information penalty is unavoidable under the stated
deterministic zero-error contract, not an inefficient partition.

The full signed-i8 geometry at admitted maximum input width 16,384 has
`B=268,435,456`, so `2B+1=536,870,913` fits all three sampled primes and
BabyBear. This proves raw signed-i8 dot products fit centered field, not
decoder equivalence. Changing a power-of-two ring to a field needs proof for
all quantization, bias, scaling, nonlinear, attention, KV and verification
boundaries. The Maverick reference's distance/LPN and storage gates remain.

Baseline all-link control counts 493,575,247 pair application-body bytes,
including 139,123,150 output bytes. Even deleting all output bodies yields only
an optimistic 28.1868% ceiling, not an achievable codec saving. These outputs
already use row residues, include framing and lack a replayed per-row field
layout here. Evidence therefore leaves new full-pair bytes unknown. Claiming
an independent 28% gain would double-count preventing a wide-field regression.

## Replay and evidence

```
.venv/bin/python -m pytest -q tests/test_cyclic_mask_codec_probe.py
python3 scripts/probe_cyclic_mask_codec.py --output docs/evidence/research-cyclic-mask-codec.json
```

Tests cover every tiny width across even/odd/composite moduli, shifted and
asymmetric ranges, singleton/maximal bounds, same-color cyclic spacing,
independent oracle rejection, large-prime boundaries, uint64 limits, actual
heterogeneous and zero-width serializer round trips, canonical framing/padding,
layout storage and malformed responses. Evidence includes source hashes,
rechecks the parent's 23,096 bounded cases, separates public synthetic rows
from source-derived shape bounds, and records actual payload/frame/storage
counts. Four tiny public rows also run actual signed-i8 dot products with
public alternating `[-128,127]` inputs and serialize/decode their replies;
frame bytes/hashes are recorded. One-row-per-stage aggregate prices are not
full-response counts. No benchmarks, downloads, wheels, or private adaptive
widths are used.
