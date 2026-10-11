# Private byte ingress with exact carry lifting: bounded rejection

**Result:** strongest concrete fused additive repair is a reversible recoding of
the existing stage-packed request, with **zero payload saving**. Separately
delegating the carry vector costs at least the entire apparent ingress saving
again, plus a fresh correction. This does not prove every cryptographic private
lifting construction impossible. No distributed protocol or protected primitive
is implemented or benchmarked by this probe.

## Exact construction and topology

For signed byte `x,r`, define `a=(x-r) mod256`, `b=[x<r]`. Then
`x=a+r-256b`, including both signed endpoints. The client already knows `x,r,b`;
there is no need for a comparison or Boolean-to-arithmetic primitive merely to
compute its own carry. Revealing `b` to the provider is nevertheless unsafe.

For compiler output residue width `h_j`, only `Wb mod2^(h_j-8)` is needed to
repair that output. Let `k=max_j h_j`. Sample fresh independent
`q_i` uniform in `Z/(2^(k-8))`, and send `c_i=(b_i-q_i) mod2^(k-8)`.
One byte `a_i` plus `k-8` bits `c_i` costs exactly `k` bits, not nine bits.
The provider computes the two public matrix products and combines them before
returning the original row-residue output. Original dequantization, activation
scales, weight scales and biases must stay unchanged.

**Fully fused Preparation:** set `R_i=r_i-256q_i` modulo the original wide ring,
and prepare `C_j=(WR)_j-s_j` using the original fresh full-domain output mask.
Provider returns `Wa-256Wc+C`; client adds `s`. Thus `C` and output payload have
exactly original widths. `R` is uniform; its unique signed-low-byte/quotient
decomposition yields independent uniform `r,q`. Existing fresh seed/ticket can
generate this one wide mask, rather than claiming an extra free correlation.
Each request consumes its own original material. No mask/ticket reuse is allowed.
This is ordinary wide-ring delegation written in two fields.

**Separate carry delegation:** if the original `Wr-s` is retained as specified
instead, Preparation must separately deliver `Wq-t` with fresh independent `t`.
The provider can fuse outputs, eliminating an additional online response, but
cannot discard the new offline correction. Client adds `s-256t`. Returning carry
output separately adds another quotient row payload. Both paths use offline
Preparation and one online provider; client participates normally. Asking
Preparation to compare/evaluate the private carry online changes topology.

All toy masks in the executable are deterministic test values. These functions
are not protected handles, PRGs, transport, or security-reviewed implementation.
A real integrated path must bind its distinct request layout and compiler numeric
contract to fresh issuance, own non-cloneable protected material, and burn it on
success, malformed input, replay, cancellation and failed attempts. No real
protected primitive is executed here, so no synthetic ownership wrapper is
presented as a security result.

## Bound and exhaustive falsification

For one odd-weight coordinate in this additive quotient layout, observed
`z=a-256c` is bijective with the two-field request, and equals `x-R mod2^k`.
Identical transcript distributions for consecutive private inputs require the
distribution of `R` to be invariant under addition by one, hence uniform over
the entire ring. This requires `k` bits of mask entropy and `k` bits for its
uncompressed uniform request. Correlating byte and quotient masks cannot evade
this argument. It is not a bound on arbitrary FSS, garbling, or specialized
whole-matrix encodings. Existing public-weight annihilator packing remains a
different, already investigated small optimization.

Executable checks:

- All **65,536** signed-byte `(x,r)` identities.
- **24,576** independent tiny matrix-product outputs for both separately prepared
  and fully fused layouts, with varying output widths, plus byte endpoints.
- Exact distributions for all signed-three-bit inputs/masks: full quotient
  masks give TV `0`; disclosed carry gives maximum TV `7/8`; binary masks in
  quotient ring eight give maximum TV `7/16`.
- Full provider transcript `(a,c,3R-s mod64)`, including its correction:
  eight input distributions, 4,096 mask combinations each, TV `0`.
- A single scalar pad on all carry-lookup entries exposes table differences:
  eight distinct patterns identify all eight possible byte masks, then recover
  the private input from its byte opening.
- Reusing a byte mask reveals `(x1-x0) mod256`. Omitting repair changes exact
  integers: `x=-128,r=127` yields uncorrected `128`, not `-128`.

Perfect toy hiding assumes an independent full-domain, trusted noncolluding
offline dealer and fresh one-use masks. A seeded implementation would additionally
need its stated computational PRG assumption. Neither toy correctness nor these
distribution checks prove a deployed distributed system secure.

## Geometry and complete-cost gate

Pinned body: Qwen2.5-0.5B-Instruct revision
`7ae557604adf67be50417f59c2c2f167def9a775`, fingerprint
`5d631be30158b3ea2a72cae355ce09a3b304bbfddde48b5758c339c346a34974`.
Compiler-derived saved geometry has 96 stages: **39 at 23 bits, 33 at 24 bits,
24 at 26 bits**; quotient stage widths are 15, 16 and 18, respectively.
Output residue means vary from 22.6571 to 25.0971 bits. The probe recovers exact
per-stage sum of row widths from saved arithmetic counts and validates the
packing equation; it does not invent an unavailable per-row histogram.
The separate serial checkpoint worker completed all **96 stages and 304,128
output rows**, independently checking weight hashes and compiler residues and
recording per-row histograms. It also derives exact `Wb` signed bounds
`[-sum negative magnitudes, sum positive coefficients]`, rather than assuming
byte or original-wide numeric bounds fit this new bit-vector delegation.
This exact signed-bit-vector ring cannot improve the quotient widths for this
numeric contract: if `Wb` fits signed width `d`, its positive sum is at most
`2^(d-1)-1` and negative magnitude at most `2^(d-1)`, so `S=sum|W|<2^d`.
The original row width `h=bit_length(127S)+1` is therefore at most `d+8`, giving
`d>=h-8`. The worker checks this bound independently for every output row.

The strongest cheap exact alternative avoids the unnecessarily signed range:
the public affine offset `Wb+sum negative magnitudes` maps every row to `[0,S]`,
where `S=sum|W|`. Its required unsigned width is `d=bit_length(S)`; `S=0` is a
constant zero requiring zero payload bits. Since `S<2^d`,
`127S<2^(d+7)`, hence `h=bit_length(127S)+1<=d+8`. Thus this shifted unsigned
layout also cannot beat `h-8`. The exhaustive two-coefficient test checks both
width bounds, both ranges and the zero special case across all 65,536 signed-byte
coefficient pairs. The completed checkpoint report includes shifted unsigned
widths too; neither exact layout beats the quotient-width bound.

Historical **46 rows per stage** (not the cold/repeat workload):

| Payload | Bytes |
| --- | ---: |
| Existing stage-packed ingress | 26,153,760 |
| New byte ingress | 8,337,408 |
| Carry ingress | 17,816,352 |
| Extra carry correction, if separate | 26,351,719 |
| Extra carry output, if separate | 26,351,719 |
| Fully fused net saving | **0** |

These are arithmetic projections, not measured HTTP bodies. The conserved disk
ledger records **493,575,247 bytes cold/repeat**, **90,443,518 ingress bytes** and
**124,905,091 public-bundle bytes**. Hypothetical uniform 24→8 saving is
60,295,678.67 bytes, not an actual packing projection. Earlier assistant handoff
estimates differed by 3,148 total bytes and 1,337 ingress bytes; they were not
user-supplied measured evidence. The canonical result uses only the conserved
ledger. This control already uses stage-packed inputs and row residues; the
46-row projection must not be presented as that measured pair.

Complete accounting:

- Fully fused layout retains original input/correction/output, public bundle,
  admission, seeds, preparation exchange and startup charges. Best-case total
  equals the matching control; framing and an additional dense provider pass can
  worsen it. No second party keys, OT, peer exchange or extra seed is needed.
- Separate layout retains every control charge and adds carry ingress, correction,
  and optionally output. Raw dealer-to-client `q,t` adds **44,168,071 bytes** in
  the 46-row projection if transmitted directly. A specified seeded distribution
  needs at least a fresh 16-byte carry seed per issuance plus authenticated
  context/framing instead; no zero-byte seeded correlation is assumed. Precise
  new transport/setup bytes remain unmeasured. Failure occurs even granting them
  an optimistic zero cost.
- Conventional fresh 128-bit garbling needs **133,398,528 bytes** for just one
  selected carry label per coordinate in the 46-row screen, versus 17,816,352
  bytes apparent byte-ingress saving. Delivering both input labels to the client
  adds 266,797,056 bytes; tables, private selection/OT if needed, output decoding,
  dealer headers and complete setup are additional. This rejects this specified
  label layout, not all garbling protocols. Labels cannot be reused across rows.
- Per-coordinate OT producing arithmetic shares still needs full quotient masks
  and private selection material; there is no installed zero-cost conversion.
  Uniform binary shares lifted directly into a wide ring fail the exhaustive test.
- Installed SharedRescale has two evaluators, dealer keys and framed openings.
  Its sign predicate of a signed shared difference requires a nine-bit domain,
  not an eight-bit difference; it does not directly implement this client-owned
  carry delegation. A second provider would be a changed topology. No benchmark
  body is borrowed from this mismatched contract.
- SecretReplicationPcf supports one scalar across a vector in fields 2/17/257,
  not independent carry lanes in exact wide rings. No tiny-field PCF, sparse mask,
  public subspace or uniform-share compression licenses this lifting.
- Local clear `Wb` needs a complete additional dense body MAC pass and public
  body weights at the client, conflicting with >90% remote body MAC placement.
  Client CPU, storage and lifetime peak RSS are unmeasured; no win is claimed.

Thus none of these concrete layouts fits the complete carry-repair budget.
No whole-decoder speed, full-wire saving, quantization change or model-quality
result is asserted.

## Reproduce / parent-owned serial work

Tiny functional checks only (no checkpoint or protected crypto allocation):

```sh
target/frontier-sdk-env/bin/python -m pytest -q -o pythonpath=. tests/test_narrow_ingress_probe.py
target/frontier-sdk-env/bin/python scripts/probe_narrow_ingress.py --output target/narrow-ingress-fresh.json
```

Replay checkpoint-derived histograms serially, never concurrently with
other memory/timing cohorts:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 target/frontier-sdk-env/bin/python \
  scripts/probe_narrow_ingress.py --serial-geometry --output target/narrow-ingress-geometry-fresh.json
```

The completed worker uses `fresh_workers()`, `frontier-probe.lock`, memory/swap
guard, 6 GiB post-reserve admission and a 300-second timeout. The default report
still uses only tiny tests and existing evidence under the requested ~512 MiB
scope. Actual probe/helper files, installed Python runtime,
native extension, all installed package Python sources, dependency versions and
control/evidence hashes are frozen and checked in parent and around the worker.
Reports expose no live masks, seeds, keys or prompt values. All new output paths
must be fresh. The committed evidence includes actual checkpoint-derived geometry;
timing/RSS and private full-stage execution remain unmeasured.

Evidence: `docs/evidence/research-narrow-ingress.json`.
