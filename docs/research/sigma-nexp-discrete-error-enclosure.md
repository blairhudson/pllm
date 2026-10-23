# D29 local nExp: source-level absolute-error enclosure

This is a diagnostic theorem for the **selected discrete scalar function**, not
an error-acceptance target, a new numeric map, a native implementation result, or
quality/privacy/source-authentication evidence. The
[approved contract](../plans/sigma-nexp.md), tables, endpoints and scales are
unchanged. The existing 512 floor certificates and completed 65,536-class integer
enumeration suffice; no further Decimal or full-class sweep is needed here.

## Pinned inputs and completed observations

Source declarations are the SIGMA/EzPC revision
`aa0f2f860266fb79901405ef5e9965b8c9fe2e88` and paper SHA-256
`f911b715b0e0523fa4d77f21c28c6a0440c5be781e67b5b30419f20e505a6ebf`,
as recorded in the approved contract. Table identity is SHA-256 of the 256 unsigned
little-endian 64-bit payloads, not a caller label:

| Table | Formula | SHA-256/u64LE |
| --- | --- | --- |
| E0 | `floor(4096*exp(-j/4096))`, `j=0..255` | `e094819aa436afcf59d46bfd2937b91860045f5c905387d98e271731259a963b` |
| E1 | `floor(4096*exp(-j/16))`, `j=0..255` | `21a20b930ba1cd6dd57601ec54b4b6b85f206e553304b7c58da5dff2f45c5ed9` |

The completed integer run covered all 65,536 effective coordinates and 772
clip/TR-boundary graph points. Adjacent outputs had 70 upward steps (+1), 4,166
downward steps (-1), and 61,299 flat transitions. Every increase was a byte carry;
each table and each 256-coordinate row was nonincreasing. There were 34,292 zero
classes and 11 zero-to-positive resets: first zero 31233, last nonzero 34048,
permanent zero from 34049. Six product half-code ties used floor, not nearest-even. These are
observations about the pinned integer map, not an upstream binary execution or a
model-quality result. Public literal identity alone authenticates neither source
execution nor a native owner's material.

## Definitions and full-domain result

Let `S=4096`, with raw Q12 integer input `0 <= X < 2^38`. Define

```text
C = min(X,65535)
l = C mod 256; h = floor(C/256)
a = E0[l] = floor(S exp(-l/S))
b = E1[h] = floor(S exp(-h/16))
P = a b                       # raw Q24 integer
E = floor(P/S)                # returned Q12 code; real value E/S
```

Conditional on the selected floor and exact-arithmetic premises detailed below,
the selected mathematical function obeys both enclosures:

```text
0 <= exp(-C/S) - E/S < 1489/2097152
0 <= exp(-X/S) - E/S < 1489/2097152
```

The first reference is **clamped ideal exp**; the second is **unclipped ideal
exp**. Neither reference replaces the discrete function. These are absolute
scalar bounds, not observed maximum errors or full-model guarantees. At `X=0`,
both errors are exactly zero. The strict upper bound is not asserted sharp.
They compare the decoded integer input `X/S`; no external input-encoding,
learned-weight, normalization or generation error is included.

## Floor-cell proof, including carries and final rounding

Let `A=S exp(-l/S)` and `B=S exp(-h/16)`. Previously certified table floors give
`a <= A < a+1` and `b <= B < b+1`. At coordinate zero the corresponding factor
equals `S` exactly. Put `d0=[l!=0]` and `d1=[h!=0]`; thus use upper factors
`a+d0` and `b+d1`, with a singleton when its coordinate is zero.

Since `C=l+256h`, `AB/S=S exp(-C/S)`. Write `P=SE+r`, `0 <= r < S`. For `C>0`,

```text
L(C) = r/S
U(C) = ((a+d0)(b+d1) - S E)/S
L(C) <= S exp(-C/S) - E < U(C).
```

For `C=0`, `L=U=0` and the enclosure is closed/exact. Nonzero-coordinate upper
factors are positive, so the strict upper product bound remains valid even when
the high table's lower endpoint `b` is zero. This includes the product's final
floor; omitting `r/S` would miss up to `4095/4096` of a code.

Without enumeration, the floor cells already give the weaker universal
`S exp(-C/S)-E < 3` codes. The **completed exact integer enumeration**, independently
checked against Fraction arithmetic in
[`test_sigma_nexp_integer_classes.py`](../../tests/test_sigma_nexp_integer_classes.py),
finds `max U = 1489/512` codes. This is a maximum of the floor-cell **upper
enclosures**, not of the unknown actual error. Its witness is `C=272`, `(l,h)=(16,1)`:

```text
(a,b) = (4080,3847); P = 15695760; E = 3831; r = 3984
L = 249/256
U = (4081*3848 - 4096*3831)/4096 = 1489/512.
```

Dividing the code bound by `S` gives `1489/2097152` in real-value units. All
classes, including byte carries and zero resets, were covered by that prior
enumeration. This note's small witness tests do not claim to repeat that sweep.

## Saturation is a separate, discharged premise

For `X <= 65535`, `C=X`; the two ideal references coincide. For larger inputs,
the selected literals satisfy `E1[255]=0`, hence `E(65535)=0` and every saturated
input returns zero. Therefore

```text
0 < exp(-X/S) <= exp(-65535/S)        (X > 65535)
```

and the unclipped absolute error is no greater than the already bounded clamped
error. In particular, the terminal floor cells give the coarser tail bound
`S exp(-65535/S) < 3849/4096` codes, which is below `1489/512`.
**The zero-tail fact is necessary for this simple extension**: a generic positive
clamped output need not remain below an unclipped exponential tending to zero.
Here the required fact is established by the selected floor-certified literals;
it is not an extra assumed input/quality criterion.

## Actual integer/GapARS premises, not masked-word bounds

Under the approved root domain, `X40=X` and the unwrapped affine value
`X40-65536` lies in `[-65536,2^38-65536)`, inside signed40. DReLU includes zero.
For `X<65536`, `B=1` and modular Select16/affine reconstruction gives `C=X`;
otherwise `B=0` and `C=65535`. W, V and C use prescribed ring16 arithmetic,
not a signed16 no-wrap claim. Unsigned TR16.8 gives `h in [0,256)`; it is not
the public typed TR's signed Q4 output or a sign extension of a masked byte.

Certified payloads have `3848 <= a <= 4096` and `0 <= b <= 4096`. Consequently
`P in [0,2^24+1)` as a half-open true-integer interval, including `P=2^24` at
zero. This fits signed50 and lies inside the complete GapARS50 quarter-ring
`[-2^48,2^48)`. The composition has **one GapARS**, its final 50-bit/12-bit
shift, so exact signed floor equals `floor(P/4096)` without wrap or negative
rounding ambiguity. No statement here bounds a uniform mask, a common masked
word, or an opened D19 masked word by the secret's small range.

## Scientific limitations and evidence scope

- Selected output is not globally nonincreasing: codes `1279 -> 1280` produce
  `2995 -> 2996`. First zero is `31233`, but `31487 -> 31488` produces `0 -> 1`.
  These are inherited factorized-floor/byte-carry effects, not demonstrated
  native divergence. No repair, extra clipping or guard change is authorized.
- Ideal exp stays positive at every finite root. At a zero output, its relative
  scalar error is therefore exactly one. A small absolute enclosure does not
  establish a small relative error, a quality target or a model-quality result.
- Numerical floor premises are established for these public literals by the
  existing project rational Taylor/tail machinery in
  [`gate-set/reference.py`](../../paper/reproduction-contracts/fusefss/gate-set/reference.py).
  The research-only normal regression
  [`test_sigma_nexp_selected_floors.py`](../../tests/test_sigma_nexp_selected_floors.py)
  checks those same 512 floor cells against the original fixture; it has no Rust
  owner or generated-file dependency. It is a reusable mathematical test, not a
  native `Verified` token or source-authentication grant.
  Completed ordinary output is `target/sigma-exp-floor-python-tests-r1.log` in
  the retained floor-proof worktree. The integer enumeration command/source is
  [`sigma_nexp_integer_classes.py`](../../benchmarks/research/sigma_nexp_integer_classes.py),
  with ordinary output `target/sigma-nexp-integer-classes-r1.json` in the proof
  worktree. No receipt/archive/replay apparatus is added.
- **Remaining execution premise:** any native owner claiming this bound must
  implement exactly this map on its approved true-input domain and satisfy the
  original table-proof/ownership/range/scale/provenance gates. This work does not
  discharge that premise, execute Rust's issuance gate or authenticate upstream
  evaluated-source identity. No native counterexample is claimed without a run.

Small independent Fraction witnesses (no sweep or Decimal baseline):

```sh
python3 -m unittest discover -s tests -p test_sigma_nexp_error_enclosure.py -v
```
