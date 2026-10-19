# F2 source gate set: independent clear contracts

**Non-production, standard-library reference only. No native gates, models,
wheels, cryptographic execution, transport or security qualification run here.**
Owned files are confined to this directory; original GELU descriptor, identity
Phi, fixtures, crates and SDK remain unchanged. This is D8's named source-derived
profile, not the missing evaluated FuseFSS descriptor manifest.

## Source and reproducible artifacts

`source-lock.json` binds the existing FuseFSS F/S/S-CPU lock, approved local
resolution §2 and PLLM-authored math/encoding files. Normative equations are
SIGMA Fig.6/App.F, Fig.8/§5.2 and Figs.11–12; FuseFSS Protocol3/typed Phi appear
on PDF pp.16–19. Author source is inert corroboration only, never imported,
executed or copied. Tests verify local lock bytes, not remote availability.

`reference.py` reuses **PLLM-authored** `shared_profile_oracle.py`'s independent
Decimal table generation and exact integer rsqrt, plus original scalar external
encoding. 80/96-digit table generations agree. New independent rational positive
Taylor enclosures certify every SiLU/E0/E1 floor: omitted exponential tail is
bounded by its next term divided by `1-x/(n+2)` once this ratio is below one.
Both resulting rational floor endpoints must agree. No host-libm rounding guess.
GELU is not regenerated here or changed; its original descriptor digest and
identity Phi are regression-checked.

Frozen `fixtures.json`: **579,821 bytes**, including newline; SHA-256
`ae1d8ba29b2563761ce1362688c4ff893812294d76066b0f398fb3388ace2a87`.
Contains table integer values, hashes and **275 cases**. Public table bytes are
exactly `b''.join(struct.pack('<Q', v) for v in fixture['tables'][name])`;
`sha256_u64le[name]` binds those bytes, not JSON layout or native packing.
No extra binary fixture is needed. Reciprocal tables include every slot, not
only reachable values. Norm tables include every one of 8,192 slots.

| Table | Entries | SHA-256 u64LE |
| --- | ---: | --- |
| silu | 1,024 | `ff696cf63579aa28571af2c56895c06b1ed2e64ce6fab942fcbbe9454744f63f` |
| exp0 | 256 | `e094819aa436afcf59d46bfd2937b91860045f5c905387d98e271731259a963b` |
| exp1 | 256 | `21a20b930ba1cd6dd57601ec54b4b6b85f206e553304b7c58da5dff2f45c5ed9` |
| reciprocal-k1 | 128 | `575fdfbd73f29854307559f4b3d50556936d75e781522d8468c40ab0d35ea492` |
| reciprocal-k2 / k3 | 256 | `b03af89917afce4cad1e1923bc09064e7d0b19e7f5b906f7694c3a421c3e8f3d` |
| reciprocal-k768 | 65,536 | `13b5289f3142cc67c8e84a5920c391122cf04f9a2fc488abd587124d764f8aa3` |
| reciprocal-k1024 | 131,072 | `99ba9f81c1600061434b027452386d75cb1fd9a4a56a0cb5d07ce1368aa1bb89` |
| rsqrt-k4 | 8,192 | `f7db0190afe1f99ef24a4207acc0cbdb1dbcf648e06a7f549e38c225234ec852` |
| rsqrt-k768 | 8,192 | `d2cc9e5c9c70cd8ff3f2323e4e5c6b643690eff21ccb1abb224f874b76c12f31` |

Other public widths use the same exact generator; k is part of table identity.
Supported reference bounds: softmax k=1..1024; norm k=1..4096. These are bounded
reference workloads, not source permission to silently limit a complete model.

```sh
python3 -B -m unittest discover -s paper/reproduction-contracts/fusefss/gate-set -p 'test_*.py' -v
python3 -B paper/reproduction-contracts/fusefss/gate-set/reference.py
python3 -B paper/reproduction-contracts/fusefss/gate-set/check_outputs.py OUTPUT.jsonl
```

Generator writes deterministic JSON to stdout, never overwrites frozen files.
Run this suite separately from original scalar/native suites: they also use a
module named `reference` and are intended for separate discovery processes.
Observed validation after partial/table checker additions: **24 unittest tests
passed**, zero failures/errors/skips, approximately 5 seconds locally. Native
replay count for this track: **zero**.

The later [native gate-primitives archive](../../../artifacts/fusefss-gates/local-v1/README.md)
separately records actual protected execution: 66 selected cases, including six
local external encodings, and all 18,560 table outputs across eight tables.
Its independent checkers pass. Those results do not replace this fixture-only
track or activate the full 275-case protected-gate claim.

The later [native composition archive](../../../artifacts/fusefss-composition/local-v1/README.md)
separately executes all 275 cases through protected owners, except for the six
explicitly local external encodings. The unchanged full checker verifies that
output. `generate-native-tables.py` reproduces the 1,536 frozen activation/exp
constants as Rust data, and separately records `source-tables-supplement.json` for
the original width-two norm case. Its 8,192 `rsqrt-k2` entries use the exact integer
formula below; no original fixture or lock is replaced. These finite compositions
retain caller range promises and native capacity limits, with complete models,
authenticated roles and SDK qualification outside the fixture/replay contract.

## Exact numeric contracts

All real activation/weight/learned affine values use Q12 signed ring50 codes.
**D8 external encoding is nearest/ties-even**, unlike SIGMA §2's external floor.
Exact `Fraction` inputs only: `-1/8192` encodes zero, `-3/8192` encodes -2.
Internal floors, table constants and public reciprocal mean are never changed
to nearest-even. Source tensor/binary float/checkpoint bridge remains unresolved.

### Activation and nonidentity Phi

For signed X: `Y=X//64`, `D=[Y>=0]`, `J=min(abs(Y),1023)`,
`T[J]=floor(64*J/(1+exp(J/64)))`, output `D*X-T[J] mod 2^50`.
Negative floor occurs **before abs**: X=-1 gives J=1 and output -31, not zero.
Source-effective signed38/Q12 input must be proven by lowering. Total signed50
extension in this clear interpreter is not a signed38 admission proof.
Early TR has A32/Q6 output from A38/Q12; index is A10/Q6; T/output A50/Q12.
Boolean D is B, not arithmetic sharing of a bit.

Existing GELU remains its frozen **511-source/512-padded affine descriptor,
Q0 slope, Q12 constant, identity Phi**, without an extra ARS12. SiLU's direct
maximal `(D,J)` partition would need 2,047 source intervals, so it is not admitted
as one current 512-source descriptor. Proposed composed lowering instead uses
protected floor/sign/abs/clip, public-chunk lookup, and typed nonidentity Phi
`B2A(D)*original_X-T[J]`; this is not an alteration to GELU.

`B2A(B(b))` yields A_n/Q0 b. For XOR shares b0,b1, arithmetic reconstruction is
`b0+b1-2*b0*b1 mod 2^n`, not arithmetic sum; `(1,1)` converts to zero.
Native conversion needs fresh correlated material/product and no clear bit open.
`PhiSelect(B(d),A(y),A(n))=n+B2A(d)*(y-n)`; arithmetic widths/scales must match.
The clear functions establish values/types only, not source-native conversion
material or a new serialized general Phi descriptor format.

### nExp, denominator and reciprocal

Public causal mask must contain at least one active entry. Maximum ignores
masked entries, which neither enter nExp nor denominator and output zero.
For active input X and maximum M:

```
C = min(M-X,65535)                    # subtraction must not wrap
E0[i] = floor(4096*exp(-i/4096))       # 256 entries
E1[i] = floor(4096*exp(-i/16))         # 256 entries
E = floor(E0[C%256]*E1[C//256]/4096)
Z = floor(sum(E)/64)
q = 6 + ceil(log2(k+1))               # exactly 6+k.bit_length()
R = floor(2^18/Z)
output = floor(E*R/4096)
```

E0/E1 are Q12; raw product <=2^24, E<=4096. An active maximum gives E=4096,
therefore **64<=Z<=64*k<2^q**, and R<=4096; final raw product <=2^24. Sum<=2^22
for k<=1024, so the displayed sufficient unwrapped domain is proved. Active
`M-X<2^49` is separately required before signed50 subtraction; tests reject its
overflow. q is based on public active count; arbitrary secret masks not covered.
Source workloads may instead bind q from public row width and use that larger
table; values agree at every admitted Z, but table identity must name that k.

Total reciprocal table slots are `R[0]=0` **sentinel only**, then
`R[j]=floor(2^18/j)` for 1<=j<2^q. Slots 1..63 and >64*k are mathematical
extensions, not reachable denominator inputs. Softmax **must not renormalize**:
`[0,-4096] -> [3013,1107]`, sum=4120; equal triple returns 1365 each.

### Norm exponent, mantissa, exact rsqrt and affine order

For LayerNorm, `mu=floor((4096//k)*sum(X)/4096)`, Z=X-mu. RMSNorm uses
Z=X with no centering. Both use the paper-defined `S=sum(Z^2)+1` (Q24 raw sum).
The integer +1 is a source guard, not Q12 epsilon or common-Qwen substitution.

```
e = S.bit_length()-1
m = floor(128*S/2^e)-128
index = 64*m+e
T[index] = isqrt((2^55*k)//((128+m)*2^e))
Q_i = floor(Z_i*T[index]/4096)
Y_i = floor(Q_i*gamma_i/4096)+beta_i
```

For unsigned nonzero A50 S, `0<=e<=49`, `0<=m<=127`, index<8192. No float log2
or approximate log2(k). The standalone index oracle admits 1<=S<2^50; the norm
interpreter conservatively requires **S<2^49** plus signed50 no-wrap at each
sum/mean-product/centering/product/affine-add stage. Thus its e<=48 is a
sufficient admitted subset, not a claim the source forbids unsigned e=49.
Full table slots e=50..63 remain exact formula-defined bytes, unreachable from
unsigned ring50 S. For every k<=4096 and every slot, `T<=sqrt(2^48*k)<=2^30`:
**no huge-slot ring50 modulo or clipping is needed**. For k768 max T=464943848.
Native same-ring index/output lookup must use **ring50**, not ring13: index has
13 bits but payload can exceed 13 bits. Likewise reciprocal input q bits does
not imply output q bits: unreachable j=1 has payload 262144; use ring50 there.

Frozen k768 counterexample: 768 copies of code4096 give mu3840, centered256,
S50331649, T65536, normalized4096. Exact division would change the function.
Four copies normalize to zero. Zero row has S=1, e=m=index=0 and output zero;
without +1 the index is undefined. Two floor products cannot be combined into
one double-width shift, and beta is Q12 **after** the second shift.
Author CPU snippets omit +1; this contract explicitly keeps paper behavior.
RMSNorm equation here does not settle RoPE or authorize Qwen epsilon/constants.

### Signed shifts and comparisons

TR(A_n,d) returns **A_(n-d)**, signed meaning floor(signed_n(x)/2^d);
canonical word equals unsigned `x>>d`. ARS returns **A_n**, same signed floor,
sign-extended to n bits. Their ring words generally differ on negatives.
`0<=d<n<=50`; no implicit GapARS shortcut. GapARS needs separately proven
quarter-ring interval, never a revealed predicate used as admission.
Comparisons return B: unsigned `<` includes sentinel thresholds 0 and 2^n;
signed `<` uses thresholds [-2^(n-1),2^(n-1)] and signed_n interpretation;
low view uses x mod 2^k with its own threshold/sentinel domain. Existing frozen
compiler supports MSB/Boolean compositions; those original tests remain intact.

## Protected composition and bounded ownership proposal (not production code)

Use existing universal public-boundary DCF only with its approved original
public table. A candidate public chunk j stores <=**256 u64 entries =2,048 bytes**.
Extend it to the whole ring with zero outside `[256*j,min(256*(j+1),L))`.
Singleton entry intervals plus up to two zero tails require <=258 source
intervals (<=259 if a compiler pads), safely below 512/513. A RingTable's actual
boundary and two-coefficient owners are **larger than 2 KiB** and must be charged;
2 KiB refers only to this table-value payload, not all allocations or a raised cap.
511-entry chunks fit interval bounds but **fail** this 2-KiB u64-value bound.

Each chunk/lane issues fresh independent mask shares and fresh universal DCF
owners for each party, opens only its masked input, evaluates every public
boundary on that **same** consuming input, and accumulates output shares.
Never reuse a consumed DCF/mask for another chunk, lane, input, gate or request.
Both parties evaluate **every chunk**, even where selected output is zero;
never reveal/select the private chunk, skip private work or open an index/value.
Disjoint public supports make the sum exactly the table value for valid index,
zero outside table; source lowering must prove index validity separately.

Full public table/source/plan/width/chunk order must bind native issuance and
frames. These proposal details are not frozen existing scalar descriptor bytes.
Existing `RingParty` has one universal DCF owner per party/chunk, with one base
evaluation plus each original interior boundary: <=258 DCF evals, <=12,900
path levels, <=514 public coefficient multiplications **per party/chunk/lane**
at ring50 (the generic two-coordinate implementation). These are calculated
source counts, not measured timings or per-call fresh keys for every boundary.

| Lookup | Chunks at 256 | Max DCF evals/party/lane |
| --- | ---: | ---: |
| SiLU | 4 | 1,032 |
| E0 or E1 | 1 | 258 |
| rsqrt | 32 | 8,256 |
| reciprocal k768 | 256 | 66,048 |
| reciprocal k1024 | 512 | 132,096 |

These conservative counts include zero-tail overhead and may overcount first/
last chunk intervals. nExp still needs a fresh shared E0*E1 product and floor
ARS12. Softmax still needs protected max, clipping, summation, denominator floor,
reciprocal and final fresh product/ARS12. Norm still needs all centered squares,
exponent/mantissa extraction, inverse lookup, two products/floors per coordinate.
Every conversion/AND/product/correction has its own fresh consuming material;
lookup parity alone supplies none of these gates.

Bounded reusable schedule: process public chunks in fixed order, <=4 lanes at
once (existing backend limit). Before randomness, reserve public descriptor/
fixture import, source RingTable, **both** party table copies/DCF owners, mask
shares, issuance temporaries, both simultaneous eval scratch, all input/output/
accumulator owners, buffered frames and final export owners. Keep old reservation
until consuming owner/frame retires; abort burns all issued material. Stream
bounded chunks only if native protocol/public schedule supports it; otherwise
preflight cumulative live all-chunk keys and fail early. Source LUT storage is
reusable public data; keys/material are not. Aggregate live envelope stays
<=64MiB; retained JSON parse arrays and caller copies count too. Full-model
cumulative material/work is separately charged even under streaming. Parent must
obtain actual `RingResources`/allocator counts, not treat these estimates as
admission or claim full model feasible. Large reciprocal work and frame/key
retention are explicit capacity blockers, not solved by bounded per-chunk LUTs.

## Finite replay protocol and evidence limits

Cases: 24 SiLU, 9 nExp, 17 reciprocal, 6 softmax, 6 norm, 147 norm-index,
42 signed shifts, 4 B2A, 12 comparisons, 2 Phi-select, 6 external encodings.
Native adapters must execute the gates, not echo expected results. Header:

```json
{"schema":"pllm.fusefss.gate-set.v1/native-output","fixture_sha256":"ae1d8ba29b2563761ce1362688c4ff893812294d76066b0f398fb3388ace2a87","execution":"protected-local-reconstruction","implementation":"actual native build identity"}
```

Then exactly one JSONL record per frozen case, in order, with exactly
`{"id":case.id,"gate":case.gate,"output":ACTUAL_RESULT}`. Output shape is
`expected`'s shape. Activation uses canonical ring50 output words; norm arrays
use signed unwrapped codes; shifts have explicit output bits/word; B outputs
are integer 0/1. This test-only reconstruction must not become online disclosure.
No footer/blank/extra lines, every line newline-terminated. Checker bounds output
to8MiB, lines256KiB, tokens18 characters; rejects duplicate keys/noninteger numbers,
Boolean-as-integer, wrong bindings/order/types/values. Synthetic checker acceptance
and mutations do not claim a native producer or prove cryptography/provenance.

Validation additionally exhausts all65,536 nExp inputs, SiLU floor boundaries and
neighbors, small-ring signed shifts/comparison domains, all admitted denominators
for five widths, every rsqrt slot at k1/4/768/4096, and all realizable exponent/
mantissa bins for unsigned50. It covers all local API branches for malformed
types/ranges/shapes and representative no-wrap failures, **not every conceivable
malformed native frame/key**. Source-width provenance/full graph ranges, native
conversion/predicate lifecycle/material, authenticated transport and frontend
remain required separately.

Native mapping priority: (1) exact public dense lookup bytes plus independent
bounded-domain replay; (2) typed signed floor/TR/ARS, comparisons, B2A/Phi;
(3) nExp products/floors and full reciprocal softmax; (4) normalization index and
full two-stage affine norm; (5) graph range provenance and complete admission.
RoPE/Qwen and original descriptor identity remain unresolved. No cap changes,
security claims, native parity, quality or performance claims follow here.

## Explicit partial native gate replay (v2 selected contract)

Original default v1 header and exact all-275 behavior above are unchanged.
Partial output is accepted **only** with explicit selection:

```sh
python3 -B paper/reproduction-contracts/fusefss/gate-set/check_outputs.py --gates shift,b2a,compare,phi-select,external OUTPUT.jsonl
```

That selection contains exactly **66 cases**: 42 shift, 4 B2A, 12 comparison,
2 Phi-select, 6 public external encoding. Filter `fixtures.json["cases"]` by
the selected labels, preserving original frozen order and original case IDs.
Do not renumber or group records independently. Header has exactly these fields:

```json
{"schema":"pllm.fusefss.gate-set.v2/selected-native-output","fixture_sha256":"ae1d8ba29b2563761ce1362688c4ff893812294d76066b0f398fb3388ace2a87","source_lock_sha256":"8080069e2edf58fb0b6b72a0a27bcbfa636db25018a0696681335a7dfe1ddcbd","execution":"protected-local-reconstruction","implementation":"actual native build identity","coverage":"selected-gates-only","selected_gates":["shift","b2a","compare","phi-select","external"],"selected_case_count":66}
```

Records retain exact v1 shape `{id,gate,output}` and all numeric/type rules.
`execution` describes protected operations' test-only reconstruction; `external`
is explicitly **public encoding**, not a claim of protected input conversion.
Actual protected owners must supply shift/B2A/comparison/Phi outputs. Fixture
`expected` values are checker oracles, never producer outputs to echo.

`--gates` accepts only comma-delimited exact known labels. No whitespace aliases,
empty selection/token, trailing comma, duplicate or unknown label is allowed.
Known labels, in canonical frozen first-occurrence order:
`silu,nexp,reciprocal,softmax,norm,norm-index,shift,b2a,compare,phi-select,external`.
CLI label order may vary; **header list must use that canonical order**, contain
exactly the CLI-selected names and bind exact selected count. Even selecting
all labels explicitly uses v2 `selected-gates-only`; it is not a complete-method
claim. Default mode rejects v2; selected mode rejects v1. Missing, duplicate,
reordered, extra, wrong-type/value cases and unsupported full-coverage header
claims are rejected. Neither mode silently falls back to the other.

## Independent bounded public-table lookup replay

This separate checker checks **table lookups**, never complete SiLU, nExp,
softmax or norm execution. It excludes reciprocal-k768/k1024 because those
tables exceed the parent's current `Table<=8192` domain. No table cap change.

```sh
python3 -B paper/reproduction-contracts/fusefss/gate-set/check_tables.py --fixture
python3 -B paper/reproduction-contracts/fusefss/gate-set/check_tables.py TABLE_OUTPUT.jsonl
```

`table-fixtures.json` is a new **1,707-byte** deterministic manifest, SHA-256
`161cb9531a63f5937f090494a5fc1a37a4d4ab8e34d22ed4f2a628f2081ae283`.
It binds original gate fixture/source lock/profile, exact table names/values'
u64LE hashes, ring50 input/output widths, ranges and total counts. Table bytes
remain in unchanged original `fixtures.json["tables"]`; no duplicate expected
arrays are needed in this manifest. Exact required order and lookup counts:

| Table | Input indices (inclusive) | Count |
| --- | --- | ---: |
| silu | 0..1023 | 1,024 |
| exp0 | 0..255 | 256 |
| exp1 | 0..255 | 256 |
| reciprocal-k1 | 0..127 | 128 |
| reciprocal-k2 | 0..255 | 256 |
| reciprocal-k3 | 0..255 | 256 |
| rsqrt-k4 | 0..8191 | 8,192 |
| rsqrt-k768 | 0..8191 | 8,192 |
| **Total** | **8 tables** | **18,560** |

Producer runs actual fresh protected lookup owners for each index in ascending
order, using <=256-entry public chunks, independent per-chunk/lane masks/DCF,
ring50 inputs/outputs, and local reconstruction **only for tests**. No copying
expected arrays into outputs. Public table inputs necessarily carry table values
for key issuance; those values are not evidence of evaluation. Table identity
must bind actual native plan/source; echoed hash fields alone cannot prove that.

Header has exactly these fields:

```json
{"schema":"pllm.fusefss.table-replay.v1/native-output","fixture_sha256":"161cb9531a63f5937f090494a5fc1a37a4d4ab8e34d22ed4f2a628f2081ae283","gate_fixture_sha256":"ae1d8ba29b2563761ce1362688c4ff893812294d76066b0f398fb3388ace2a87","source_lock_sha256":"8080069e2edf58fb0b6b72a0a27bcbfa636db25018a0696681335a7dfe1ddcbd","execution":"protected-local-reconstruction","implementation":"actual native build identity","coverage":"public-table-lookup-only","table_count":8,"case_count":18560}
```

Then exactly eight records, one per table in manifest order, no footer:

```json
{"table":"silu","sha256_u64le":"ff696cf63579aa28571af2c56895c06b1ed2e64ce6fab942fcbbe9454744f63f","outputs":["REPLACE WITH 1024 ACTUAL INTEGER OUTPUT WORDS IN ASCENDING INPUT ORDER"]}
```

The illustrative string above is **not valid producer data**: outputs must be an
array of exact integer words. Every manifest slot is checked, including reciprocal
zero sentinel and unreachable norm exponent slots. Those are total-table tests,
not admission of these indices to source softmax/norm gates. No out-of-table
lookup behavior or native frame/lifecycle security is established by this suite.

Read-only checker permits file or `-` stdin, bounds stream to2MiB, line256KiB,
integer token18 characters, and rejects duplicate JSON keys, floats/nonfinite,
bool-as-int, missing/extra/reordered/duplicate tables, wrong count/source/fixture/
table bindings and wrong outputs. It rebuilds and byte-checks both fixtures before
acceptance. Success summary says `public-table-lookup-only`; complete-gate/full
coverage claims are not accepted. Neither checker launches a native process or
verifies provenance. Tests use explicitly synthetic streams plus negative
mutations; **no native output has been run by this track**.
