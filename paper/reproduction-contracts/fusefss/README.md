# FuseFSS: bounded descriptor and scalar compiler reference

**Non-production specification only.** This directory independently implements
canonical typed descriptors, Protocol-3 compilation and clear scalar interpreters
for the first GELU slice of D8's `fusefss.source-derived-ring50-q12.v1`.
It contains no FSS backend, shares, cryptographic randomness, native integration,
network codec, model execution or SDK selection. The clear compiled record includes
the mask and private thresholds; it is unsuitable for disclosure to an evaluator.

[D8/D9](../../methods/DECISIONS.md#d8-approved-fusefss-source-derived-descriptor-profile)
close the specified profile/feedback **approval** gates. Neither decision makes
this scalar slice a complete FuseFSS implementation or an evaluated-build replica.

## Sources and frozen identities

[`source-lock.json`](source-lock.json) binds the reviewed F/S PDF bytes, immutable
SIGMA CPU source revision/file hash and exact D8/D9 decision-section snapshots.
The selected evidence is FuseFSS Protocol 3 (PDF p.17), Lemma 4.3 and Lemmas
D.1/D.2/E.1 (p.16), typed expressions/Phi (pp.18–19), and SIGMA Fig.6/CPU
optimizations (p.8). The local governing interpretation is
[source resolution §§2.1, 5.1, 7.1](../shared-profile-resolution.md#51-approved-f-n1-descriptor-profile).
Author code was fetched/read/hashed as inert evidence only; none is imported,
executed, linked or vendored here.

- LUT: 256 entries, SHA-256 over consecutive unsigned **u64 little-endian** entries:
  `b6a45a82dc8f660724943c1eb76654704c2e49563ffc0b1a0bc5388b3425a01b`.
- Canonical GELU descriptor: **32,900 bytes**. Identity is
  `SHA256(b"FuseFSS-reference-descriptor-v1\0" || canonical_JSON_bytes)`:
  `5f1e888e7bdabe3cbad4b6bbf1abd6749c8dff0918c6f243916db071398e79d3`.
- Canonical bytes are reproducibly generated from the frozen LUT, source lock
  and descriptor rules below; the complete byte digest is asserted by tests.
  [`vectors.json`](vectors.json) also freezes clear compiled-record hashes.
  Those compiler hashes are regression identities, **not independent correctness
  or authentication evidence**.

## Exact D8 GELU arithmetic

Input/output arithmetic type is signed **A50/Q12**. For its signed code X:

```text
Y = floor(X / 64)                       # signed floor BEFORE absolute value
D = [Y >= 0]
J = min(abs(Y), 255)
T[j] = floor(2048 * (j/64) * erfc((j/64)/sqrt(2)))
output = D*X - T[J]                     # represented modulo 2^50
```

External input **and weight** encoding is nearest/ties-even of `real*4096`,
followed by representability checking and canonical ring representation. The
reference requires an explicit `Fraction`: an external binary32/binary64/decimal
tensor bridge is not selected implicitly. Internal table entries and signed
shifts use floor. In particular external `-1/8192` encodes as zero, while the
internal GELU input code `X=-1` returns **-31**, not zero.

The source's effective post-linear input is signed38/Q12, with a sign-extension
provenance requirement `-2^37 <= X < 2^37`. Fig.6 then has early TR output
A32/Q6, clipped index A8/Q6, a Boolean D, and LUT/output A50/Q12. These widths
and the required range are bound in descriptor metadata. The affine lowering
absorbs the early-floor selector into its intervals; it does not insert a new
truncation or claim the same source instruction costs.

Protocol 3 requires a total ring partition. The descriptor therefore defines the
displayed map over all A50; full-ring tests outside signed38 verify that mathematical
extension only. They do **not** admit those inputs to a narrowed source CPU kernel.
For example X=`2^37+64` maps to X under the total map but to zero if its selector
is incorrectly narrowed to signed38. Future graph lowering must establish the
metadata precondition, as well as all upstream product/accumulator range rules.

### Canonical 511-interval descriptor

Let N=`2^50`. Boundaries, in canonical unsigned order, are exactly:

```text
(0, 64, 128, ..., 16320,
 N/2,
 N-16256, N-16192, ..., N-64,
 N)
```

Every interval is **[lower, upper)**; a boundary belongs to the interval starting
there. The first 256 intervals cover nonnegative signed values. The remaining
255 cover negative values. The negative clipped tail includes Y<=-255, so its
last boundary is `N-16256`, not `N-16320`. Intervals are maximal for the pair
`(D,J)`; different indices remain separate even when LUT entries happen to agree.

For each interval, payload is **(constant, linear)** = `(-T[J] mod N, D)`.
Both payload slots are A50; their numerical scales are **Q12 and Q0**, respectively.
The padded degree is one for every piece. Horner evaluates `linear*x+constant`
in the original unmasked ring x. Its product is already Q12. **No ARS12 follows.**
The post-circuit **Phi is identity A50/Q12 → A50/Q12**, with no Boolean outputs,
B2A or A2B. Non-identity Phi is rejected rather than assigned guessed semantics.

## Typed Protocol-3 reference

`reference.py` validates a complete canonical partition, fixed two-slot affine
payloads, ring/fractional widths and a uniform Boolean output arity. Synthetic
conformance descriptors also admit bounded typed Boolean expressions:
`const`, unsigned `lt`, `low_lt`, explicit signed `slt`, `msb(x+c)`, `not`,
`xor`, `and`, `or`. B is distinct from A_n; Python booleans cannot masquerade as
integer coefficients or descriptor bit constants. Implicit mixed-domain
conversion, higher degree and other Phi circuits are rejected.

Compilation follows Protocol 3 exactly:

1. Sort `((alpha_i+r) mod N, i)`. If zero is absent, split the wrapping interval
   at 0/N and duplicate its payload. If zero is present and m<N, split the
   **first** interval of length at least two at `lower+1`, duplicating its payload.
   If m=N, retain N intervals. Always M=`min(m+1,N)`.
2. Normalize each piecewise Boolean output to
   `XOR_i (([x<alpha_(i+1)] XOR [x<alpha_i]) AND B_i(x))`.
3. Rewrite each nonsentinel threshold in its k-bit view using
   `theta=(r_k+beta) mod 2^k` and carry `[r_k+beta>=2^k]`:
   `[view(xhat)<theta] XOR [view(xhat)<r_k] XOR carry`.
   Low views reduce both xhat and r. Signed comparisons shift the view by N/2;
   MSB uses a public shifted view followed by NOT of comparison with N/2.
4. Keep two query slots for **every** nonsentinel atom, even when a sampled
   threshold becomes zero. Only public descriptor-level 0/N sentinels simplify.
   Descriptor order fixes the query/Boolean program order; no deduplication occurs.
5. Lookup the two coefficients, recover `x=xhat-r mod N`, evaluate original-x
   Horner and then identity Phi. The independent unfused oracle instead scans
   source intervals, directly evaluates an expanded affine formula and directly
   interprets the original predicates.

The reference interpreter uses clear scalar lookup/branches. It models an ideal
backend's values, not constant-time or protected execution. A mandatory expected
descriptor digest detects wrong-descriptor substitution; a caller rewriting a
trusted compiled object is not an authenticated adversary model.

## Bounds, serialization and explicit work

Before mask-dependent compilation, admission validates/counts:

| Bound | Limit |
| --- | ---: |
| Source / padded intervals | 512 / 513 |
| Arithmetic degree / slots per payload | 1 / 2 |
| Reference ring widths | 1–50; D8 gate exactly 50/Q12 |
| Boolean outputs / expression depth / total source nodes | 4 / 12 / 8,192 |
| Comparison queries / Boolean VM operations | 32,768 / 65,536 |
| Canonical descriptor / clear instance bytes | 256 KiB / 1 MiB |
| Exact external rational numerator/denominator width | 4,096 bits |

Canonical descriptor JSON uses sorted keys, compact separators, ASCII and exact
integer values; parsing rejects duplicate keys, floats, extra fields and alternate
encodings. The clear compiled-record format has fixed-width fields determined
solely by its descriptor. Boundaries reserve **n+1 bits** for sentinel N; ring
coefficients reserve n bits, with byte rounding. All masks yield equal lengths.
This is a bounded **reference format**, not a proposed wire format or FSS key codec.
Python object/allocator overhead and process-wide residency are not budgeted here.

Actual GELU has **511 source / 512 padded intervals**, zero comparison queries
and **11,846 clear-record bytes**. It requires one ideal interval-lookup interface
call, one coefficient-selected arithmetic Beaver product, one ring addition and
local original-x recovery. Phi has zero operations/conversions. Degree-one
Horner retains its product slot even for constant pieces. Synthetic Boolean
normalization charges one AND per interval/output, additional expression ANDs,
and one AND plus two XORs per OR. No B2A/A2B is silently supplied. These are
reference circuit counts, not backend keys, rounds, traffic or measured costs.

## Checks and finite negative controls

From repository root:

```sh
python3 -B -m unittest discover -s paper/reproduction-contracts/fusefss -p 'test_*.py' -v
python3 -B paper/reproduction-contracts/fusefss/freeze.py
```

The second command prints reproducible vectors; it never updates frozen files.
Generate canonical descriptor bytes on stdout, from this directory, with:

```sh
python3 -B -c 'import reference; print(reference.gelu_descriptor().canonical_bytes().decode())'
```

The digest excludes print's trailing newline. Table generation uses independent
exact rational enclosures: Machin arctangent bounds for pi, integer-root bounds,
and an alternating integral series. Both bounds must have the same floor for
each entry. Tests check 128/160-bit enclosures, the existing pinned table hash,
and a separate standard-library erfc formula. Expected GELU vectors use the
direct mathematical function; compiled-transcript hashes are separately labelled
regression checks. The remote CPU source is not needed or executed by tests.

Coverage includes all **8,192** ring8 partition/mask/input combinations with
piecewise Boolean outputs, all small signed/unsigned/low-bit views and shifts,
every GELU boundary's immediate neighbors under seven masks, and **32,771**
central signed input codes. Controls distinguish early floor from abs-first;
Q0 gains from Q12 gains/extra rescale; x from xhat in Horner; XOR from arithmetic
addition; reduced from full low-bit masks; ring-wrap-before-rescale from an
unwrapped product; floor-of-sum from sum-of-floors; and A44 TR from A50 ARS.
Malformed/type/shape/range cases, descriptor substitution and fixed serialized
lengths are tested. Ignored PDF inputs, when absent, cause an explicit source-hash
test skip; no network fetch or unreported source verification is implied.

## Exact remaining gates

| Cell | State / required next evidence |
| --- | --- |
| D8 profile and D9 boundary choices | Approved; not reopened by this reference. |
| Original evaluated FuseFSS descriptor manifest | **Blocked: missing source artifact**. This named interpretation is not original-table numerical identity. |
| External tensor/checkpoint representation and frontend | **Unresolved here**. Exact rational nearest-even is fixed; no implicit float-format, checkpoint or tokenizer bridge is invented. |
| Source effective-width and full-graph range provenance | Required. Prove sign-extension from signed38 here and every pre-wrap product/accumulator/rescale bound upstream. |
| Remaining gate descriptors / non-identity Phi | Not implemented. Freeze complete LUTs, types, constants, correction/conversion circuits and order for each gate. |
| Protected lookup/predicates/Horner/Phi | Not implemented. Qualify actual source-compatible FSS and Beaver/AND/conversion backends and fixed key shapes. |
| Roles, material and transport | Not implemented. Preserve client + two online servers + dealer, public model, fresh per-instance material, retirement/abort and authenticated private transport; admit all live owners. |
| D9 complete transaction | Not implemented. Charged protected embedding, final norm/full head, lowest-index argmax, private fixed-count full-prefix feedback and final client-only IDs remain required. |
| Native / D5 SDK / original and common-model qualification | Not implemented or measured by this slice. No security, performance, quality or completed-method status follows. |
