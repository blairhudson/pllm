# Compact setup for private linear execution

This follow-up to the [bootstrap frontier](bootstrap-frontier-reference.md)
asks whether compact reusable setup can remove **20–50% of all-link traffic**
while preserving exact decoder arithmetic, predominantly remote body computation
and the **256 MiB Low-tier whole-client ceiling**.

Three separate investigations examine
[transciphered byte ingress](transciphered-linear-reference.md),
[streaming fixed-matrix correlations](streaming-matrix-correlation-reference.md)
and [matrix-triple prefill](matrix-triple-prefill-reference.md). Their complete
consumers, additional links and client work must be charged alongside setup.
Compact primitive seeds alone do not establish compact private inference.

## Findings

**Encrypted byte ingress needs a different reply representation.** Independent
tiny BFV projections preserve ten signed outputs with separate client/provider
contexts. Even granting unimplemented full-slot packing, their cheapest measured
reply serialization projects about **879 MB** over the saved historical stage
geometry. The recurring 20%-saving reply allowance is only **265.56 MB**, before
new setup, conversion and helper costs. This rejects ordinary BFV replies, while
leaving tightly packed/extracted output as a separate unimplemented possibility.
PETCHA/Coral cipher execution and full-width noise have not been established.

**Streaming correlations need the complete fixed-weight relation.** The desired
one-use outputs are `Client: (r,s)` and `Inference: W*r-s`, with independent fresh
full-ring masks. The ideal consumer reconstructs exact `W*x`, but a compact key
for this relation with light client expansion is still missing. The tested
chosen-vector-OT adapter costs terabytes per historical pair; this is a cost of
that adapter, not a communication lower bound. Bypassing its OT with public
matrix data moves matrix-dependent expansion back to the client. Dory's streaming
COT memory figure does not supply arbitrary chosen matrix correlations. The DCR
abstract promises arbitrary-modulus COT/VOLE, while its general quadratic/OLE
extension is specifically over `F2`. These paper claims were checked at abstract
level; full algorithm/parameter verification remains open.

**Fresh matrix triples lose even when limited to prefill.** The explicit
random-to-fixed adaptation opens `D=W-A` to both endpoints. One such opening per
stage costs **2.103 GB** in optimistically packed rings, before inputs, replies,
keys or any decode work. Adding the fourteen sequential decode uses brings the
pair's opening payload to **31.546 GB**. The field31 versions cost 2.773 GB and
41.597 GB respectively. These figures reject the specified dense-opening layout,
not every matrix PCG. Disclosable A-only generation keys could change that cost,
but require a source-supported interface that keeps B/C shares private. They
would still leave dense client adaptation: **56.616 billion additional MACs**
for the complete historical pair, before PCG expansion. Original online body
computation stays remote; the extra client work remains separately charged.

The prefill-only check uses the matrix report's one-opening-per-stage count
(also reported as the unsupported reusable-A setup). It needs no reusable-A
assumption for a single prefill. At 150 rows, each stage's two dense openings
cost at least `2*896/150 = 11.9467` times its removed correction **payload**.
Thus excluding decode does not rescue that explicit prefill construction.

None of these screens establishes 20–50% serving savings or a Low-tier candidate. Their
historical 493.575 MB control is distinct from the newer controls below. Primitive
sizes, hypothetical output densities and measured whole-response traffic retain
their separate scopes.

## Next useful research gate

Prioritize a **fixed-public-matrix, exact-ring correlation generator with
asymmetric expansion**: the client cheaply derives independent fresh `(r,s)`,
while Inference derives `W*r-s` without learning either mask. The complete
construction must price both endpoint expansion, setup delivery and renewal.
Generic COT/VOLE seeds and random matrix triples do not yet supply this contract.
Full primary papers and concrete parameter sets are prerequisites to selecting
an implementation; the blocked PDFs prevent a paper-level algorithm audit.

Encrypted ingress remains a secondary lead only with a genuinely compact reply.
The existing control demonstrates signed integer correctness and serialization
cost for tiny BFV projections. It does not execute PETCHA/Coral, conversion into
arithmetic ciphertexts, or full-width noise. A hypothetical 1.5-times reply
expansion would leave room for 26.54% first-pair/35.53% recurring savings on its
historical control before all new costs; no tested representation attains it.

## Matched Low-tier control budgets

The new [control audit](../evidence/research-silent-linear-controls.json)
independently reconciles twelve existing fresh-client samples from the
[five-way combination screen](sota-combinations-reference.md). Each sample owns
one new-prefix request with eight outputs and its exact repeat. Every selected
control replicate is below 256 MiB process-lifetime peak RSS, with zero observed
swap growth. These measurements qualify the historical controls only.

| Control | Input tokens | Covered pair bodies | Correction bodies | Public delivery aggregate | Free-correction first-pair ceiling | Free-correction recurring ceiling |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Tiling + digit8 | 150 | 493.618 MB | 139.061 MB | 124.905 MB | 28.17% | 37.72% |
| Tiling + digit8 + placement | 150 | 471.830 MB | 121.105 MB | 155.465 MB | 25.67% | 38.28% |
| Tiling + digit8 | 288 | 802.351 MB | 255.489 MB | 124.905 MB | 31.84% | 37.71% |
| Tiling + digit8 + placement | 288 | 736.652 MB | 222.468 MB | 155.465 MB | 30.20% | 38.28% |

Values are three-replicate medians in decimal MB, rounded for display; every
replicate's exact counts remain in the audit. These are application bodies, not TCP-stream or physical-wire
counts. The recurring ceiling assumes that the entire original public-delivery
aggregate becomes reusable for both candidates. It is a sensitivity endpoint:
the recorded aggregate does not certify that every manifest or metadata byte is
reusable. With all public delivery recurring instead, the recurring ceiling equals
the first-pair ceiling. Both endpoints have the same first pair.

All arithmetic correction bodies are optimistically removed in this table.
Existing acknowledgements and other controls remain. New keys, helper traffic,
renewals, distribution and failed-attempt costs reduce the available saving.
Even this free replacement cannot reach 50%; another link must shrink as well.

For the 150-token placement control under full public reuse, a 20% asymptotic
target leaves **57.832 MB per pair** for all additional traffic. The 288-token
control leaves **106.231 MB**. At a finite horizon, reusable setup consumes some
of that allowance. These are budgets for an unspecified candidate, not savings.

### Exact horizon rule

For `R` workload pairs, let `C` be original common reusable delivery, `T` recurring
control bodies, `S` removed corrections and `E(R)` every additional candidate byte:

```text
control   = C + R*T
candidate = C + R*(T-S) + E(R)
target g requires E(R)/R <= S - g*(T + C/R)
```

`E(R)` includes initial setup, repeated helpers, renewals, failure retirement,
additional framing and newly required public artifacts on all links. Renewal
boundaries can make savings nonmonotonic. Any control cost omitted from the
historical measurement, including checkpoint distribution, needs a new matched
measurement before entering a complete-wire claim.

The audit checks 20%, 35% and 50% targets at 1, 2, 4, 16, 64 and 4,096 pairs for
both public-reuse endpoints, retaining exact rational byte budgets. Negative
budgets explicitly reject that target before any new traffic is charged.

## Base and per-output accounting

Known base, output-dependent and unattributed bodies conserve each control's
total. Unclassified envelopes, padding and controls remain unattributed.
**Complete per-request base and per-generated-token rates remain unknown.**
A correction category is not itself a per-token marginal cost: prefill rows,
prepaid output capacity, exact-repeat reuse and generated output counts retain
their separate contracts. None of the hypothetical candidates inherits measured
traffic attribution merely by removing a category from a control ledger.

## Replay

The joint audit passes **77 focused tests** and Ruff, verifies **24 source-file
hash bindings**, and reproduces the three deterministic reports exactly. A fresh
SEAL replay also preserves all ten signed outputs and the recorded 55/55/19-bit
noise budgets. Randomized ciphertext serialization changes by a few bytes; the
frozen original sample remains the source of the quoted cost projection.

```sh
target/frontier-sdk-env/bin/python scripts/probe_silent_linear_controls.py \
  --output target/silent-linear-controls-replay.json
target/frontier-sdk-env/bin/python -m pytest --noconftest -o pythonpath= -q \
  tests/test_silent_linear_controls.py
```

The output path must be new. Tests reject incomplete cohorts, duplicated or
missing replicates, changed numeric identities, scope substitution, nonconserving
accounting, conserving correction-category substitution and swap growth. They
also check the horizon inequality independently. Source hashes bind the frozen
cohort and the audit script. No candidate cryptography or model executes here;
candidate RSS, resource admission, latency and measured savings remain unknown.
