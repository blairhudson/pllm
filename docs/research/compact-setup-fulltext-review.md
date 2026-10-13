# Compact setup: full-text review and fixed-weight consumer gate

**Outcome:** all eight supplied PDFs are readable, including their appendices.
Exact-ring correlation generation is supported by published constructions. A
compact generator with cheap **client expansion for fixed checkpoint weights**
is still missing from the proposed PLLM consumer. No candidate has measured
20–50% all-link savings or qualified under the 256 MiB client ceiling.

The key correction is bounded matrix-mask reuse: **2026/2298 allows fresh
columns of one matrix triple to be consumed sequentially**. The earlier
31.55 GB opening estimate applies to the explicitly fresh-per-use adaptation;
it does not reject every consumer of that paper. Indefinite reuse across
independent triples is a different claim.

## Source identity and scope

[Source manifest](../evidence/compact-setup-fulltext-review.json) records each
local filename, full PDF SHA-256, length, page count and reviewed locations.
Page numbers below are **1-based PDF pages**. Fresh `pdftotext -layout`
extractions agree with the eight working copies. These are the supplied
revisions; their hashes do not certify that no newer revision exists.

This is a construction/cost review, not an independent proof audit or paper
implementation. Historical [encrypted-ingress](transciphered-linear-reference.md),
[streaming-correlation](streaming-matrix-correlation-reference.md) and
[fresh-triple](matrix-triple-prefill-reference.md) records describe access and
evidence when those screens ran. Their abstract-only access limitations are
resolved here; their measured controls retain their original scope.

## Full-text findings

### Exact-ring PCGs — 2025/1223

[Efficient Pseudorandom Correlation Generators over Z/p^k Z](https://eprint.iacr.org/2025/1223),
pp. 7, 33–36, gives Galois-ring OLE and SPDZ2k-type authenticated triples.
Page 7 also describes circuit-dependent preprocessing and an extension to
matrix triples over `Z/2^k`. Exact-ring generation is therefore a real lead.

Table 5 reports **109,000 triples/s** at `(k,s)=(32,26)`, `c=5,t=27`, in its
`3^15` column. The experiment uses an AMD EPYC 9754 and **512 GB RAM** (p. 34);
machine capacity is not a measured working set. The faster `c=3,t=27,m=2`
results are explicitly **broken benchmark-only parameters**. Some larger
configurations run out of memory.

The roughly **115 MB for one billion triples** in Table 2 is a seed-size
projection. Page 36 uses `c=9,t=9,m=2,n=15` for that communication point,
which Tables 3–4 classify at **80-bit security**, whereas `c=5,t=27` targets
128 bits. They cannot be combined into one measured configuration or cited as
115 MB at 128-bit security. Distributed setup costs more than delivered seeds.

**Next use:** investigate its programmable/circuit-dependent construction for
the asymmetric functionality below. Random triples and short seeds do not
automatically supply that specialization or its client expansion cost.

### DCR PCFs — 2026/2030

[New Pseudorandom Correlation Functions and Exponent VRFs from DCR](https://eprint.iacr.org/2026/2030),
§4.1 and Appendix F, supports COT-to-VOLE over arbitrary integer moduli.
Page 15 charges approximately `ceil(log2 M)` COT samples for a power-of-two
modulus, with statistical slack for general moduli. Integer lifting and its
failure probability must compose over all uses.

The OLE/general degree-two extension is over **F2** (pp. 6, 17–18), not an
arbitrary-ring matrix generator. Table 1, p. 21, reports **522.80 ms receiver /
1.53 ms sender** per COT at a 2048-bit RSA modulus; batched amortized times
are **31.02 ms / 73.87 microseconds**, with batch size 256. The key pair is
approximately **195 KiB**. These are COT costs before a matrix consumer;
moving the receiver between roles moves its cost.

Corollary 3 (p. 14) explicitly invokes **DCR and the Generalized Riemann
hypothesis**, while §4 assumes an inner-product-membership weak PRF. The
abstract's “solely DCR” shorthand is insufficient for a concrete security
declaration. **Decision:** modulus-compatible primitive, but no client-light
fixed-`W` generator or applicable cheap expansion measurement follows.

### Ring-LWR PCFs — 2025/1637

[Pseudorandom Correlation Functions from Ring-LWR](https://eprint.iacr.org/2025/1637),
§§6.6–7, supports additive shares of arithmetic-circuit outputs, including
matrix triples, over rings and fields. Definition 7 (p. 23) requires **each
party** to bootstrap pseudociphertexts, evaluate the correlation circuit and
decrypt with its own key share. It does not replace client computation with
an ordinary cheap PRG.

Table 3, p. 22, **estimates** 1,839 seconds for one encrypted ring element
at plaintext modulus `2^64`, translating to about **9 triples/s** with the
paper's packing. Some subroutines are microbenchmarked and others extrapolated;
this is not a measured PLLM stage. Share conversion requires negligible
relative noise (§7.1), not unrestricted sharewise rounding. **Decision:** an
expressive specification reference whose supplied evaluation strategy does
not meet the lightweight-client objective.

### Matrix-triple PCGs — 2026/2298

[Pseudorandom Correlation Generators for Matrix Triples](https://eprint.iacr.org/2026/2298)
resolves several previously unknown details:

* **Rectangles:** pp. 70–71 use padded square blocks. An `(m by k)(k by r)`
  product consumes `ceil(m/n)*ceil(k/n)*ceil(r/n)` square triples. Padding,
  additions and block-specific openings remain costs.
* **Sequential columns:** §7.2, p. 72, packs `s <= n` vector products into
  fresh columns of one right-hand matrix. The left mask can be opened once
  for those columns even when the vectors arrive sequentially.
* **Reuse limits:** pp. 74–75 leave reuse of operand masks **across block
  products/instances** outside the present programmable proof. Corollary 4.6's
  public-multiplier reuse is within a polynomially bounded batch (p. 42),
  not permanent setup or reuse of private activation masks.
* **Domain:** p. 72 explicitly leaves the `Z/2^64` extension to future work.
  Existing ring work cannot silently supply a proof for twisted QA-SD.
* **Cost:** Table 4, p. 68, estimates **2 GiB per party** at `N=2^12` and
  **32 GiB** at `N=2^16` for the natural twisted implementation with 128-square
  matrices over `2^64-2^32+1`. Estimated expansion is 14–60 core-ms/triple,
  versus a 2.1 ms clear product. These are microbenchmark-derived estimates,
  not complete-implementation process RSS or latency measurements.
* **Setup:** Table 7, p. 70, separately estimates distributed generation.
  Twisted `(c,t)=(2,64),N=2^12` has **200.3 MiB FSS generation plus 49.5 MiB
  payload-phase communication**, both parties combined. Amortized seed bytes
  are not the complete setup. Trusted-dealer delivery needs its own ledger.
* **Verification:** p. 72 distinguishes its semi-honest unauthenticated
  64-bit triples from authenticated, roughly 128-bit-field comparators.
  Page 75 leaves the authentication correlation unpriced.

The natural published working sets exceed Low; this is not a lower bound
against future streaming implementations. A bounded-column consumer deserves
reconsideration, with exact-ring compatibility and client computation priced
first. Neither cross-instance reuse nor safely publishable A-only keys is
supplied by this review.

### Dory — 2025/1660

[Dory: Streaming PCG with Small Memory](https://eprint.iacr.org/2025/1660),
§§5–6, includes the attack reducing old regular-noise EA-LPN parameters to
14–22 bits and updated relaxed-regular/low-weight-noise parameters. The supplied
PDF includes the correction described in its abstract.

Tables 2–3, p. 21, specify bounded output counts and distinguish base-COT
communication from silent extension. LPN2 with eight threads and malicious
security reports **20 MB memory**, **0.419 + 0.389 MB** setup communication,
and capacity **10,295,705 COTs**. Memory/compute are the maximum of either
party, not the whole application. The implemented correlation is `F2` into
`F_(2^128)` COT. **Decision:** strongest streaming/ownership reference here,
but it still needs a ring conversion and complete fixed-weight consumer.

### Preprocessing for Life — 2025/787

[Preprocessing for Life](https://eprint.iacr.org/2025/787), pp. 1–3, combines
PCGs and FLIOPs in a two-online-party-plus-offline-dealer protocol. Its
implemented comparison is for **Boolean circuits**. Page 3 still charges
**two online bits per party per AND gate**; the thousandfold improvement
concerns correlation storage/dealer communication, not removal of online
work. Page 2 notes richer circuits as a blueprint generalization, not a
measured exact-ring decoder.

Client plus Inference could occupy the online roles only by charging the
client's circuit evaluation. Making Preparation the other online party changes
the role graph. **Decision:** lifecycle and dealer-checking reference, not a
drop-in prepared correction generator.

### PETCHA and Coral — 2026/112 and 2026/1917

[PETCHA](https://eprint.iacr.org/2026/112), p. 9, specifies the original
**64-bit counter / 64-bit nonce** ChaCha state. A 96-bit-nonce IETF variant
is not interchangeable. Its conservative ChaCha20 3-LUT row in Table 2
(p. 20) reports **92.7 s single-thread / 25.8 s with four threads per 512-bit
block**, at its negligible-failure setting. It reports 128-bit post-quantum
parameter estimates; faster/high-failure rows have different contracts.

[Coral](https://eprint.iacr.org/2026/1917), pp. 12–13, reports **339.061 s
single-thread / 7.806 s multithread per 512-bit ChaCha20 block**. The machine
offers 192 hardware threads without an extra concurrency limit. Its single-thread
ChaCha result is slower than its baseline; the 5.517x headline applies to its
multithread comparison. Parameters target at least 128-bit LWE/GLWE security
and estimated failure at most `2^-40`. This is a different security, hardware
and threading cohort from PETCHA's conservative row.

Both reduce client **ingress** expansion. Neither supplies our measured
full-width signed W8A8 projection, compact reply, TFHE-to-packed-arithmetic
conversion, whole-session failure bound or all-link setup accounting. The
historical BFV reply rejection concerns that BFV control, not every conceivable
transciphering scheme. **Decision:** lower priority than exact-ring correlations
at the supplied cipher costs.

## Required asymmetric fixed-W functionality

For public signed-i8 `W` of shape `m by n`, let `R = Z/(2^k)` be its compiled
ring. Finite setup binds source/weight digest, stage, numeric layout, roles,
epoch and maximum issuance count `Q`, issuing distinct role keys. At fresh
public index `j`:

```text
Client expansion:      (r_j, s_j) in R^n x R^m
Inference expansion:   q_j = W r_j - s_j in R^m
Client -> Inference:   z_j = x_j - r_j
Inference -> Client:   y_j = W z_j + q_j
Client reconstruction: y_j + s_j = W x_j
```

The ideal distribution samples full-ring `r_j` and independent `s_j` anew
for each index. A construction must emulate the **joint multi-index**
distribution against each role's complete key/transcript view. Marginally
random-looking masks are insufficient. Inference must not recover `r_j`
from its key, correction, openings or published setup data.

Per-output residues may project this relation into compiled quotient rings
only after layout validation. Field masks cannot simply be reduced into `2^k`:
wraparound, reconstruction, widths and failure bounds need their own proof and
cost. Original signed decode, bias, scales and float32 boundaries still apply.

**Client-light is an additional requirement.** Price both key expansion and
consumer work. A fresh dense `W r` or `D B_client` is not cheap mask expansion.
Low admission includes the existing SDK, state, keys, output overlap, scratch
and renewal overlap. A 20 MB primitive or small seed does not establish the
256 MiB whole-process maximum. At least 90% of original body MACs must remain
remote in prefill and decode separately; additional cryptographic/client work
must also be reported rather than hidden in that placement ratio.

Index allocation commits before exposure and survives restart, restore and
cloning. Cancellation, malformed replies, expiry and failure burn reserved
indices. Exhausting `Q` requires fresh setup; different session labels alone
cannot prevent rollback. Setup/renewal include both role deliveries,
proofs/conversions and failed attempts. Offline trusted, noncolluding
Preparation remains a separate assumption from a paper's distributed setup.

This is a **target functionality and admission contract**, not a construction
of `Gen`, `Expand_client` or `Expand_inference`. Those algorithms and concrete
security parameters remain the missing implementation prerequisite.

## Next experiment and stop conditions

1. Start with **2025/1223**, selecting one non-broken parameter set throughout.
   Derive explicit party keys/expansions; identify the programmability theorem
   applicable to arbitrary checkpoint `W`. Price source-dependent setup and
   every generated row. If it only gives random triples, state that gap.
2. Keep a **bounded-column matrix-triple control** from 2026/2298: at most `n`
   fresh sequential columns per triple, rectangular blocks and renewal at
   exhaustion. Opening `D=W-A` once reduces traffic within that capacity but
   leaves `D B_client` work and field compatibility. Reuse across distinct
   triples requires a further construction/proof.
3. Only after paper costing passes, independently implement one bounded native
   stage. Check all integers, role-local views, exhaustion and retirement.
   Measure fresh-process client peak, both expansions, setup/renewal delivery
   and complete online bodies before SDK promotion.
4. Retain the [matched traffic budget](silent-linear-frontier-reference.md):
   correction-only savings cap at **25.67–31.84% first-pair** and
   **37.71–38.28% recurring**, before replacement costs. **50% requires savings
   on another link.** Seed-size ratios cannot remove this ledger constraint.

The next question is specific: **can programmable exact-ring correlations
bind fixed W without dense client expansion?** Full-text access is resolved;
a priced construction is the gate.
