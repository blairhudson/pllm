# Transciphered exact linear execution: bounded screen

This screen separates symbolic symmetric ingress from independently measured
downstream BFV projection. It implements no PETCHA/Coral cipher, scheme conversion,
production transport, decoder, checkpoint loader or upstream algorithm dependency.
`complete` means the falsification/cost screen completed, not protocol admission.

## Primary sources and access

* PETCHA, Guimarães/Jacob/Pereira, [2026/112](https://eprint.iacr.org/2026/112):
  received January 23, revised June 8, 2026; PQCrypto 2026. Its abstract specifies
  **ChaCha**, not a generic AES/ChaCha API; it claims 128-bit quantum security
  from a 256-bit classical cipher and post-quantum FHE parameters.
* Coral, Bai/Wang/Lu/Wang, [2026/1917](https://eprint.iacr.org/2026/1917):
  received September 8, approved September 10, 2026; preprint. Its abstract
  specifies **TFHE**, instantiated for Trivium, AES, ChaCha20 and ZUC. General
  Boolean-module support is not certification of arbitrary ciphers or exact
  packed W8A8 matvec conversion.

Both primary HTML pages were retrieved and hashed. Direct PDFs, query-string
download and archive attempts returned HTTP 403. Search indexed PDF excerpts
were available, but are not a retrieved full paper: **direct access is abstract-only**.
No cipher round/nonce variant, parameter table, key serialization, failure budget,
paper throughput transfer or section-level implementation claim is made for either.
In particular, ChaCha's exact nonce/counter layout must be selected from the
reviewed source, not silently replaced with another library's ChaCha20 variant.

[BumbleBee's NDSS 2025 full PDF](https://www.ndss-symposium.org/wp-content/uploads/2025-57-paper.pdf)
was retrieved and extracted under `target/transciphered-linear/`. Section III's
PackLWEs/interleaving discussion explains why useful coefficients must be packed,
not sent as sparse full RLWE ciphertexts. Its circuit-privacy remark explicitly
distinguishes hiding computation noise from plaintext correctness. Its ring
protocol and H2A functionality are not the prime-field BFV control here; no cost
or security theorem transfers automatically. The project
[packed embedding reference](packed-embedding-reference.md) already distinguishes
extracted-LWE packing, interleaving, keys and full-width noise admission.

[SEAL v4.1.2 Serializable documentation](https://raw.githubusercontent.com/microsoft/SEAL/v4.1.2/native/src/seal/serializable.h)
was retrieved and hashed: fresh symmetric-key HE encryption can seed its random
half and expand on load. **Evaluated replies do not inherit that seed form.**
Compression of serialization is distinct from seeded generation, coefficient
extraction and homomorphic packing. Sending only b from an extracted LWE sample
omits the required a vector; sending `(b,a)` per scalar may expand traffic greatly.
No undocumented seeded-output discount is taken.

## Exact numeric and ownership contract

Client encrypts signed i8 activation bytes with a reviewed symmetric cipher and
sends its FHE-encrypted symmetric key once per admitted epoch. Inference receives
only public/evaluation keys, evaluates decryption and all public integer W*x.
Client alone decrypts final outputs and retains original nonlinear/scaling work.
No intermediate client decrypt/reencrypt or client W*carry is allowed. Preparation
may build public layouts offline; secretly generating keys on its behalf changes
the trust contract and is not granted. A TFHE cipher output is not automatically
a BFV SIMD vector: bit/byte sign extension, modulus conversion, key switching,
packing, bootstrap keys, their delivery and noise must be independently specified.
All original body MACs can remain remote in this design, but latency and memory
are unknown. Placement admission requires at least 90% remote original body MACs;
tests admit exactly 10% client dense ownership and reject any greater fraction.

The bounded control uses degree 8192, prime t=268369921 and coefficient bits
[55,54,54,55], admitted by SEAL TC128. This is **classical parameter admission**,
not PETCHA post-quantum certification. A public bound B requires t>2B; center the
decrypted integer first, then reduce modulo 2^16/2^24/2^32. Arbitrary full-ring
masked values cannot be substituted into this proof. Saved input widths up to
4864 satisfy the coarse symmetric W8A8 bound 4864*127^2; the installed symmetric
activation contract is [-127,127]. Toy -128 endpoints stress arithmetic only,
not extension of an installed quantizer contract. Saved row residue decoding,
bias, scales, float rounding and complete logits/KV remain separate gates.

Provider and client use separate SEAL contexts. Provider reloads only parameters
and rotation keys. Two synthetic 5x8 projections exercise signed endpoints, zero
rows, first/last coordinates, packed zero tail and reused keys. Public direct-HE
input encryption is a downstream control, **not transciphering**. Every modulus
level is serialized, decrypted and checked; zero noise or wrong outputs remain
failures rather than cheap admissible replies. Raw SEAL compression-mode byte,
setup components, trial times and bounded process peak are in JSON. No secret key
is transmitted. Synthetic public fixtures and ephemeral keys stay in target.

## Matched traffic accounting

Historical sensitivity only, not latest Low/full-wire: first new 150+8 prefix
request plus exact repeat is 493575247 body bytes; assumed common reusable public
C0=124905091, recurring T0=368670156. Remove correction bodies including metadata
139060689, narrow input payload 90104112 to 28732928, and retain original other
overheads. Original output payload O=138864059. Thus S=200431873 and:

```text
control   = C0 + R*T0
candidate = C0 + R*(T0-S+(r-1)*O+H) + B
            + (ceil(R/K)-1)*E + F
```

B is initial setup across every link; E subsequent renewal, not double-counted
initial issuance. H includes new frames, provider artifacts, conversion, output
verification and other recurring traffic; F all charged failed attempts. Unknown
actual costs are null/explicit unknown. Zero-extra-cost rows are labeled hypothetical
ceilings. The alternative endpoint makes all original public delivery recur on
**both** sides; both endpoints reproduce exactly the same first pair. Prepositioning
does not remove candidate setup delivery. Warm baseline warms too.

At hypothetical r=1.5, S-(r-1)O=130999843.5 removed bytes: **26.54% first-pair,
35.53% recurring** under reusable public endpoint, before every new cost.
At 20%, r must be <=1.73249208 first-pair or <=1.91238757 asymptotically.
Total setup/helper/renewal/failure allowance at r=1.5 is **32.285 MB first-pair**;
asymptotic recurring helper allowance is **57.266 MB/pair**. These are budget
limits, not measured output expansion or cryptographic performance.

Geometry validates all 96 stage/weight identities, payload conservation and 15297
executed rows against saved ingress/narrow controls. Ordinary stages have 150
prefill rows plus 14 decode rows; terminal stages have one prefill plus 14 decode.
Each dependent decode row must be sent separately; it cannot pool with future
tokens or dependent stages. Output pricing uses ceil(prefill*m/slots) plus
14*ceil(m/slots), not ceil(total outputs/slots). Compare actual toy's 4096-slot
first batching row with hypothetical full 8192-slot density. Public padding and
a hypothetical 96-byte frame per ciphertext are charged separately. Full-width
dense packing and corresponding noise are **not certified** by tiny measurements.

## Missing promotion gates and next experiment

Best surviving design is reviewed ChaCha transciphering with server-only exact
bounded-integer matvec, tightly packed final outputs, reusable admitted keys and
client nonlinear work. This is a **conditional architecture**, not a qualified win.
Plain compact ingress survives algebra; naive sparse outputs, free conversion,
generic seeded evaluated replies and hidden client dense work do not.

Before implementation promotion:

1. Retrieve full PETCHA/Coral source and freeze cipher variant, TFHE parameters,
   complete key/serialization contract, security level and total-use failure bound.
2. Independently implement one transciphered signed-byte block and exact conversion
   into the chosen matvec representation; oracle upstream only, never link/build it.
3. Evaluate synthetic source-width 896x896, 1152x896, 9728x896 and 896x4864 stages,
   especially single-row decode. Measure dense reply packing noise and delivered
   bytes, not only final-slot counts; reject if budgets above cannot hold.
4. Measure fresh/reused/ephemeral contexts, all bootstrap/distribution/renewal links,
   failures and authenticated frames, and whole client peak below 256 MiB including
   SDK/tokenizer/KV/context/key copies and overlapping serialization. Tiny process
RSS cannot establish whole-client admission.

### Actual downstream rejection

One bounded run measured setup delivery **18.262 MB** (parameters 83 bytes,
public key 0.494 MB, rotation keys 17.769 MB); client secret serialization 0.247 MB
is retained, not sent. Two 5x8 projections preserve ten outputs and every unused
slot exactly. At three/two/one data primes, compressed size-two replies were
approximately **370.2/246.9/123.5 kB**, with minimum measured noise **55/55/19 bits**.
Bounded process lifetime peak was **about 133 MB**, not whole-client RSS admission.
Serialization randomness causes small byte-size variation between runs.
These compressed measurements are not fixed public padding bounds. Actual frames
must pad to an admitted public maximum, never a private-value-selected length.
Uncompressed size-two RNS coefficient storage alone is 131072 bytes at one prime,
before SEAL metadata; changing compression/padding cannot be priced as free.

Saved source geometry needs **12782** ciphertexts using the implemented first
4096 slots, or **7111** granting unimplemented full 8192-slot output density.
For the cheapest measured one-prime frame, those imply **about 1.58 GB / 879 MB**
output bodies including hypothetical 96-byte frames (approximately 879 MB for the
full-slot case; exact saved sample is 879012043 bytes at 123517 bytes/reply).
Full-slot expansion is
**6.33x**, not 1.5x. This naive BFV serialization route loses traffic even with
free transciphering and no extra setup: **rejected**. It does not reject every
compact/extracted output scheme. At full-slot counts a 20% target requires at most
about **33.83 kB first-pair / 37.35 kB recurring per ciphertext**, including frames
and before new costs. A different compact representation must demonstrate exact
decoding, safe noise and these sizes; referring to packing alone does not do so.

One matvec context per stage would already deliver about **1.753 GB**; one per
executed stage row about **279.35 GB**, before TFHE/key conversion. Reusable contexts
are necessary for this setup, but their security lifetime is not certified.
One-prime 19-bit toy noise cannot certify 896/4864-wide products or 9728-output
packing. It is downstream feasibility evidence only, not full-width noise bound.

Symmetric key/nonce/counter uniqueness must be durable across retries, forks,
restart and context epochs; allocate fresh nonce before each attempt and never
reuse stream bytes. Bind cipher/key/weights/ring/stage/shape/padding/request/use
ordinal and direction to authenticated envelopes; burn failed/cancelled attempts,
reject replay/foreign/malformed context, and never expose a decryption-error oracle.
Client-verifiable transport authentication does not itself prove provider homomorphic
AEAD evaluation or correct W*x. Semi-honest provider privacy is weaker than malicious
output verification. Public W alone may not require W-hiding circuit privacy, but
that does not prove compositional safety of noise, extra slots or secret circuit
state. Any flooding/re-randomization/verification costs need explicit charging.
Key renewals follow nonce capacity, cryptographic total-use advantage and union
bound Q*epsilon, not chosen amortization convenience. No reviewed lifetime exists
yet; lifelong reuse is not assumed as an implementation guarantee.

## Replay

```sh
target/frontier-sdk-env/bin/python -m pytest --noconftest -o pythonpath='' -q tests/test_transciphered_linear_budget_probe.py
target/frontier-sdk-env/bin/python scripts/probe_transciphered_linear_budget.py --measure-seal --output NEW.json
target/frontier-sdk-env/bin/python -m ruff check scripts/probe_transciphered_linear_budget.py tests/test_transciphered_linear_budget_probe.py
```

Runner refuses existing output. Final JSON binds probe/tests/document, source
evidence and retrieved primary snapshots by SHA256 after content finalization.
Only bounded synthetic crypto executes; no dependencies, checkpoints or upstream
code are installed, loaded, built or vendored.

Focused tests: **13 passed**. Existing-environment Ruff: **all checks passed**.
System `ruff` was absent; no installation was needed. Final evidence command uses
`--measure-seal --output docs/evidence/research-transciphered-linear.json`.
Audit amendments preserve that real crypto sample: recompute geometry projections
from the saved `seal_downstream_control`, then refresh `source_sha256` after final
probe/test/document edits. No randomized crypto measurement is regenerated merely
to update prose locks. Invalid measured modulus levels are explicitly recorded
as rejected and excluded from every cost projection; stage identity sets must be
unique/equal and original ring/residue contracts must match saved geometry.
