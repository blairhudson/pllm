# Bottom-bit-only fixed-W correction: bounded conditional result

## Verdict

**Exact one-bit replacement is algebraically possible with an ordinary trusted
offline dealer still computing every high correction bit. Security parameters
and Low qualification remain unknown. Reject 20% savings for this selected
subset even with free setup/expansion.** This is not a full-ring client-light
generator, new serving measurement, or impossibility theorem for all generators.

The proposed `P=(W mod2)A` trick computes only parity. It cannot supply the carry
into the retained high correction. Simply retaining high material prepared for
an independent high mask is wrong. Nothing here programs arbitrary checkpoint W
into an existing PCG theorem.

## Precise hypothetical protocol and custody

For each source-bound stage, let W be its original public signed-i8 matrix,
input ring `Z/(2^k)` and original output residue widths `h_j<=k`. Let public
binary `A` have shape `n by d`, and publish `P=(W mod2)A`, shape `m by d`.
For each fresh ticket:

1. Client and trusted noncolluding Preparation receive private material for fresh
   `u in GF(2)^d`, noise vector e from the specified distribution, and independent
   uniform high input mask `v in (Z/(2^(k-1)))^n`. Inference must not receive their
   seeds, u, e, support or resulting low mask. Define `l=A*u+e mod2`, `r=l+2v`.
2. Independent role material lets Client, Preparation and Inference expand
   `q_low in GF(2)^m`. Inference can own this seed; it cannot derive r from it.
   Client and Preparation separately expand uniform independent high output
   shares `s_high,j mod2^(h_j-1)`. Inference must not own that seed.
3. Client computes `a=(P*u+(W mod2)*e) mod2`, then
   `s_low=a-q_low mod2`, and `s=s_low+2*s_high`.
4. Preparation computes original exact `Wr`, and delivers only
   `q_high,j = floor(((Wr-s) mod2^h_j)/2)` to Inference. Inference combines
   `q=q_low+2*q_high` in each original output residue ring.
5. Client sends original full-ring `D=x-r`. Inference returns original
   `WD+q`; Client adds s. Existing numeric contracts, placement and verifier
   remain unchanged. Output and opening widths do not shrink.

This replaces the low bit of the **Preparation-to-Inference correction body**,
not low bits of every transport. Preparation must know the low/private mask
material to prepare the joined high frame. Client still expands full high masks
and high output shares. No dealer CPU saving is claimed. Seed issuance, role
delivery, source binding and extra controls are not free; reusing an existing
seed requires proven independent domains and appropriate existing custody.
There is no online second evaluator here. Dealer/client/inference collusion is
outside this privacy contract.

### Exact carry and borrow

With signed integer `t=W*l`, per output:

```text
Wr mod2^h = (t mod2) + 2*(Wv + floor(t/2)) mod2^h
s_low = ((t mod2) - q_low) mod2
q_high = Wv + floor(t/2) - s_high - [t mod2 < s_low] mod2^(h-1)
```

Signed floor division matters. Counterexamples:

- W=`[1,1]`, l=`[1,1]`, v=0, q_low=s_high=0: correct q=2;
  independent-high-only join gives 0.
- W=`[1]`, r=0, q_low=1, s_high=0 in ring eight: s_low=1,
  q=7 and q_high=3; omitting borrow gives q_high=0.

Preparation can compute this bridge normally. Making Client compute
`floor(Wl/2)` instead brings back full-precision work and weight publication;
`P*u` does not provide it. Multi-bit replacement needs its own carry/borrow
construction and security analysis, not repetition of independent GF(2) slices.

## Joint view and security gate

In the ideal uniform-r experiment, for any fixed q_low, independent uniform
s_high makes q_high uniform and independent of r. Thus Inference's entire
`(A,P,q_low-seed,D,q_high,WD+q)` view reduces to the distribution of the opening
D plus independent pads. Returned value is a deterministic view function.
Independent uniform v makes all high bits of D uniform even after conditioning
on its low bit: subtracting l creates a borrow, but translates uniform v rather
than exposing it. This argument requires **unobserved independent s_high**;
conditioning on that private seed invalidates it.

Replacing uniform l by `Au+e` requires computational pseudorandomness for the
**complete multi-sample noisy-code distribution `(A,{Au_i+e_i})`**, with actual
dimensions, seeded A ensemble, noise law, fresh secrets, horizon and source
bindings. P is a deterministic function of already-public W,A, so does not add
information under that assumption; it does not make the assumption true.
Transpose/code formulations and ordinary LPN parameter statements are not
interchangeable without their reduction and concrete bounds. Bernoulli noise
and exact-weight noise are different distributions. No applicable primary-source
parameter certificate was established here: d, noise, security bits and allowable
query count are **unknown**, not "128-bit". Primary-source PDF/text hashes from
the existing full-text review are retained as provenance, not a binary-LPN
certificate. Its exact-ring PCG and matrix-triple parameters cannot be imported.

Freshness must cover u, e, v and s_high at every ticket and every retry. Shared A
requires one joint budget across all stages/queries, including adaptive inputs
and exposed prior openings. q_low may be a PRG output from Inference's own seed,
independent of private seeds; its secrecy from Inference is not needed. Full
input-mask reuse reveals input differences. Reused e with fresh u leaks dual-code
projections of input differences. No noise plus deficient public A exposes input
parity, matching the `PublicSubspaceMaskRegression` failure mode. Public sparse
noise support is not safe seed custody. Authentication and malicious-result
soundness remain the original verifier's separate obligations.

## Deterministic tiny checks

`scripts/probe_binary_correction_slices.py` executes:

- 131,072 exact ring output checks: two-coordinate signed weights/inputs,
  every ring-eight input mask and every low/high output share choice; oracle is
  independent signed integer dot product reduced modulo eight.
- 2,304 binary image checks across all tiny A,u,e, signed weights.
- All 64 input distributions for full `(D,q)` in n=2, ring eight,
  W=`[1,-1]`: uniform-low/noise control has TV=0.
- Same complete view with A=`(1,1)^T`, fresh uniform scalar u and independent
  Bernoulli(1/4) noise: neighboring-input TV=1/4. This intentionally insecure
  tiny example rejects security certification, not real-size LPN.
- Revealed low mask gives TV=1; revealed high output share gives TV=1/2.

Independent tests additionally cover signed-i8 endpoints with heterogeneous
16/17-bit output residues, carry/borrow counterexamples, exact mask-reuse
differences, noise-reuse dual witness, two fresh scalar queries (all 16 input-pair
distributions, 256 views each), byte rounding, renewal/burn boundaries and pinned
source drift. Toy probabilities are exhaustively counted, not Monte Carlo.
No protected primitive, security parameter estimator or process benchmark runs.

## Conditional byte ceiling and complete cost

Pinned selective geometry uses 23 bulk gate/up stages, each `9728 by 896`.
Mandatory correction bits can lose at most one bit/output lane. For one packed
frame with B bits and L replaced bits, exact removal is
`ceil(B/8)-ceil((B-L)/8)`, at most `ceil(L/8)`. Fragmented frames, envelopes,
padding and low/high stream descriptors need their own attribution; no wholesale
correction-residual removal is credited. New q_low seed/frame traffic is extra.

| Workload | Lanes/pair | Optimistic removed bytes S | Free first-pair saving | Free warm saving |
| --- | ---: | ---: | ---: | ---: |
| 150 unplaced | 33,561,600 | 4,195,200 | 0.850% | 1.138% |
| 150 placement | 33,561,600 | 4,195,200 | 0.889% | 1.326% |
| 288 unplaced | 64,438,272 | 8,054,784 | 1.004% | 1.189% |
| 288 placement | 64,438,272 | 8,054,784 | 1.093% | 1.386% |

Twenty-percent complete-cost allowance is negative at horizons
1,2,4,16,64,4096 in every cohort; even the infinite warm limit rejects it.
The earlier 24.30–37.44 MB allowance concerned replacing **all** selected
correction bits, not this one-bit slice.

For R pairs, original public C and recurring T, additional all-owner candidate
traffic E(R), positive net saving requires `E(R)<R*S`; target g requires
`E(R)<=R*S-g*(C+R*T)`. This is a necessary optimistic byte gate, not measured
serving gain. Retained high correction/setup remain in the baseline; any
additional high preparation traffic belongs in E, never counted twice or omitted.
With ticket capacity Q, burns b, N live rows, each fresh pool costs delivery D:
`pools=ceil((N+b)/Q)`, `E=setup+pools*D+(N+b)*per_row+other`. Per-stage pools,
unused slots, renewal overlap and authenticated role delivery must be priced.

Explicit dense publication costs for a shared bulk A and separate P per W:

```text
A: ceil(896*d/8) bytes
23 P: 23*ceil(9728*d/8) bytes
A+P: 28,080*d bytes  (before headers, copies and multiple owners)
23 binary W: 25,059,328 bytes
all original 96-stage binary W: 44,728,320 bytes
```

If all selected binary W must be newly delivered, W alone exceeds either first
pair saving. Ignoring all other costs, positive saving needs at least six
150-token pairs or four 288-token pairs. With dense A+P and W, necessary strict
break-even is `R > (25,059,328+28,080*d+other_setup)/S`, plus recurring costs.
Already-owned parity columns could lower publication, but their presence and
lifetime are not established by the existing public-bundle byte total.

For a noise vector of t nonzeros, uncached packed-column access costs
`t*ceil(m/8)` logical W bytes per stage/row; streamed P costs `ceil(m*d/8)`;
dense A expansion costs n*d bit operations and dense P expansion m*d. Binary
W should be column-oriented for this access pattern; transpose/index publication
and scratch are extra. Naive setup W*A costs m*n*d bit operations. Private u/e
sampling, PRGs, support indices, high mask/share expansion, ordinary dealer W*r,
authentication and all-owner creation/delivery are additional. Bernoulli t is a
random count, not a silently fixed capacity. Disk spills trade RAM for logical
reads; caches/page owners and multiple mapped representations stay charged.
Across a bulk pair, constant t gives `t*S` uncached W-column read bytes and
`d*S` dense P read bytes. These are logical local reads, not network bytes or
performance lower bounds; caching/vectorization can change them. They expose
read amplification relative to the single saved correction bit.

256 MiB is whole-client, not an artifact admission threshold. No value of d/noise
or lifetime peak RAM, extra disk, CPU, device I/O or serving gain is established.
Original body remains remote; additional client bit work must be reported
separately. No performance/RSS/native/checkpoint runs launched amid workers.
Real geometry experiments require parent-only serialized opt-in `fresh_workers`.

## Replay and unresolved gates

```sh
target/frontier-sdk-env/bin/python scripts/probe_binary_correction_slices.py \
  --output target/binary-correction-slices-fresh.json
target/frontier-sdk-env/bin/python -m pytest --noconftest -o pythonpath=. -q \
  tests/test_binary_correction_slices_probe.py
```

Output must be new; existing files reject overwrite. Replay JSON must match
`docs/evidence/research-binary-correction-slices.json` byte-for-byte when own code
and locked inputs match. Evidence pins both source JSON hashes, own script/test
hashes, the historical cartridge reference hash, and existing primary-source
PDF/text hashes. It does not read/download
those PDFs anew or certify their parameters. Unresolved promotion gates: reviewed
source/shape-bound binary security parameters and finite horizon; authenticated
seed custody/burns; retained high bridge integration; exact frame attribution;
all-owner setup/renewal cost; measured expansion and <=256 MiB whole-client peak;
unchanged numeric/verifier identity and matched serving measurements.
