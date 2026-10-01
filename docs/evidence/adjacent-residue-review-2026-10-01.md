# Adjacent residue track: delayed reconstruction, MPC and HE

Date: 2026-10-01. **Decision: no admitted tenfold decoder.** CRT supports exact
bounded integer/rational polynomial islands. It is neither a privacy mechanism
nor an information-width compression. The tested full-capacity, two-source
resident interface fails both matched budgets before conversion/nonlinear work.
Narrow interfaces leave room only conditionally: their private signed lifting,
dynamic scales, rounding and complete-layer correlations remain unspecified.

## Scope, control and evidence

This review owns only this document, `scripts/probe_residue_feasibility.py` and
`tests/test_residue_feasibility.py`. It uses the existing semantic compiler and
cached public configuration; it does not load weights, train, activate a
pipeline, change runtime arithmetic or execute a secure decoder. Arithmetic
inputs are synthetic. Two independent, honest-but-curious, non-colluding online
workers must hold only their own shares. Providers receive no plaintext private
input/activation, and transformer-body computation remains remote. No TEE.
HE below is a distinct cryptographic contract, not an implicit share conversion.

Read contracts: root `NETWORK_IO_10X_RESEARCH.md`,
`python/pllm/runtime/{semantic_numeric,semantic_executor,quantization,semantic_stages,region_contract_cost}.py`,
`crates/pllm-compiler/src/decoder_runtime_schedule.rs`, and
`docs/content/docs/sdk/plans/numerics.mdx`.

Pinned checkpoint: `Qwen/Qwen2.5-0.5B-Instruct` at
`7ae557604adf67be50417f59c2c2f167def9a775`, W8A8, 24 layers, hidden width 896,
intermediate width 4,864, 14 query heads and 2 KV heads. Body fingerprint:
`5d631be30158b3ea2a72cae355ce09a3b304bbfddde48b5758c339c346a34974`.
Composition digest:
`bd91e1659773faabcbb998a350893cbbc6df313a830da7cb7bfb87463c8f0b36`.
Probe rejects plan/schedule/composition mismatch against
[`compiler-region-contract-qwen25-2026-09-30.json`](compiler-region-contract-qwen25-2026-09-30.json).

| Cohort | Covered all-link / online control bytes | All-link / online budgets | Body query rows |
| --- | ---: | ---: | ---: |
| 39+8 | 116,843,966 / 73,831,744 | 11,684,396 / 7,383,174 | 46 |
| 39+32 | 178,970,558 / 113,545,024 | 17,897,055 / 11,354,502 | 70 |

Each response executes one prefill and `generated_tokens - 1` decode steps.
Plan/schedule digests checked by the probe:

- 39+8: `d292d42933e35239655e2a2d09e7bb8ac3c0d29ceb3b96ddd5512836148c82b1`
  / `19d45c42975b80997c85b9c7bf4d594905e485588e7d8dce13b39de3bf478327`.
- 39+32: `62381dbd11470bd150723cfb0d7bab096c52f331ec8e9f5dd23e178df28a0a4d`
  / `bf3bc2e91b6e95a8d0b9ac7eebfb2866a3b3b8c2fe4c2ff401dc625f130a0d5c`.

## 1. What CRT can defer exactly

For pairwise-coprime moduli `m_i`, let `M = product(m_i)`. The ring isomorphism
`Z_M = product(Z_m_i)` preserves addition, public integer multiplication and
polynomial evaluation. Public affine maps can run independently on each
worker's additive share in every channel. Inject public bias once across the
two shares. Secret quadratic multiplication still needs cross terms, e.g.
Beaver/OLE/HE machinery: `(a+b)^2` is not `a^2+b^2`. Small rings do not remove
that work or generate joint attention/MLP correlations.

Exact integer lifting requires a public certified interval `[L,U]` with
`U-L < M`. For a symmetric interval, require `M > 2B`. Equal-width endpoint
domains fail: under an even modulus, `-M/2` and `+M/2` collide. The probe rejects
the entire ambiguous declared domain and residues outside an admitted interval;
it never chooses a nearest lift from observed prompt behavior.

An exact rational region can carry `(secret numerator, public denominator)`:
addition uses a common denominator/LCM; multiplication multiplies denominators;
a polynomial with public rational coefficients can remain a numerator circuit.
Bound the resulting numerator, including biases and residual alignment, before
choosing `M`. Modular inversion of a denominator is not integer division or
rounding. Some channels cannot invert it at all (`2` modulo `16`). Rational
reconstruction from a modular quotient also needs numerator/denominator bounds;
the oracle instead retains the explicit public denominator.

The probe executes linear → quadratic rational → linear channelwise, with
**only final offline oracle reconstruction**, and checks against independent
`Fraction` arithmetic. This is an arithmetic-island witness, not a private
quadratic protocol or a replacement for checkpoint SiLU.

Deferral stops being exact whenever the baseline observes a rounded edge:

- `RN_even(1/2)+RN_even(1/2)=0`, but `RN_even((1+1)/2)=1`.
- `3*RN_even(1/2)=0`, but `RN_even(3/2)=2`.
- `floor(-1/2)=-1`, truncation toward zero gives `0`, ties-to-even gives `0`.
- Multiplication by `2^-1 mod 5` maps `1` to `3`, not rounded integer `0`.

All finite float32 values can be described as dyadic rationals, but exact
rational accumulation with one final rounding does not reproduce stepwise
float32 execution. A universal float32 denominator may require `2^149`, with
large numerators across the exponent range; it does not produce a narrow
fixed-point representation. Private dynamic scales cannot become public
denominators without revealing input-dependent information. A new fixed-scale
quadratic path needs complete numeric/checkpoint admission, not a Q7-island test.

## 2. Private conversion, comparisons and mixed radix

CRT reconstruction of **one party's own random share** is local and permissible.
CRT reconstruction of both shares by a worker reveals the private value. The
oracle explicitly labels reconstruction as offline-only; no intermediate client
reconstruction is licensed to move norms/attention back to the client.

For canonical residues define `M_i=M/m_i`, `v_i=M_i^-1 mod m_i`:

```text
S = sum_i r_i v_i M_i
x_unsigned = S - alpha M,  alpha = floor(S/M)
x_signed = x_unsigned - sign_bit M
x mod t = (S - alpha M - sign_bit M) mod t
```

`alpha`, signed representative selection and remainders are secret functions.
Ignoring them is not exact base extension. Likewise for shares `a,b` modulo
4096 of signed-i8 `x`:

```text
x = a + b - 4096 k,    k in {0,1,2}
```

All 20,480 checked splits at `x ∈ {-127,-1,0,1,127}` require correct carry
determination; the correction is zero for the 131 splits with `k=0`.
Histogram is `k=0:131`, `k=1:20,223`, `k=2:126`. This histogram is public
synthetic evidence only. Publishing actual `k` leaks: `k=0` excludes negative
inputs; `k=2` excludes nonnegative inputs. CRT simply moves this correction into
base-extension/rank/order circuitry. For source modulus 15 and target 17,
shares `(14,2)` of `1` naively extend to `16`; `(14,0)` of `-1` extend to `14`
instead of `16`. Reconstructing each share separately does not fix their secret
sum's wrap.

Mixed radix expresses the canonical value as
`d0 + m0*d1 + m0*m1*d2 + ...`. Garner-style extraction is triangular:
`d_i = (r_i - sum_{j<i} d_j product_{h<j} m_h) / product_{h<i} m_h mod m_i`.
For plaintext or a party-local share this costs roughly `O(k²)` small modular
work; private extraction additionally needs cross-ring digit conversion and
carry/sign handling. Lexicographic high-digit comparison works only after
that private work. It is not a per-channel sign test: `-2` and `1` both have
residue `1 mod 3` but opposite integer signs.

Concrete protocol leads, checked against primary sources:

1. **RNS-MPC lifting/truncation:** Frederiksen et al. [S1], Theorem 1,
   Sections 3–4 and Algorithms 3–6. For `0<=x<pq`,
   `floor(x/p)=p^-1*((x mod q)-(x mod p)) mod q`; this requires the exact
   remainder from `Z_p` represented in `Z_q`. Their `Lift_p→q` opens a masked
   value using a correlated noise pair; `Lift_q→p` also needs a statistical
   pad and constrained no-wrap range. `NoisyTrunc_U` has additive error up to
   `U`; the refinement reduces it to at most one, not ties-to-even parity.
   Even a noiseless pair `rbar=r` gives a wrap error: `p=5,q=17,x=4,r=3`
   opens `2` and lifts to `16`, rather than `4`. The paper's optimized
   preprocessing allows leakage and covert robustness; its headroom loses
   approximately `2log2(n)+s_tilde` usable bits. This is real protocol evidence,
   but not a contract-compatible exact Qwen conversion.
2. **Exact order/conversion components:** edaBits [S2] correlate arithmetic
   integers with secret-shared binary decompositions, covering signed/unsigned
   comparisons, truncation and conversions under dishonest-majority-compatible
   MPC. A possible exact fallback converts channel values to secret bits,
   evaluates CRT/mixed-radix reduction and signed round-to-even circuitry, then
   converts output bits back to channels. This is a compositional design lead;
   no optimized two-worker RNS implementation or concrete byte schedule was
   measured here. All correlated bits, Boolean gates and return conversions
   remain charged obligations.
3. **RNS encrypted sign:** Chartier et al. [S3], Section 4, Algorithm 1,
   combine encrypted channels into scaled estimates, apply functional
   bootstrappings/blind rotations, and combine bootstrapped estimates. They
   explicitly explain that signs of components do not determine integer sign.
   The PDF reports high-probability correctness and sub-100-ms 32-bit sign on
   its laptop parameters, with 128-bit security. This is encrypted cross-channel
   work, not a deterministic exact MPC comparison, not our timing, and not a
   whole-decoder budget. The PDF clarifies a malformed probability phrase on
   its landing page.
4. **Fast RNS base extension/scaling:** Bajard et al. [S4], Halevi–Polyakov–Shoup
   [S5], and Cox–Rower bounds [S6] provide concrete basis-extension/scaling
   algorithms. [S4,S5] operate on ciphertext coefficients; [S6] is modular
   arithmetic hardware. Provider-local manipulation of encrypted coefficients
   is permitted, whereas applying those algorithms to private activation
   shares requires protected rank/order corrections. Their speedups cannot be
   imported as zero-communication two-worker base extension.

### DPF/FSS over small moduli

DPF/FSS [S7–S9] can protect point lookup, comparisons, shifts and offset-gate
families. With a public unary table on one `m_i`-sized channel, evaluation may
be cheap. A masked opening still needs party-local keys, fresh independent
offset/output masks, complete issuance bytes, expansion CPU and burn-on-use.
Any simple full table scan costs `m_i` entries per channel per invocation.

Global sign, rounding, row maximum and cross-ring base extension are generally
not unary functions of each channel separately. A channel with two indistinguishable
inputs requiring different outputs cannot implement that function via its local
table. A joint table has domain `M`, or `M²` for two arbitrary operands; DPF
key compression does not eliminate table evaluation or magically yield efficient
FSS for an arbitrary multivariate function. [S8] makes efficiency conditional
on the gate's offset family. [S9] treats shifts/bit decomposition in power-of-two
rings, and identifies a barrier for its single-round multiply-then-truncate
construction; it does not claim a universal impossibility theorem. Extending it
to odd/mixed CRT channels requires a specified conversion protocol.

Redundant residue checks can detect some errors under an explicit integrity
model. They add capacity/storage/work and **do not hide plaintext**, refresh
shares, authenticate a malicious worker or supply private carry correction.

## 3. Capacity, storage and directed-body accounting

For independently bit-packed channel residues:

```text
capacity = log2(M) = sum_i log2(m_i)
s = sum_i ceil(log2(m_i))                 # bits per party's residue tuple
s >= ceil(log2(M))
two-party shares for N values: 2 ceil(N*s/8) bytes
```

Byte-aligned channels instead use `sum_i 8 ceil(ceil(log2(m_i))/8)` bits.
Four u32 channels occupy 128 bits **per share**, 256 across workers. SIMD/RNS
parallelism is a CPU-layout benefit, not a tenfold reduction in information.
Lossless joint encoding of each party's canonical share can use
`ceil(log2 M)` bits; this is a different share codec, with local CRT/mixed-radix
encode/decode CPU to price. It reveals no secret if it processes only own share.

| Basis | Product M | Capacity bits | Packed channel bits | Byte-aligned bits | u32 storage bits/share |
| --- | ---: | ---: | ---: | ---: | ---: |
| 16,17 | 272 | 8.087 | 9 | 16 | 64 |
| 16,17,19 | 5,168 | 12.335 | 14 | 24 | 96 |
| 251,241,239 | 14,457,349 | 23.785 | 24 | 24 | 96 |
| 251,241,239,3 | 43,372,047 | 25.370 | 26 | 32 | 128 |
| 251,241,239,13 | 187,945,537 | 27.486 | 28 | 32 | 128 |
| 251,241,239,233 | 3,368,562,317 | 31.649 | 32 | 32 | 128 |

W8A8 symmetric quantization uses `[-127,127]`. Bias-free dot envelopes are
`B=d*127²`: for `d=896`, `B=14,451,584` needs at least 25 information bits
for its symmetric envelope; for `d=4,864`, `B=78,451,456` needs 28. These
are coefficient-class envelopes, **not measured minimum widths for fixed
checkpoint rows**. A tighter fixed-weight certificate is
`B_j=127 sum_i |W_ji|`; weights were not loaded here. Bias/dequantization,
floating scales, quadratic numerators and attention require separate bounds.
The 24-bit three-prime basis fails even the 896-wide unrestricted envelope.
The 9-bit basis contains the signed-i8 source, but not its linear output.
The 14-bit basis provides more than 12 bits of range and cannot be transmitted
as three independent residues in 12 bits.

### Existing resident two-source interface, replaced by CRT tuples

This screen reuses semantic attention-query and post-attention-MLP source
lineage, **not stage-name heuristics**. Each worker sends its own masked tuple
to the other. Per-worker source elements:
`N=2*24*896*executed_rows`, i.e. 1,978,368 / 3,010,560. Client ingress remains
24-bit embedding shares to each worker; egress remains 32-bit final-hidden
shares from each worker. Their combined token-boundary bodies are
304,640 / 605,696 bytes. This is an interface scenario, not a protocol lower
bound applying to all MPC/HE layouts.

| Source tuple bits | 39+8 peer + boundary bytes | Online / all-link room | 39+32 peer + boundary bytes | Online / all-link room |
| --- | ---: | ---: | ---: | ---: |
| 9 | 4,755,968 | 2,627,206 / 6,928,428 | 7,379,456 | 3,975,046 / 10,517,599 |
| 14 | 7,228,928 | 154,246 / 4,455,468 | 11,142,656 | 211,846 / 6,754,399 |
| 28 | **14,153,216** | **-6,770,042 / -2,468,820** | **21,679,616** | **-10,325,114 / -3,782,561** |

The tested 28-bit full-dot-capacity basis **fails both budgets on known bodies**.
The 14-bit independent-channel layout leaves only 0.078 / 0.070 bytes per
source coordinate, aggregated across both workers, for all omitted online
work. That is not enough evidence to price even one actual conversion.
Jointly encoding the 5,168-state share takes 13 bits instead of 14; its
optimistic totals are 6,734,336 / 10,390,016 bytes, leaving 648,838 / 964,486
online bytes, with codec CPU unmeasured. This distinction prevents treating
per-channel padding as a universal bound. Neither codec proves narrow sources
can execute wide dots or baseline float32 norms.

**Known:** tuple products/widths, semantic shapes and source counts, duplicated
directed peer/body counts, existing token-boundary widths, public-dot envelopes.
**Unpriced, never zero:** signed base extension; CRT↔binary conversion;
secret maximum/abs/zero handling; exact division, scaling and ties-to-even;
quadratic/private attention multiplications; RMSNorm reciprocal square root;
softmax exp/division; checkpoint SiLU; private KV scale consistency/share
renewal; one-use dealer material to each worker; key/correlation expansion;
controls/authentication/framing/full wire; cold distribution and two worker
checkpoint snapshots; actual independent-worker latency/CPU/peak memory.

For `k` channels, each public MAC becomes `k` small modular MACs per worker;
each shared multiplication needs channelwise material unless a concrete joint
generator proves otherwise. A generic Beaver multiplication comparator opens
two operands in both peer directions: `4 ceil(N*s/8)` online bytes, plus
roughly `6 ceil(N*s/8)` dealer bytes for two parties' three triple shares.
These are costs of that generic construction, not cryptographic lower bounds.
Conversion circuitry adds its own work; it cannot be hidden in a free CRT call.

## 4. Semantic boundaries that defeat whole-layer deferral

The probe checks every declared operator in every layer in both phases, graph
dependency order, complete semantic stage coverage, and locked digests. Dynamic
quantization groups follow compiled `input_ids`: Q/K/V share one input; gate/up
share another. Seven linear operators therefore have **four** dynamic activation
quantization boundaries per layer, not seven and not zero.

The actual W8A8 contract chooses each row's private `max(abs(x))/127` (zero row
uses scale 1), performs float32 division, `rint` ties-to-even and clipping, then
preserves float32 dequantization multiplication order and public biases.
Baseline body operator order is:

```text
RMSNorm -> dynamic quantization -> Q/K/V integer maps -> dequantization/bias
-> public rotary -> shared persistent KV -> private QK dot -> scale by 1/8
-> causal mask -> stable softmax -> probability*V
-> dynamic quantization -> output map/dequantization -> residual add
-> RMSNorm -> dynamic quantization -> gate/up maps/dequantization
-> SiLU(gate) -> SiLU(gate)*up -> dynamic quantization
-> down map/dequantization -> residual add -> next layer
```

The head-dimension-64 scale is rational `1/8`; this does not permit deleting its
float32 rounded edge. RoPE has a `q30.libm` descriptor in the graph, while the
current semantic executor computes float32 trigonometric coefficients and
multiply/add results; a protected candidate must pin actual coefficient and
rounding behavior instead of inferring exact rational semantics from that label.

| Obligation per layer | Sites | Why residue-only local polynomial execution stops |
| --- | ---: | --- |
| Dynamic activation quantization | 4 grouped inputs | private order, abs/max, zero test, divide, round/clip; secret scales |
| Body RMSNorm | 2 | square reduction followed by inverse sqrt, learned scale, float32 output |
| Stable softmax | 1 | private max, masked exp, private sum/division and float32 output |
| SiLU | 1 | private sign branch and exp/division; no exact quadratic equivalence |
| Linear output/dequantization | 7 | signed accumulator lift, int→float32, scale multiplications/bias |
| Rotary | 2 | public coefficients but stepwise floating multiplies/addition |
| QK, attention scale, probability×V | 3 | private products/reductions and rounded scale/output |
| Residual adds and gated multiply | 3 | baseline floating rounding before next norm/quantization |
| Total represented obligation sites | **23** | may fuse cryptographically; baseline rounded values must still be preserved |

These are **23 semantic obligation sites, not 23 plaintext openings or forced
network rounds**. Within each site there can be many rounded scalar operations.
Reshape/cache layout and public causal positions are not falsely counted as
comparisons; masked `-inf` needs a protected representation, not a CRT integer
sentinel that silently changes softmax. KV can remain shared but retains its
original numeric interpretation across decode. A norm/softmax can be implemented
without reconstructing plaintext: it still needs non-residue order/numeric work.

| Compiler-derived response work | 39+8 | 39+32 |
| --- | ---: | ---: |
| Dynamic quantization layer-query rows | 4,416 | 6,720 |
| Body norm rows | 2,208 | 3,360 |
| Softmax head-query rows | 15,456 | 23,520 |
| SiLU scalar elements | 5,369,856 | 8,171,520 |
| Representation sites, counted once per layer per phase invocation | 4,416 | 17,664 |
| Representation sites weighted by query rows | 25,392 | 38,640 |

A straightforward binary tournament for dynamic row maxima uses
`3*(896-1)+(4864-1)=7,548` comparisons per layer-query row, hence
8,332,992 / 12,680,640 comparisons across these cohorts, before softmax maxima
or rounding. This is work for that algorithm, not a universal comparison-gate
lower bound. No key sizes or bytes were assigned to those comparisons.

Required plaintext internal reconstruction count is **zero** under the desired
resident contract. The trusted client reconstructs only 8 / 32 final hidden
boundaries and performs final norm/head/token selection. Token feedback cannot
be deferred across generation steps. The semantic counts price obligations,
not permission to reconstruct intermediate norms, MLP inputs or KV.

## 5. HE: plaintext RNS versus ciphertext RNS

- **Ciphertext RNS** [S4,S5]: a BFV/BGV/CKKS ciphertext has polynomial
  coefficients modulo a large ciphertext modulus `Q=product(q_j)`, represented
  internally across RNS limbs. Basis extension, modulus switching, rescaling
  and key switching act on encrypted coefficients. Reducing limb machine width
  can improve arithmetic throughput while increasing limb count. It does not
  narrow the model's plaintext dynamic range, make CKKS exact, provide plaintext
  comparison, eliminate depth/noise refresh or imply smaller serialized bodies.
- **Plaintext/message RNS**: encode a model integer modulo several `m_i` in
  independent ciphertexts or appropriately separated encrypted slots. Its
  message capacity is `product(m_i)`, and all encrypted channels, packing,
  cross-channel sign/scale conversion, noise and refresh must be counted.
  Encrypting one narrow channel is insufficient to recover an unrestricted
  wide result. Mixed-channel encrypted sign [S3] is a concrete conversion lead.
- **Nested message RNS** [S10]: Boneh–Kim combine three residue layers for
  large prescribed-modulus encrypted arithmetic. Their gains are for large
  integer modular arithmetic, not measured fixed-Qwen attention/SiLU/rounding
  parity or our response traffic. It is genuinely adjacent arithmetic research,
  not evidence that existing ciphertext-RNS limbs compress private activations.

No HE context/key/ciphertext was instantiated by this track. Existing sampled
TenSEAL depth and token-boundary vetoes in the root research contract remain
separate evidence; RNS terminology does not resolve them.

## 6. Reproduction and bounded verification

```sh
uv run --no-sync pytest -q tests/test_residue_feasibility.py
uv run --no-sync python scripts/probe_residue_feasibility.py
/usr/bin/time -l uv run --no-sync python scripts/probe_residue_feasibility.py --semantic
uv run --no-sync ruff check scripts/probe_residue_feasibility.py tests/test_residue_feasibility.py
git diff --check -- docs/evidence/adjacent-residue-review-2026-10-01.md scripts/probe_residue_feasibility.py tests/test_residue_feasibility.py
```

The arithmetic probe verifies 2,025 exact signed affine cases, 225 rational
quadratic cases, 225 delayed linear→quadratic→linear cases, 225 modular
public-linear share cases, 5,168 full-domain mixed-radix cases, and 20,480
signed narrow-share carry splits. Tests separately enumerate signed endpoints,
malformed/noncoprime bases, ambiguous lifts, all small share splits, negative
ties, denominator nonunits, lost quadratic cross terms, base-extension/sign
counterexamples and the exact RNS quotient identity versus noisy lifting.
Independent oracles use Python exact integers and `Fraction`, not floating
approximations or CRT reconstruction reused as the expected result.

Final measured cached-config probe: 0.67 seconds wall,
0.61 seconds user + 0.05 seconds system, 65,306,624 bytes maximum resident set
size on this macOS host. This is well below 1 GiB. No model weights or large
tables allocated. Timing is local feasibility-screen timing, not independently
distributed protected execution. Final verification:

- `pytest -q tests/test_residue_feasibility.py`: **20 passed in 0.16s**.
- Both default arithmetic and `--semantic` executions passed all internal
  exact-arithmetic assertions; both pinned cohorts passed digest checks.
- `ruff check` on the two new Python files: **All checks passed**.
- Scoped status: exactly these three paths are new/untracked. Existing work
  remains present. `git diff --check` passed; because new files are untracked,
  a separate `git diff --no-index --check /dev/null <path>` checks their content.

## 7. Strongest falsifiable reopening gate

**Reject the tested wide resident CRT interface now.** Narrow CRT alone is not
a positive gate: it preserves the exact same secret carry/range problem and
adds cross-channel numeric work. Literature approximate/noisy truncation is not
an exact or privacy-contract-preserving replacement.

Reopen only with one concrete two-worker complete-layer protocol that:

1. Binds all admitted source/output ranges, scales, checkpoint coefficients and
   rounded semantic edges; proves exact signed lift and rounding including
   carries 0/1/2, negative endpoints, zero norms/rows, ties and masked softmax.
   No input-dependent public fallback, order disclosure or intermediate client
   reconstruction. Each provider holds only own fresh shares/material.
2. Specifies protected CRT base extension/order/scale conversion and private
   multiplication, not just local residue arithmetic. Give both worker views,
   seed independence, one-use cancellation/replay burn and measured serialization.
3. Fits **both** cohort budgets after all directed peer, client and dealer
   bodies, setup, keys and complete controls. For the tested independent
   14-bit tuple schedule, all remaining online work must fit 154,246 / 211,846
   bytes, all remaining all-link work 4,455,468 / 6,754,399 bytes. An admitted
   13-bit share codec would change these ceilings but must be implemented and
   measured. Using the existing hypothetical 12-bit ring schedule instead
   gives only 1,143,430 / 1,717,126 online bytes; do not mix that floor with a
   14-bit CRT tuple. Cold checkpoint/key delivery is separately reported.
4. Preserves complete-layer local numeric parity before protected execution,
   then passes held-out prefill, same-token decode and free-generation checks
   for fixed weights, with most body MACs remote and bounded CPU/memory.

This gate can be falsified by the first excess body count or rounding/range
counterexample. No universal impossibility of RNS-MPC/HE is claimed; no tested
construction here currently meets the complete fixed-model target.

## Primary sources retrieved 2026-10-01

- **S1:** Frederiksen, Lindstrøm, Madsen, Spangsberg, *A New Approach to Efficient
  and Secure Fixed-point Computation*, ACNS 2024.
  <https://eprint.iacr.org/2024/035>;
  <https://eprint.iacr.org/2024/035.pdf>. Landing page fetched with `webfetch`;
  PDF examined (Theorem 1, Algorithms 3–6, Sections 4.2–4.4).
- **S2:** Escudero, Ghosh, Keller, Rachuri, Scholl, *Improved Primitives for MPC
  over Mixed Arithmetic-Binary Circuits*, CRYPTO 2020.
  <https://eprint.iacr.org/2020/338>. Primary abstract checked; no concrete
  RNS-specific implementation costs imported.
- **S3:** Chartier, Koskas, Lemou, Méhats, *Homomorphic sign evaluation with a
  RNS representation of integers*, 2024.
  <https://eprint.iacr.org/2024/156>;
  <https://eprint.iacr.org/2024/156.pdf>. Landing page fetched with `webfetch`;
  PDF Algorithm 1/Section 4 and correctness discussion examined.
- **S4:** Bajard, Eynard, Hasan, Zucca, *A Full RNS Variant of FV like Somewhat
  Homomorphic Encryption Schemes*, SAC 2016.
  <https://eprint.iacr.org/2016/510>. Primary abstract checked.
- **S5:** Halevi, Polyakov, Shoup, *An Improved RNS Variant of the BFV
  Homomorphic Encryption Scheme*, CT-RSA 2019.
  <https://eprint.iacr.org/2018/117>. Primary abstract checked.
- **S6:** Bajard, Merkiche, *Double Level Montgomery Cox-Rower Architecture,
  New Bounds*, 2014. <https://eprint.iacr.org/2014/440>.
  Primary abstract checked; hardware bounds are not MPC privacy protocols.
- **S7:** Boyle, Gilboa, Ishai, *Function Secret Sharing: Improvements and
  Extensions*, CCS 2016 version. <https://eprint.iacr.org/2018/707>.
- **S8:** Boyle, Gilboa, Ishai, *Secure Computation with Preprocessing via
  Function Secret Sharing*, TCC 2019. <https://eprint.iacr.org/2019/1095>.
- **S9:** Boyle et al., *Function Secret Sharing for Mixed-Mode and Fixed-Point
  Secure Computation*, 2020. <https://eprint.iacr.org/2020/1392>.
  Primary abstracts for S7–S9 checked; no byte figures transplanted.
- **S10:** Boneh, Kim, *Homomorphic Encryption for Large Integers from Nested
  Residue Number Systems*, CRYPTO 2025. <https://eprint.iacr.org/2025/346>.
  Primary abstract checked; large-integer performance claims not reproduced.
- **Historical mixed-radix provenance:** Garner, *The Residue Number System*,
  IEEE Transactions on Electronic Computers EC-8(2), 140–147, 1959.
  <https://doi.org/10.1109/TEC.1959.5219515>. Publisher-deposited title/author/
  DOI metadata checked through Crossref; full historical paper not inspected.
