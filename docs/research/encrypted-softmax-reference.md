# Bounded encrypted softmax

`pllm.nonlinear.PackedEncryptedSoftmax` independently composes source exponential
primitives with encrypted reciprocal and complete normalization. Rust admits
parameters and allocations; SEAL performs ciphertext arithmetic; Python owns
immutable configuration, distinct client/provider contexts and independent oracles.
Only the client decrypts, after the final normalized ciphertext. No upstream code
is linked or executed.

## Source and numeric contracts

- `method="repeated_squaring"` uses NEXUS Equation (4):
  `v_i = (1 + (x_i - 1/8)/256)^256`. The constant public shift replaces private
  maximum evaluation. Three unrelaxed Goldschmidt iterations normalize `v/n`.
- `method="square_normalize"` uses THOR Appendix C's degree-15 exponential and
  Algorithm 3's square/normalize structure with `delta1=1`, `delta2=2`.
  It evaluates `p15(x/2)`, normalizes, squares the probabilities, then normalizes
  again. Both reciprocals use two unrelaxed iterations. Balanced powers replace
  a depth-15 Horner ciphertext schedule.

These narrow-domain choices adapt the papers. They do not implement their full
relaxation schedules, secure maximum or bootstrap. Source hashes and exact sections
appear in reports; numeric identities differ between methods. NEXUS is an all-CKKS
system, not an HE–garbled-circuit boundary. THOR's coefficient source is the pinned
`papers/thor.pdf`, SHA-256
`ddb57004b56ee8e00beffc4f563fd0921242753fda6c704065d4900b792439cd`.

Inputs are caller-owned or public scores in **[-1/8,1/8]**, checked before any
secret key material. Width is one of 1, 2, 4, 8, 16, 32; lanes count 1–64 serial
rows. There is no clipping, private range fitting, or packed causal mask. Each row
repeats across all SIMD slots, allowing sequential one-slot rotations with a single
Galois key. Input values, decrypted probabilities and their hashes do not enter
reports. Aggregate errors may depend on caller input; replay needs those inputs
again if they were supplied.

### Public reciprocal bounds

For `u` in `(0,2)`, the product `(1+e)(1+e²)...(1+e^(2^(k-1)))`, `e=1-u`,
equals `(1-e^(2^k))/u`. It uses only ciphertext products and public constants.
For repeated squaring, every positive `v_i` and their mean lie in
`[(1-1/1024)^256,1]`, inside `(0,2)` independently of private scores.

For the THOR polynomial at `|x/2| <= 1/16`, summing absolute nonconstant terms
places its output conservatively in `[0.93,1.07]`. The first reciprocal residual
has magnitude at most `0.07^4 = 0.00002401`. After squaring, the second reciprocal
input is `n * sum(probability_i²)`. Its lower bound is
`(1-0.00002401)^2`; the positive-vector inequality
`n*sum(v_i²)/(sum(v_i))² <= (a+b)^2/(4ab)` gives an upper bound below `1.005`
for `a=0.93`, `b=1.07`. It too lies in `(0,2)`.
These are analytic profile bounds, not universal CKKS error bounds. Decoded results
are checked separately against 60-digit Decimal Horner/direct-power profile and
Decimal `exp/sum` softmax oracles. Reports expose profile error, true softmax error
and probability-mass error separately. CKKS is approximate; exact-integer count
stays zero. The observed-result tolerance is `0.0002`.

## Complete depth and ownership

Both schedules consume fourteen rescale levels, including normalization and all
products. SEAL validates TC128 at degree 32,768, coefficient bits
`[50] + [40]*14 + [50]` (660 total), and scale `2^40`. The last prime is for key
switching. The ordinary eight-level/512 MiB linear admission stays separate;
this explicitly selected schedule preflights contexts, key copies, delivery
overlap, scratch and 32 live ciphertexts against a **2 GiB combined allocation
ceiling**. This is neither process peak RSS nor a client-memory improvement.

Each sample has a fresh client key, while its serial rows reuse public evaluation
keys with fresh randomized ciphertexts. The provider imports parameter, public,
Galois and relinearization frames into a public-only context. Final-level replies,
sequence, binding and bounded padding are checked; malformed/replayed/cancelled
sessions retire. Polynomial powers retire before reduction temporaries, and
completed sample owners are released before the next key setup without relying
on cyclic garbage collection.
Local framing does not authenticate transport, provide circuit privacy or verify
a malicious evaluator. Independent cryptographic review remains absent.

Setup counts every padded public/evaluation-key frame once per sample. Online
counts include context headers and complete encrypted query/reply frames. Provider
evaluation includes frame decoding, homomorphic operations and frame encoding;
complete local online time additionally includes client encryption and final
decryption. Every rotation, multiplication, relinearization, modulus switch and
rescale is counted. Secret-polynomial and public/evaluation-key payload owners
are separate from the 128-byte public THOR coefficient table. Physical wire and
measured peak memory remain unknown.

## Decoder boundary

This complete softmax operator does not execute attention QK/value products,
causal masks, normalization layers, SiLU, private persistent state or token
feedback. SEAL cannot bootstrap CKKS. The existing eight-prompt Qwen diagnostic
already rejected the broader historical [-16,16] range; that evidence retains its
original scope. A separate fresh twelve-prompt pinned W8A8 range audit now checks
this operator's [-1/8,1/8] contract across twelve prefills and 36 decode positions.
All 48 positions exceed the range: 3,521,405 of 3,559,248 valid scores are outside,
with maximum absolute score 1,701.7544. Masked negative-infinity entries are excluded.
The unchanged clear decoder alone executes, with zero observed swap growth;
no encrypted candidate, clipping or private range fitting occurs. This is a range
rejection, not a candidate-quality result. Decoder Experiments reject the component.
MOAI full-source access remains blocked; this work supplies no MOAI-specific claim.

## Measured bounded costs

Eighteen fresh processes check 738 outputs across width 1/8/32 and both profiles.
Worst true-softmax error is `3.21e-5`, worst profile error `3.27e-5`, and worst
probability-mass error `4.30e-5` (rounded up). At width 32, both profiles send
260.19 MB of setup frames and 25.19 MB online for three rows. Median complete local
online time is 3.56 seconds for repeated squaring and 8.38 seconds for
square/normalize, with approximately 4.5 seconds setup each. Their allocation
estimate is 1.334 GB. Width one omits rotation keys and uses 134.29 MB setup;
ciphertext frame sizes and online bodies are unchanged.

Different approximations retain different numeric identities. These observed
synthetic costs do not compare full paper systems, establish a general winner,
measure model quality, or validate actual process memory.

## Reproduction

```sh
uv run --no-sync python -m benchmarks.research.encrypted_softmax \
  --output NEW-encrypted-softmax-cost.json
uv run --no-sync python examples/benchmarks/encrypted_softmax.py \
  --output NEW-encrypted-softmax-sdk.json
uv run --no-sync python examples/benchmarks/encrypted_softmax.py \
  --method square_normalize --output NEW-encrypted-softmax-thor-sdk.json
uv run --no-sync python -m benchmarks.research.encrypted_attention_range \
  --output NEW-encrypted-attention-range-qwen25.json
```

`docs/evidence/research-encrypted-softmax-cost.json` retains fresh-process width
1/8/32 controls, three rows per process and both numeric profiles. The two
`research-encrypted-softmax-*-sdk.json`/`research-encrypted-softmax-sdk.json` reports
retain installed-wheel replay. These are synthetic bounded operator measurements.
`docs/evidence/research-encrypted-attention-range-qwen25.json` separately retains
the source-locked clear-checkpoint range audit and its exact public configuration.
