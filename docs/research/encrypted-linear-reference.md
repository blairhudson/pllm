# Bounded encrypted linear foundation

`pllm.protocols.PackedEncryptedLinear` is an independently implemented diagonal
SIMD CKKS matrix operator. Rust owns public packing, parameter/resource admission
and source identity; SEAL executes cryptography as a separate optional dependency.
Python owns public configuration, key/role orchestration, bounded frame delivery
and independent float64 oracles. Upstream paper code is not a runtime dependency.

## Source correction

The pinned NEXUS NDSS 2025 source (`papers/r21-nexus.pdf`, SHA-256
`c809182345f27f76e21d983d97dffa0d086dbcd17dabc265f0df9fb6ca9b70f2`) uses
RNS-CKKS throughout: ciphertext/slot operations, coefficient packing, encrypted
polynomial nonlinear operators and bootstrapping. It does **not** use garbled
circuits. Earlier PLLM HE–GC summaries were incorrect and are corrected, including
the unavailable `NexusCkksBootstrap` method replacing the inaccurate conversion
placeholder. The source reports BERT-base and Llama-3-8B one-output forward
workloads; those do not establish KV-cached Qwen generation or private token
feedback.

This operator cites NEXUS Section II only for background HE primitives. Its
diagonal-slot layout is separate from NEXUS Section III's coefficient-packing and
offline-online construction. MOAI's full source remains unavailable through the
OpenReview browser challenge; neither this operator nor its measurements claim a
MOAI reproduction. SEAL has no CKKS bootstrap implementation. Encrypted depth is
finite and no secret-key refresh or intermediate decryption substitutes for it.

## Contract

- Public matrices and private vectors are finite float64 values in `[-1,1]`.
- Rectangular shapes are 1–32 by 1–32. Rust pads to the next power-of-two width
  and emits public diagonals. The client repeats its padded vector across slots.
- One client-owned secret key per sample; the provider loads parameter, public
  and Galois-key frames into its own public-only context. Keys are reused across
  1–64 fresh encrypted queries in the declared sample.
- `rotation_keys="single"` uses one sequential one-slot Galois key; `"full"`
  rotates the original query with one key per shift. Neither changes the public
  float64 numeric contract; CKKS errors and complete costs are measured separately.
- The linear path uses degree 8,192, coefficient bits `[50,40,50]`, scale `2^40`
  and one rescale level. SEAL TC128 validation is mandatory. Generic foundation
  admission supports up to eight levels, chooses degree 16,384 when needed and
  checks a 512 MiB combined allocation estimate before material.
- Every multiplication rescales; ciphertext products additionally relinearize.
  Alignment explicitly modulus-switches operands. Tiny scale normalization is
  approximate and included in decoded-output error. Exhausted depth rejects.
- The client decrypts only final ciphertexts. A separate `math.fsum` dense-product
  oracle checks a `0.0002` absolute tolerance. This is a result check, not a proved
  universal error bound or exact integer parity.

Public padding bounds every serialized key/ciphertext frame. A request binds the
fresh session, configuration, ordinal and direction. Wrong contexts, replay,
malformed lengths/padding, wrong levels/scales and cancellation retire the session.
The checked arithmetic also composes ciphertext multiplication, relinearization,
alignment and plaintext products without intermediate decoding. These in-process
owners and framing are not authenticated role transport, a malicious-result proof,
circuit privacy or independent cryptographic review.

## Accounting

`setup_frame_bytes_by_operation` charges parameters, public keys and Galois keys
once per sample. Query/reply frames include their 81-byte context headers and fixed
public padding. Provider evaluations include deserialization, encrypted arithmetic
and reply serialization; complete local online time also includes client encoding,
encryption and final decoding. Key generation/serialization and public layout
construction are separate timings.

Raw key fields price the client secret polynomial and padded public/evaluation
representations. They do not count a SEAL context as a serialized secret key or
claim measured memory residency. Native resource admission separately prices
context/RNS scratch, key copies, delivery overlap, live ciphertexts and pool slack.
Public matrix/layout payloads are separate; model distribution is unmeasured.
Process CPU includes Python and SEAL orchestration. Process RSS, physical wire,
multi-host latency, checkpoint quality and protected decoder execution are unknown.

## Measured bounded cohort

Eighteen fresh processes check 1,224 decoded outputs from public 3×5, 16×16 and
32×32 matrices, four queries per process and both key layouts. The maximum
absolute error against the independent float64 oracle is `9.37e-7` (rounded up).
At 32×32, the single-key layout reduces setup frames from 25.16 to 1.20 MB and
median issuance from 0.2834 to 0.0266 seconds. Online frames remain 1.606 MB for
four queries; median local online time is 0.2301/0.2289 seconds for full/single
keys. The public matrix/layout occupies 16,384 bytes separately. These are local
operator costs and observed errors, not full-model memory or latency results.

## Reproduction

```sh
uv run --no-sync python -m benchmarks.research.encrypted_linear \
  --output NEW-encrypted-linear-cost.json
uv run --no-sync python examples/benchmarks/encrypted_linear.py \
  --output NEW-encrypted-linear-sdk.json
```

`docs/evidence/research-encrypted-linear-cost.json` retains fresh-process public
3×5, 16×16 and 32×32 matrix controls with four queries each and both key layouts.
`docs/evidence/research-encrypted-linear-sdk.json` retains installed-wheel replay.
Complete operator availability does not activate an Experiment or satisfy the
remaining MOAI/NEXUS/THOR operator, state, bootstrap or model-quality contracts.
