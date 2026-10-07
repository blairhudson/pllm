# FastQuery packed embedding reference

## Source and implementation scope

Primary source: *FastQuery: Communication-efficient Embedding Table Query for
Private LLM Inference*, DAC 2024, Sections 4.1–4.3. The checked source is
`papers/fastquery.pdf`, SHA-256
`5a242b1a65ceaaaa256a3f02fd0d3cc6eb40c249ca39c0e645502176520711be`.
The source paper is specification only; no upstream implementation is linked,
vendored or run. SEAL/TenSEAL remains the separate existing crypto dependency.

The installed `pllm.protocols.PrivateEmbeddingLookup` independently implements
one-hot-aware negacyclic coefficient packing at degree 4,096 and plaintext
modulus 8,192. A single 50-bit coefficient prime passes SEAL's TC128 parameter
validation; every decoded response must retain positive noise budget. This is
parameter admission, not independent security review.

An encrypted one-hot monomial selects one row. The provider multiplies public
table polynomials and masks **every** response coefficient with independent
OS-random residues, so unselected convolution coefficients are not opened.
Distinct provider/client contexts exchange bounded padded SEAL serializations;
the provider receives no client secret key. It receives the full encrypted query,
not a plaintext token ID or token-indexed table gather.

Signed layouts use one 8-bit channel, two 4-bit channels, or three 4/3/3-bit
channels, each with a guard bit. A bounded half-gates circuit combines the two
13-bit shares internally, extracts and sign-extends the signed fields, and
subtracts independent ring32 output masks. Evaluator outputs remain opaque until
the trusted benchmark decodes its output share. Sharewise extraction is invalid:
the native regression exhibits cross-slot borrow errors. Fresh issuance identity
and non-cloneable state reject foreign labels, replay and malformed inputs;
failure/drop consumes and retires material. The local issuer sees both input
shares and is not a distributed production protocol.

**Missing paper components:** FastQuery's PackLWEs output consolidation, SiRNN
distributed bit-width extension, frequency-weighted STE fine-tuning and provider
transport. Full ciphertext replies and local half-gates conversion are explicit
adaptations. They cannot establish FastQuery's paper-system costs or quality.
Honest one-hot inputs remain an assumption; no malicious-query proof or reviewed
circuit-privacy construction is claimed.

## Bounded cost evidence

`docs/evidence/research-packed-embedding-cost.json` retains nine fresh processes,
three independent samples each: 17×7, 256×64 and 4,096×64 tables under all three
layouts. All **1,215** returned integers match an independent row-lookup oracle.
Each sample has a fresh HE keypair, fresh complete output masks and serial fresh
conversion circuits. The public integer fixtures differ across bit-widths, so
their costs compare layouts rather than a common model numeric identity.

Public geometry and input bounds precede allocation. The SDK limits vocabulary
to 4,096, width to 256 and samples to 16, with a conservative combined 128 MiB
allocation estimate. Some combinations reject even inside those dimension
limits. Each conversion separately admits under a 4 MiB estimate. These bounds
are not measured RSS or an OOM proof.

The following **symbolic, non-executable** projection uses 151,936 vocabulary
rows, 896 features and 70 executed lookup rows for historical 39+32 Qwen geometry.
It would encrypt all 38 vocabulary blocks, multiply every provider block and sum
their matching outputs. It is outside the native one-block admission and is not
evidence of a full-width noise budget or successful decoder execution.

| Layout | HE query/reply frames | Both-party conversion keys | Conversion boundary labels |
| --- | ---: | ---: | ---: |
| Int8 | 4.553 GB | 363.27 MB | 90.32 MB |
| Int4 | 2.369 GB | 310.59 MB | 77.27 MB |
| Mixed 4/3/3 | 1.643 GB | 293.35 MB | 73.00 MB |

Those are distinct owners/scopes, not a complete wire total. Context/key setup,
public circuit metadata, authentication, model distribution and the rest of the
decoder are extra. The benchmark uses fixed 69,632-byte public-key/ciphertext
frames and a 512-byte parameter frame, including SEAL serialization and padding.
HE frames are measured from the bounded operator. Material sizes come from its
actual converter; full-width counts are geometry projections. Raw provider table
codes and packed coefficients are additional retained owners. No measured
full-response client/aggregate memory result follows.

## Source-only numeric and ownership diagnostic

`benchmarks/research/packed_embedding.py --mode quality` locks the existing
compiled Qwen2.5 W8A8 source, reconstructs its complete per-token embedding table
in bounded 512-row float32 chunks and fits public per-channel statistics. It
freezes Int8, Int4 and mixed 4/3/3 profiles before evaluating twelve new public
prompts. No prompt participates in fitting.

Mixed precision assigns four bits to the highest-mean-absolute-value third of
channels, using stable original-order ties; the rest use three bits. Public scales
round to the nearest power of two; scalar codes use ties-to-even and explicit
symmetric saturation. Rust statistics/quantization are checked against independent
NumPy oracles. The source is already W8 quantized, and this diagnostic restores
channel order before the original first normalization and retains the original
body/head. It is not the paper's BF16-to-quantized training procedure or a
protected checkpoint run. Twelve prompts cannot establish representative quality.

`docs/evidence/research-packed-embedding-quality-qwen25.json` records the fresh
cohort. The untouched W8A8 control matches all 48 logit/KV positions and all twelve
eight-token trajectories. Each candidate checks **1,007,104** quantized elements
against its independent NumPy oracle, with zero observed swap growth:

| Profile | Prefill selections | Same-token decode selections | Exact eight-token trajectories | Worst absolute logit error |
| --- | ---: | ---: | ---: | ---: |
| Int8 | 11/12 | 35/36 | 9/12 | 3.4780 |
| Int4 | 10/12 | 36/36 | 8/12 | 4.1603 |
| Mixed 4/3/3 | 10/12 | 34/36 | 6/12 | 5.8638 |

No candidate has exact complete logits or KV. These frozen public requantizers
fail the exact-control fidelity screen; they do not measure the paper's trained
quantizer. The checkpoint diagnostic does not invoke HE or garbled conversion.

The diagnostic also checks that token lookup and the client head share the same
136,134,656-byte integer matrix. Embedding-only relocation releases **zero** of
that matrix while the client head remains. This structural ownership result is
separate from paging, native head snapshots, process peak RSS and an unimplemented
combined private lookup/head placement.

## Reproduction

Install a release wheel with `[he]` for SDK-only execution. The exact example
compares all layouts and replays each immutable configuration:

```sh
python examples/benchmarks/packed_embedding.py
```

Repository research commands refuse to overwrite evidence:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run --no-sync python -m benchmarks.research.packed_embedding --mode cost --output new-embedding-cost.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false uv run --no-sync python -m benchmarks.research.packed_embedding --mode quality --output new-embedding-quality.json
```

The quality command admits four GiB of headroom beyond the host reserve and
monitors reserve loss/new swap. Reports retain exact source/configuration/native
identities and public profile commitments. Private caller inputs and outputs
never enter operator reports. Decoder placement remains unavailable until the
missing conversion/packing/transport contracts, fidelity and complete resource
owners are implemented and measured.
