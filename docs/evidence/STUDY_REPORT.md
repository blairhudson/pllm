# Private inference from preparation to decoded output

## Main result

The coefficient matrix backend now completes actual BFV encryption, server
matrix evaluation, ciphertext reconstruction, and client decryption. On the
matched 768 to 256 stage it reduces preparation from 3.122 seconds to 0.633
seconds for 2,048 correlations: 4.93 times faster under the same parameters.

The large stage experiments use the actual matrix dimensions of Qwen3.5 27B.
The weights are synthetic signed four bit integers. No full Qwen checkpoint was
loaded, generated from, or evaluated for language quality.

| Stage | Dimensions | Complete batch seconds | Correlations per second |
|---|---:|---:|---:|
| DeltaNet input | 5120 to 16480 | 16.56 | 123.70 |
| Attention input | 5120 to 14336 | 15.24 | 134.39 |
| Mixer output | 6144 to 5120 | 8.39 | 244.22 |
| MLP expansion | 5120 to 34816 | 32.26 | 63.48 |
| MLP contraction | 17408 to 5120 | 19.03 | 107.60 |

Batch size is 2048. These rates include mask sampling, encryption, coefficient
conversion, evaluation and decryption. They exclude key generation, reference
verification and network transfer. Each full size result contains two samples.
Every output column was compared with independent NumPy arithmetic for 16 mask
rows. Small tests compare every mask row and every output column.

## Eight bit activations without changing the encrypted evaluator

A separate full input-width experiment uses 17408 inputs, W4 weights and
activation extrema of plus or minus 127. Actual BFV preparation and masked
execution recover the exact signed extrema of plus or minus 15475712. The
plaintext modulus is 33554467; the coefficient program and its 54 bit prime
are unchanged. The minimum measured noise budget is 14 bits.

Online fields grow from three bytes to four. With compact preparation, that
adds approximately 6.8% to the total projected bytes per token. This supports
a lower-cost W4A8 compatibility experiment, not a model-quality claim. No Qwen
checkpoint was evaluated at either precision.

## The complete process

1. The client validates a public model plan and owns the tokenizer, embedding,
   private state, activation scales and HE key. The server owns dense matrices.
2. The client samples independent masks for a future stage and encrypts one
   polynomial per input coordinate. Its coefficients contain future masks.
3. The service multiplies the matrix by ciphertext coefficient arrays using
   integer GEMM. This is exactly the same ciphertext linear map as SEAL additions.
4. The client reconstructs valid SEAL ciphertexts and decrypts the transformed
   masks. Each pair is bound to a stage, session and unique identity.
5. Prompt prefill consumes distinct correlations for every prompt row. Local
   convolution, recurrence, attention, normalization and activation operations
   connect the private matrix stages.
6. During decoding, the client sends masked integer activations, receives
   masked results, subtracts the matching output masks, applies its private
   scales and samples or selects output tokens locally.
7. An exhausted inventory triggers another real HE preparation batch. Tokens
   rejected during proposal verification still consume their correlations.
8. The session ends by discarding remaining preparation. Nothing from a previous
   session is accepted by the next server epoch.

## Useful size changes the conclusion

The verified text graph has 64 blocks: 48 use Gated DeltaNet and 16 use gated
full attention. Fusing projections that consume exactly the same activation
leaves 257 sequential private exchanges per decoded token, including the head.
This plan covers 25.62 billion dense weight elements. The embedding remains local.

The protocol sends 18.96 MB of online residues for each decoded token. Its raw
coordinate preparation adds 101.12 MB per useful token set at full occupancy.
With standard seeded inputs and seven-byte output coefficients, preparation
traffic is projected at 74.63 MB per token set. This reduction changes the byte
encoding, not the model or arithmetic result.

At one gigabit per second shared between directions, the combined compact
preparation and inference traffic alone limits sustained throughput to 1.34
new tokens per second. With one gigabit available independently in each
direction, the bound is 1.77 tokens per second. These are bandwidth ceilings
with zero compute cost and ideal overlap, not achieved model rates.

The measured CPU components project to 2.49 seconds of preparation work per
complete token set, before online inference or transfer. The vocabulary
preparation estimate scales output work from a measured 1024-row tile and
counts the common input work only once. The complete model was not allocated.
Generating 2048 complete token sets this way projects to about 85 minutes of
serial preparation work on this host. A larger preparation batch does not
provide an instant first answer.

Storing both masks as uint32 for those 2048 sets requires 48.22 GiB. Regenerating
input masks from a correctly managed cryptographic seed still leaves 31.68 GiB
of transformed masks. Recurrent state requires 144 MiB; attention KV at BF16
requires 512 MiB at 8192 positions. The client is not a thin browser runtime.

## Whole lifecycle experiment

A four block model was trained with causal convolution, gated DeltaNet, gated
attention, partial RoPE, RMS normalization and SwiGLU. All dense projections
and the head are evaluated in a separately spawned server process. The client
has no dense projection matrices in its runtime state. Its public embedding
and small normalization/convolution parameters stay local.

Training uses a short authored corpus and only establishes a nonrandom
functional checkpoint. It is not evidence of Qwen language quality.

The experiment generates 96 tokens after an 18-token prompt. It begins with no
prepared material. It prepares 2176 correlations, consumes 1904, and discards
272 unused entries. The second preparation cycle appears as a visible pause in
the output trace. All generated tokens and final logits match the clear W4A4
execution for the ordinary decoding path.

| Added delay per exchange | Ordinary decode | Four proposals | Result |
|---:|---:|---:|---|
| 0 ms | 2.90 s | 4.27 s | Speculation is slower |
| 1 ms | 4.91 s | 5.35 s | Speculation is slower |
| 10 ms | 20.11 s | 11.29 s | Speculation is 1.78 times faster |

Delay is injected in the server application, not measured over a wide area
network. These are whole-run measurements with preparation, not only the
period when the inventory happens to be full. The repeated corpus makes prompt
lookup more favourable than it would be on arbitrary tasks.

Proposal verification reduces online calls from 1632 to 646. But it consumes
3434 correlations instead of 1904 and prepares almost twice as many entries.
That is why fewer network exchanges do not guarantee a faster response.

## Recurrent state replay

For eight proposals, saving one complete recurrent state after each position
would require 1.125 GiB across the 48 real-size DeltaNet layers. Keeping the
required local projection trace requires 27.14 MiB, plus the existing 144 MiB
base state. This is about 42 times less incremental memory.

The client replays only the accepted prefix using already decrypted results.
It makes no additional private matrix query and never reuses a mask. A four
position replay takes 1.96 ms for one real-size recurrent layer. Replaying with
the same kernel gives exactly the saved prefix state. Switching the recurrence
from elementwise reductions to matrix kernels introduces only floating
summation differences in this test; this is not a proof of BF16 Qwen parity.

## Consequences for the SDK

Keep `OpenAI().responses.create(...)` and `pllm chat`, but make the SDK execution
plan reflect the whole request. Preparation, prefill, decoding and inventory
waits should be distinct local states. Report time to first token, total request
time, generation rate, bytes, preparation spent and preparation discarded.
Do not advertise only the rate obtained after masks are available.

Use a persistent client agent within the customer's trusted environment. Its
key may serve several authorized sessions in that same trust boundary. Never
pack unrelated customer keys into one ciphertext. A customer gateway is still
a trusted client: moving it onto an untrusted provider does not preserve the
threat model.

For prompt prefill, generate and consume one stage's mask batch at a time.
This avoids storing a whole model's inventory, but does not remove preparation
work or traffic. Decode inventories need a bounded horizon. Fixed large batches
make short conversations pay for material they never use.

Enable proposal verification only when its reduction in interaction latency
outweighs extra matrix rows, preparation, traffic and local replay. The
experiment implements greedy verification. It does not implement the rejection
sampler needed to claim distribution equivalence for arbitrary sampling.

## Next work, in priority order

1. Reduce preparation communication using a reviewed matrix protocol. Faster
   server GEMM alone cannot remove the measured link ceiling. Scalar VOLE
   expansion is not automatically a cheap arbitrary matrix correlation factory.
2. Complete a native batch bridge for client encryption and decryption. Client
   work is now comparable to server preparation. GPU work on only the server
   cannot remove that cost.
3. Compile larger secure execution regions to remove serial exchanges. A
   persistent connection does not remove dependencies between model stages.
4. Implement bounded stage streaming and inventory-aware proposal scheduling
   behind the existing client API.
5. Load an actual Qwen checkpoint, validate all hybrid operators and quantization
   quality, then measure long prefill and sustained decode on the intended
   client and provider hardware.
6. Keep sparse experts separate until private dispatch is implemented. The
   35B-A3B model has 256 experts and activates 8. Evaluating all experts multiplies
   expert computation by 32; exposing their identities leaks routing decisions.

## Security boundary

This is public-weight inference against a server that follows the protocol.
The binary channel authenticates messages and rejects replay; it does not
verify arbitrary provider computation. Timing, dimensions, request counts and
scheduling remain visible. The client keeps its activation scales private and
uses fresh cryptographic masks. The experiment does not claim model secrecy,
security against malicious providers, or resistance to restoring a whole VM
snapshot.
