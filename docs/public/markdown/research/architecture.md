# How PLLM works

A local Responses interface around a private matrix protocol.


## The fast path

For an integer matrix `W`, private activation `x`, and fresh mask `r`, the client obtains `W r` through HE during preparation. At inference time:

```text
client → provider:  d = x + r mod p
provider → client:  u = W d mod p
client recovers:    u − W r = W x mod p
```

The activation scale needed for dequantization stays local. The modulus must cover the intended signed result; equality modulo `p` alone does not prevent wraparound when converting back to a signed integer.

## Coordinate preparation

Place one coordinate of many future masks in the coefficients of one plaintext polynomial. A model row becomes a linear combination of ciphertexts. The completed study turns this into integer GEMM over ciphertext coefficient arrays and reconstructs valid ciphertexts for client decryption.

The matrix weights do not change. The layout changes how future work is grouped. The batch belongs to one encryption key and cannot contain arbitrary independent customer keys.

## Why the client remains substantial

Attention, recurrent state, nonlinearities, sampling, and decryption still run locally. The 27B capacity study also keeps a public embedding local. Removing dense projection weights from the client does not make it a thin browser client.

## What the API changes

The local gateway presents familiar Response objects and streaming events. It does not make an ordinary remote vLLM or Ollama endpoint capable of private tensor execution. Such endpoints receive plaintext unless a different protocol is implemented inside their execution path.
