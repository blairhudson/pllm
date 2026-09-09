# Benchmarks

Read the workload and denominator before reading the speedup.


Historical figures come from the supplied lifecycle study. A current installed
prepared-inference acceptance run is reported separately below. Download [the
raw lifecycle summary](/downloads/lifecycle-summary.json) and [evidence
records](/downloads/evidence.zip).

## Prepared Qwen acceptance

An installed macOS arm64 wheel generated 32 tokens with
`Qwen/Qwen2.5-0.5B-Instruct` using separate loopback preparation and inference
processes. Warm time to first token was **1.54 seconds** and decode throughput was
**7.06 tokens/s**. Preparation upload was **326,304 bytes**, masked inference
upload was **52,370,400 bytes**, and inference returned **72,404,832 bytes**.
Preparation relayed **72,962,400 bytes** directly to inference, leaving total
online client traffic at **125,373,502 bytes** instead of two full result
downloads. The tied boundary bundle was **145,970,989 bytes** cold and zero
bytes warm from the fingerprinted disk cache. This validates arithmetic,
packaging, compact rings, relay accounting, and local performance, not
production non-collusion. See `qwen-prepared-inference.json` in the evidence
archive.

## Complete preparation

For a 768 input, 256 output matrix and 2,048 future masks, the matched coordinate addition path took **3.122 seconds**. The completed coefficient GEMM path took **0.633 seconds**, or **4.93 times faster**. Both include encryption, conversion, server arithmetic, ciphertext reconstruction, and client decryption. Physical network transfer is excluded.

## Large stage measurements

Weights are synthetic W4 matrices with Qwen3.5-27B stage dimensions. A correlation means one complete matrix input mask and its complete output mask, not a generated language token.

| Stage | Batch seconds | Correlations/s |
| --- | ---: | ---: |
| DeltaNet input, 5120 to 16480 | 16.56 | 123.70 |
| Attention input, 5120 to 14336 | 15.24 | 134.39 |
| Mixer output, 6144 to 5120 | 8.39 | 244.22 |
| MLP expansion, 5120 to 34816 | 32.26 | 63.48 |
| MLP contraction, 17408 to 5120 | 19.03 | 107.60 |

Each large result contains two timing samples. Every output column was checked for 16 selected mask rows. Small tests check the complete batch. There is no measured complete Qwen generation rate.

## A complete small request

A trained four block model generated 96 tokens after an 18 token prompt. It started without prepared material and used separate client and provider processes.

| Injected delay per exchange | Ordinary generation | Four proposals |
| --- | ---: | ---: |
| 0 ms | 2.90 s | 4.27 s |
| 1 ms | 4.91 s | 5.35 s |
| 10 ms | 20.11 s | 11.29 s |

These are complete request durations with preparation. Delay was injected in the application, not measured on a real WAN. The ordinary path matched the clear W4A4 tokens and final logits. The short authored corpus is not a language quality benchmark.

## Why fewer exchanges did not always win

Proposals reduced online calls from 1,632 to 646, but consumed 3,434 correlations instead of 1,904. Rejected work still needs fresh masks. At low latency, the extra preparation outweighed the saved exchanges.

## Model scale projections

The study's 27B plan requires 257 sequential private exchanges and 93.59 MB of compact preparation plus online traffic per token set. A shared 1 Gbit/s link therefore has a **1.34 tokens/s bandwidth ceiling**, even with zero compute cost. This ceiling is not delivered TPS.
