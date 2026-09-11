# Benchmarks

Read the workload and denominator before reading the speedup.


Historical figures come from the supplied lifecycle study. Download [the raw
lifecycle summary](/downloads/lifecycle-summary.json) and [evidence
records](/downloads/evidence.zip).

## Current runtime

No current offline-inventory model run is published as retained benchmark
evidence in this revision. Development acceptance runs informed the runtime, but
their sanitized records, exact commands, model revisions, and host manifests
were not retained together. Quoting their numbers here would make them less
reproducible than the historical evidence below.

Use `pllm benchmark dashboard` to create new immutable, text-free records. For a
comparison, hold the model-body fingerprint, cold/warm mode, prompt-token count,
output allowance, host, and software revision constant; repeat each cell; then
export the sanitized records with an environment manifest before publishing a
claim. The model/fingerprint/context matrix is intended to make that missing
study straightforward rather than imply it has already been completed.

## Superseded relay canary

The 8 September 2026 installed canary used the earlier just-in-time seeded
relay: 32 output tokens from a 16-token prompt, 1.54 s warm TTFT, 7.06 tokens/s,
and 125.37 MB of client/inference traffic. It remains a historical regression
point for compact rings, tied boundary matrices, and bundle caching. It does not
describe the current offline Preparation-to-Inference inventory lifecycle. See
`qwen-prepared-inference.json` in the evidence archive.

## Complete preparation

For a 768 input, 256 output matrix and 2,048 future masks, the matched coordinate addition path took **3.122 seconds**. The completed coefficient GEMM path took **0.633 seconds**, a **4.93x speedup**. Both include encryption, conversion, server arithmetic, ciphertext reconstruction, and client decryption. Physical network transfer is excluded.

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
