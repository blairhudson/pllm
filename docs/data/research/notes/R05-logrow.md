# R05 · Garbled Circuit Lookup Tables with Logarithmic Number of Ciphertexts

**Priority 5 · 2024 · single_evaluator · full text reviewed 2026-09-23**

Authors: David Heath, Vladimir Kolesnikov, Lucien K. L. Ng.  
Primary source: https://eprint.iacr.org/2024/369  
Full-text source: `papers/r05-logrow.pdf` (ignored local library). The related
one-hot setup source is David Heath and Vladimir Kolesnikov, *One Hot Garbling*,
CCS 2021; [ePrint 2022/798](https://eprint.iacr.org/2022/798), retained locally as
`papers/one-hot-garbling.pdf`.

## What the source contributes

The evaluator can see a randomly masked row index, but not the true row. A
punctured one-hot encoding selects the row privately; a random-function pad
allows masked table bits to be sent without revealing its plaintext entries.
The optimized paper cost is `(n−1)κ + nmκ + Nm` bits for an N-row m-bit table
with `n = ⌈log2 N⌉`; the table-data term is unavoidable.

This short source summary is separate from the proposed PLLM design below. The source has not been reproduced merely by adding this card.

## Native implementation scope

PLLM has a bounded in-process Rust reference in `pllm-garble::logrow` for
1–9-bit inputs/outputs. It uses the 2022 **two-row** one-hot setup (so costs
`[2(n−1)+1+nm]κ+Nm` bits), one-use input/program/decoder ownership, and
a 9-bit wrapper for the public-calibrated Compact Q7 SiLU table. The masked
index is exposed only inside the reference evaluator; original input, mask,
and plaintext table remain client-owned. A specialized compiler reference binds
one fitted-Q7 element or one **complete bounded semantic SiLU tensor** to the
operation, phase, plan and profile digests, implementation source, resource
policy and fresh issuance. The tensor preflights the complete body before
issuance and burns all rows together on invalid input. The public
`pllm.protocols.prepare_logrow_q7_tensor_reference` API exercises the bound
tensor in-process through an opaque one-use native handle. No reviewed
distributed 2PC, secure provider input transfer, general arithmetic conversion,
whole-session schedule or complete compiled decoder is established.
The opt-in float32 bridge is only defined on `[-1, 1]`: ties-to-even Q7
quantization rejects out-of-range values rather than saturating, adds at most
1/256 input error, and burns the complete tensor when any input is invalid.
This does not establish a wider-range conversion for actual checkpoint MLPs.
An immutable session-cost estimator counts all semantic SiLU evaluator bodies
for bounded prefill and decode steps and rejects totals above an explicit cap.
The separate offline `prepare_logrow_q7_session_reference` preissues bounded
material for all those tensors, orders one-use consumption, and burns the
remainder on failure or abort. Neither API runs a complete compiled decoder.

Data flow: **Boolean index labels + prepared lookup → Boolean output labels; explicit arithmetic conversions around it.**

Remaining candidate components, pending full contract admission:

- `gate.logrow`
- `convert.lookup_io`

Dependencies: R04. Public compilation/model-data scans and all online evaluation happen in Rust or reviewed native accelerator libraries called from Rust. Python selects immutable parameters and displays public results. It does not implement a timed alternative or silently repair unsupported native execution.

## Required fixture and benchmark specification

Matched 8/12/16-bit tables versus flat, Boolean and weighted-path variants. Include the table-data term and both conversion directions.

Mandatory paper-specific gates:

- Small exhaustive tables
- Serialized table-data term versus ciphertext count
- Secret-index queries, malformed labels and repeated use

Each fixture locks input/output representations, ring/field parameters, numeric graph, party count and material lifetime. Compare an optimized implementation with identical functionality and applicable privacy assumptions; record original-artifact numbers and adaptations separately. A source-advertised speedup is not an expected threshold for different hardware.

## Privacy, correctness and proof obligations

Lookup privacy and input-binding argument, including arithmetic bridge obligations and observable access behavior.

Do not translate logarithmic ciphertext count into logarithmic total bytes or claim our flat-table improvements beat this unimplemented baseline.

Provide the corrupted party's permitted view, known plaintext/public inputs, randomness model, allowed leakage and attack budget. Formal or empirical results have explicit scope. Passing functional tests cannot grant a production privacy badge. A changed hash, field, layout, actor role or precision creates a documented adaptation obligation.

## What PLLM already has

Functional parity holds on all 257 signed-Q7 values for a fitted Compact
profile. For the padded 512-row by 9-bit table, evaluator material is 2,144
bytes: 272 one-hot tree bytes, 1,296 random-function row bytes, and 576
masked table bytes. Input labels (144 bytes), output labels (144 bytes),
framing and local workspace are additional. The older Compact oracle has
73,728 half-gate ciphertext bytes before other material. These are not
controlled network, CPU, memory, disk, generation-quality, or checkpoint
measurements; the `LogRowGarbledLookup` Python class stays pending.

Bounded Qwen2 and dense Qwen3 compiler tests evaluate 32-element prefill and
16-element decode tensors, with exact 68,608- and 34,304-byte evaluator-body
preflight. A 64 MiB per-tensor hard ceiling protects this research reference;
it is **not** a measured per-token or full-decoder memory bound.

This is an adapted native bounded-tensor reference, **not a reproduction of the paper's
optimized setup, measured experiments or whole-system security**.

## Reproduction gates

`R05.acquire → R05.specify → R05.reference → R05.native → R05.assure → R05.benchmark → R05.document`.

The public research backlog records the remaining implementation and validation work. Native integration requires the actual PLLM command, source locks, raw measurements, and assurance report.

## Developer-facing explainer requirement

The eventual method page needs a worked operator example, a role diagram in prose or graphics, source section links, why/when to use it, configuration parameters, unsupported cases, numeric envelope, failure behavior and source-vs-PLLM differences. Generate tables from evidence rather than copying old performance ratios. Show `not implemented`, `adapted`, `reference only` or `reproduced at scope X` honestly.

## Default policy

`eligible_default = false` until implementation, full coverage, source/security review and deployment gates are met. The track restriction remains binding even after successful reproduction. Experimental or comparison methods cannot silently enter the preferred single-evaluator/no-HE profile.
