# Implementation checks and experiment results

Use ordinary tests for correctness and reusable experiment `.py` files with
standard benchmark outputs for measurements. Build wheels with uv/Maturin as
described in [CONTRIBUTING.md](../CONTRIBUTING.md). Git history contains earlier
source versions; separate build receipts and copied-source archives are not
required.

## Current entry points

- [`examples/prepared_sdk.py`](../examples/prepared_sdk.py): prepared SDK example.
- [`examples/papers/`](../examples/papers/): per-paper experiment definitions.
- [`examples/benchmarks/`](../examples/benchmarks/): ordinary benchmark drivers.
- [`tests/`](../tests/): SDK, gateway, arithmetic, admission and lifecycle checks.
- [`docs/evidence/`](../docs/evidence/): benchmark outputs used by research tables.
- [`IMPLEMENTATION-STATUS.md`](IMPLEMENTATION-STATUS.md): current coverage and
  remaining method gates.
- [`TEN-METHOD-COVERAGE.md`](TEN-METHOD-COVERAGE.md) and
  [`reproduction-contracts/`](reproduction-contracts/): source-method mapping and
  numeric contracts.

## Historical generated checks — 16 October 2026

The original [conformance results](artifacts/conformance/README.md) retain their
separate workloads and implementation identities. They are functionality checks,
not checkpoint-quality or matched system-performance measurements.

The Qwen2, Qwen3 and Llama generated fixtures compared 29,592 complete logit and
active-KV values between tiled and untiled W8A8 execution. All compared values
were bit-identical. Against upstream FP32, the declared absolute tolerance was
0.1; maximum errors were:

| Generated family | Logits | Keys | Values |
| --- | ---: | ---: | ---: |
| Qwen2 | 0.042739 | 0.022892 | 0.023220 |
| Qwen3 | 0.051512 | 0.051037 | 0.021957 |
| Llama | 0.047203 | 0.021044 | 0.019906 |

The later generated run passed 29 focused tests and the family run passed 46.
These counts describe separate historical executions. The earlier installed-v4
run did not establish provider-child import identities. New source changes need
their own ordinary tests; historical results do not qualify the current build.

## Reusable checks

| Capability | Tests or reference |
| --- | --- |
| MLP tiling, complete logits/KV, admission and retirement | `tests/test_mlp_tiling.py` |
| Freivalds integer verification | `tests/test_native_freivalds.py` |
| Public channel equalization and exact frame accounting | `tests/test_public_equalization.py`, `tests/test_public_equalization_accounting.py` |
| Curl, LogRow and Compact gated operators | `docs/research/nonlinear-frontier.md` and linked operator tests |
| Masked-wire FSS operators | `docs/research/masked-wire-reference.md` |
| Private embedding lookup and repacking | `docs/research/packed-embedding-reference.md` |
| Encrypted softmax | `tests/test_encrypted_softmax.py` |

The historical equalization framing failure came before the current accounting
controls. A later synthetic clock-width control demonstrated a possible two-byte
difference; it did not establish the cause of the original uncaptured frames.
The repaired checks compare arithmetic and non-timing framing separately from
encoded timing fields.

Bounded operators remain distinct from complete paper-system implementations.
Generated SDK/gateway success does not establish independent operators, broad
checkpoint quality, client-memory tiers, or a new performance ranking. Keep these
scopes in ordinary benchmark reports and concise method documentation.
