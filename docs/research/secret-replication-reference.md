# Secret Replication scalar/vector reference

`pllm.correlation.SecretReplicationPcf` exposes a bounded two-party correlation
construction and its compatible scalar/vector multiplication consumer through
`benchmark_operator` and `replay_operator_benchmark`. One scalar applies to the
entire vector. The installed operator accepts the complete fields 2, 17 and 257;
it does not accept an independently varying scalar for each vector coordinate.

## Primary source and construction

The source is **Compressing Correlations via Secret Replication: PCFs from
Symmetric Cryptography**, Sections 4.1–4.2 and Remark 25:
<https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.ITC.2026.7>.
The pinned PDF SHA-256 is
`6adaab779eae9ec559d735e0984de8d33e8d757b8ca7f1551057fb79a6eec9d2`.
The Rust implementation is independent; no upstream code is built or linked.

For scalar domain `D`, the construction uses minimal-support lines `L_j(t)=t-j`.
One direction gives its sender all `D` independent-looking leaf keys and its
receiver every key except the one indexed by the receiver's private scalar.
The omitted line vanishes at that scalar. Two independent directions produce
additive shares of scalar `x`, vector `a`, and product `x*a`. Specifically,
`z_i=a_i*x_i+y_other(x_i)-b_i` gives `z_0+z_1=(x_0+x_1)*(a_0+a_1)`.

The expanded control holds the leaf keys. The punctured layout instead holds
one sender GGM root and the sibling seeds along the missing receiver leaf's
path. A padded power-of-two tree has the same depth for every private scalar.
Both layouts use independently seeded AES-128 trees and domain-separated AES
field-output PRFs with bounded rejection sampling. AES expansion, field choices,
one-use context binding and the local dealer are explicit engineering adaptations.
This is not the paper's distributed setup or an independently reviewed protocol.

The consumer opens one masked scalar and one masked vector, then returns
party-local output shares using the usual Beaver identity. The trusted local
benchmark reconstructs only its final outputs and checks independent Python
integer multiplication modulo the selected field.

## Privacy and compatibility checks

Each party owns a separate non-cloneable state. Plan/session/operation/index,
field, scalar domain, encoding, lane count and fresh issuance bind the exchange.
Malformed, cross-issuance, wrong-role and replayed frames consume material;
cancellation drops all remaining state. These checks are not authenticated
network transport or malicious-result verification.

The Rust reference can generate restricted-domain correlations for algebra and
leakage tests, but its generic consumer rejects and burns them. Conditional on
one party's own mask share, an arbitrary-input opening is hidden only when the
other scalar mask covers the full field. Exact support enumeration records:

| Field | Scalar domain | Distinguishing input shift | Total variation distance |
| ---: | ---: | ---: | ---: |
| 17 | 2 | 2 | 1 |
| 17 | 12 | 5 | 5/12 |
| 17 | 17 | 0 | 0 |
| 257 | 16 | 16 | 1 |
| 257 | 256 | 1 | 1/256 |
| 257 | 257 | 0 | 0 |

The report additionally retains exact counterexamples for reusing one scalar
mask across independent elementwise inputs and substituting prime-field
arithmetic for ring32 arithmetic. Opening `u_i-x` and `u_j-x` reveals
`u_i-u_j`. Thus this scalar/vector construction cannot simply replace the
independent ring32 triples or FSS keys used by the nonlinear blocks.

## Complete costs

`docs/evidence/research-secret-replication-cost.json` contains eighteen fresh
processes: fields 2/17/257, both encodings, 1/64/256 lanes, and three samples per
process. All **5,778 outputs** match the independent modular-integer oracle.

For field 257 and 64 lanes:

| Quantity | Expanded keys | Punctured keys |
| --- | ---: | ---: |
| Raw keys, both parties | 16,424 bytes | 328 bytes |
| Expanded correlations, both parties | 1,032 bytes | 1,032 bytes |
| Header-inclusive peer frames | 676 bytes | 676 bytes |
| Field-output PRF evaluations, both parties | 65,664 | 65,664 |
| Median local dealer issuance | 2.572 ms | 0.032 ms |
| Median worker correlation expansion | 26.038 ms | 28.481 ms |
| Median local online exchange | 0.078 ms | 0.083 ms |
| Median complete native-call process CPU | 28.694 ms | 28.632 ms |

Key payload falls 50.07-fold, but expansion work, expanded storage and peer
traffic remain. The same conservative 832,960-byte allocation estimate prices
both layouts. This is not a measured peak RSS or client-memory improvement.
Raw keys count secret scalar shares and seeds; external key framing, authenticated
delivery, client input/output sharing and physical wire remain separate. Reports
include client-boundary estimates and preserve unknown wire/RSS fields.

Each party performs `lanes*(2D-1)` field-output PRF evaluations, before rejection
sampling and GGM-tree overhead. Full-field symbolic counts at 64 lanes grow from
65,664 combined evaluations at 257 to 16,777,344 at 65,537 and
1,099,511,626,368 at 4,294,967,291. The latter widths are not admitted or executed.
Small keys do not remove the scalar-domain expansion cost. No Qwen material,
checkpoint quality, provider transport or whole-response saving is claimed.

## Bounds and reproduction

The native reference preflights fields, domain, at most 4,096 lanes and a 16 MiB
combined allocation estimate. The installed benchmark narrows this to 1–256
lanes and 1–16 fresh samples. Python retains only immutable aggregate reports;
native execution releases the interpreter lock and exports no secret handles.

```sh
uv run python -m benchmarks.research.secret_replication \
  --output docs/evidence/research-secret-replication-cost.json
uv run python examples/benchmarks/secret_replication.py
cargo test --locked --release -p pllm-garble secret_replication
uv run pytest -q tests/test_secret_replication.py
```

The cost runner refuses to replace an existing report. The installed example's
collection is retained in `docs/evidence/research-secret-replication-sdk.json`.
SDK availability is operator-only: an Experiment can serialize the component,
but decoder resolution rejects it until a compatible compiler-bound consumer
and complete role lifecycle exist.
