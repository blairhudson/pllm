# Five more client-work offload screens

This second round tests five new PLLM engineering candidates, not literature-first
novelty. It follows the [indexed-tokenizer round](client-offload-five.md), which
reduced the measured paged Qwen3-4B client/dashboard peak to 225.28 MB. That result
does not establish that the remaining private arithmetic is cheap to delegate.

The new screen uses pinned `Qwen/Qwen3-4B` revision
`1cfa9a7208912126459214e8b04321603b3df60c`, its native semantic plan, explicit
W8A8 quantization, and a public 16-input/8-output-token workload. This executes
23 body rows and eight output-head rows. Private-value stand-ins are synthetic;
the embedding, head, layer-0 norm and Q-projection weights come from the actual
public checkpoint. No full decoder response is executed in this round.

## Results — 6 October 2026

Raw record: [five-probe Qwen3-4B evidence](../evidence/client-offload-round2-qwen3-4b-2026-10-06.json).
All sizes use decimal MB/GB unless stated otherwise. Timings are local operator
samples, not network latency. No new host swap was observed.

| Candidate | What can move | Exactness/privacy gate | Measured result and decision |
| --- | --- | --- | --- |
| Public RoPE tables | Position-dependent sine/cosine construction to an offline public compiler | Download the entire trusted, plan-bound table. Private positions and activations stay local. Every checked operator bit matches. | Response-shaped rotary CPU **12.81 → 10.97 ms**, saving **1.84 ms (14.38%)**. Import adds **0.295 ms** and the table **25,093 bytes**. A small operator improvement; no whole-response speed or memory claim. |
| Private RMSNorm square-root pages | Positive-normal float32 square root to two non-colluding lookup workers | Normalize the mantissa locally; retrieve its page with fresh DPF keys. The exponent and within-page offset stay local. **10,008** clear-table cases and **8** private queries match float32 bits. | **1,418 bytes** and **1.329 s** combined worker CPU per query. The 34,799-scalar schedule projects **49.34 MB** and **12.85 worker CPU hours**. Reject this layout. |
| Private tied embedding/head | Both uses of the shared vocabulary matrix to two online workers | DPF lookup returns the selected quantized embedding and scale. Fresh exact-ring shares delegate the head; client reconstructs and selects tokens. Checked embedding, integer-head and float32-logit bits match. | Could remove a **388.96 MB client weight artifact**, while retaining **0.608 MB** head scales. Adds **10.03 MB** projected online bodies. One head falls **89.23 → 0.297 ms client CPU**, but costs **331.83 ms combined worker CPU**; one private embedding lookup costs **789.32 ms** combined worker CPU. No total-compute win or measured client-RSS saving. |
| Garbled activation absmax | Float32 magnitude comparison for activation quantization to an opaque-label evaluator | A trusted offline helper issues one-use encodings; only the client decodes the maximum. Finite float32 magnitude bit ordering is exact. | At width 2,560, client encode/decode takes **0.978 ms** versus **0.00223 ms** clear maximum; evaluation takes **854.20 ms** and issuance **1,855.53 ms**. Full-width projection needs **7.79 GB online labels** and **62.31 GB one-use ciphertexts**. Reject before quantization division or transport. |
| Public norm-weight folding | Multiplication by public RMSNorm gamma into the next public matrix | Only public weights enter compilation; private inputs stay local. Real-number algebra is valid, but reassociation and W8A8 rounding change execution. | All **94,208** checked W8A8 outputs change, with **0.01983** maximum absolute error; input codes and stage commitments change too. The removed multiply costs only **0.01195 ms** for this 23-row operator. Reject as a substitution under the original numeric plan. |

The public compiler, shared by these screens, takes **3.526 CPU seconds** and
peaks at **849.15 MB** process RSS. That includes cached checkpoint reading,
bounded quantization, derived lookup tables and artifact hashing. This work is
moved rather than eliminated. Original checkpoint distribution is not measured.

## What the measurements cover

### Public rotary compilation

The artifact binds the semantic plan, public capacity, source rotary attributes,
float32 layout and NumPy version to a trusted compiler digest. Import checks its
exact length, header, digest and finite values. The client fetches the complete
table and performs all activation arithmetic locally; it never makes a
position-indexed remote request.

Four ordinary/precompiled/precompiled/ordinary samples check **4,239,360** Q/K
elements apiece across every layer's prefill and seven decode positions. Separate
timing windows repeat the same tensors five times and exclude random generation
and oracle comparisons. Reported values are two-sample medians per mode. This is
a same-host microbenchmark; cross-platform coefficient fidelity, authenticated
artifact delivery and an executable component contract remain unimplemented.

### Private exact square root

Normalizing a positive normal float32 value to `[1,4)` leaves 24 lookup bits.
The public table contains 16,777,216 float32 outputs, or **64 MiB per worker**.
Packing 64 outputs per 256-byte page reduces the DPF domain while keeping its
offset private. The restored power-of-two exponent is exact for this domain.
Zero, negative, subnormal and non-finite inputs are rejected.

The eight timed queries include the smallest and largest positive normal
float32 values. Combined worker CPU is the median sum of both worker phases;
the 12.85-hour figure extrapolates that median, not an executed tensor schedule.
The current native worker admits only 4,096 queries, so the full 34,799-query
projection also requires additional worker instances. It excludes their setup
and distribution, plus RMSNorm sums, divisions and remaining decoder work.
Client query/reconstruction CPU is **0.187 ms per scalar**; an optimistic local
batched square-root control takes **0.00971 ms for all 34,799 scalars**. That
control excludes the ordinary decoder's dispatch and other normalization work.

### Tied vocabulary boundary

The public compiler streams the actual BF16 embedding in bounded row chunks and
uses the ordinary W8 per-token quantizer. Each private lookup record contains
2,560 i8 coefficients and its float32 scale. Both workers scan their complete
**389.56 MB** record table for each query, with fresh one-use DPF keys. Three
checked token locations include both table edges.

Three synthetic hidden-vector queries separately compare the paged clear head
with two paged wrap32 kernels. Fresh OS-random shares hide the quantized input;
activation scales, reconstruction, dequantization and token selection stay
client-local. The exact signed dot bound is `2560 × 127² = 41,290,240`, within
the centered 32-bit ring. The 10.03 MB projection counts all 23 embedding lookups
and eight heads. Head bodies count arithmetic payloads only; lookup bodies also
include the existing DPF framing. Neither is full wire accounting.

The reference owns a **388.96 MB private head snapshot per worker** in addition
to each lookup table. Provider table/snapshot duplication and cold distribution
must be charged in any deployment. Removing the client's already-paged matrix
does not imply removing 388.96 MB of RAM. The isolated phase timings show a
client-CPU/storage trade, not a whole-response improvement. A new tied-boundary
role graph, resource admission, independent worker deployment and complete
response parity would be required before selection.

### Exact garbled magnitude maximum

For finite float32 inputs, stripping the sign bit makes unsigned bit order equal
to absolute-value order. The native half-gates reference compares 31-bit words
and returns opaque output labels, consuming material on evaluation and rejecting
cross-circuit inputs. Tests cover signed zero, subnormals, equal magnitudes and
float32 extremes; invalid domains and shapes consume the client handle.

The circuit is built and measured at widths 8, 128 and 2,560. Compiler-derived
input widths price all grouped remote stages plus the local head quantizer.
Some real stages are width 9,728, beyond this reference's 2,560 limit; the
7.79/62.31 GB totals are algebraic projections, not an admitted tensor schedule.
Uncompressed helper-to-client encoding material adds another **15.58 GB**.
Circuit instructions, transport and the rest of quantization are excluded.
These are Rust wall-time samples; native process peak RSS is unknown. The
separately reported Python-launcher RSS does not measure the Rust process.

### Public gamma folding

The screen uses the pinned first-layer 4,096-by-2,560 Q projection, its public
input-norm gamma and 23 synthetic normalized rows. Even float32 reassociation
changes 79,745 values, though its maximum error is only `7.45e-8`. Requantization
then changes 57,903 input codes and all checked W8A8 outputs. An identity-gamma
control preserves the outputs. A differently digested numeric component with
fresh held-out model-quality evidence could test this trade, but algebraic
equivalence alone cannot authorize the existing W8A8 plan.

## Privacy and promotion boundary

These are bounded in-process research references. DPF and additive-share workers
require non-collusion; the garbling helper retains the existing trusted-issuance
assumption. Co-location in a measurement process does not establish independent
operators or malicious-participant security. No sensitive value is deliberately
opened to an evaluator. Public-table compilation does not authorize delegating
private table indices or token-dependent lookups in plaintext.

Reports retain public hashes, counts, exactness outcomes and aggregate costs,
not query indices, prompts, activations, shares, labels, masks or credentials.
None of the five candidates activates a Pipeline. This round establishes a
small public-setup opportunity and rejects the tested costly private-arithmetic
layouts; it does not beat the naive two-worker whole-response compute comparator
or establish a 10× client improvement.

## Reproduction

From the repository root with the pinned public checkpoint cached and the
current native extension installed:

```bash
cargo build --release -p pllm-garble --example client_absmax_probe
HF_HUB_OFFLINE=1 uv run --no-sync python scripts/probe_client_offload_round2.py \
  --output docs/evidence/client-offload-round2-rerun.json
uv run --no-sync pytest -q tests/test_client_offload_round2.py
cargo test --release -p pllm-garble --example client_absmax_probe
```

Choose a new output path. `--scratch-dir` sets temporary artifact storage.
The controller requires 4 GiB of physical headroom beyond the ordinary host
reserve and 4 GiB of disk headroom, retains the memory/swap watchdog, and bounds
child duration. Each screen uses a fresh process; both logical private workers
remain inside that screen's process. Python worker RSS therefore includes all
its logical roles rather than a separately measured client. Incomplete runs
retain their status. Source, native extension and Rust executable hashes bind
the measured implementation even before its subsequent evidence commit.
