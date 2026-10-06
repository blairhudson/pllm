# SIGMA-style masked-wire gated block

This independently implemented **bounded component reference** follows the
primary-paper cost audit. It replaces the earlier generic ties-even/Beaver
layout with early-terminated Boolean DPF comparisons, public-gap floor shifts,
the residual SiLU lookup and masked-wire composition. It is not a complete
SIGMA or FuseFSS system or protected Qwen decoder. The bounded operator is now
available through `pllm.metrics.MaskedWireGatedBlockProbe`; its installed SDK
benchmark status is independent of its cost/quality and whole-model coverage.

## Installed operator benchmark

`MaskedWireGatedBlockProbe(lanes=64, samples=3).run()` returns an immutable
`OperatorBenchmarkReport`, with independent integer parity, separate float64
approximation error, both-party raw keys, phase/direction peer frames, local wall
and process CPU measurements, public-table storage and client-boundary estimates.
Unknown full wire and peak memory remain null. `from_dict(report.configuration)`
replays the public configuration. Custom Q24 inputs are checked before issuance
and are not retained in reports; they must be supplied again for replay.
Every sample consumes fresh material. No reusable keys leave the native binding.

The SDK guide is `docs/content/docs/sdk/evaluate/paper-operators.mdx`, the installed
example is `examples/benchmarks/masked_wire.py`, and the saved SDK evidence is
`docs/evidence/research-masked-wire-sdk.json`. These local operator timings are
separate from the historical fresh-process Qwen projection below. Operator
availability does not activate a Pipeline or replace the retained negative quality result.

Implementation:

- `crates/pllm-garble/src/compact_dpf.rs`
- `crates/pllm-garble/src/masked_wire_reference.rs`
- `crates/pllm-core/src/residual_silu_reference.rs`
- `benchmarks/research/masked_wire_reference.py`
- `docs/evidence/research-masked-wire-reference-qwen25.json`

## Primary-source contract

The retained PDF hashes appear in the report. The primary sources are SIGMA
Sections 2.5–2.6, 4.2.2, 5.1, Figure 2/7 and Appendix F; Grotto Sections 5–6;
and Pika's Boolean-DPF lookup/sign correction. Upstream implementations are
not dependencies.

The independent Boolean DPF stops its binary tree at 128 Boolean leaves.
Grotto's suffix-parity comparator visits one child per level instead of
expanding the domain. This version retains **128 secret seed bits plus separate
control bits**. SIGMA instead packs a 127-bit seed with its flag. Its per-party
encoded key size is therefore:

```text
DPF(n,1) = 2^n bits                         for n <= 7
         = 256 + 130*(n-7) bits             otherwise.
```

The CPU AES expansion uses two calls per child, recomputing the control block;
it does not reproduce the paper's half-PRG optimization or GPU throughput.
Keys are freshly issued and never exported through a public reusable evaluator.
The internal comparison key serves only the specified tests within one lane
and one operation. Every lane and subsequent invocation gets fresh material.

### Floor shift with a public gap

For signed `v` in `[-2^(n-2), 2^(n-2))`, add the public offset
`2^(n-2)` before applying SIGMA's positive half-ring GapARS. The offset is
subtracted in the output-mask correction. One Boolean comparison identifies
the low-bit carry; one masked-bit opening converts it to arithmetic shares.
The high wrap correction is determined by the opened masked value's sign and
the secret input mask's sign. It is not silently omitted.

The resulting output is exactly `floor(v / 2^f)` in the same ring. Its
per-party key is `DPF(f,1)+3n` bits. Thus the published **32/7 example is
28 bytes/party** here too. The complete block uses a different **40/12**
contract, costing 1,026 bits per shift. Exhaustive small-ring tests cover all
input masks and gap-valid values; an explicit outside-gap counterexample
demonstrates why the range condition cannot be discarded.

Truncate-and-reduce uses `DPF(f,1)+2(n-f)` bits and preserves the reduced ring.
This block reduces an effective 19-bit Q12 gate to a 13-bit Q6 lookup input.

## Fixed public numeric mapping

The mapping was frozen before twelve fresh public prompts were evaluated:

- Signed 40-bit ring; gate and up enter as **Q24** and shift to **Q12**.
- Gate range `[-48,48]`; up range `[-128,128]`; non-finite/outside inputs reject.
- `T[i] = floor(4096 * (i/64) / (1 + exp(i/64)))`, for `0 <= i < 1024`.
- Let `g=floor(g24/4096)`, `u=floor(u24/4096)`.
- Following SIGMA Figure 7's GPU order, `j=min(abs(floor(g/64)),1023)`.
- Return `floor((max(g,0)-T[j])*u/4096)` in Q12.

The table and bounds are public, mathematical and digest-bound; no private
prompt sets a scale or tail. Moving floor before absolute value introduces
signed asymmetry, retained explicitly rather than repaired with a different
numeric rule. The 1,023 tail entry is zero. Exhausting all admitted Q12 gate
codes proves the effective width and final product's public gap; both Q24
input bounds also fit the 40-bit gap.

The Qwen diagnostic uses a **trusted clear float32-to-Q24 bridge**, then returns
the result to the compiled W8A8 decoder. This is not a protected conversion,
fixed-point linear decoder or upstream-float32 quality comparison. Same-token
decode uses the unmodified control's selections. A separately constructed fresh
runtime verifies every control logit and KV value. The Python integer oracle
checks every native candidate element.

## Party-local protocol and permitted views

The local research dealer knows masks and both parties' preprocessing material.
It receives no online model prompt. The trusted client encodes bounded inputs
with those masks and alone removes the final output mask. Each evaluator holds
only its own non-cloneable keys and arithmetic/Boolean correction shares.

Both evaluators see public parameters, bindings, frame lengths and freshly
masked intermediate values. They do not receive unmasked input/output masks,
the other party's keys, plaintext gate/up values or an opened table index.
The trusted dealer/client must not collude with an evaluator. This stated view
and functional tests are not an independent proof of security.

The composed schedule opens eleven ordered peer phases:

1. Masked carry bits for both input GapARS operations.
2. Their masked Q12 gate/up outputs.
3. The truncate/reduce carry bit.
4. Its masked Q6 output.
5. Masked nonnegative and in-interval bits.
6. A masked clipped absolute table index.
7. The lookup's masked inner-product value.
8. The masked `ReLU - residual` activation.
9. The masked product with up.
10. Its GapARS carry bit.
11. The masked Q12 block result.

The sign and interval predicates share one DPF, but use independent output
masks; only the combined membership share is sent. Opening two comparisons
with a cancelling shared output mask would leak the membership bit.
SelectLin supplies the clipped index; selection supplies ReLU. The lookup
evaluates the **whole public table** locally, never making a private-indexed
remote query. Its Boolean DPF difference is a signed point function. Local
bit-count differences supply sign shares; a fresh masked opening and two
arithmetic correction shares restore the table value with its output mask.
This explicit layout does not implement the paper's seeded distribution.

Masked multiplication expands `(xhat-rx)*(yhat-ry)+rout`; its coefficients are
secret-shared, not published. Adjacent gates bind the same wire mask. Addition
composes ReLU and residual shares directly, with no standalone unmasked
activation or generic Beaver reshare at that boundary.

Frames bind the public context, immutable numeric profile, random issuance,
party, phase and lane count. Public contexts are caller declarations, not
compiler validation or remote attestation. Malformed input/frame, replay,
cancellation and completion burn that party's retained keys. The reference
caller cancels its peer when an attempt fails. There is no authenticated
transport or malicious-result verification.

## Complete raw material accounting

All fields are counted from actual packed secret-payload serialization; metadata
and eventual dealer transport framing are additional. Expanded lookup vectors
and allocator padding are not confused with persistent key payload.

| Component | Bits per party per element |
| --- | ---: |
| Three 40/12 GapARS operations | 3,078 |
| 19/6 truncate/reduce | 90 |
| Shared sign/interval DPF plus two output corrections | 908 |
| Clipped-index SelectLin | 80 |
| ReLU selection | 160 |
| Boolean-DPF lookup and two arithmetic correction shares | 726 |
| Masked multiplication | 120 |
| **Complete block** | **5,162 = 645.25 bytes** |

Bit packing spans lanes. Sixty-four lanes retain **41,296 bytes per party**;
eleven peer phases transmit **6,020 bytes across both directions**, including
their 78-byte frame headers. Admission checks a conservative combined material,
scratch and framing allocation before random issuance; at most 256 lanes and
64 MiB estimated allocation are allowed per attempt. This bounds the local
reference, not full-width decoder execution.

## Measured reference and Qwen-shaped projection

Twelve fresh native processes check **5,136 protected block outputs** against
the independent Python integer oracle. Median-of-process-medians for 64 lanes
is **12.70 ms issuance** and **7.15 ms local online execution**, including both
evaluators and the integer client boundary. These are isolated CPU component
times, not paper-GPU, WAN or full-decoder measurements. One-scalar clear timings
can fall below the timer resolution; no speedup claim uses them.

The compiler counts `24 * 4864 * (39+8-1) = 5,369,856` gated elements. At
256-lane batches, the new layout projects:

| Cost owner / scope | 39+8 tokens | 39+32 tokens |
| --- | ---: | ---: |
| Both parties' raw one-use material | **6.9298 GB** | **10.5453 GB** |
| Per-party raw one-use material | 3.4649 GB | 5.2727 GB |
| Both-direction peer frames | **397.12 MB** | **604.31 MB** |
| Additional client input/output payload if not resident | 134.25 MB | 204.29 MB |
| Historical prepared whole-response covered bodies | 116.84 MB | 178.97 MB |

An 8,192-byte public table per party is additional. Metadata, offline transport
framing, protected float conversions and other decoder operators remain extra
or unknown. The peer projection includes 361.12 MB arithmetic payload and
36.00 MB frame headers at 39+8. Even its peer payload alone exceeds the permissive
whole-response comparator. Raw material distribution is a separate cost; these
figures are neither peak memory nor measured full-wire traffic.

**6.93 GB versus the historical 60.10 GB is an 8.67-fold layout reduction**, not
an apples-to-apples runtime speedup: rounding, precision and approximation all
changed. The 60.10 GB figure still belongs to its original reference; neither
number is a lower bound on SIGMA, FuseFSS or general private inference.

On twelve fresh Qwen2.5 public prompts, the unmodified control preserves every
logit/KV value in all 48 positions. The frozen residual profile has no range
rejections and checks **61,870,080 elements**, but matches only **10/12 prefill
and 34/36 same-token decode selections**. No checked full logits or KV are
bit-exact; worst absolute logit error is **2.7158**, mean top-5 recall **0.9083**.
Zero new swap is observed. This fails the predeclared numeric screen.

**Optimization decision: no-go for recommending this fixed mapping and layout.**
Both fidelity and the selected complete-material/traffic budget fail. The
correct bounded reference remains useful for **opt-in user-driven SDK operator
benchmarks**; poor results should be reported, not hidden. Installed bindings and
typed benchmark reports are the next SDK delivery. Current reproduction remains
repository-only; it does not yet claim SDK availability or complete-paper coverage.

Then compare source-specified FuseFSS fusion and distribution on this baseline,
accounting for every conversion and round. Improved numeric mappings need new
held-out prompts. Unknown costs remain unknown and offline costs remain charged.

## Reproduction and remaining gates

```bash
uv run python -m benchmarks.research.masked_wire_reference \
  --output .artifacts/masked-wire-reproduction.json
cargo test --locked -p pllm-core -p pllm-garble
uv run pytest -q tests/test_masked_wire_reference.py
```

The Python command builds the native example, runs fresh-material native cases
in twelve fresh processes, then executes the clear same-token checkpoint
diagnostic under the existing host-memory/swap watchdog. It never activates a
Pipeline choice. Public sources must already be available in the shared model
and paper caches. Existing evidence files are not overwritten.

The reference still needs numeric fidelity, complete operator cost improvement,
independent cryptographic review, protected input conversions or an admitted
fixed-point decoder, compiler validation, independent issuance and role transport.
FuseFSS's general fused schedule, the rest of SIGMA's transformer, and paper GPU
benchmarks remain separate work. These results cannot reject all constructions
from either paper.
