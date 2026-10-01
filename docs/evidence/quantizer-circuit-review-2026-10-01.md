# Quantizer-aware circuit synthesis: bounded encoded-function screen

Date: 2026-10-01. Track-local research; no runtime component is activated.

## Decision

**Finite-domain encoded-function synthesis works exactly, and common-subexpression
sharing materially reduces circuit size. This construction fails the traffic
gate by orders of magnitude. It establishes neither an exact Qwen circuit nor a
network saving over the prepared baseline.**

The compiled circuit computes output code bits directly. Float32 nonlinear
intermediates are evaluated only while compiling the public truth table, not
inside the Boolean circuit. The strongest measured fixed-scale implementation
has **7,657 non-XOR gates** per scalar gate/up pair. At a byte-packed label width
that retains at least 128 hidden bits after exposing the selection bit, fresh
half-gate ciphertexts alone cost **260,338 bytes per pair**.

Repeated across compiler-pinned Qwen MLP element counts, that specified
construction projects **1,397,977,571,328 / 2,127,357,173,760 bytes** for 39+8 /
39+32. Actual Qwen's input domain is different, and its dynamic scales and
conversion costs are not implemented. These are conditional construction prices,
not measured protected decoder traffic or lower bounds on all synthesis methods.

The prepared path already performs SiLU, gate multiplication and activation
quantization at the client. There are no garbled gates there to remove. A smaller
Boolean circuit therefore does not itself remove a masked-linear stage message:
**zero stage crossings are demonstrated eliminated** by this probe.

## Numeric contracts and independent checks

### Fixed public-scale scalar

The input domain is the complete two's-complement 8-bit pair `(g,u)`, including
the extra minimum value -128. This is a total circuit with no secret validity
branch or unspecified don't-care inputs. The symmetric W8A8 quantizer ordinarily
emits only -127..127; including -128 is explicitly a domain extension here.

```text
x = float32(g) * float32(1/128)
y = float32(u) * float32(1/128)
a = actual SemanticDecoderRuntime._local("silu", x)
p = float32(a * y)
q = clip[-127,127](round_even(float32(p / float32(1/128))))
output = signed i8 q, encoded in two's complement
```

SiLU preserves the actual stable float32 operation order: nonnegative inputs use
`x / (1 + exp(-x))`; negative inputs use `(x * exp(x)) / (1 + exp(x))`. Each
multiply, addition and division retains its rounded boundary. It is not a
float64 real-number SiLU table or the earlier rational/quadratic approximation.

The public table calls the actual semantic SiLU and multiply operators plus
`quantize_activation_per_row(bits=8, scales=1/128)`, including installed native
quantizer dispatch. A separate scalar arithmetic oracle computes those rounded
operations independently, using `np.exp` for the exponential primitive and
Python `round(float(...))` for ties-even after the required float32 division.
It does not call the vector runtime or `np.rint` quantizer. **All 65,536 pairs
agree in both product float32 bits and encoded output bits.** The two arithmetic
oracles share NumPy's exponential primitive, so this is not an independent
correct-rounding proof of exp or portability across arbitrary exp backends.

### Whole tiny vector, with dynamic scale retained

The second contract is an entire two-element vector with four signed 4-bit
inputs, each dequantized at public scale 1/8. It applies the same actual
float32 SiLU→multiply program separately to both elements, then the actual
dynamic signed-i8 row quantizer:

```text
maximum = max(abs(p0), abs(p1))
scale = float32(maximum / 127) if maximum > 0 else float32(1)
q_i = clip[-127,127](round_even(float32(p_i / scale)))
output = q0 bits || q1 bits || exact 32-bit float32 scale word
```

The independent scalar oracle checks **131,072 float32 products** and **65,536
complete code-plus-scale outputs**, with zero mismatches. There are 121 distinct
scale words and 41,444 distinct encoded vector outputs. The circuit has all
**48 output bits**, not only the codes. Scale is never assumed public; it is
merely represented as output bits in the clear correctness evaluator. A real
protocol must keep it shared or visible only to an authorized client.

This demonstrates that tiny dynamic max and rounded scale can be synthesized,
but not that the method scales to a 4,864-element row. Both cases have 16 input
bits, but different numeric domains and output widths. Their gate-count ratio
must not be described as the isolated overhead of dynamic scaling.

### Independent edge tests

The probe and focused tests independently check **780** hand-derived cases:
260 signed half-integer thresholds, each exact tie and its immediately adjacent
float32 values, including saturation. Expected tie results come from integer
parity, not the implementation's rounding function. Zero dynamic rows retain
scale 1 and zero codes. An actual float32 counterexample remains explicit:

```text
row = [1, 0.035433072596788406]
dynamic quantization:                 [127, 4]
divide row by float32(3), then quantize: [127, 5]
```

Thus ideal real-number common-scale cancellation is not silently substituted
for the rounded program. Tests also cover all 16 two-input Boolean functions,
complemented outputs, reversed variable order, partial evaluator chunks,
two's-complement minimum inputs, vector feature permutation and scale identity,
bounded-width rejection and evidence accounting.

## Circuit construction and measured results

The new Python builder emits an explicit topologically ordered XOR/AND program
with constants and free complemented edges. A Shannon branch becomes:

```text
mux(s, low, high) = low XOR (s AND (low XOR high))
```

The generic comparator is deliberately unoptimized: separate output-bit trees
with constant folding but no global gate reuse. The genuinely shared variant
hash-conses normalized XOR/AND expressions across subtrees and output bits;
repeated Boolean subfunctions are evaluated once. The first comparison uses
identical public tables, variable order, constant folding, domain and limits.
An additional predeclared interleaved ordering reports order sensitivity rather
than claiming global optimality. This is a shared Shannon circuit, not a claim
to minimum ROBDD size, arithmetic circuit synthesis or threshold optimality.

The independent bit-sliced clear evaluator executes every generated gate and
exhaustively checks all assignments for every variant. It uses no table lookup
and never follows a secret-dependent branch through the decision diagram.

| Contract / variant | ANDs | XORs | AND / total logic depth | Fresh ciphertext bytes |
| --- | ---: | ---: | ---: | ---: |
| Fixed scalar, generic grouped | 64,636 | 71,772 | 15 / 44 | 2,197,624 |
| Fixed scalar, shared grouped | 8,473 | 16,729 | 15 / 44 | 288,082 |
| Fixed scalar, shared interleaved | **7,657** | 14,022 | 15 / 44 | **260,338** |
| Dynamic two-element vector, generic grouped | 595,388 | 762,797 | 15 / 44 | 20,243,192 |
| Dynamic two-element vector, shared grouped | **27,850** | 53,897 | 15 / 44 | **946,900** |
| Dynamic two-element vector, shared interleaved | 42,396 | 78,453 | 15 / 44 | 1,441,464 |

Same-order CSE reduces AND counts **7.63×** for the fixed scalar and **21.38×**
for the dynamic vector. Depth does not improve. XOR and complemented edges are
free only in ciphertext count, not in local compute or storage.

| Contract / variant | Synthesis seconds | Tracemalloc peak bytes | Public circuit JSON bytes | Exhaustive clear evaluation seconds |
| --- | ---: | ---: | ---: | ---: |
| Fixed generic | 3.057 | 19,918,296 | 2,441,444 | 1.085 |
| Fixed shared grouped | 1.974 | 8,281,656 | 458,050 | 0.203 |
| Fixed shared interleaved | 1.818 | 6,824,408 | 388,492 | 0.178 |
| Dynamic generic | 25.389 | 179,126,256 | 26,691,985 | 10.595 |
| Dynamic shared grouped | 13.409 | 19,511,824 | 1,558,904 | 0.663 |
| Dynamic shared interleaved | 14.069 | 28,399,960 | 2,327,275 | 1.031 |

These are one-run Python diagnostics; synthesis timings include tracemalloc
instrumentation. Peak is tracked Python allocation, not process RSS or total
NumPy/native allocation. Public circuit serialization occurs after tracing.
Clear evaluation is not cryptographic backend latency. Compilation is bounded
to 16 input bits, 2,000,000 wires and 1,000,000 AND gates. Exhaustive truth-table
compilation is exponential and is not proposed for a whole Qwen layer.

## Fresh material, hidden-label entropy and directed cost obligations

Primary source: Zahur, Rosulek & Evans, *Two Halves Make a Whole*,
[ePrint 2014/756 PDF](https://eprint.iacr.org/2014/756.pdf), Table 1 and Figure 2:
two ciphertexts per AND, four garbler hashes and two evaluator hashes, compatible
with free XOR. Sections 3.2/4 describe label/selection-bit and hash assumptions;
page 19 explicitly notes the one-bit degradation from point-and-permute. Only
the PDF and local numeric oracles are used; no upstream implementation is
downloaded or imported.

The requested **128 hidden-bit entropy** is stronger than simply quoting a
128-bit serialized label: an exposed selection bit leaves 127 other bits in
that conventional label. The main projection conservatively uses a hypothetical
129-bit label padded to **17 bytes**, costing **34 bytes per AND**. No reviewed
129-bit hash/garbling backend exists in this probe. This width is a payload
projection under the source's parameterized construction, not a security claim.
The evidence separately records the conventional 16-byte-label **32 B/AND**
comparator and a 32-byte-label **64 B/AND** alternative. Even the 32 B comparator
is overwhelmingly above the control.

Each core has 16 input bits. At 17 B/label, selected input labels cost 272 B,
and input encoding plus global delta storage costs 289 B. Returning all output
labels costs 136 B for the fixed scalar or 816 B for the vector; authorized
plain decoding metadata is 1 B or 6 B before framing. This metadata must not be
given to providers when it would expose activations or row scales. Depending on
abstract input ownership, 16 or 8 function-input OTs are required; their protocol
bytes remain **unknown**. They are not added to the selected-label count as if
one were a complete OT transcript.

The core assumes encoded function inputs are already available as labels. Actual
resident worker inputs would be separately held arithmetic shares. Reconstruction
inside a circuit needs share inputs, private carries and additional gates/OTs;
output conversion must restore suitable shares for the down projection. Those
costs are explicitly **unpriced**, not replaced by the 16 core input bits. If
labels remain internal to a larger region, repeated input/output transfer may be
avoided, but the core's fresh AND ciphertexts remain charged for this construction.

Fresh circuit tables flow garbler→evaluator. Input delivery/OT and output labels
have their own directions and ownership-dependent placement. A dealer topology
must additionally charge dealer→each worker, independent input/output material,
setup, cold checkpoint distribution, framing and full wire. The experiment does
not instantiate that topology or declare those unknown links free. Public circuit
structure can be reused; ciphertexts, labels, global offsets and input correlations
must be fresh per invocation and burn on cancellation/replay. No private Python
table lookup, branch, or reusable low-entropy masking seed is admitted.

## Pinned whole-MLP repetition and baseline comparison

Source: `Qwen/Qwen2.5-0.5B-Instruct@7ae557604adf67be50417f59c2c2f167def9a775`.
Cached official config only; weights are not loaded. Existing
`probe_projective_feasibility.pinned_inventory` validates actual compiler operator
coverage, semantic lineage, dimensions and runtime schedule against the matched
control. Existing `probe_region_contract_cost.run` independently supplies the
semantic MLP projection MAC inventory. No operation-ID string parsing is used.

| Compiler / cost quantity | 39+8 | 39+32 |
| --- | ---: | ---: |
| Executed rows, `39 + generated - 1` | 46 | 70 |
| Layers / hidden / intermediate | 24 / 896 / 4,864 | 24 / 896 / 4,864 |
| SiLU / gate×up elements each | 5,369,856 | 8,171,520 |
| MLP rows requiring whole-row dynamic scale | 1,104 | 1,680 |
| MLP integer projection MACs per resident worker | 14,434,172,928 | 21,965,045,760 |
| Matched prepared covered all-link bodies | 116,843,966 B | 178,970,558 B |
| Matched prepared online all-link bodies | 73,831,744 B | 113,545,024 B |
| Historical tenfold all-link / online budgets | 11,684,396 / 7,383,174 B | 17,897,055 / 11,354,502 B |
| Fixed generic fresh core ciphertexts | 11,800,924,422,144 B | 17,957,928,468,480 B |
| Fixed shared grouped fresh core ciphertexts | 1,546,958,856,192 B | 2,354,067,824,640 B |
| Fixed shared interleaved fresh core ciphertexts | **1,397,977,571,328 B** | **2,127,357,173,760 B** |
| Selected core input labels if all re-encoded | 1,460,600,832 B | 2,222,653,440 B |
| Core output labels if all returned | 730,300,416 B | 1,111,326,720 B |
| OTs if all core inputs evaluator-owned | 85,917,696 | 130,744,320 |

Best fixed core ciphertexts alone exceed the **full**, not tenfold, prepared
covered control by **11,964× / 11,887×**. Their garbler/evaluator hash counts are
164,467,949,568 / 82,233,974,784 and 250,277,314,560 / 125,138,657,280.
Backend execution time remains unknown; ordinary projection MACs also remain.

Even a conditional comparator with **one fresh AND per scalar element** at 34 B
would cost **182,575,104 / 277,831,680 B**, above the entire matched prepared
covered bodies before anything else. This rules out pursuing merely a modest
gate-count improvement within that per-element fresh half-gates placement. It
is not a universal circuit-complexity or protocol lower bound: whole-region
encodings, amortization across outputs and other constructions need separate
accounting. The historical tenfold budgets permit only **2.176 / 2.190 B per
gate-product element** for the entire response; incremental savings would still
need to beat the matched full control.

The full MLP includes norm, gate/up projections, SiLU, product, dynamic
quantization, down projection and residual addition. Qwen gate/up outputs are
float32 values from integer accumulators, rounded activation/learned weight
scales and biases, not public-scale i8 values. Private 4,864-way max selection,
rounded rescaling, scale output sharing and linear backend conversion remain
missing. Attention, rotary, protected KV, other norms and final client
head/selection/feedback also remain required in the whole response. If selection
is moved away from the client, a private selector and feedback protocol must be
priced explicitly. Unknown complete-region cost is recorded as `null`.

## Source locks

| Lock | Value |
| --- | --- |
| Official config SHA-256 | `18e18afcaccafade98daf13a54092927904649e1dd4eba8299ab717d5d94ff45` |
| Existing control body fingerprint; weights not reloaded | `5d631be30158b3ea2a72cae355ce09a3b304bbfddde48b5758c339c346a34974` |
| W8A8 composition | `bd91e1659773faabcbb998a350893cbbc6df313a830da7cb7bfb87463c8f0b36` |
| 39+8 plan | `d292d42933e35239655e2a2d09e7bb8ac3c0d29ceb3b96ddd5512836148c82b1` |
| 39+8 schedule | `19d45c42975b80997c85b9c7bf4d594905e485588e7d8dce13b39de3bf478327` |
| 39+32 plan | `62381dbd11470bd150723cfb0d7bab096c52f331ec8e9f5dd23e178df28a0a4d` |
| 39+32 schedule | `bf3bc2e91b6e95a8d0b9ac7eebfb2866a3b3b8c2fe4c2ff401dc625f130a0d5c` |
| Primary half-gates PDF SHA-256 | `789ecaf8651b6dc8341d6033f951d76c0e6f4b04427cdde26cf91ecfb0af3083` |

The JSON additionally locks local semantic/quantizer source hashes, both truth
tables and all six public circuit structures. Recorded evidence factors common
protocol obligations and retains selected stdout fields; it is not a byte-for-byte
copy of full probe stdout. Deterministic counts/digests are reproducible; timings
and peak allocations vary. Evidence class is clear numeric measurement and public
circuit synthesis plus compiler-derived conditional body projections. Protected
execution, security, whole-checkpoint parity and transport savings are unmeasured.

## Reproduction and next gate

New files:

- `scripts/probe_quantizer_circuits.py`
- `tests/test_quantizer_circuits.py`
- `docs/evidence/quantizer-circuit-screen-2026-10-01.json`
- `docs/evidence/quantizer-circuit-review-2026-10-01.md`

```sh
uv run --no-sync python scripts/probe_quantizer_circuits.py --with-pinned-config
uv run --no-sync pytest -q tests/test_quantizer_circuits.py
uv run --no-sync ruff check scripts/probe_quantizer_circuits.py tests/test_quantizer_circuits.py
uv run --no-sync ruff format --check scripts/probe_quantizer_circuits.py tests/test_quantizer_circuits.py
```

Optional `--source-pdf PATH` hashes the locally retrieved primary PDF; the run used
`/var/folders/3d/16hqtng12ng_bfz1n99jg04w0000gn/T/opencode/quantizer-half-gates-2014-756.pdf`.
Without `--with-pinned-config`, only synthetic bounded contracts run. With it,
`local_files_only=True` reads the existing official config and checks controls.
The probe prints JSON and does not issue material or write evidence files.
Verification: **27 focused test cases passed**; focused Ruff lint and format
checks passed. No shared registries, runtime modules, staging or commits changed.

**Recommended next check:** before building a cryptographic backend, derive a
bounded *actual accumulator/rescale→SiLU→multiply→dynamic-quantization* function,
including every float32 edge and its private scale output. Compile at least a
tiny whole row to shared output encodings suitable for down projection, then
price the complete label/share conversion boundary. A viable proposal must avoid
fresh per-element half-gate material at the measured scale, remove enough actual
prepared messages, and account for private max, scale, output sharing and local
compute. Uniform interval certificates or structured threshold predicates might
help synthesis, but neither is implemented or credited here. Another successful
small public LUT, with the real conversion boundary still unknown, is insufficient.
