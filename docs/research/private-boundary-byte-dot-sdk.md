# Opt-in exact byte-dot private boundary head

Status: native/compiler, installed-wheel decoder and SDK lifecycle checks pass.
A separate six-client pinned Qwen2.5 screen preserves complete state within observed
Low RAM, but establishes no serving-speed improvement: median decode falls 0.65%.
Synthetic v3 probe evidence remains separate: six-worker median API CPU 6.54875 to
4.228 ms (35.44% lower), including input packing and owned output allocation.
That is neither isolated SDOT instruction evidence nor an SDK/end-to-end benefit.

## Selection and ownership

```python
from pllm.roles import PrivateLookupAndFullHead

PrivateLookupAndFullHead(head_kernel="byte_dot")
# Orthogonal existing choices remain available:
PrivateLookupAndFullHead(head_kernel="byte_dot", head_dispatch="parallel",
                         head_output_encoding="uniform_residues")
```

Default `head_kernel="existing"` is omitted from component, native contract and
boundary descriptor serialization. Historical serial/u32 default bytes remain
unchanged. `Matrix::wrap32` and all ordinary body products remain unchanged.
The additive model-neutral native API is
`Matrix::wrap32_byte_dot(&Executor, &[u32]) -> Result<Vec<u32>, String>`;
PyO3 exposes `Executor.wrap32_byte_dot(matrix, input_bytes)`, releasing the GIL
through packing and computation. `CompiledMatrix.wrap32_byte_dot` validates a
single `[1,H]` input and returns `[1,V]` u32 output. No environment/global target
feature switch or persistent prepared-weight representation is introduced.

The component selector enters the composition digest, compiler source schedule
contract and schedule digest. Client binding and head-only source loader compare
the complete expected native contract independently, then derive the boundary
descriptor from that validated schedule, not untrusted bundle metadata. The
descriptor digest enters admission/context binding; the worker dispatches only
from its actual descriptor. Unknown, malformed, forged or selection-mismatched
contracts reject. An older installed extension missing the new API rejects the
opt-in worker at construction, rather than silently routing to a different
runtime implementation. Existing one-use reservation, burn-before-work,
shielded cancellation and capacity-retirement code is unchanged.

## Exactness, bounds and fallback

Each invocation borrows the existing immutable `Matrix` allocation, packs four
contiguous signed-byte limbs and allocates owned output. For each u32 input,
repeat `d=((carry+128)&255)-128; carry=(carry-d)>>8` four times. The final carry
multiplies 2^32 and vanishes modulo the ring. Each row reconstructs its four
signed dot products using wrapping shifts/adds. Reinterpreting raw bytes without
balanced carries would be incorrect and is not used.

Width is at most 131071, so every limb's full absolute sum is bounded by
`H*128*128 <= i32::MAX`, including vector horizontal reduction and scalar tails.
Input length must exactly equal matrix width. Weights are at most 256 MiB;
the conservative weights + input + limbs + output live payload is at most
512 MiB, checked before slab/output allocation. Matrix construction already
requires positive dimensions and checked weight product. SDK admission retains
its stricter existing table/frame/hidden bounds, plus the byte-dot weight cap.
The Python wrapper requires an already-shaped `[1,H]` NumPy array and checks the
same geometry/live-payload ceiling before integer scanning or conversion. Its
explicit scalar correctness reference retains the i8 weights and one u32 output;
it never expands the full matrix into object arrays. Malformed multi-row batches
therefore cannot allocate conversion copies before rejection.

ARM SDOT uses a locally scoped target-feature function and stable inline assembly
(matching the independently implemented probe's Rust 1.85-compatible approach).
Runtime `dotprod` detection gates entry. SIMD-disabled executors force exact
balanced-limb scalar execution; non-aarch64 and no-dotprod hosts take the same
scalar fallback. Full sixteen-byte loads occur only within slice bounds; scalar
tails handle every remaining element. No unsupported instruction can be reached
through selector alone. Physical no-dotprod/non-aarch64 hosts remain untested by
this worker; portable cfg tests and a parent cross-target command are provided.

## Resource admission: existing conservative frame bound covers scratch

There is no new resident weight owner. The only additional algorithmic scratch
is `4H` bytes, shared across all output rows and executor threads, released per
call. Output is `4V`, versus the existing API's transposed accumulator and output.

The production resource model in `runtime/benchmark_memory.py` already reserves
`8*max(lookup_frame, head_frame)` bytes per concurrent worker request, independently
of lookup scratch, import peak, process allowance and residue codec allowance.
Let `F = 8192 + 4(H+V)` be its full-width head frame bound. Conservatively retaining
the parsed input share, Python serialized input, native decoded input and new
limbs totals `16H`. Native output, temporary output serialization and returned
PyBytes total at most `12V`: together `16H+12V <= 4F`. NumPy input/result views
do not own another payload. Request chunks/join/unpack overlap precedes compute
and fits three frames; they do not coexist with native packing. After native
return, decoded input/limbs/native output/serialization temporaries are released;
even retaining the input share, result bytes, head-codec input copy, codec output
and packed HTTP response fits five full-width frames. Thus the unchanged eight
frame allowance covers the additional slab without borrowing lookup scratch or
assuming `V >= H`. Uniform residue encode scratch retains its separately charged
existing allowance. Parallel dispatch runs one head per boundary process, with
independent per-request allowance on each worker; executor threads share the
single immutable slab. No resource bound was reduced or measurement helper edited.
These are logical payload bounds, not RSS or a measured Low-client claim.

## Tests and actual checks

Added Rust unit cases cover full-ring values, carry edges, signed -128 weights,
wrapping outputs, widths/tails 1/3/4/15/16/17/31/32/33/127/896/131071, maximal
positive limb accumulation, scalar/SIMD and one/two executor threads, malformed
lengths, excessive width and non-aarch64 fallback. Installed-wheel native tests
compare complete integer vectors against Python arbitrary-precision arithmetic.
Compiler tests bind byte-dot across generated Qwen2/Qwen3/Llama schedules while
preserving body steps. Generated decoder tests compare every head share's whole
integer output against ordinary wrap32, then full logits, KV, tokens and RNG
state. SDK role fixtures add byte-dot serial/u32 and parallel/residue cohorts,
including existing real cancellation/one-use lifecycle cases.

Serial parent validation passed four native arithmetic tests and eight compiler
tests, then 192 installed-wheel native, option, decoder and SDK checks. Generated
Qwen2/Qwen3/Llama controls compare complete state, including full-vocabulary logits.
The guarded six-client tiny matched benchmark and pinned-checkpoint admission also
pass. Malformed geometry and resource-limit regressions prove rejection precedes
conversion. Initial worker validation used 29 source mocked/unit checks:

```sh
target/frontier-sdk-env/bin/python -m pytest -o 'pythonpath=python .' -q tests/test_private_boundary_byte_dot.py tests/test_private_boundary_options.py -k 'not native_contract and not loader_derives and not client_descriptor'
```

## Serial verification commands

Use a rebuilt installed wheel for native and SDK checks. Source-only mocked
checks cannot establish native integration. Keep builds and model jobs serial:

```sh
cargo test -p pllm-core --lib byte_dot::tests
cargo test -p pllm-compiler --lib decoder_boundary::tests
# On a host/toolchain with the target installed; no forced target features:
cargo check -p pllm-core --target x86_64-unknown-linux-gnu

# Use a fresh wheel output directory; rebuild/install only under parent ownership.
target/frontier-sdk-env/bin/python -m maturin build --release --out target/byte-dot-wheels
uv pip install --python target/frontier-sdk-env/bin/python --reinstall --no-deps target/byte-dot-wheels/pllm_run-*.whl
target/frontier-sdk-env/bin/python -m pytest -o pythonpath=. -q tests/test_native_byte_dot.py tests/test_private_boundary_options.py tests/test_private_boundary_byte_dot.py
target/frontier-sdk-env/bin/python -m pytest -o pythonpath=. -q tests/test_private_boundary_decoder.py -k byte-dot
target/frontier-sdk-env/bin/python -m pytest -o pythonpath=. -q tests/test_private_boundary_sdk.py -k byte-dot
# Baseline compatibility and lifecycle regressions:
target/frontier-sdk-env/bin/python -m pytest -o pythonpath=. -q tests/test_private_boundary_protocol.py tests/test_private_boundary_worker.py tests/test_private_boundary_parallel.py tests/test_private_boundary_head_codec.py
```

Verify the installed-wheel glob matches one current wheel before installing.
The frontier environment is pip-free; the actual parent installation used `uv pip`.
Physical no-dotprod/non-aarch64 execution remains unmeasured. Forced scalar controls
establish arithmetic fallback on the tested Apple Silicon host.

## Matched installed SDK screen

Six fresh pinned Qwen2.5-0.5B clients, three balanced replicates per kernel, share
one installed build and private cohort. Every request reaches 150+8 tokens and
seven decode intervals. Both arms use parallel dispatch and uniform-residue replies;
the only component change is `head_kernel="byte_dot"`. Shared native 100/40 caps
cover all five roles. All **52,561,920** checked full-logit/active-KV values and
history, position, selection, RNG and output commitments match. Observed swap
growth is zero. Each request claims all 14,625 issued rows; none burn or discard.

| Metric, median unless stated | Existing head | Byte-dot head |
| --- | ---: | ---: |
| N−1 decode, tokens/s | 2.7452 | 2.7275 |
| Full request, s | 82.561 | 82.664 |
| Startup through cleanup, s | 110.329 | 109.828 |
| Online first text, s | 47.386 | 47.322 |
| Client cold CPU, s | 20.614 | 20.868 |
| Five-role cold-first CPU, s | 65.186 | 65.438 |
| Client lifetime peak, MiB | 223.97 | 218.53 |
| **Largest** complete-client peak, MiB | **225.625** | **222.422** |
| Preparation cold CPU, s | 15.407 | 15.194 |
| Preparation sampled peak, MB | 209.62 | 215.97 |
| All-link TCP streams, MB | 365.221 | 365.221 |

Decode falls **0.65%**; request time rises **0.13%**, client CPU **1.23%** and
five-role cold-first CPU **0.39%**. Head and prepared arithmetic bodies are equal;
paired covered-body deltas are **0/1/0 bytes**, entirely lifecycle control. The
reported one-byte median traffic difference is retained, not a codec saving.
Median client RSS falls **2.43%**, but one paired sample rises and no client weight
owner was removed. These observations establish no causal memory or speed gain.
Both arms qualify observed Low/Medium using their largest complete worker peaks.
The default stays unchanged; this explicit selector remains available for research.

Client RSS includes SDK, supervisor, dashboard, WAN relay and full-state audit;
the idle launcher and provider processes are separate. Cold-first CPU ends after
the response; startup-through-cleanup time includes retirement. Source checkpoint
and indexed tokenizer are pre-positioned. Distribution, physical wire, Internet RTT,
independent operators and representative quality remain unmeasured. Historical
cohorts and their different timings do not supply controls for this screen.

The ordinary `BenchmarkTarget.response`/`benchmark_suite` driver and stricter
Python/JavaScript readers reconcile raw metrics, state, bodies, TCP streams,
material, all five CPU owners and largest-worker tier qualification. Research
scorecard registration preserves this non-winning speed result as a **screen**.
Original bytes: `docs/evidence/private-boundary-byte-dot-qwen25-sdk.json`,
`.plan.json`, `.summary.json`, `.progress.json`.

```sh
target/frontier-sdk-env/bin/python scripts/benchmark_private_boundary_byte_dot.py \
  --artifacts target/prepared-frontier-artifacts \
  --output target/private-boundary-byte-dot-new.json

# Read-only archived replay; no model or provider execution.
target/frontier-sdk-env/bin/python - <<'PY'
import json
from pathlib import Path
from benchmarks.research.private_boundary_byte_dot import audit
p = Path('docs/evidence/private-boundary-byte-dot-qwen25-sdk.json')
assert audit(json.loads(p.read_bytes())) == json.loads(p.with_suffix('.summary.json').read_bytes())
print('Byte-dot SDK summary replays exactly.')
PY
```

The separate [role profile](private-boundary-role-profile.md) measures the original
native head's cost within the exchange; the [full-size fusion reference](boundary-scan-fusion-full-reference.md)
tests scan sharing and its lookup-delivery delay. Their observations are not added
to this cohort or treated as byte-dot attribution.
