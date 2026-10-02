# E3 exact packed public-linear HE review — 2026-10-01

**Decision: veto; research nonselectable.** Stop reason: `full_projection_exact_but_region_body_gate_failed`.

## Actual execution

Pinned `Qwen/Qwen2.5-0.5B-Instruct@7ae557604adf67be50417f59c2c2f167def9a775`, layer 0 `model.layers.0.mlp.down_proj.weight`,
existing SymmetricPerRow W8A8. Raw BF16 shape `[896,4864]`, output-by-input;
loaded float32 source, quantized int8 weights and original output scales are
SHA-pinned and match prior prepared-stage evidence.

One **full 4864-input row → 896 ciphertext outputs**, packed into one output
ciphertext: actual full-shape executed = **True**;
exact verified outputs = **896**.
Input is an adversarial synthetic A8 row `127*sign(W[max-L1-row])`, not a
prompt-derived Qwen activation. It saturates actual signed bound **18,993,866**.
Full symmetric W8A8 shape bound is **78,451,456**. No prompt/full-decoder parity claim.

| Term | Actual result |
| --- | ---: |
| Two input BFV ciphertext bytes | 740,340 |
| Packed output BFV ciphertext bytes | 370,148 |
| Total online ciphertext bodies | 1,110,488 |
| One-client public context + Galois-key setup bytes | 36,030,317 |
| HE client online CPU seconds | 0.008357 |
| HE provider CPU seconds | 42.781816 |
| HE complete stage-operation CPU seconds | 42.790173 |
| Client/provider setup CPU seconds | 0.358909 |
| Prepared same-stage measured covered body bytes | 27,649 |
| Prepared native stage-operation aggregate CPU seconds | 0.008003 |
| Two-offset native arithmetic CPU seconds (partial comparator) | 0.002991 |
| Sampled parent + worker peak RSS bytes | 556,367,872 |
| Observed worker CPU seconds | 46.748368 |

## Arithmetic, packing and resources

TenSEAL 0.3.17 / SEAL BFV, degree 8192, plaintext prime **268369921**
(`1 mod 16384`), coefficient bit sizes `[55,54,54,55]`, total **218**:
SEAL default tc128 degree-8192 coefficient bound. Parameter validity and batching
are checked natively; actual `parms_id`, binary/source SHA pins are in screen JSON.
TenSEAL cannot expose coefficient Modulus values or security enum through its
bindings; those getters are null, not fabricated measurements.

`t > 2*78,451,456` proves centered BFV output uniquely represents every public
domain dot, including intermediate partial sums. Bound is below signed int32
half-ring; converting centered output mod `2^32` reproduces prepared semantics.
Int64 oracle cannot overflow. Original client activation/weight scales and bias
semantics remain outside encrypted integer evaluation; no modular lift of random
prepared masks into BFV is assumed.

Scheme uses two padded 4096-slot inputs. Each public output uses two plaintext
dots (12 rotate/add steps each), then adds partials. Groups of 32 outputs pack,
then 28 groups pack to final 896-slot output. Full-width non-power-of-two input
cannot use direct TenSEAL dot (`step count too large`); an unpadded 768-slot tail
also corrupts later packed outputs. Padding to full BFV row is compulsory.
Fresh-context positive/negative boundary tests cover this regression and nested
packing. Provider receives public/Galois context only; no secret or relin keys.
Client-created default relin keys remain resident but are unused and not sent.

One CPU thread per native kernel/context. Preflight prices key/context copies,
transient encoded weights, ciphertext groups and allocator allowance. Encoding
is streamed, not all public plaintext weights retained. Full retained encoded
copies would cost 234,881,024 bytes for this layout. Supervisor samples aggregate
parent/worker RSS and worker CPU every 20 ms, terminates at 1 GiB / 120 CPU seconds,
and preserves partial evidence. Sampled peak is not an exact allocator high-water mark.
32×4864 native substage is preflight evidence only; full 4864→896 result is separate.

Encoded-copy sizes are analytical buffer estimates, not backend allocation measurements;
exact backend encoded-weight allocation and allocator-exact peak remain null. Native
scratch allowance and aggregate sampled RSS include transient conversions and copies.

## Gate and projection scope

25% targeted-region body reduction **fails**, even excluding key setup. Matching
control executes fresh full-ring seeded masks, native offline correction and online
masked multiplication with existing request/response/correction codecs. Control
does not include live authorization/acks/transport. Client retains **zero dense
stage weights** in both designs, plus **3584 bytes** of output scales. Public stage
weights cost 4,358,144 bytes at each weight-owning provider; HE streaming int64/
plaintext copies, padded zero lanes and evaluation key expansion are charged separately.
Zero client dense-weight retention describes proposed evaluation contract. Co-located
research process also holds public weights for adversarial-input construction and
independent parity oracles; that experimental working set is included in measured RSS.

JSON contains 1/10/100-response horizons for decode first, prefill39 and 39+32
(39 prefill + 31 executed decode rows). Prefill uses **rowwise projection**, no
measured batched HE prefill or cross-client key sharing. Whole-response known-body
projections replace one or all 24 down stages against pinned measured prepared
178,970,558-byte cohort. Only layer 0 weights execute; other layers, full decoder,
sampling/quality and transport remain unmeasured. Shared setup charged once per
client/provider, not once per stage. Actual per-key serialized sizes vary with fresh keys.

Known-body projections below use decimal MB; complete bodies remain unknown.
One-layer response means 39 prefill plus 31 executed decode rows. All-24-layer
projection charges shared key setup once. Single-layer whole-response subtraction
uses average down-stage attribution (`45,226,088 / 24`), not measured layer-0 attribution.

| Setup horizon (responses) | One-row decode MB | Prefill39 one-layer MB | 70-row one-layer MB | Whole-response, all 24 down stages replaced MB |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 37.141 | 79.339 | 113.764 | 2,035.395 |
| 10 | 4.714 | 46.912 | 81.337 | 2,002.967 |
| 100 | 1.471 | 43.669 | 78.094 | 1,999.725 |

Matched measured prepared whole-response bodies: **178.971 MB**. All-layer
known-body projection is **11.17–11.37× larger**, not a decoder measurement.

One-row client online screen cap is 1 CPU second; complete client cap and aggregate
compute comparison against complete two-offset protocol remain unadmitted. Native
two-offset arithmetic is explicitly partial. Missing admission/control, full wire,
cold weight distribution, prompt activation parity, prefill execution, complete
two-offset and full decoder CPU costs stay **null**. No missing term priced as zero.
Honest-but-curious research contract; malicious integrity cost unknown. No live
protocol, SDK/runtime/compiler integration or selectable component.

## Reproduction

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/probe_exact_linear_he.py --write-evidence
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_exact_linear_he.py
.venv/bin/ruff check scripts/probe_exact_linear_he.py tests/test_exact_linear_he.py
.venv/bin/ruff format --check scripts/probe_exact_linear_he.py tests/test_exact_linear_he.py
```

Screen SHA256: `f03b0ea4775dbda03b157722d2d9da079982c88be391a5c008adfb5c7ba54d8d`.

Validation completed: **11 tests passed** (fresh-context positive/negative full-width
boundaries, nested packing, insufficient modulus/domain rejection, native prepared
u32 masks/offsets, archived full-shape parity/resource gate and SHA audit). Ruff lint
and format checks passed for both owned Python files.
