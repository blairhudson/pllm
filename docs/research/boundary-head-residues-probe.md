# Source-certified full-head low residues: isolated probe

## Result and scope

**Codec correctness established on small synthetic fixtures only. No production
integration, actual Qwen width histogram, whole-response measurement, security
promotion, or performance promotion.** Only this note, the probe, and its tests
are added. No runtime, shared manifests, or core changes are required.

The current private-boundary head returns two full `u32[V]` outputs. For
`V=151936`, `H=896`, W8A8, each reply is 607,744 bytes. A public conservative
25-bit ring reduces each payload to **474,800 bytes**, saving **21.875%** of
head-output payload. Per-row public L1 certificates can only improve that ceiling.
These are geometry calculations, not measured model histogram statistics.

Two distinct certificate profiles now execute. The minimal-L1 profile requires
actual pinned weights for verification. The **uniform public-geometry profile
requires no client weight read**, using the already-admitted W8A8 dtype,
quantizer, geometry, and source/matrix/scales/descriptor identities. It transmits
one width byte, expanded locally for the existing native codecs.

With proposed 100-byte headers and delivery to both workers and client, eight
uniform-geometry heads save **2,122,510 application bytes** versus headerless raw
reply payloads. Minimal-L1's conservative 25-bit ceiling saves **1,666,666 bytes**
after its larger layout delivery. Neither is a TCP/all-link whole-response result.

The maintained comparison remains
[`private-boundary-tiers-reference.md`](private-boundary-tiers-reference.md):
24 source-present 150+8 clients, exact full-state parity, maximum whole-client
210.22 MiB, 19.93% less traffic and 20.26% shorter full request than measured
incumbents; coupled decode 1.9235 TPS versus maintained approximately 3.794 TPS.
Do not apply the head percentage to those whole-request measurements.

## Exact arithmetic argument

For each public signed-i8 weight row `w_j`, use the public quantizer contract
`q_i ∈ [-127,127]` and compute using integers:

```
B_j = 127 * sum_i abs(int(w_ji))
b_j = max(1, B_j.bit_length() + 1)
```

Thus `B_j < 2**(b_j-1)`, including the positive endpoint. A zero row uses one bit.
`-128` weights contribute 128, not 127. Reject any bound requiring more than 32
bits or violating the independently admitted wrap32 geometry. The Qwen uniform
bound is `896*128*127 = 14,565,376 < 2**24`, hence 25 bits.

Given existing additive inputs `q = x_a+x_b mod 2**32`, workers retain their
original exact wrap32 matrix multiplication. For each output share `y_pj`, send
`y_pj mod 2**b_j`. Since `2**b_j` divides `2**32`, the client adds the two low
residues modulo `2**b_j`, then centers at `2**(b_j-1)`. The strict bound above
makes that signed integer exactly the original full dot, including negative
wraps. No independently signed interpretation of either share is valid.

The client must retain full vocabulary order and call the existing
`dequantize_matmul` unchanged:

```
integer.astype(float32) * activation_scale[:, None] * weight_scales[None, :]
```

Do not reassociate multiplication, dequantize shares separately, substitute
float64 products, truncate vocabulary, or change logits/selection semantics.
The probe compares complete float32 byte strings, not tolerances or argmax only.

## Compiler-neutral certificate and trust boundary

`scripts/probe_boundary_head_residues.py` implements a research namespace, not an
installed boundary descriptor namespace. Its canonical JSON metadata binds:

- source-lock, checkpoint, model-config, and existing boundary descriptor digests;
- actual row-major signed-i8 matrix and little-endian float32 scale digests;
- vocabulary, hidden dimension, public qmax=127, one row/call, bias absent;
- named bound method, named encoding, SHA-256 of the ordered width bytes;
- SHA-256 of locally recomputed little-endian `u64[V]` absolute bounds.

Both profiles bind `weight_dtype=i8` and the explicit
`symmetric-per-row-i8-qmax127-v1` activation quantizer. The certificate ID is
SHA-256 of a domain separator, canonical metadata, and **transmitted** layout.
Profile `minimal-row-l1-i8-v1` transmits ordered width bytes, one per row. Full
bounds need not travel: the verifying recipient recomputes them from the pinned
actual weights. Its verifier still requires minimal widths exactly.

**A matching self-declared hash is not source authentication.**
`verify_certificate` requires an independent trusted pin and actual weights and
scales, hashes those bytes, independently recomputes every bound and minimal
width in Python integers, and compares the installed native width derivation.
It then requires exact metadata/layout equality. Changing a source digest,
weights, scales, bound digest, qmax, width, row count, or encoding does not install
an alternative certificate. Even wider-than-required layouts are rejected by
this fixed canonical profile.

The independent pin must come from the existing verified source/compiled
descriptor path, never be copied from the untrusted certificate. This helper
does not prove the source-to-weight relation on its own. Anyone can manufacture
a consistent pin and matching synthetic certificate; that is explicitly not
admission. The synthetic report uses conspicuous placeholder source digests.

For the L1 profile, workers already holding source can verify locally. A source-present trusted
client can do the same, paying verification CPU and weight-map/copy cost. A
source-free client cannot validate a self-declared L1 certificate merely by
checking its hash: it needs an independently authenticated compiled admission
that binds the recomputed widths, or an additional verified-proof mechanism.
Neither new trust mechanism for source-free **L1** verification is implemented
here. If L1 verification requires new
client weight delivery, charge those bytes separately; the source-present
traffic arithmetic below does not conceal or include that delivery. This would
defeat the thin-client coupling advantage and is not the preferred thin-client path.

### Distinct source-free uniform-geometry profile

`geometry_certificate(trusted_pin)` and
`verify_geometry_certificate(candidate, trusted_pin)` deliberately have no weight
or scale byte arguments. Their profile is `uniform-public-geometry-i8-v1`, with
bound method `hidden-times-i8-abs128-times-qmax127-v1`:

```
B = H * 128 * 127
b = B.bit_length() + 1
transmitted_layout = bytes([b])
native_widths = transmitted_layout * V
```

The certificate explicitly includes B and b, the profile, dtype/quantizer, H/V,
qmax, bias-free/one-row restrictions, and all existing trusted source, matrix,
scales, checkpoint/config/descriptor identities. Its width digest hashes the
single transmitted width byte. Verification recomputes the scalar bound/width
from the **externally admitted** pin and requires exact equality of metadata,
transmitted layout, and expanded widths. No actual weight magnitude measurement
or model histogram is needed: every admitted i8 value has magnitude at most 128.
Scale values still undergo the existing client scale-commitment/finite-positive
checks when logits are dequantized; residue admission adds no weight download.

This does not authenticate opaque hashes. An attacker supplying both a pin and
certificate can make a self-consistent pair; tests explicitly demonstrate that
such a pair is rejected against the independently admitted pin, not magically
authenticated. The external admission must already guarantee W8A8 dtype and
activation range, not merely carry arbitrary strings. Workers retain existing
actual weight/scales installation checks.

Uniform widths can be wider than actual minimal L1 widths, but only in this
separately named contract. Neither verifier accepts the other's certificate or
an altered profile/bound/dtype/quantizer. Client postchecking uses one scalar B,
not a retained u64[V] bound vector. The executable exercises both native codec
paths against the same synthetic wrap32 shares and full float32 baseline.

Do not choose widths from private activations, observed shares, logits, token
selection, or per-call magnitude. Public fixture RNG is reproducible; all
private additive masks use fresh `secrets.token_bytes`, including tests.

## Frame contract

The experimental header is exactly 100 bytes, little-endian
`<8sBBH32s32sIIQQ`:

| Field | Bytes | Expected value |
| --- | ---: | --- |
| magic/version namespace | 8 | `PLLMHR01` |
| encoding ID | 1 | 1, low residues LSB-first |
| party | 1 | expected A=0 or B=1 |
| reserved | 2 | zero |
| certificate ID | 32 | independently verified certificate |
| context digest | 32 | independently expected head context |
| activation row count | 4 | exactly 1 |
| vocabulary row count | 4 | admitted V |
| sequence | 8 | expected one-use sequence |
| packed payload length | 8 | `ceil(sum(widths)/8)` |

The context digest must bind existing descriptor/admission, paired session,
epoch, operation=head, and sequence through the installed boundary's canonical
context construction, not through worker-supplied arbitrary context. Party is
bound separately in the header. The probe supplies a clearly synthetic context.

Payload iterates vocabulary rows, emits low bits LSB-first, and zero-fills only
the final byte's unused high bits. Exact dimensions, full header, byte length,
and tail padding are checked before native reconstruction. No alternate
encoding, implicit width negotiation, trailing extension, cross-party swap,
cross-context replay, or extra activation row is accepted. Existing transport
must still own replay state, bounded reads, role-specific authenticated channel,
abort/cleanup, and reservation accounting.

These are context checks, not a MAC or proof of honest computation. In-range
payload corruption may remain undetectable. The usual honest-worker/non-collusion
assumptions do not improve. `reconstruct_checked` also checks locally recomputed
public bounds after reconstruction; that detects out-of-bound sums but does not
make malicious matrix multiplication verifiable.

## Costs: exact application bytes and explicit memory inventories

Let `S=sum(widths)`, `P=ceil(S/8)`, metadata length `C`, head calls `N`,
transmitted layout length `L=V` for L1 or `L=1` for uniform geometry.

```
raw two-party output payload                 = 8V per call
new two-party framed output                 = 2P + 200 per call
layout/certificate delivery to A, B, client   = 3*(8+C+L) once
padding per party                           = 8P-S bits
net bytes saved vs headerless raw           = N*(8V-2P-200)-3*(8+C+L)
unchanged two-party head input payload       = 8H per call
```

The eight-byte delivery prefix carries two `u32` lengths for bounded metadata
and layout. The executable charges this prefix but does not install a setup
endpoint. Metadata parsing must be bounded and duplicate-key/canonical-byte
strict when integrated. Existing setup/control exchanges, HTTP/TLS/TCP overhead,
delivery acknowledgements, failures, and stream duplication remain unmeasured.
No compression or certificate-cache hit is assumed.

For Qwen geometry metadata is exactly 1,002 bytes for L1 and 989 bytes for
uniform geometry in these fixed schemas. All digests have fixed lengths. L1's
25-bit layout is a conservative payload ceiling, not an actual histogram or
fabricated certificate accepted by the minimal-L1 verifier. The separately
named uniform profile validly uses 25 bits for every row.

| Item | Bytes |
| --- | ---: |
| raw reply, one party | 607,744 |
| uniform 25-bit packed reply, one party | 474,800 |
| both framed replies/call | 949,800 |
| both-party payload-only reduction/call | 265,888 |
| padding, uniform geometry | 0 bits |
| L1 certificate + layout + prefix, one recipient | 152,946 |
| L1 cold delivery to three recipients | 458,838 |
| L1 eight-call saving, conservative payload ceiling | 1,666,666 |
| L1 break-even calls against headerless raw | 2 |
| uniform certificate + one width + prefix, one recipient | 998 |
| uniform cold delivery to three recipients | 2,994 |
| uniform eight-call saving after new framing and cold delivery | 2,122,510 |
| uniform break-even calls against headerless raw | 1 |
| unchanged both-party input payload/call | 7,168 |

Inventories, **not measured RSS or lifetime peaks**:

- Two packed payloads + native i64 reconstruction + final f32 vector: `2P+12V`
  = 2,772,832 bytes at uniform Qwen geometry.
- Retaining two received frames plus their sliced payload copies adds another
  `2P+200`. Counting f32 cast and both multiply outputs gives
  `4P+200+20V` = 4,938,120 bytes of owned arrays/buffers. Lifetimes may overlap
  differently; CPython allocator, native capacity, HTTP chunk joins, dashboard,
  relay, and already-live model/KV state are extra.
- L1 retained locally recomputed u64 bounds: `8V` = 1,215,488 bytes; uniform uses
  only a scalar. Both expand/retain V native width bytes locally, **not** on wire
  for uniform. Bound checking
  can also allocate an i64 absolute-value vector and a boolean comparison vector;
  those are additional transient arrays, not silently free.
- L1 public source verification weight+scales: `V*(H+4)` = 136,742,400 bytes, before
  Python/native copies, row temporaries, hash traversal, and width/bound buffers
  (`9V` = 1,367,424 bytes). The Python oracle is intentionally not a production
  low-memory compiler. A streaming implementation still needs source provenance.
  Uniform client certificate verification reads **zero** weight/scale bytes;
  ordinary head scales remain necessary for unchanged dequantization.
- Each worker retains its existing raw u32 output while packing: `4V+P`
  = 1,082,544 bytes, plus existing weights/native matrix and framing copies.
- No expanded second client share tensor is required by native reconstruction.
  No claim that these inventories preserve the existing 210.22 MiB tier margin.

CPU reporting separately includes both workers' native pack cost, client native
reconstruction, checked reconstruction, frame validation, unchanged float32
dequantization, original u32 centering, and certificate build plus independent
second recomputation. It excludes matrix multiplication (unchanged), actual
network/HTTP scheduling, and whole SDK overhead. Tiny process-time samples are
correctness smoke diagnostics, not evidence of a speedup.

## Executed evidence and source lock

Installed SDK interpreter, small geometry `V=257,H=32`, three codec repetitions:

- 38 tests pass (small native-codec suite).
- Native widths/packing/reconstruction matched independent Python integer oracle.
- Complete float32 logits matched original wrap32 reconstruction byte-for-byte.
- Tests cover actual input-share matrix products, negative wraps, -128 weights,
  zero rows, widths 1 through 32, signed minima, cross-byte boundaries, forged
  source/weight/bound metadata, altered widths/encoding/row counts, both parties'
  padding, trailing/truncated frames, context/party/sequence substitutions, and
  static geometry/MAC limits, modeled-memory preflight, cross-profile
  substitution, uniform source-free verification (L1/native weight readers
  replaced with failing stubs), and exact ±B versus one-past-bound rejection.
- Synthetic histogram only: `{1:1,13:1,19:118,20:137}`. Packed payload 625 bytes
  per party with four padding bits; L1 metadata 998 bytes. Eight-call net saving
  after framing/delivery is 1,059 bytes. Uniform transmits one width `20`, uses
  983-byte metadata and 643-byte party payloads (four padding bits), saves 1,584
  bytes after framing/delivery. Both yield byte-identical full float32 logits.
- Smoke CPU: L1 build+recompute 0.779 ms; uniform build+source-free verification
  0.039 ms. Tiny codec process times varied, including a 54 µs A-pack outlier;
  three repetitions under concurrent work cannot support timing claims. JSON
  reports each party, reconstruction/checking, frame checks and dequantization
  separately. No microbenchmark numbers are promoted to a speed claim.

No actual quantized Qwen head was opened and independently admitted in this
bounded pass. The existing plan names Qwen2.5-0.5B-Instruct revision
`7ae557604adf67be50417f59c2c2f167def9a775`; that string is **not** proof of an
actual head artifact. Existing frontier artifacts inspected here exposed control
documents/tokenizer rather than a separately admitted quantized head. No
checkpoint quantization, full model loading, downloads, or checkpoint jobs ran.
There is no actual-Qwen histogram in this evidence.

SHA-256 locks at smoke time (the JSON CLI recomputes locks on every replay):

| Component | SHA-256 |
| --- | --- |
| probe script | `c55e86ec7389edfc35f2cc4958c14044a85f6eae4ab8b36ad114b5f1f9fdd164` |
| `crates/pllm-core/src/offset_transport.rs` | `5b379541fa4de7390cc048a564a44616cae7cf6b82c3c869993df2ec76c660c2` |
| source `private_boundary_transport.py` | `6c9c34d974a8e2dee2769bbc45b3af0de2340bc256838d1839181ab4397718d6` |
| installed `_native.abi3.so` | `6b9370fba12ddd3df25979631d2ae9d37097d60bf0ae9bba1cd1bc2e8389a41e` |
| installed `quantization.py` | `89c7013e569cf434d6cd54f6fc3015bc89e5a2e08e351d3b4a48d071d398d412` |

## Replay and later serial measurement

Small correctness/replay commands, safe for concurrent investigations:

```sh
target/frontier-sdk-env/bin/python -m pytest -q -o pythonpath= tests/test_boundary_head_residues_probe.py
target/frontier-sdk-env/bin/python scripts/probe_boundary_head_residues.py --vocabulary 257 --hidden 32 --head-calls 8 --repetitions 3 --output target/boundary-head-residues-smoke.json
```

Static geometry limits: at most 160 MiB i8 weights, 262,144 vocabulary rows,
65,532 hidden columns subject to signed bound, two million Python-oracle MACs,
and 100 timing repetitions. These are **not host-headroom checks**. Available
host memory is not queried. Before fixture allocation, the probe also requires
its modeled temporary envelope to fit `--memory-budget-mib` (default 512):

```
modeled bytes = 4*V*H + 1024*V + 32*H + 64 MiB
```

The 4VH term allows ndarray/bytes and conservative copy/allocator overhead. The
1024V term prices Python mask/complement/dot/oracle lists and tuples, two codec
profiles' packed replies/frames/slices, outputs, float32 temporaries, public
bounds/widths/scales, and timing temporaries. 32H allows row-local work; 64 MiB
allows interpreter/imports, source-lock reads, and miscellaneous bookkeeping.
This is a deliberately conservative admission model, **not measured RSS or a
proof against OS memory pressure**. Native/allocator retention and concurrent
processes require externally checking real machine headroom before a large run.

Initial weight generation now allocates the persistent i8 array and processes
only 64 rows of int16 RNG plus i8 conversion at once; it no longer creates a
full int16 matrix. Independent scalar dot oracle processes one weight row at a
time and retains only V resulting dot integers, not a V×H Python-scalar matrix.
At Qwen geometry RNG temporaries are at most 172,032 bytes. The modeled total is
767,258,624 bytes (approximately 731.7 MiB), so default 512 MiB admission rejects
the full geometry before allocating. Explicit 1,024 MiB budget is required.

The full-geometry command below raises MAC and modeled-memory budgets; it is
**synthetic Qwen-sized codec evidence, not actual Qwen weights**. Reserve a serial
window; do not run it alongside other workers:

```sh
target/frontier-sdk-env/bin/python scripts/probe_boundary_head_residues.py --vocabulary 151936 --hidden 896 --head-calls 8 --repetitions 100 --oracle-mac-limit 136134656 --memory-budget-mib 1024 --output target/boundary-head-residues-full-geometry.json
```

Not executed here. Timing excludes Python dot-oracle work; public certificate
recomputation is separately timed. No heavy SDK benchmark command is represented
as already usable: the runtime has no residue namespace installed yet.

## Integration gate before any response claim

1. Choose the uniform geometry profile for source-free clients unless an existing
   independently authenticated L1 installation already exists. Admit W8A8
   geometry/dtype/quantizer and source/matrix/scales/descriptor identities through
   the trusted boundary admission, never through a self-supplied pin. For actual
   L1 evidence separately admit a local quantized head, stream/recompute bounds,
   and record real histogram/cold CPU. Parent search found no `target/**/head.i8`;
   this work remains synthetic-only.
2. Extend the compiled boundary descriptor/admission and both workers' installers
   with an explicit residue namespace/certificate ID; reject self-declared pins.
   Verify clients' source-present versus source-free trust/delivery distinction.
3. Install bounded setup delivery, immutable layouts, context-bound one-use
   frames, both-party length/padding checks, certified postchecks, failure ledger,
   and existing abort/cleanup. Preserve freshly sampled input shares and original
   W8A8/u32 worker products; do not introduce secret-dependent width selection.
4. Replace only head reply coding and client integer reconstruction. Preserve
   exact full logits, float32 multiplication order, sampler, full KV, and audit
   commitments. Differentially test Qwen/Llama/Qwen3 decoder paths as applicable.
5. Run a new matched serial fresh-client 150+8 cohort after integration. Charge
   every role's delivery/online/cleanup bytes, both pack CPUs, verification CPU,
   whole-client lifetime RSS including relay/dashboard, and full-state parity.
   Compare complete request, TTFT, N−1 decode TPS, TCP streams, and tier margin.
   Do not combine this codec-only hypothesis with other angles' unmeasured gains.
