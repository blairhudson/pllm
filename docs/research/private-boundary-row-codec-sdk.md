# Source-certified per-row private-head replies: SDK implementation gate

## Status and scope

Optional `PrivateLookupAndFullHead(head_output_encoding="row_residues",
head_certificate=...)` implements source-certified heterogeneous low-residue
full-head replies. Historical default serial/u32/existing component, descriptor
and native schedule serialization omit all new fields and retain their identity.
Both serial/parallel dispatch and existing/byte_dot kernels compose; the existing
byte-dot geometry/API admission remains mandatory. Source importer supports tied,
bias-free W8A8 heads and local children only; A4 certificates and unsupported
network graphs fail explicitly. No family-named execution branch is introduced.

The unified native/compiler build and installed-wheel generated
decoder/SDK/gateway gates now pass, including publication admission and raw head
accounting. The latest installed build also contains versioned public/private
workload witnesses. A separate six-client pinned Qwen 150+8 initial screen preserves
52,561,920 checked logit/KV values with zero observed swap growth: head payload falls
8.31%, total TCP streams 0.17% and decode throughput rises 2.41%, while full request
time rises 1.18%. Largest complete-client peak is 221.66 MiB. A separate six-client
150+64 screen preserves 486,555,648 checked values with zero swap growth: total
TCP traffic falls 0.91%, decode rises 0.87% and full request time is effectively
unchanged, while five-owner cold CPU rises 2.86%. Largest candidate peak is
217.83 MiB. Mixed paired RSS changes establish no memory improvement in either
screen. Independent confirmation remains open.
The scanner's 7.88 ms analytical upload-floor difference is not measured latency
attribution. Full results, publication costs, original rejected attempts and the
separate longer-workload cohort are in the
[matched benchmark reference](private-boundary-row-codec-benchmark.md).

## Public API and publication

Offline trusted publisher, using an ordinary locked/local-source Experiment with
`PrivateLookupAndFullHead()` (or a non-row head recipe):

```python
from pllm.roles import PrivateLookupAndFullHead

candidate_pin = PrivateLookupAndFullHead.publish_head_certificate(
    publisher_experiment, "/absolute/public/head.cert",
)
# candidate_pin = {"path": ..., "digest": <SHA256 of actual artifact bytes>,
#                  "size_bytes": ...}
```

Publisher uses the existing head-only source importer and native compiled-contract
bridge. It validates locked source, semantic tied matrix, exact owned native
weight digest, finite positive float32 scales, source config and native schedule;
then scans every immutable signed-i8 row in bounded chunks. Output creation is
exclusive. No decoder loading or full Python matrix export is added. Publication
source import/quantization/checkpoint hashing and subsequent row scan are separate
offline work, not free client setup.

### Publisher admission before tensor values

The public publisher now performs metadata-only locked-source, native schedule,
semantic shape/dtype and canonical tied-head validation before importing tensor
values. It requires an existing local checkpoint or an explicitly offline cached
source; it does not materialize `Model.tiny()` or fetch a missing online source.
Undeclared head bias is rejected from headers, before the ordinary importer's
optional-bias read. No decoder weights or decoder-session memory plan is admitted.

`publication_memory(source, spec, store)` prices the head from headers and tokenizer
file sizes. Let E=V*H, D be source dtype bytes (2 or 4), T be the sum of the four
possible tokenizer files, and C=`min(V,64,max(1,floor(1 MiB/H)))*H`. Its conservative
incremental reservation adds all these owners, rather than taking an optimistic
maximum or crediting streamed/mapped source reuse:

| Owner | Bytes |
| --- | ---: |
| Source head/mapped-input allowance | D*E |
| Full float32 import allowance | 4E |
| Quantized/native copying owners | 4E |
| Scales and row metadata | 32V |
| Source conversion/quantizer chunk workspace | 32C |
| Tokenizer bytes/JSON/native metadata expansion | 32T |
| Row scan, layout, compression and artifact workspace | 2 MiB + 8V + 4 MiB |
| Importer metadata/native runtime reserve | 128 MiB |

Each tokenizer file is limited to 64 MiB before loading. The conservative full-head
allowances cover the ordinary streamed quantizer without claiming its scratch is
the whole publisher footprint. Source hashing still streams raw bytes as part of
source identity validation; metadata preflight never calls `SafeTensorStore.get`,
`get_slice`, `iter_slices` or `_load_stage` to inspect tensor values.

`reserve_incremental_memory(bytes)` in `pllm.deployment.resources` uses the existing
process-wide SDK reservation ledger, not a fabricated planning record. Under its
lock it observes current RAM after the host reserve, intersects visible cgroup
headroom, subtracts live same-process reservations, and either acquires one owner
or rejects. Acquisition occurs before importer/native construction and `_load_stage`.
An outer owner stays live through quantization, scales/tokenizer loading, source
contract hashing, final row scan and publication. Source caches clear on loader
success and failure; the reservation releases on every exit. This is conservative
admission, not a hard RSS ceiling or interprocess memory lock.

Publication writes a private sibling, flushes/fsyncs it, and atomically hard-links
the complete file into the exclusive destination. Failed writes/fsync/link leave
no partial public artifact and remove the sibling; an existing destination is
never overwritten. The internal `publish(loaded, ...)` helper still assumes its
caller already owns/admitted the supplied head. Ordinary callers use the public
`publish_head_certificate` wrapper, which owns the complete import/publication gate.

**Trust contract:** an independently trusted publisher must endorse the actual-byte
digest through an authenticated release channel or an out-of-band caller-approved
pin. A digest received from an untrusted worker, a local path, a checksum, or this
publication function's return value alone is not source authentication. This is a
trusted publisher assertion, not a new cryptographic proof. A malicious trusted
publisher can lie to a thin client; this construction does not solve that threat.
Workers additionally recompute all bounds/layout commitments from their own actual
native weights, independently of publisher assertions. Existing source admission,
authenticated distinct roles and non-collusion assumptions remain unchanged.

After that independent approval, deploy identical public artifact bytes to each
recipient and select the approved pin in the caller's Experiment:

```python
approved_pin = {
    "path": "/absolute/public/head.cert",
    "digest": authenticated_release_sha256,
    "size_bytes": authenticated_release_size,
}
boundary = PrivateLookupAndFullHead(
    head_dispatch="parallel",
    head_output_encoding="row_residues",
    head_kernel="existing",  # "byte_dot" is independently selectable
    head_certificate=approved_pin,
)
# Use boundary with PreparedBoundaryRoles in the ordinary local Experiment.
# build_roles(experiment).client() and gateway_app() use the same contract.
```

This slice imports a bounded local public artifact; it does not add a remote
artifact-fetch/authentication endpoint. Local child roles share the absolute
path specified by their Experiment. External deployment graphs stay unsupported.
Explicit low-level worker/transport APIs take `head_certificate=approved_pin` too;
they cannot silently discover or trust a pin from a response. Source-certified
descriptors carry only `head_certificate_digest`, never peer-supplied row widths.

## Exact certificate and session binding

Artifact is `PLLMBRC1`, little-endian u32 metadata length, strict JSON metadata and
one zlib stream of exactly V unsigned one-byte widths. Maximum artifact is 1 MiB;
metadata is at most 16 KiB; V is at most 262,144. Import checks declared bounded
size and actual SHA-256 **before parsing/decompression**. Expansion is capped at
V+1; missing EOF, trailing streams/data, overrun, invalid widths, duplicate fields
and invalid provenance commitments fail closed. A pre-parse lexical bound permits
only three shallow objects and no arrays. No vocabulary matrix is fetched.

Internal telemetry may call `import_certificate(pin, descriptor,
with_metadata=True)` to receive `(widths, validated_metadata)` from this **same
authenticated parse**, with no artifact reread. Metadata retains `source`,
`numeric`, `widths_digest`, `bounds_digest` and `implementation` provenance, and
adds local `artifact_digest` / `artifact_size_bytes` from the authenticated pin.
Default `with_metadata=False` preserves the original width-byte return. All
source/layout/native-row checks finish before either form returns.

Metadata pins source-lock, checkpoint, lowered config, exact row-major signed-i8
matrix and little-endian f32 scale hashes, tied lookup commitment, V/H geometry,
W8A8 activation range, global signed bound, bias-free numeric/layout contract,
width hash and hash of all little-endian u32 per-row bounds. It also retains
publisher loader, quantizer, native extension and certificate implementation
digests. These provenance digests are covered by the authenticated artifact pin;
they are not independent signatures or worker-provided authentication evidence.

For every row, including `-128` and zero rows:

```
B_i = 127 * sum(abs(int(w_ij)))
b_i = max(1, bit_length(B_i) + 1)
```

Workers recompute every b_i and the complete bounds hash from immutable native
weights before allocating sessions. Client imports certified widths and verifies
their geometric ceiling but does not reconstruct a proof from public weights.
Certificate source identity intentionally excludes composition/schedule to avoid
a hash cycle: publication precedes selecting its artifact. Instead the **new
native composition and complete runtime schedule bind the entire pin**, and the
boundary descriptor binds its digest alongside independently derived source,
matrix/scales, schedule, encoding, dispatch and kernel identities. Substituted,
stale, absent or unauthenticated artifacts fail before boundary session material;
compiled thin-client binding also imports/validates the artifact before use.

Inputs remain full ring32 shares. Existing native `offset_pack_rows` and
`offset_reconstruct_rows` pack each output share and reconstruct all signed
integers exactly. Per-worker payload is `ceil(sum(b_i)/8)`. Zero rows retain the
native one-bit minimum. All logits remain present. Original float32 cast and two
separately rounded scale multiplies and the original sampler/RNG remain unchanged.

Replies use new `PLLMBH02` magic and
`pllm/private-boundary/source-row-head-residues/v1` namespace; heterogeneous width
marker is zero, not an untrusted width. The 74-byte header's context digest binds
certificate through descriptor plus admission, role, session, epoch and sequence.
Existing uniform `PLLMBH01` and raw/u32 responses cannot downgrade this mode.
Length, context, checksum and unused high padding bits are checked before native
reconstruction. Hashes detect corruption, not malicious-worker computation.

Existing one-use worker ledgers, attempt retirement, error/cancellation handling
and serial lookup ordering remain in force. Parallel head calls drain both
outstanding requests before cancellation/failure retirement; cancelled pairs
cannot deliver logits. No persistent executor, speculative head or mask reuse is
introduced.

## Allocation and delivery accounting

`private_boundary_memory(..., head_output_encoding="row_residues",
head_certificate=pin)` is used by ordinary role admission. It keeps original
full-u32 frame/workspace allowances; smaller network replies do not lower those
owners. Without opening artifacts, admission uses the safe uniform geometric
payload ceiling P, not a worker's claimed bit sum. Let S be pinned artifact size.
Additional per-concurrent-request allowances are:

* Client: `24V + 4P + 4S + 128 KiB`, including retained width bytes, both native and
  Python i64 decoder output owners, sliced payloads, certificate input/hash/JSON,
  bounded expansion and conversion overlap.
* Worker: `16V + 2P + 4S + 2 MiB`, including retained widths, packing owners,
  certificate import and bounded native-weight row scan. Native/Python chunk
  accounting includes int16 absolute-value/reduction scratch, not a full matrix.
* Existing parallel thread stacks/frame owners, full logits/dequantization,
  original lookup scratch, native matrix/scales, retained source metadata and
  session/replay ledgers remain separately charged.

Returned accounting exposes one V-byte layout owner, S bytes of public artifact
delivery per recipient, **3S** for two workers plus client, and a separate
`2 MiB + 8V + 4S` publication workspace allowance. Original publisher source import
is explicitly separate. These are conservative allocation-owner estimates, not
measured RSS, physical wire counts or memory-tier qualification. Artifact delivery
is caller-managed and is **not** silently included in HTTP role-link body counters;
filesystem publication/storage/distribution, authenticated release costs and any
repeated artifact verification must remain separately recorded in measurements.

## Tests and pending serial parent gates

Bounded source-tree extension run:

```sh
.venv/bin/python -m pytest -q \
  tests/test_private_boundary_row_certificate.py \
  tests/test_private_boundary_head_codec.py tests/test_private_boundary_options.py \
  tests/test_private_boundary_transport.py tests/test_private_boundary_parallel.py \
  tests/test_private_boundary_worker.py tests/test_private_boundary_topology.py \
  tests/test_private_boundary_accounting.py
```

Focused tests cover actual generated Qwen2/Qwen3/Llama publication matching an
independent thin bundle, all-row native integer endpoints/random masks and float32
parity, zero/-128 rows, missing/forged/stale/substituted certificates, malformed
authenticated artifacts, bounded decompression, duplicate keys, wrong provenance,
native row-bound disagreement, downgrade/session/replay/padding checks, actual
existing-kernel worker transport, pair drainage/cancellation and separate admission
owners. Source-tree extension lacks byte-dot API: those two cases skip explicitly.

Initial source-tree run: **320 passed, 2 skipped**. The same bounded suite also runs
against the already-installed frontier native extension with current Python
sources, without building, installing or changing either binary:

```sh
target/frontier-sdk-env/bin/python -c '
import importlib.util, pathlib, sys
path = pathlib.Path("target/frontier-sdk-env/lib/python3.13/site-packages/pllm/_native.abi3.so").resolve()
spec = importlib.util.spec_from_file_location("pllm._native", path)
native = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native)
sys.modules["pllm._native"] = native
sys.path.insert(0, str(pathlib.Path("python").resolve()))
import pytest
raise SystemExit(pytest.main(["-q",
    "tests/test_private_boundary_row_certificate.py",
    "tests/test_private_boundary_head_codec.py", "tests/test_private_boundary_options.py",
    "tests/test_private_boundary_transport.py", "tests/test_private_boundary_parallel.py",
    "tests/test_private_boundary_worker.py", "tests/test_private_boundary_topology.py",
    "tests/test_private_boundary_accounting.py"]))
'
```

This mixed-source/native unit gate exercises actual byte-dot workers as well as
existing workers: initial gate **322 passed**. It does not pretend to be a unified installed
SDK build or validate the newly added native compiler selector.

Unified native build/test/install subsequently passed. Installed-wheel validation
uses the environment's package and extension rather than source-tree imports:

```sh
# Use newly installed unified wheel for child-process SDK/gateway tests.
target/frontier-sdk-env/bin/python -m pytest -q -o pythonpath=. \
  tests/test_private_boundary_row_certificate.py \
  tests/test_private_boundary_options.py tests/test_private_boundary_decoder.py \
  tests/test_private_boundary_sdk.py
```

Rust compiler regression `source_row_certificate_binds_all_native_schedules_without_changing_steps`
checks all three generated model structures, both dispatches and both kernels,
unchanged decoder steps, pin substitution/digest changes and malformed pin rejection.
Extended generated decoder tests require exact complete logits/KV, tokens and RNG
for every row-mode dispatch/kernel combination. Extended supervised SDK fixtures
exercise both parallel kernels through ordinary SDK and gateway lifecycle tests.
The installed parity gate passed 156 checks; the expanded publisher/accounting
gate passed 339 checks. Both explicitly verified `pllm` and `_native` resolve from
the environment's `site-packages`. The latest unified build is recorded in
`target/certified-head-sdk-build-v3/progress.json`, with wheel SHA256
`55bc8e2b1f760fc4f96c9e88ea322e01408e24fe25c9154850c40e3748c21a50`.
The final unified installed-wheel check passes **577 tests**, covering workload
witnesses, benchmark replay, publication, raw head accounting, bundle validation,
decoder/SDK/gateway parity and incremental client-resource reservations. Ruff's
CI scope plus both new research modules and the benchmark coverage drift check pass.
Real-checkpoint publication, exact whole-request audits and matched measurements
remain separate from generated functionality and replay-auditor tests.

The subsequent six-client generated-source response smoke preserves 83,352
logit/KV values and complete output/state commitments with zero observed swap
growth. Source certificates, actual frame lengths and directed counters reconcile;
receipt-backed replay reproduces the archived summary. The first Qwen certificate
publication rejects host-memory admission before tensor loading. After host memory
is freed, a fresh admitted publication matches the independent full-row source
scan and records zero observed swap growth. Separate 150+8 and 150+64 performance
cohorts retain their own acceptance gate. Original attempts, publication
artifacts and replay are linked from the [benchmark reference](private-boundary-row-codec-benchmark.md).

### Publication follow-up source regression

Using the same existing-native bootstrap above, the updated source suite runs:

```python
pytest.main(["-q",
    "tests/test_private_boundary_row_certificate.py",
    "tests/test_incremental_resources.py", "tests/test_client_resources.py",
    "tests/test_private_boundary_topology.py", "tests/test_private_boundary_head_codec.py",
    "tests/test_private_boundary_options.py", "tests/test_private_boundary_transport.py",
    "tests/test_private_boundary_parallel.py", "tests/test_private_boundary_worker.py",
    "tests/test_private_boundary_accounting.py"])
```

**362 passed**. Added regressions deny host/cgroup headroom before `_load_stage` or
any tensor-value method, reject materializing/online sources and undeclared bias,
check metadata-only pricing, exercise concurrent public publishers competing for
one reservation, and retain ownership through successful publication and injected
import/tokenizer/scan/fsync/link failures. Raw-byte resource tests check atomic
concurrency, host reserve/cgroup intersection, invalid costs and observation failure.
Metadata telemetry checks same-parse provenance and pin fields, unchanged default
return, and native-bound disagreement rejection. No build/install, public checkpoint
values, provider call, benchmark or commit belongs to this follow-up run.
