# Bounded full-share wire adapter contract

Core exports from `pllm_core::gkm`:

```rust,ignore
Publication::export_wire(&self) -> Result<WireBytes, Error>
Publication::import_wire(bytes: &[u8], authenticated_owner_digest: [u8; 32])
    -> Result<Publication, Error>
Client::begin_wire(&self, context: [u8; 32], plan: &[Step])
    -> Result<ClientWireWindow, Error>
Window::into_wire(self) -> Result<ClientWireWindow, Error>

ClientWireWindow::start_frames(&mut self) -> Result<[WireBytes; 2], Error>
ClientWireWindow::lookup(&mut self, token: usize) -> Result<[WireBytes; 2], Error>
ClientWireWindow::linear(&mut self, input: &[i64]) -> Result<[WireBytes; 2], Error>
ClientWireWindow::accept_replies(&mut self, zero: &[u8], one: &[u8])
    -> Result<WireTentative<'_>, Error> // public borrowed .values and .signed
ClientWireWindow::finish(&mut self) -> Result<AcceptedWindow, Error>
ClientWireWindow::abort(&mut self)
ClientWireWindow::request_id(&self) -> [u8; 32]

WorkerWire::new(publication: Arc<Publication>, authenticated_owner_digest: [u8; 32],
    worker: Worker, stages: &[Stage<'_>], table: &FloatTable) -> Result<WorkerWire, Error>
WorkerWire::begin(&mut self, authenticated_start: &[u8]) -> Result<WorkerWireWindow, Error>
WorkerWire::abort(&mut self)
WorkerWireWindow::execute(&mut self, request: &[u8]) -> Result<WireBytes, Error>
WorkerWireWindow::finish(&mut self) -> Result<WorkerWireReceipt, Error>
WorkerWireWindow::abort(&mut self)
WireBytes::as_bytes(&self) -> &[u8]
WireResources::current() -> WireResources
```

### Explicit Decoder capacity

Existing constructors and `MAX_*` constants retain Capsule policy. Larger owners
use an explicit local `Capacity::Decoder` argument (capacity is never inferred
from wire data):

```rust,ignore
PublicationResources::for_shapes_with_capacity(shapes: &[(usize, usize)],
    vocabulary: usize, width: usize, capacity: Capacity) -> Result<PublicationResources, Error>
Publication::publish_with_capacity(context: [u8; 32], stages: &[Stage<'_>],
    table: &FloatTable, capacity: Capacity) -> Result<Publication, Error>
Publication::import_wire_with_capacity(bytes: &[u8], pin: [u8; 32],
    capacity: Capacity) -> Result<Publication, Error>
I16Matrix::from_encoded_owned_with_capacity(rows: usize, columns: usize,
    weights: Vec<i16>, input_bounds: Vec<u64>, capacity: Capacity) -> Result<I16Matrix, Error>
FloatTable::from_le_bytes_owned_with_capacity(vocabulary: usize, width: usize,
    bytes: Vec<u8>, capacity: Capacity) -> Result<FloatTable, Error>
WireResources::current_with_capacity(capacity: Capacity) -> WireResources
```

Borrowing/copying counterparts `I16Matrix::from_encoded_with_capacity` and
`FloatTable::from_le_bytes_with_capacity` take slices. Owned source imports require
exact vector capacity. `WorkerWire::from_owned` transfers those owners without
another source copy; `WorkerWire::new` explicitly prices and copies them. Both
inherit publication capacity, as do clients, native responses and wire windows.
`Publication`, `I16Matrix`, `FloatTable` and `WireBytes` expose `capacity()`.
`PublicationResources` also reports `capacity` and `window_accumulator_bytes`.

Capacity exposes `max_rows`, `max_columns` (also table width), `max_vocabulary`,
`max_stages`, `max_window_steps`, `max_matrix_elements`, `max_source_bytes`,
`max_publication_bytes`, `max_window_bytes`, `max_process_publication_bytes`,
`max_process_wire_bytes`, `max_publication_wire_bytes` and `max_wire_frame_bytes`.
Decoder limits: 262144 rows/vocabulary, 16384 columns/table width, 256 stages,
65536 window steps, 2^28 elements per matrix, 2 GiB aggregate owned source
(including bounds), 128 MiB native window, 4 GiB publication/process-publication
envelope, 4 GiB process wire pool, 32 MiB encoded publication and 4 MiB frames.
Admission also checks a conservative encoded-publication upper bound. Numeric
profiles, stage/matrix/table/publication digests and canonical wire bytes do not
include capacity. An identical small artifact has identical identities under
both policies. Both policies also admit the explicit seeded sharing profile.

Send array index 0 only to Worker Zero, index 1 only to Worker One. Each worker
is bound to one authenticated client session and monotonically increasing window
sequences. Complete/finish the worker request before beginning its next window.
Worker receipt means schedule completion, never client result acceptance.
Client outputs remain tentative until its original native final verifier passes.
Full sharing remains the default. Explicit seeded windows use the existing
complete-window AES profile: only Worker One receives independent linear/lookup
keys, and its per-operation requests carry no expanded shares. Worker Zero
receives complementary explicit shares and no keys. Stream consumption follows
the complete scalar claim schedule without reseeding or rewinding.

Worker source arguments are **public**, immutable, publication-ordered weights
and the original finite float32 byte table. The owner makes explicitly charged
copies. Exact matrix identities/bounds and table identity/row hashes must match
the trusted publication. No token, reconstructed input, complementary share or
publisher scalar enters a worker owner. Full-share start frames contain only
schedule and framing; seeded Worker One's start additionally carries secret keys
and therefore requires confidential authenticated delivery. Starts bind
publication, session, sequence, request/window, context,
plan/count, ordinal, role, stage/artifact, geometry and numerical/sharing profile.

The caller must authenticate public pins and start schedules and provide separate
authenticated confidential role channels. Core supplies no network server,
credentials, encryption, reconnect/retry, or restart-persistent replay ledger.
Input parsing errors in an outer binding must explicitly abort the native owner.
All core state-changing errors are sticky even when ignored. Abort/drop incomplete
worker requests before releasing an outer transport. Drain external in-flight
calls when cancellation closes the native client transaction.

## Explicit resource boundaries

- Original Capsule complete-publication sub-pool remains **16 MiB**. All profiles
  additionally share a **4 GiB** process ceiling. Imports reserve their own
  complete native envelope before native payload allocations.
- Capsule wire sub-pool remains **8 MiB process-global**, under a shared **4 GiB**
  wire ceiling, independent of publication pool. It charges source copies/control, public plans, dynamic
  request scratch, client wire controls, and every retained output byte owner.
- `WireResources::current()` reports Capsule usage/limit;
  `current_with_capacity(Decoder)` reports all profiles' total process usage.
- Each worker copies source weights/bounds/table plus matrix metadata, native
  owner sizes and 1024 bytes control allowance before allocation. Worker request
  scratch charges its actual schedule and the largest `8 * columns + 16 * rows`
  across admitted matrices (or table row bytes, whichever is larger). Client wire
  scratch charges its size, two owner-sized maximum word arrays, actual encoded
  schedule length and 1024 bytes. These reservations remain live with their owners.
- Each exported frame charges exact capacity plus `size_of::<WireBytes>() + 64`;
  no unbounded queue or implicit cloning exists. Protocol frame maximum is
  **8192 bytes** for Capsule and **4 MiB** for Decoder; public artifact maximum is
  **512 KiB** for Capsule and **32 MiB** for Decoder.
- Worker requests reuse native admission counters: eight live windows shared by
  a publication, 64 admission handles, existing per-owner and process cumulative
  attempt/step limits and process live-window cap. Co-located client plus two
  workers consume three live leases. Retained request/reply frames retain leases.
- Streamed SDK publications instead select explicit single-window ownership:
  complete source, publication and publisher scratch remain charged, while import
  scratch is bounded to 131072 bytes and one complete window is reserved. The
  publication allows one lifetime window attempt and one SDK publication export;
  repeat export retires the client. Generic eager/eight-window owners and all
  per-profile/process ceilings remain unchanged. Exported frames retain leases
  independently of request cancellation or owner retirement.
- Streamed operation frames share a lifetime-window quota: at most four retained
  START/REQUEST/REPLY handles and `4 * max_wire_frame_bytes` charged bytes, plus
  four temporary scratch guards within another `4 * max_wire_frame_bytes`.
  Capacity and bounded owner/control overhead both count. Quota exhaustion
  poisons the window; refunds follow buffer zeroization and deallocation, not
  cancellation. Public artifact exports have a separate publication-owned
  one-live-handle cap of `max_publication_wire_bytes`. Generic wire behavior is
  unchanged. These quotas fit the existing SDK frame/publication allowances;
  they do not cover caller-owned Python copies or arbitrary ASGI hosting.
- Borrowed inbound bytes, caller-made copies, source import buffers, allocator
  overhead and transport/interpreter buffers require outer admission. Fixed-size
  `Arc::new` retains native abort-on-OOM exception. These are payload limits, not
  process RSS guarantees or secure erasure of incidental compiler/stack copies.

Malformed publication import checks trusted digest, canonical compressed SEC1
points (including legal identity), geometry, duplicate IDs, range bounds and exact
length without allocating (at most 2 KiB of fixed offset metadata). It then reserves its publication envelope and
copies exact row hashes. Wire encoding preserves existing publication/request/plan
identities. It exports no secret setup scalars.

## Native evidence and remaining integration gates

Focused checks cover byte-exact original rows (including negative zero), widened
integer oracles, every frame-binding field, all publication truncation positions,
canonical identity/invalid points, duplicate stages, source substitution, row
hashes, final linear corruption, partial replies, repeated/ignored errors,
replay/cancellation, sharing-profile substitution, and every new payload-allocation
failure seam. Existing native numeric identities and seeded/reference tests pass.

The independent allocator binary verifies zero-allocation malformed import and
capacity rejection, real null-allocation recovery, retained packet leases,
separately charged maximum-size source copies, and a complete 128-step window
with 256-by-256 matrices. The original native allowance plus the independently
charged wire allowance exceeds the measured requested-allocation peak. Two
maximum 4 MiB table copies plus their controls deliberately cannot coexist inside
one process's 8 MiB wire allowance; separate worker processes admit one each.

Decoder fixtures additionally exercise 1031-by-3 and 2-by-1033 matrices, 2053-wide
table rows, 4097 vocabulary, 256 stages and a 257-step wire schedule. Geometry-only
Qwen2.5-0.5B admission covers 97 matrices: 151936 R points, 182144 S points,
1532461056 source weight/table bytes and 20422656 accumulator bytes. Tests do not
import or execute the model. Null-allocation checks cover newly dynamic wide
input/output/reply buffers, peak overlap, and complete 4 GiB publication-envelope
exhaustion using tiny payloads. Retained packets/responses preserve leases after
cancellation; out-of-owner-shape frames reject even when inside Decoder maxima.

The process fixture launches two native child processes. Each imports the trusted
public artifact, constructs only its own role, receives only its own schedule and
share bytes, and completes dependent lookup/linear execution before the parent's
native final verification. Pipes establish party-local functionality, not remote
channel security or independence of operators.

Replay commands (sequential, focused native checks only):

```sh
CARGO_BUILD_JOBS=2 cargo test --offline --locked -p pllm-core --lib gkm:: -- --test-threads=1
CARGO_BUILD_JOBS=2 cargo test --offline --locked -p pllm-core --test gkm_wire --test gkm_wire_allocation --test gkm_wire_process -- --test-threads=1
CARGO_BUILD_JOBS=2 cargo test --offline --locked -p pllm-core --test gkm_allocation --test gkm_seeded_restart -- --test-threads=1
CARGO_BUILD_JOBS=2 cargo test --offline --locked -p pllm-core --test gkm_capacity -- --test-threads=1
CARGO_BUILD_JOBS=2 cargo test --offline --locked -p pllm-core --doc gkm -- --test-threads=1
```

Compiler/PyO3/SDK/gateway wrapping, outer copy and transport admission, complete
request cancellation/draining, authenticated confidential channels, independent
operator deployment and durable replay accounting remain integration gates.
Seeded network transport, batching, cryptographic review,
original-model quality and performance qualification remain separate gates.
