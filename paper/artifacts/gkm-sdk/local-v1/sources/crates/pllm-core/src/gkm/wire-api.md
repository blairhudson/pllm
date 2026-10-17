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

Send array index 0 only to Worker Zero, index 1 only to Worker One. Each worker
is bound to one authenticated client session and monotonically increasing window
sequences. Complete/finish the worker request before beginning its next window.
Worker receipt means schedule completion, never client result acceptance.
Client outputs remain tentative until its original native final verifier passes.
Seeded windows/frames reject; native full/seeded defaults and identities remain.

Worker source arguments are **public**, immutable, publication-ordered weights
and the original finite float32 byte table. The owner makes explicitly charged
copies. Exact matrix identities/bounds and table identity/row hashes must match
the trusted publication. No token, reconstructed input, complementary share or
publisher scalar enters a worker owner. Public start frames contain only schedule
and framing, and bind publication, session, sequence, request/window, context,
plan/count, ordinal, role, stage/artifact, geometry and numerical/sharing profile.

The caller must authenticate public pins and start schedules and provide separate
authenticated confidential role channels. Core supplies no network server,
credentials, encryption, reconnect/retry, or restart-persistent replay ledger.
Input parsing errors in an outer binding must explicitly abort the native owner.
All core state-changing errors are sticky even when ignored. Abort/drop incomplete
worker requests before releasing an outer transport. Drain external in-flight
calls when cancellation closes the native client transaction.

## Explicit resource boundaries

- Original complete-publication pool remains **16 MiB**, unchanged. Imports
  reserve their own complete native envelope, before native payload allocations.
- Additional wire pool is **8 MiB process-global**, independent of publication
  pool. It charges worker source copies/control, public plans, stack/dynamic
  request scratch, client wire controls, and every retained output byte owner.
- Each worker copies source weights/bounds/table plus matrix metadata, native
  owner sizes and 1024 bytes control allowance before allocation. Each worker
  request charges `128 * size_of::<Step>() + 3 * 256 * 8 +
  size_of::<WorkerWireWindow>() + 1024` bytes. Client wire owner charges its size,
  two maximum word decode arrays, one maximum schedule array and 1024 bytes.
- Each exported frame charges exact capacity plus `size_of::<WireBytes>() + 64`;
  no unbounded queue or implicit cloning exists. Protocol frame maximum is
  **8192 bytes**; public artifact maximum is **512 KiB**.
- Worker requests reuse native admission counters: eight live windows shared by
  a publication, 64 admission handles, existing per-owner and process cumulative
  attempt/step limits and process live-window cap. Co-located client plus two
  workers consume three live leases. Retained request/reply frames retain leases.
- Borrowed inbound bytes, caller-made copies, source import buffers, allocator
  overhead and transport/interpreter buffers require outer admission. Fixed-size
  `Arc::new` retains native abort-on-OOM exception. These are payload limits, not
  process RSS guarantees or secure erasure of incidental compiler/stack copies.

Malformed publication import checks trusted digest, canonical compressed SEC1
points (including legal identity), geometry, duplicate IDs, range bounds and exact
length without allocating. It then reserves unchanged publication envelope and
copies exact row hashes. Wire encoding preserves existing publication/request/plan
identities. It exports no secret setup scalars.

## Native evidence and remaining integration gates

Focused checks cover byte-exact original rows (including negative zero), widened
integer oracles, every frame-binding field, all publication truncation positions,
canonical identity/invalid points, duplicate stages, source substitution, row
hashes, final linear corruption, partial replies, repeated/ignored errors,
replay/cancellation, seeded-frame rejection, and every new payload-allocation
failure seam. Existing native numeric identities and seeded/reference tests pass.

The independent allocator binary verifies zero-allocation malformed import and
capacity rejection, real null-allocation recovery, retained packet leases,
separately charged maximum-size source copies, and a complete 128-step window
with 256-by-256 matrices. The original native allowance plus the independently
charged wire allowance exceeds the measured requested-allocation peak. Two
maximum 4 MiB table copies plus their controls deliberately cannot coexist inside
one process's 8 MiB wire allowance; separate worker processes admit one each.

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
CARGO_BUILD_JOBS=2 cargo test --offline --locked -p pllm-core --doc gkm -- --test-threads=1
```

Compiler/PyO3/SDK/gateway wrapping, outer copy and transport admission, complete
request cancellation/draining, authenticated confidential channels, independent
operator deployment and durable replay accounting remain integration gates.
Seeded network transport, enlarged geometries, batching, cryptographic review,
original-model quality and performance qualification remain separate gates.
