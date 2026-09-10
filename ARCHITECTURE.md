# Architecture

PLLM is a mixed Python and Rust project with one Python distribution. Maturin
builds the PyO3 module `pllm._native` and packages it alongside `python/pllm`.

## Dependency direction

```text
Python SDK, CLI and provider
            │
            ▼
pllm.runtime → pllm._native (PyO3)
                         │
                         ▼
                     pllm-core
```

The Rust core has no Python or web framework dependency. The binding translates
Python bytes into validated integer buffers, calls the core without holding the
Python interpreter lock, and returns immutable bytes. Python retains model and
protocol orchestration. SEAL/TenSEAL is a separate cryptographic dependency.

## Crates

`crates/pllm-core` contains the integer matrix executor, scalar reference paths,
runtime AVX2 and NEON selection, bounded coefficient arithmetic, codecs,
quantization, masking, output subtraction, and operating system random sampling.
Matrices own their validated weights. Their dimensions and contents cannot be
mutated through the public Rust API. An executor owns a persistent Rayon pool.

`crates/pllm-python` contains only the Python binding. Maturin builds this crate
as `pllm._native`. It depends on `pllm-core` and PyO3. The stable Python ABI is
configured from Python 3.11. The application dependency matrix currently limits
Python to 3.11 through 3.13.

## Python package

`python/pllm` is the only installed namespace. Its public API consists of the
client classes, configuration, response types, application factories and native
matrix interface. Public objects are imported on demand. `pllm.runtime` holds
the model graph, HE preparation, transport, protocol, scheduling and importers.
Applications should not depend on internal module locations.

`python -m pllm` and the installed `pllm` command use the same entry point.
A source checkout does not shadow an installed wheel through a second package
at the repository root.

## Hot operations

A matrix is copied into Rust once at compilation, then reused for later calls.
Input and output conversions are explicit. This is not a zero copy interface.
The present snapshot consumes one extra signed byte per weight while the source
matrix is retained for metadata and preparation-service loading. Benchmarks must
count that storage and buffer conversion work.

Moving arithmetic to Rust does not remove dependent client-to-inference stage
exchanges, offline preparation work, or client attention state. No speed claim
follows merely from the choice of implementation language.

The legacy packed-BFV correlation backend remains available to research and
confidential-weight protocol code but is not selected by public inference.
Direct-FHE and blinded execution retain their separate transport.

Public-weight deployments use offline seeded inventory preparation. Before chat,
the client asks inference to create an inventory with immutable model, body, stage,
quantization, and attempt-budget commitments. It authorizes that inventory through
trusted preparation, then sends preparation one root seed and a batch size for each
remote stage. Batches default to 64 rows and are configurable with
`PLLM_PREPARED_INVENTORY_ROWS`. Domain-separated expansion binds each row to the
inventory, model, body, stage, weight, shape, quantization, ring, modulus, and wire
width and produces one-time input mask `r`, output mask `s`, and ticket.
Preparation computes each batch of `W·r-s`, pushes it to inference's fixed endpoint,
and waits for inference's durable acknowledgement. After every stage is loaded, the
client asks inference to seal the inventory; only then does inference report `READY`.
The push WebSocket URL is derived from the validated inference HTTP(S) origin, and
only the provider push credential authenticates its upgrade.

At chat start, the client reserves inventory rows for the execution. Each online
stage message contains only the one-time ticket and `x-r`. Inference atomically
consumes the matching preloaded row and returns `W·x-s`; the client adds `s` and
center-decodes. Prefill batches this as one ticket vector and one packed matrix
per stage rather than one serialized envelope per row; decode retains one ticket
per stage over its persistent connection. Online chat is therefore
client-to-inference only. Preparation is
idle online, and refill occurs only while the client is idle between executions.
Inventory is in memory and may be reused across chats, but restart or idle expiry
discards it. Reservation is a burn boundary: cancellation, early end of stream, or
failure also burns every unused row reserved for that execution.
Stages use the smallest exact `u16`, `u24`, or `u32` ring selected from the exact
signed output bound. The preparation role is public-weight-only, holds the same
model body, and must follow the protocol, erase masks, and not collude with the
inference provider. Self-hosting keeps that trust inside the client boundary.
The online public path does not use BFV or contact preparation.

Public model bundles also carry the quantized token-lookup and output-head
matrices. The customer evaluates token lookup locally and applies the output
head only to the final prefill row. This removes vocabulary-sized HE work from
the online public-weight path. Transformer-body matrices remain provider-owned;
proprietary bundles never include either boundary matrix.

Bundle schema 2 stores quantized matrices once and lets stages reference them.
For tied embeddings, one vocabulary-by-hidden matrix uses per-token row scales;
token lookup gathers its rows and `lm_head` multiplies by the same array. This is
not numerically identical to schema 1 token lookup, which independently quantized
the transposed matrix with per-hidden-feature scales. Output-head quantization is
unchanged. Any per-layer token table remains a separate auxiliary object using
its existing orientation.

## Application boundary

The customer client owns plaintext input, inventory root seeds and masks, private
activation scales, model state, output decoding and public token-boundary matrices.
The trusted preparation service and untrusted inference provider both hold the
public transformer body. Preparation receives batched stage root seeds before
chat; inference receives the resulting corrections and later the masked integer
tensors, never the seeds. The client never receives the correction. Client-to-inference,
client-to-preparation, and preparation-to-inference credentials are distinct.
The local Responses gateway is inside the customer boundary.
An ordinary provider endpoint cannot be made private by changing its URL in an
unmodified SDK.

The public protocol assumes both roles follow the computation and do not
collude. The arithmetic simulator, guarded controls and transport authentication do
not establish security against arbitrary malicious participants. These limits
do not change with the packaging layout.

## Repository boundaries

Fumadocs lives in `docs` and has its own npm manifest. It is deployed as a static
site and is not bundled into the Python wheel. `paper/manuscript.md` is the
canonical Pandoc Markdown paper source. Pandoc generates both the website article
and PDF, using Tectonic locally or pdfLaTeX in CI. Historical measurements are
kept under `research/evidence` and are not rewritten as native Rust results.

The loopback benchmark dashboard runs the real client, preparation, and inference
roles and receives their OTLP metrics and traces directly. Protocol byte counts
are custom OTEL metrics; prompts and activation payloads are never telemetry
attributes. The generated tiny checkpoint validates transport behavior only.
Interactive dashboard runs default to Qwen2.5-0.5B-Instruct; random tiny weights
require the explicit `--tiny` transport-smoke option.

## Build and release boundaries

The Cargo workspace version is inherited by both Rust crates. The Python source
version and citation version are checked against it by `scripts/release.py`.
Release changes use that script rather than independent manual edits.

The root `cargo test` runs the Python independent core. The wheel matrix compiles
the binding for each platform and runs Python tests against the installed wheel.
The publisher receives only validated artifacts and OIDC credentials; it does
not compile source with publishing credentials in scope.

References: Maturin project layout and configuration documentation;
PyO3 0.26 parallelism documentation. These links are listed in the development
page of the Fumadocs site.
