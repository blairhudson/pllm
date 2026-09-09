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

Moving arithmetic to Rust does not remove two network round trips per linear
stage, preparation responses, or client attention state. No speed claim follows
merely from the choice of implementation language.

The legacy packed-BFV correlation backend remains available to research and
confidential-weight protocol code but is not selected by public inference.
Direct-FHE and blinded execution retain their separate transport.

Public-weight deployments use just-in-time seeded preparation. Inference first
registers a random session with immutable model, body, stage, quantization, and
attempt-budget commitments. Before stage execution, the client sends one compact
session authorization to preparation. Preparation validates it against its loaded
model and relays it to inference with the provider-only push credential. Inference
accepts that exact session once and starts no runner computation before acceptance.
For each linear stage, the client then creates a fresh 128-bit attempt ID and
256-bit seed. A domain-separated expansion binds session, model, body, stage, weight, shape,
quantization, ring, modulus, and wire width and produces input mask `r` and
output mask `s`. The client sends the seed to preparation and `x-r` to inference
concurrently. Preparation computes and pushes `W·r-s` exactly once to a fixed
inference endpoint over one persistent one-way binary WebSocket and returns only a
small client acknowledgement after the send. Each bounded correction frame carries
its session and random attempt ID; inference reports a rejection through the matching
activation request.
The WebSocket URL is derived from the validated inference HTTP(S) origin, and only
the provider push credential authenticates its upgrade. A failed or ambiguous send
is never replayed; the next independent attempt may establish a new connection.
For an authorized session, inference starts its runner on activation while awaiting the correction,
then atomically burns both halves and returns `W·x-s`; the client adds `s` and
center-decodes.
Stages use the smallest exact `u16`, `u24`, or `u32` ring selected from the exact
signed output bound. The preparation role is public-weight-only, holds the same
model body, and must follow the protocol, erase masks, and not collude with the
inference provider. Self-hosting keeps that trust inside the client boundary.
The online public path does not use BFV or durable correlation inventory.

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

The customer client owns plaintext input, fresh seeds, private activation
scales, model state, output decoding and public token-boundary matrices. The
trusted preparation service and untrusted inference provider both hold the
public transformer body. Preparation receives seeds; inference receives masked
integer tensors and the direct preparation correction, never the seed. The
client never receives the correction. Client-to-inference,
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
