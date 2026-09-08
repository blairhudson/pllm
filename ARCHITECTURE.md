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
matrix is retained for HE preparation. Benchmarks must count that storage and
buffer conversion work.

Moving arithmetic to Rust does not remove network round trips, preparation
traffic, client attention state, or the cost of HE encryption and decryption.
No speed claim follows merely from the choice of implementation language.

Public-weight BFV correlation preparation uses tiled diagonal matrix-vector
evaluation. An 8,192-coefficient BFV context packs as many as four masks per
ciphertext group, tiles matrices larger than one guarded slot segment, and
returns packed output tiles rather than one ciphertext per output coordinate.
Contexts are shared by plaintext modulus, while ciphertext envelopes bind the
matrix dimensions and tile layout. Direct-FHE and blinded execution retain
their separate legacy transport.

Public-weight deployments may instead select direct two-provider sharing. The
client splits each quantized activation into fresh uniform ring-2^32 shares,
submits both requests concurrently over separate authenticated confidential
connections, and reconstructs the exact signed W8A8 result locally. Rust uses
wrapping AVX2 or NEON multiplication for this ring. This strategy performs no
HE preparation and relies on two independently administered honest providers
not colluding. BFV remains the default and the single-provider alternative.

Public model bundles also carry the quantized token-lookup and output-head
matrices. The customer evaluates token lookup locally and applies the output
head only to the final prefill row. This removes vocabulary-sized HE work from
the online public-weight path. Transformer-body matrices remain provider-owned;
proprietary bundles never include either boundary matrix.

## Application boundary

The customer client owns plaintext input, secret keys, private activation scales,
model state, output decoding and public token-boundary matrices. The provider
owns transformer-body matrices and receives encrypted preparation data and
masked integer tensors. The local Responses gateway is inside the customer
boundary. An ordinary provider endpoint cannot be made private by changing its
URL in an unmodified SDK.

The public weight protocols assume each provider follows the computation. The
arithmetic simulator, guarded controls and HTTP authentication do not establish
security against arbitrary malicious participants. These limits do not change
with the packaging layout.

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
