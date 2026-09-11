# Security

## Current public-weight claim

PLLM's public path splits each remote linear operation across a customer client,
a trusted preparation role, and an inference role. It aims to keep plaintext
activations from either service viewed alone under an honest-but-curious,
non-colluding threat model. "Untrusted inference" means inference is not given
plaintext; it does not mean the protocol verifies a malicious provider.

| Role | Holds | Receives | Must not receive |
| --- | --- | --- | --- |
| Client or local gateway | Prompts, tokens, activation scales, attention state, inventory root seeds, masks, sampling state, decoded output, public token-boundary matrices | Model plan, acknowledgements, masked stage outputs | `W*r-s` corrections |
| Trusted preparation | Public transformer body | Inventory authorization, stage root seeds, row counts, committed metadata | Prompts, online activations, inference credential |
| Inference | Public transformer body, prepared corrections | Inventory authorization, then one-time tickets and `x-r` | Root seeds, masks, activation scales, plaintext prompts |

Public bundles expose quantized token lookup and output-head matrices to the
client. Transformer-body stages stay at preparation and inference. For tied
embeddings, bundle schema 2 stores one canonical matrix referenced by both local
boundaries; untied matrices remain separate.

## Offline preparation

Before chat, the client asks inference to register an inventory with immutable
model, body, stage, weight, shape, quantization, ring, wire-width, and attempt-budget
commitments. It sends preparation one authorization, then one root seed and batch
size for every remote stage. Domain-separated expansion produces one-time input
mask `r`, output mask `s`, and ticket per row.

Preparation computes each `W*r-s` batch and pushes it one way to inference's fixed
`/v1/he/corrections/ws` endpoint. The WebSocket URL is derived from preparation's
validated inference HTTP(S) origin; a client cannot supply a callback. Only the
provider-push credential authenticates the upgrade. Preparation waits for a bounded
durable acknowledgement, then erases expanded masks. After every stage is loaded,
the client asks inference to seal the inventory. Inference reports `READY` only
after the complete committed inventory exists.

## Online execution

At response start, the client reserves a contiguous inventory range. Each online
row carries only its one-time ticket and `x-r`. Inference atomically consumes the
matching preloaded correction and returns `W*x-s`; the client adds `s` and
center-decodes. Fresh uniform masks use the smallest exact `u16`, `u24`, or `u32`
ring selected from the signed output bound.

Prompt prefill sends a compact ticket vector and packed masked matrix per stage.
Decode sends one ticket per stage over a persistent client-to-inference connection.
Preparation receives no online request. It may prepare a spare inventory only
while the client has no active response.

Inventory is memory-only. Inference restart or configured idle expiry discards it.
A reservation is a burn boundary: successful completion consumes reached rows;
cancellation, failure, timeout, or early end also burns every unused row reserved
for that execution. Tickets and masks must never be replayed or restored from a
snapshot.

## Required assumptions

- Preparation follows the protocol, erases expanded masks, and does not collude
  with inference. If it retains masks or colludes, activation privacy is lost.
- Inference follows the stated computation. Authentication, commitments, bounded
  frames, and one-time tickets do not prove arbitrary model execution.
- Client software and its host remain trusted. A modified client changes the
  assumptions for confidential-weight modes.
- Remote inference and preparation use distinct TLS origins. Loopback HTTP is
  accepted only for local evaluation.
- Credentials are distinct for client-to-inference, client-to-preparation,
  preparation-to-inference push, and any local gateway.

Self-hosting preparation keeps its trust inside the customer boundary. Running
both services under one untrusted operator does not satisfy non-collusion.

## Observable information

Both services can observe model IDs, stage names, tensor shapes, timing, scheduling,
failure patterns, and approximate input or output lengths. Inference also stores
prepared correction sizes and sees masked online tensors. Preparation sees seed
batch sizes and timing before chat. Transport encryption does not hide this
metadata from each endpoint.

Activation scales remain local because they can fingerprint private activations.
A zero count for literal prompt bytes is useful instrumentation, not proof that all
metadata is harmless.

## Unsupported guarantees

The public protocol is not secure against arbitrary malicious or colluding
participants. The authenticated arithmetic command is a simulator, not a deployed
distributed protocol. Guarded and blinded confidential-weight modes expose
intermediate outputs to the client; query limits do not prevent a modified client
from reconstructing a matrix with enough chosen inputs. Direct BFV is a separate
slow reference path. None of these modes is selected by public seeded preparation.

Repository tests establish implementation properties, not a cryptographic audit,
provider independence, secure erasure, side-channel resistance, operational
hardening, or production readiness.

## Operational controls

Keep the plaintext gateway on loopback or a protected customer network. Disable
payload logging and external prompt tracing. Protect model and config directories,
and avoid putting credentials in command histories or captured process listings.

The benchmark dashboard is also loopback-only. It rejects non-loopback Host and
browser Origin values, authenticates OTLP ingestion with a fresh per-launch token,
and writes its sanitized SQLite archive with owner-only permissions. The live page
does display the current prompt and output to its local browser; unlike the
archive, it is not a text-free interface and must not be exposed through an
unauthenticated proxy.

Use limits large enough for committed inventory batches but still bounded. The
provider's prepared-session idle timeout controls READY inventory lifetime; the
client HTTP timeout and preparation push timeout control different operations.
Closing the correction WebSocket while idle is recoverable for a later new
inventory. Closing the client-to-inference decode WebSocket ends that execution and
burns its unused reservation.

## Reporting a vulnerability

Enable GitHub private vulnerability reporting in the repository settings. Submit
reports through **Security -> Advisories -> Report a vulnerability**. If private
reporting is unavailable, open an issue requesting a private contact without
including exploit details or sensitive data.

Identify the revision, privacy profile, assumed adversary, reproduction steps, and
observed disclosure. Do not attach real prompts, credentials, model secrets, mask
seeds, or preparation responses.

## CI and release boundary

Pull requests run without deployment credentials. PyPI publication and static
documentation deployment use separate jobs and permissions. Passing CI or
publishing an artifact is not a cryptographic or deployment audit.
