# Security boundaries

Name the protected data, the adversary, and what still leaks.


## Public weights

The client retains plaintext prompts, token IDs, activation scales, seeds, masks, and model state. The untrusted inference provider observes masked stage values, preparation corrections, shapes, stage names, timing, and scheduling. The trusted preparation service observes fresh seeds and bound stage metadata, derives `r` and `s`, and pushes `W·r-s` to inference. The client never receives this correction.

Inference registers each prepared session with immutable model/body/stage commitments, quantization, and a bounded attempt budget. Before stage traffic, the client sends preparation one compact authorization for that session. Preparation validates and relays it with its distinct provider-push credential. Inference accepts it once and performs no runner computation before authorization; inference-client credentials cannot authorize sessions.

Preparation sends corrections over one persistent binary WebSocket to a fixed
provider route. Its `ws` or `wss` URL is derived only from the validated inference
HTTP(S) origin. Only the provider-push credential authenticates the upgrade.
Each bounded `CorrectionPush` frame carries its random attempt ID and session
binding. Delivery is one-way on the critical path; inference reports any rejected
correction through the matching activation request. Payloads are not audit logged.

Fresh uniform masks in the smallest sufficient exact `u16`, `u24`, or `u32` ring hide online values from inference. Privacy requires
the preparation service to follow the protocol, erase expanded masks, and not
collude with inference. Self-hosting places that trust inside the client boundary.
Two services controlled by one untrusted operator do not satisfy the assumption.
The client rejects equal origins, non-TLS remote origins, non-public models, and
different model commitments, but cannot prove operator independence or erasure.
Anyone observing both channels can recover the activation, so only TLS is
accepted away from loopback. Authentication does not prove correct computation.

## Confidential weights

The client sees intermediate outputs in public, blinded, and guarded paths. A modified client can submit chosen layer inputs and reconstruct a matrix from enough independent queries. Blinding preparation removes one equation source but does not remove this layer oracle.

Guarded controls limit traffic and bind requests to a principal. They do not establish model secrecy. A protocol for hostile clients must enforce the computation while keeping intermediate values shared or encrypted with a suitable output policy.

## What the fixes address

The included correction patch keeps activation scales local, replaces mask generation with operating system randomness and unbiased modular sampling, and makes result check challenges unpredictable by default. The synthetic scale experiment identified 4,092 of 4,096 token rows before this correction.

That experiment demonstrates why searching payloads for literal prompt text is insufficient. Private metadata can reveal content even when the text itself is absent.

## Single use material

The client creates one seed per stage attempt and commits it before sending.
Timeout, cancellation, rejected speculative work, and an uncertain network result
burn that seed. The client never retries one channel independently and keeps no
durable pool that could be rolled back. Preparation does not retry after an
ambiguous WebSocket send. It discards that connection and reconnects only when a
later independent attempt arrives.

## Deployment assumptions

Keep the local gateway and client device trusted. Restrict network access. Disable external conversation tracing. Audit log destinations. Independent review of the complete malicious provider protocol is still required before making a claim against actively hostile model hosts.
