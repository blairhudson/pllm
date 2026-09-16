# Private inference for an open compute world

A two-page overview of PLLM prepared private inference.

[View canonical HTML](https://pllm.run/research/whitepaper/)

Document ID: `pllm.research.whitepaper`  
Release: `0.1.0`  
Build: `sha256:4a93c61285a110010f1bafefa367e198ed52465071615e2d4d9a91a45f2d82e2`  
Source hash: `sha256:044280458996ff93941a715d8d9d317143933fae9928b2f7874e0fe23a6d0607`

[Download PDF ↓](/downloads/whitepaper.pdf)

# Private inference for an open compute world

Private inference can let organizations use remote AI compute without exposing prompts, context, or generated output to the infrastructure provider. PLLM keeps sensitive data on the client and sends protected inference requests to remote services. This design could support a broader market of independent compute providers without requiring users to trust one provider with all of their data.

> Diagram: Three-role prepared inference architecture. Before chat, the client sends root seeds to trusted preparation, which sends correction inventory to inference. During chat, the client sends a one-time ticket and masked activation to inference, receives a masked result, and unmasks it locally. A non-collusion boundary separates preparation from inference.

## What PLLM changes

Ordinary hosted inference bundles computation with access to prompts and generated text. PLLM instead keeps tokenization, nonlinear operations, attention state, sampling, and decoding in the client environment. Public transformer-body matrices run remotely as masked integer projections. Compute can therefore be selected independently from willingness to disclose client language. This is a protocol boundary, not a URL setting.[1]

## Prepared inventory

Before chat, the client commits an inventory to one model, body, stage set, quantization profile, and attempt budget. It sends one domain-separated root seed per stage to preparation. Preparation expands one-time input masks *r*, output masks *s*, and tickets, computes *Wr - s*, then pushes each batch to inference. Inference acknowledges accepted stage batches and seals the inventory `READY`.

W(x - r) + (Wr - s) = Wx - s

Online, inference atomically consumes a ticket and returns the masked result; the client adds *s*. Prefill rows share a compact matrix envelope, while decode retains one ticket per stage on a persistent connection. Unused reserved rows burn on cancellation or failure. Inventory is memory-only and disappears on restart or idle expiry.[1]

## Boundary, not invisibility

**Protected from inference:**  plaintext prompts and outputs, token identities, root seeds, masks, activation scales, local attention and recurrent state, and sampling choices. Preparation sees seeds and bound stage metadata, but not online masked activations. Inference sees corrections and masked activations, but not seeds.

**Still exposed:**  both services know the public model, stage names, tensor shapes, quantization profile, scheduling, timing, traffic volume, and approximate sequence lengths. Preparation must follow the protocol, erase expanded masks, and not collude with inference. Two services under one untrusted operator do not meet this assumption; authentication and TLS protect channels but do not prove independence, erasure, or correct model execution.[2]

SHAKE-256 expands each fresh root seed into pseudorandom masks. Under that computational assumption, each masked activation hides its input from inference when viewed alone. The claim assumes protocol-conforming roles, mask non-retention, and non-collusion; Python releases masks after use but verified physical zeroization is not established. Deployment review must therefore name operators and log-access paths, not merely count endpoints.

PLLM / 01 OF 02

## A world of compute, without exposing client data.

> Diagram: Private inference market flywheel. Protect client language, admit independent compute, route by price and energy, and expand access.

## Use more sources of compute

Masked execution separates the choice of compute from access to client data. Regional clouds, renewable projects, sovereign infrastructure, colocation operators, and independent machines could compete without receiving client language. Participation still requires compatible execution, availability, credentials, and market rules, but not permission to read the workload.

## Follow available energy

Solar generation, grid congestion, and idle capacity vary by place and hour. A scheduler could move masked public-model work toward abundant renewable generation and replenish offline inventory where latency matters less. Privacy makes broader routing plausible; dependent client exchanges, correction placement, distance, and operator separation remain physical constraints. PLLM does not yet measure an energy saving.

## Price the complete lifecycle

| Client | Local model work, state, memory and traffic |
| --- | --- |
| Preparation | Body storage, matrix work, correction egress and erasure |
| Inference | Masked compute, inventory, traffic and tickets |
| Market | Discovery, settlement, verification and operator separation |

Broader supply can create price pressure and widen access. This is a market mechanism, not a measured savings result.

## Remove direct access to plaintext

Inference receives tickets and *x-r*; Preparation receives seeds and commitments. Neither service receives plaintext prompts, token IDs, or decoded output. If both services follow the protocol, do not retain masks, and do not collude, neither has the direct plaintext stream that one provider could otherwise keep for training. This is not a universal guarantee against learning: metadata remains visible, and collusion defeats the split.[2]

## Current implementation evidence

Nine warm Qwen2.5-0.5B runs used revision `277d19f` on an Apple M5, with three repetitions per exact context after one warmup.[3]

| Input tokens | Output | TTFT | Client I/O |
| --- | --- | --- | --- |
| 30 | 9 | 0.978 s | 63.33 MB |
| 63 | 16 | 1.374 s | 126.29 MB |
| 255 | 16 | 5.158 s | 432.71 MB |

All records completed authoritatively with one retained model/body fingerprint pair, zero online Preparation protocol work, and zero plaintext prompt/token-byte audit counters. These are CPU loopback results, not WAN, GPU, energy, price, or quality results.

## Deployment gates

Preparation must be independently controlled or client-hosted; target-model quality must pass; the full cost ledger must beat explicit ceilings; and credential separation, erasure, restart behavior, and incident response must survive a pilot.

## References

[1] PLLM, [Private inference](https://pllm.run/understand/architecture/).
[2] PLLM, [Security boundaries](https://pllm.run/understand/privacy-assurance/).
[3] PLLM, [Run your own benchmark](https://pllm.run/measure/reproduce/); [retained study data](https://pllm.run/downloads/current-runtime-2026-09-11.json).

PLLM / 02 OF 02
