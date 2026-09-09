# Start here

A familiar API. A different privacy boundary.


PLLM divides public-weight execution among a client you control, a trusted preparation service, and an untrusted inference provider. The client keeps conversation text, activation scales, and decoded output. Both services evaluate large matrix operations on separate masked shares.

The fast protocol uses fresh seeded `u16`, `u24`, or `u32` masks and ordinary integer arithmetic. Preparation receives the seed and pushes `W·r-s` to inference's fixed authenticated endpoint. Inference receives `x-r`, returns `W·x-s`, and never receives the seed. The client receives only a small preparation acknowledgement, adds its locally expanded `s`, and center-decodes. Preparation must not retain masks or collude with inference.

## Choose your path

| Your task | Start with |
| --- | --- |
| Use the Python client | [Install the client](/docs/client/install) |
| Keep an existing OpenAI application | [Use the local gateway](/docs/client/openai) |
| Host a checkpoint | [Install the provider](/docs/server/install) |
| Deploy on a private network | [Deployment overview](/docs/deployment/overview) |
| Understand the protocol and results | [Read the paper](/docs/research/paper) |

## Three roles

Inference listens on port **8000** and preparation commonly uses **8001**. The local gateway listens on **8080** and exposes `/v1/responses` to your application. Plaintext enters only the trusted client or local gateway.

## Current scope

The bundled reference runtime uses just-in-time seeded preparation for public weights. It keeps no durable mask inventory and performs no public-path BFV work. Historical coefficient preparation experiments remain under `research/` and are not the application default.

Public weights are the documented deployment path. Guarded confidential weights do not protect the model against a modified client. Read [security boundaries](/docs/security) before moving a workload across machines.
