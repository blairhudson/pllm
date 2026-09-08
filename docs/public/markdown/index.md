# Start here

A familiar API. A different privacy boundary.


PLLM divides model execution between a client you control and a remote provider. The client keeps conversation text, cryptographic keys, activation scales, and decoded output. The provider evaluates the large matrix operations.

The fast protocol uses homomorphic encryption during preparation. During generation it uses ordinary integer arithmetic on masked values. It is not a complete model evaluated entirely inside ciphertexts.

## Choose your path

| Your task | Start with |
| --- | --- |
| Use the Python client | [Install the client](/docs/client/install) |
| Keep an existing OpenAI application | [Use the local gateway](/docs/client/openai) |
| Host a checkpoint | [Install the provider](/docs/server/install) |
| Deploy on a private network | [Deployment overview](/docs/deployment/overview) |
| Understand the protocol and results | [Read the paper](/docs/research/paper) |

## Two endpoints, not one

The provider listens on port **8000** by default. It accepts the private stage protocol. The local gateway listens on port **8080** and exposes `/v1/responses` to your application. Plaintext enters the local gateway, not the provider.

## Current scope

The bundled reference runtime is PLLM 0.14 with the recorded privacy corrections. Its CLI and constructor signatures are the basis for these guides. The newer coefficient preparation experiments are supplied separately. They are not yet the default preparation backend of `pllm serve`.

Public weights are the documented deployment path. Guarded confidential weights do not protect the model against a modified client. Read [security boundaries](/docs/security) before moving a workload across machines.
