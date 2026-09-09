# Security

## Intended boundary

The client owns plaintext input, generated text, token IDs, activation scales,
fresh mask seeds and model state. Public-weight inference uses two service roles:
a trusted preparation service receives seeds, derives `r` and `s`, and pushes
`W·r-s` directly to a fixed inference endpoint. The untrusted inference provider
receives `x-r`, combines the correction, and returns `W·x-s`; the client adds
`s`. The client never receives the correction and supplies no callback URL. Both hold
the public projection matrices. Message authentication is not verification of
arbitrary model execution.

Before sending stage traffic, the client asks preparation to authorize one random
inference session. That compact authorization binds model/body/stage commitments,
quantization, and the bounded attempt budget. Preparation validates it against its
loaded model and relays it with a distinct provider push credential. Inference
accepts it once; inference-client credentials cannot authorize sessions, and no
prepared runner computation starts for an unauthorized session.

Corrections use a persistent binary WebSocket from preparation to the fixed
inference origin. Preparation derives `ws` only from a validated loopback `http`
origin and `wss` only from a validated `https` origin; configuration cannot supply
a separate WebSocket host or path. The upgrade accepts only the provider push
credential and a fixed subprotocol. Each bounded frame contains a bounded session
identifier and random attempt ID in one `CorrectionPush`. Delivery is one-way;
inference burns rejected frames and reports failure through the matching activation
request. Neither service logs correction payloads.

Both services can observe model identifiers, stage names, tensor shapes,
scheduling, timing and approximate lengths. Preparation must erase expanded
masks and must not collude with inference; either condition failing reveals the
activation. Self-hosting preparation keeps that trust inside the client boundary.
Quantization parity refers to the clear integer reference, not the original
floating checkpoint.

## Unsupported guarantees

Guarded and blinded profiles do not protect confidential weights from a client
that can replace its runtime and query layers. The authenticated arithmetic
preview is a simulator; it is not a distributed protocol demonstrating security
against malicious participants. Its tests do not establish such a guarantee.

There is no durable public-path preparation inventory. Every stage creates a
fresh seed, and any failed or ambiguous attempt burns that seed and both channel
requests. A disconnect after a send is ambiguous: preparation never retries that
correction. It drops the socket and reconnects only for a later independent
attempt. Never retry only one channel or reuse a mask. Remote endpoints require
TLS and distinct origins; loopback is allowed for local evaluation. Use three
distinct credentials for client-to-inference, client-to-preparation, and
preparation-to-inference traffic. Inference bounds pending attempts and bytes,
times out unmatched halves, rejects authorization replay, and tombstones consumed
or burned attempt IDs. Keep the
local plaintext gateway on loopback or a protected customer network.

## Reporting a vulnerability

Enable GitHub private vulnerability reporting in the repository's security
settings. Submit reports through the repository **Security → Advisories → Report
a vulnerability** page. If private reporting has not been enabled, open an issue
asking for a private contact, without exploit details or sensitive data.

Reports should identify the revision, affected privacy profile, adversary,
reproduction steps and observed disclosure. Do not attach production prompts,
keys, model secrets, mask seeds or preparation responses.

## CI and release boundary

Pull requests run without deployment credentials. PyPI publication has a separate
job, an environment approval gate and an OIDC identity. GitHub Pages has a
separate deployment job. A successful test run is not a cryptographic audit.
