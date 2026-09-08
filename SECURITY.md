# Security

## Intended boundary

The client owns plaintext input, generated text, token IDs, activation scales,
HE secrets, preparation inventory and model state. The provider owns large
projection matrices. Public weight inference assumes that the provider follows
the protocol. Message authentication is not verification of arbitrary model
execution.

The provider can observe model identifiers, tensor shapes, scheduling, timing,
approximate lengths and cache traffic. Quantization parity refers to the clear
integer reference, not the original floating checkpoint.

## Unsupported guarantees

Guarded and blinded profiles do not protect confidential weights from a client
that can replace its runtime and query layers. The authenticated arithmetic
preview is a simulator; it is not a distributed protocol demonstrating security
against malicious participants. Its tests do not establish such a guarantee.

Restoring a complete VM or an old inventory snapshot is outside the restart
model. Never restore prepared material into an existing live session. Never
reuse a reserved mask. Keep gateway and provider credentials distinct. Keep the
local plaintext gateway on loopback or a protected customer network.

## Reporting a vulnerability

Enable GitHub private vulnerability reporting in the repository's security
settings. Submit reports through the repository **Security → Advisories → Report
a vulnerability** page. If private reporting has not been enabled, open an issue
asking for a private contact, without exploit details or sensitive data.

Reports should identify the revision, affected privacy profile, adversary,
reproduction steps and observed disclosure. Do not attach production prompts,
keys, model secrets or correlation inventories.

## CI and release boundary

Pull requests run without deployment credentials. PyPI publication has a separate
job, an environment approval gate and an OIDC identity. GitHub Pages has a
separate deployment job. A successful test run is not a cryptographic audit.
