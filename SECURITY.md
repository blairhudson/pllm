# Security Policy

## Supported versions

Security fixes are provided for the latest release and the default branch.
Pre-release (`0.x`) versions are alpha software: treat the documented
[support status](https://pllm.run/sdk/reference/status/) as authoritative for
which protocols, models, and deployments are actually covered.

## Reporting a vulnerability

Use GitHub private vulnerability reporting under **Security > Advisories > Report
a vulnerability**. Do not disclose vulnerability details in a public issue.

Include the affected version, potential impact, and steps to reproduce. Do not
include credentials, private data, or other secrets.

If private reporting is unavailable, open an issue requesting a private contact
without including vulnerability details.

## Security model

PLLM's premise is a trust boundary: the client-controlled gateway keeps prompts
inside the client boundary, while preparation and inference roles may run under
independent operators. The meaningful security questions are documented, not
assumed:

- [Architecture and trust boundary](https://pllm.run/learn/understand/trust-boundary/) —
  which components are trusted, which are provider-side, and what "non-collusion"
  requires operationally.
- [Privacy assurance](https://pllm.run/learn/understand/privacy-assurance/) —
  what is and is not claimed about privacy for each path.
- [Current support status](https://pllm.run/sdk/reference/status/) — the
  per-integration, per-protocol checklist of what has been verified. Entries
  marked **Not evaluated** or **In progress** should be treated as unverified.

## Scope notes

- A `--local` gateway run co-locates every role under one operator. It exercises
  the protocol but provides no operator non-collusion.
- Homomorphic-encryption paths are optional (`he` extra, TenSEAL/SEAL). The
  default local-test correlation mode is a development transport, not a
  privacy guarantee.
- Python configuration targets execute local code; noninteractive use requires
  explicit `--trust-python`.
- Released wheels are built by trusted publishing from git tags; see
  [RELEASING.md](RELEASING.md) for the provenance chain.

## Cryptographic dependencies

`cryptography`, `tenseal` (optional `he` extra), `safetensors`, `tokenizers`.
Dependency updates are handled through Dependabot; report vulnerabilities in a
dependency through the same private reporting path so they can be tracked and
disclosed together.
