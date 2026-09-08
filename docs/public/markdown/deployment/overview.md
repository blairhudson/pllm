# Deployment overview

Separate the provider, trusted client, and documentation site.


## Three processes with different responsibilities

| Component | Accepts plaintext? | Recommended location |
| --- | --- | --- |
| Provider | No prompt text on the private path | Machine that hosts the model matrices |
| Client SDK or gateway | Yes | Customer workstation or trusted customer service |
| Documentation site | No inference input | Static hosting, separate from either runtime |

## Local evaluation

Use the provider on port 8000 and the local gateway on port 8080. Keep both bound to loopback. Run one client per conversation while checking memory and lifecycle metrics.

## Private network evaluation

Place the client inside the customer boundary and the provider close enough that network latency is practical. Protect the provider connection with TLS, firewall it, and keep administrative credentials separate from application credentials. A customer gateway is trusted with plaintext; hosting it at the model provider changes the threat model.

## Containers and system services

Use [Docker](/docs/deployment/docker) for repeatable environments or [systemd](/docs/deployment/systemd) for a managed Linux host. The supplied deployment files are templates checked for command alignment, not evidence of a live production deployment.

## The docs are not a gateway

The Fumadocs application is a static site. It does not expose a prompt box that forwards secrets to a public server. Search runs in the browser over a static documentation index, and no analytics or remote font service is configured.

[Deploy the site](/docs/deployment/site) independently on a static host.
