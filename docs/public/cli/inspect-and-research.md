# Inspect configuration and components

Validate public experiment data and inspect built-in component records.

[View canonical HTML](https://pllm.run/cli/inspect-and-research/)

Document ID: `pllm.docs.cli.inspect-and-research`  
Release: `0.1.0`  
Build: `sha256:55dedf191ed9de69d95eb69a68160a404194419bf2f721a57103e95fdce4a4e0`  
Source hash: `sha256:9b0a5fbfec8bdc3688b22b3b76c90dd117f05b0ec7089fbcb881152c80fb0ea9`

Inspect or export public experiment configuration:

```bash
pllm config show examples/pllm.yaml
pllm config export examples/pllm.yaml --output experiment.json
```

Inspect built-in components:

```bash
pllm components list --format json
pllm components show pllm/cpu --format json
```

These commands report static or author-supplied records. Presence does not prove
runtime coverage, evidence quality, security, or publication readiness.

Continue to generated [configuration](/cli/reference/config/) and
[component](/cli/reference/components/) references. For research work, read the
[paper catalog](/research/papers/), [backlog](/research/backlog/), and
[method boundaries](/research/methods/).
