# Inspect configuration and research

Validate public experiment data and inspect built-in component or research records.

[View canonical HTML](https://pllm.run/cli/inspect-and-research/)

Document ID: `pllm.docs.cli.inspect-and-research`  
Release: `0.1.0`  
Build: `sha256:4a93c61285a110010f1bafefa367e198ed52465071615e2d4d9a91a45f2d82e2`  
Source hash: `sha256:f4133bf92204d7f5b4f16dac4011f42117b90a5ab456dc46a3f5086dd98f2e36`

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

Inspect research records without executing their workflows:

```bash
pllm research sources list --format json
pllm research methods list --format json
pllm research recipes list --format json
```

These commands report static or author-supplied records. Presence does not prove
runtime coverage, evidence quality, security, or publication readiness.

Continue to the generated [configuration](/cli/reference/config/),
[component](/cli/reference/components/), and
[research](/cli/reference/research/) references. For autonomous research work,
read [research agents](/sdk/contribute/agents/) and the
[publication workflow](/research/publications/).
