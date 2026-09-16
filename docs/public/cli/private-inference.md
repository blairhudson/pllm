# Run private inference

Run the trusted gateway locally or connect it to independently operated PLLM roles.

[View canonical HTML](https://pllm.run/cli/private-inference/)

Document ID: `pllm.docs.cli.private-inference`  
Release: `0.1.0`  
Build: `sha256:dab1bcb88380ca9a4c2b79e3de085391f482bb97d841a9f0741979579c87dbfb`  
Source hash: `sha256:d38d48f9661d3c10599c881dc798f22ea20e556f218f95aca3730f87056c0517`

Applications connect to the trusted loopback gateway. They do not send ordinary
Responses API or Chat Completions API requests to an inference provider.

## Local development

```bash
pllm gateway --local --model Qwen/Qwen2.5-0.5B-Instruct --local-files-only --api-key local
```

This command starts the gateway plus co-located inference and preparation roles.
It exercises the private protocol, but one machine does not demonstrate operator
non-collusion.

## Configured roles

```bash
pllm gateway --config client.toml
```

The client configuration identifies independently operated HTTPS role endpoints
and credentials. The gateway remains inside the client boundary and exposes
`/v1/responses` and `/v1/chat/completions` on loopback.

Use the exact [`gateway` command reference](/cli/reference/gateway/) for flags.
Then choose a tested [consumer integration](/learn/integrations/) or use the
[native client boundary](/sdk/operate/client-boundary/) directly.
