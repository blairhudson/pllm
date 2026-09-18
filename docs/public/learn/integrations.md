# Integrations

Connect OpenAI-compatible SDKs and coding agents to the trusted local PLLM gateway.

[View canonical HTML](https://pllm.run/learn/integrations/)

Document ID: `pllm.docs.learn.integrations`  
Release: `0.1.0`  
Build: `sha256:fa1208fc732ce6403c8c82d95417818355eb7253231b6d031bfc704a15c0a95f`  
Source hash: `sha256:67e583054b763f0a58823eb8151468ce268b4953749f688088b29ffb181e0277`

PLLM puts a client-controlled gateway between an application and the preparation
and inference services. The gateway binds to loopback, holds plaintext application
state inside the trusted client boundary, and exposes:

- `POST /v1/responses`
- `POST /v1/responses/compact`
- `POST /v1/chat/completions`
- `GET /v1/models`

Preparation and inference roles do not receive plaintext prompts, tool schemas,
tool arguments, or tool results. They still observe contract-specific metadata
such as public model identity, tensor shapes, timing, traffic volume, and
approximate sequence length. Review the [trust boundary](/learn/understand/trust-boundary/)
before moving either role.

## Choose a path

- [Local gateway](/learn/integrations/local-gateway/) starts all roles for development or connects the gateway to separately operated services.
- [Responses API](/learn/integrations/responses-api/) describes the Responses API surface and conformance evidence.
- [Chat Completions API](/learn/integrations/chat-completions/) covers clients that use `/v1/chat/completions`.
- [OpenAI Python SDK](/learn/integrations/openai-python/) uses the official `openai` client.
- [OpenAI Agents SDK](/learn/integrations/openai-agents/) uses its Responses API model adapter.
- [Codex](/learn/integrations/codex/) configures a custom Responses API provider.
- [OpenCode](/learn/integrations/opencode/) configures an OpenAI-compatible Chat Completions API provider.

All-local mode co-locates preparation and inference. It tests transport and
development workflows; it does not establish operator separation or
non-collusion.
