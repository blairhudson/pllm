# Chat Completions API

Connect text and local function-tool clients through PLLM's Chat Completions API adapter.

[View canonical HTML](https://pllm.run/learn/integrations/chat-completions/)

Document ID: `pllm.docs.learn.integrations.chat-completions`  
Release: `0.1.0`  
Build: `sha256:27f84427615190d0e8c08970a963d10d2ec7d1c4bb9a2db6446fba32419c8ada`  
Source hash: `sha256:3e9f8e2b306538cd423838c727dbd741ccc2d9b0da5fe9ab4b623f998b99a570`

`POST /v1/chat/completions` maps OpenAI-compatible chat messages onto the trusted
gateway's Responses API path. It supports non-streaming and SSE streaming text,
system, developer, user, assistant, and tool roles, and local function tools.

The adapter maps assistant tool calls and tool results without sending plaintext
schemas, arguments, or results to either provider role. Unsupported request
controls and non-text message content fail with an error. API resemblance does
not establish compatibility with every OpenAI Chat Completions API feature.

Use base URL `http://127.0.0.1:8080/v1`, API key `local`, and the model ID loaded
by the gateway. The official OpenAI Python SDK's non-streaming and streaming Chat
Completions API paths pass repository tests.

For applications that can use the Responses API directly, prefer
[Responses API](/learn/integrations/responses-api/). For OpenAI-compatible AI SDK
consumers, see [OpenCode](/learn/integrations/opencode/).
