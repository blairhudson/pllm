# Responses API

Use PLLM's Responses API surface and understand its tested conformance scope.

[View canonical HTML](https://pllm.run/learn/integrations/responses-api/)

Document ID: `pllm.docs.learn.integrations.responses-api`  
Release: `0.1.0`  
Build: `sha256:55dedf191ed9de69d95eb69a68160a404194419bf2f721a57103e95fdce4a4e0`  
Source hash: `sha256:b5c26a66a5b12c51c0db6694ebd66c0925fbb95c68b8048928cf4745a1775861`

Send Responses API requests to `http://127.0.0.1:8080/v1/responses` with a bearer
token for the local gateway. The endpoint supports non-streaming and SSE
streaming text responses, stored response retrieval and cancellation,
continuation with `previous_response_id`, and local function-call items.

`POST /v1/responses/compact` performs bounded local history compaction. It keeps
recent items and marks elided history; it does not ask a provider to generate a
semantic summary.

## Conformance result

The gateway passed all 17 conformance tests from
[OpenResponses](https://www.openresponses.org/) against its 2026-04-24
specification at pinned upstream commit
[`92c12d96d7b61d6d15e2214daa5e9c6000ab6e1c`](https://github.com/openresponses/openresponses/commit/92c12d96d7b61d6d15e2214daa5e9c6000ab6e1c).
The run covered the gateway contract exercised by those tests. It does not prove
support for every Responses API field, hosted tool, transport, or model modality.

Current model traffic is text plus local function tools. Image and file inputs
are not model-understood modalities, and provider-built-in tools are not
available. Unsupported fields fail instead of being silently accepted.

Use the [OpenAI Python SDK](/learn/integrations/openai-python/) or
[OpenAI Agents SDK](/learn/integrations/openai-agents/) for tested Python client
paths.
