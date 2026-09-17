# Client boundary

Understand which data and runtime state must stay with the client.

[View canonical HTML](https://pllm.run/sdk/operate/client-boundary/)

Document ID: `pllm.docs.operate.client-boundary`  
Release: `0.1.0`  
Build: `sha256:55dedf191ed9de69d95eb69a68160a404194419bf2f721a57103e95fdce4a4e0`  
Source hash: `sha256:61f75639fd02b152d6deea7dfcfc40ce69a56e14ba8e5e5bd44edd963bccfcb5`

`pllm.runtime` exposes clients, transport code, privacy declarations, gateways,
and application factories. The [Python API inventory](/sdk/reference/python/pllm/)
lists each public object. Some interfaces predate complete plan-based deployment;
an importable object is not necessarily supported for production use.

Plaintext requests terminate inside trusted client boundary. Client owns tokenization, state, sampling, and decoded output under methods that claim this placement. Gateway compatibility must be tested against exact application/SDK semantics; API resemblance does not establish full OpenAI compatibility.

The [local gateway](/learn/integrations/local-gateway/) binds to loopback and is
client-controlled. Provider roles do not receive plaintext prompts, tool
schemas, arguments, or results. See [status](/sdk/reference/status/) for runtime
limits.

## Python SDK example

```python
import pllm

mode = pllm.PrivacyMode.PUBLIC
assert mode.value == "public"
assert mode.protocol == "masked_w4a4"
```

API: [`pllm.PrivacyMode`](/sdk/reference/python/pllm/#objects-and-signatures)

This reads a public privacy declaration; it does not create a protected client session.
