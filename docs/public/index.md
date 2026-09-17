# PLLM — Private LLM Inference

Private language model inference. Client and server guides, the protocol, and reproducible research.

[View canonical HTML](https://pllm.run/)

Document ID: `pllm.home`  
Release: `0.1.0`  
Build: `sha256:bab6f73b33765ac11862794abf645d4eb324a2fd52abc43cc2a9cd21c09e27c7`  
Source hash: `sha256:8e4a8e12314eff4ae182cecc488c67484e8358825edc7cfdf5b6a03f2ef77285`

# Keep your data private. Open compute to the world.

PLLM is a high-performance private LLM multi-party inference runtime and autonomous research harness.

[Start building →](#start-building)
[Run research →](#run-research)

[Learn how PLLM works](/learn/)

## Separate the compute from the data.

Remote AI usually means one provider gets both the work and the data. PLLM separates them.

When providers do not need the full request, more providers can compete on price, location, speed, and energy use. Users get more choice. Less private data is exposed to one company.

These are goals. Speed, cost, quality, privacy, and energy use must each be measured.

## Split the work across clear roles.

Client
Holds plaintext prompts, context, model state, masks, and output.

Preparation
Creates one-time protected work before inference.

Inference
Runs masked model stages without receiving the seeds.

Neither remote service receives the full request. This depends on both services following the protocol and not colluding.

[Learn how private inference works](/learn/masked-linear-inference/)

## Start building.

Install PLLM from PyPI, then run the trusted client-side gateway. It gives your apps one local OpenAI-compatible endpoint while PLLM handles the private protocol on the client.

**Local gateway**

**Client config**

**Operate services**

```text
pllm gateway --local --model Qwen/Qwen2.5-0.5B-Instruct
```

```text
pllm gateway --config client.toml
```

```text
pllm serve inference --config inference.json
pllm serve preparation --config preparation.json
```

Use client config to connect the gateway to separately operated services. Local mode co-locates the roles for development; co-location does not provide role separation or non-collusion.

[Install PLLM](/learn/start/installation/)
[Send a private request](/learn/start/first-private-request/)
[Review the trust boundary](/learn/understand/trust-boundary/)

## Use your tools.

Use PLLM's native client directly, or point a compatible client at the trusted local gateway. Gateway consumers receive one endpoint and one local key, never preparation or inference service details.

**PLLM SDK**

**Responses API**

**Chat Completions API**

**OpenAI SDK**

**Agents SDK**

**Codex**

**OpenCode**

```text
from pllm import OpenAI

client = OpenAI()
response = client.responses.create(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    input="Hello from PLLM",
)
print(response.output_text)
```

```text
curl http://127.0.0.1:8080/v1/responses \
  -H 'Authorization: Bearer local' \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen2.5-0.5B-Instruct","input":"Hello from PLLM"}'
```

```text
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Authorization: Bearer local' \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen2.5-0.5B-Instruct","messages":[{"role":"user","content":"Hello from PLLM"}]}'
```

```text
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8080/v1",
    api_key="local",
)
response = client.responses.create(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    input="Hello from PLLM",
)
print(response.output_text)
```

```text
from agents import Agent, Runner
from agents.models.openai_responses import OpenAIResponsesModel
from openai import AsyncOpenAI

client = AsyncOpenAI(
    base_url="http://127.0.0.1:8080/v1",
    api_key="local",
)
agent = Agent(
    name="PLLM",
    model=OpenAIResponsesModel(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        openai_client=client,
    ),
)
result = await Runner.run(agent, "Hello from PLLM")
print(result.final_output)
```

```text
model = "Qwen/Qwen2.5-0.5B-Instruct"
model_provider = "pllm"
web_search = "disabled"

[model_providers.pllm]
name = "PLLM"
base_url = "http://127.0.0.1:8080/v1"
env_key = "PLLM_GATEWAY_API_KEY" # Set to local.
wire_api = "responses"
```

```text
{
  "$schema": "https://opencode.ai/config.json",
  "model": "pllm/Qwen/Qwen2.5-0.5B-Instruct",
  "enabled_providers": ["pllm"],
  "provider": {
    "pllm": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "PLLM (Chat Completions API)",
      "options": {
        "baseURL": "http://127.0.0.1:8080/v1",
        "apiKey": "local"
      },
      "models": {
        "Qwen/Qwen2.5-0.5B-Instruct": {
          "name": "Qwen 2.5 0.5B Instruct"
        }
      }
    }
  }
}
```

[PLLM SDK guide](/sdk/operate/client-boundary/)
[Responses API guide](/learn/integrations/responses-api/)
[Chat Completions API guide](/learn/integrations/chat-completions/)
[OpenAI SDK guide](/learn/integrations/openai-python/)
[Agents SDK guide](/learn/integrations/openai-agents/)
[Codex guide](/learn/integrations/codex/)
[OpenCode guide](/learn/integrations/opencode/)

## Run autonomous research.

Use the paper catalog, ordered backlog, and public component APIs to find strong methods, add capability-based components, test them, and run fair benchmarks.

- Find the strongest relevant methods.

- Add them to PLLM.

- Test correctness, privacy, and model quality.

- Compare them under the same conditions.

Comparisons must use the same model, workload, privacy rules, numeric settings, and hardware. Automation cannot invent claims or publish results without review.

[Research overview](/research/)
[Research papers](/research/papers/)
[Research backlog](/research/backlog/)
[Method boundaries](/research/methods/)

## Continue with PLLM.

[Documentation](/)
[Source](https://github.com/blairhudson/pllm)
[Technical paper](/research/paper/)
[Whitepaper](/research/whitepaper/)
[Benchmarks](/sdk/research/benchmarks/)
