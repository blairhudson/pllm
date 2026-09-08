import os

from pllm import OpenAI

model = os.environ.get("HE_MODEL", "gemma-4-e2b-he")
with OpenAI(
    base_url=os.environ.get("PLLM_BASE_URL", "http://127.0.0.1:8000"),
    api_key=os.environ.get("PLLM_API_KEY", "he-local"),
    correlation_mode=os.environ.get("PLLM_CORRELATION_MODE", "bfv"),
    he_transport="websocket",
    correlation_prefetch=8,
    token_cache_size=4096,
) as client:
    response = client.responses.create(
        model=model,
        input="Explain why homomorphic encryption is useful in one paragraph.",
        max_output_tokens=128,
        stream=True,
    )
    for event in response:
        if event.type == "response.output_text.delta":
            print(event.delta, end="", flush=True)
    print()
    print(client.privacy_audit.to_dict())
