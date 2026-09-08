import os

from pllm import create_openai_client

client = create_openai_client(
    gateway_url=os.environ.get("PLLM_BASE_URL", "http://127.0.0.1:8000"),
    gateway_api_key=os.environ.get("PLLM_API_KEY", "he-local"),
    correlation_mode=os.environ.get("PLLM_CORRELATION_MODE", "bfv"),
    he_transport="websocket",
    correlation_prefetch=8,
    token_cache_size=4096,
)
try:
    response = client.responses.create(
        model=os.environ.get("HE_MODEL", "gemma-4-e2b-he"),
        input="Write a concise explanation of private inference.",
    )
    print(response.output_text)
finally:
    client.close()
