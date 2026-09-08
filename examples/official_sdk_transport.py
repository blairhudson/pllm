from pllm import create_openai_client

client = create_openai_client(
    gateway_url="http://127.0.0.1:8000",
    gateway_api_key="he-local",
    correlation_mode="bfv",
)
response = client.responses.create(
    model="he-bigram-demo",
    input="The OpenAI SDK serializes this, but HETransport intercepts it before the network.",
)
print(response.output_text)
