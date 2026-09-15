from pllm import create_openai_client

client = create_openai_client(
    gateway_url="http://127.0.0.1:8000",
    gateway_api_key="pllm-local",
    correlation_mode="bfv",
)
response = client.responses.create(model="pllm-bigram-demo", input="private")
print(response.output_text)
client.close()
