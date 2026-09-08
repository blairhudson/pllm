from pllm import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000",
    api_key="he-local",
    correlation_mode="bfv",
    he_transport="websocket",
)
try:
    stream = client.responses.create(model="he-bigram-demo", input="private", stream=True)
    for event in stream:
        if event.type == "response.output_text.delta":
            print(event.delta, end="", flush=True)
finally:
    client.close()
