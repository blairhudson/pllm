from pllm import OpenAI

with OpenAI(
    base_url="http://127.0.0.1:8000",
    api_key="he-local",
    correlation_mode="bfv",
    he_transport="websocket",
) as client:
    response = client.responses.create(
        model="he-bigram-demo",
        input="This prompt stays on the client.",
    )
    print(response.output_text)
    print(client.privacy_audit.to_dict())
