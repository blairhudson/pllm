from pllm import OpenAI

with OpenAI(
    base_url="http://127.0.0.1:8000",
    api_key="pllm-local",
    correlation_mode="bfv",
    session_transport="websocket",
) as client:
    response = client.responses.create(
        model="pllm-bigram-demo",
        input="This prompt stays on the client.",
    )
    print(response.output_text)
    print(client.privacy_audit.to_dict())
