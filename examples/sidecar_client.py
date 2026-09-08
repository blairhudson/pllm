from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="local")
response = client.responses.create(model="he-bigram-demo", input="Private prompt")
print(response.output_text)
