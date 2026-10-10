"""Use the deployment configured in PLLM_CONFIG; negotiation belongs to the SDK.

The TOML file's [client] table sets network and experiment JSON paths. Provider
credentials remain in the environment references declared by the network file.
"""

from pllm import OpenAI


def main():
    with OpenAI() as client:
        response = client.responses.create(input="Hi", max_output_tokens=2, temperature=0)
        print(response.output_text)
        decision = client.planning(response.id)
        if decision is not None:
            assert decision.status == "feasible"
            print(f"Selected plan: {decision.digest}")


if __name__ == "__main__":
    main()
