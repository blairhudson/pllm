"""Run a bounded, client-owned conversation through ordinary prepared roles."""

from __future__ import annotations

import argparse
import json

import pllm
from pllm.preparation import PreparedInventory
from pllm.profiles import MaskedLinearCpu
from pllm.quantization import SymmetricPerRow
from pllm.state import ClientPrefixReuse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".pllm/conversation-example")
    args = parser.parse_args()
    experiment = pllm.Experiment(
        "managed-conversation",
        MaskedLinearCpu(
            pllm.Model.tiny(),
            quantization=SymmetricPerRow(causal_reduction="prefix_f32"),
            cache=ClientPrefixReuse(max_bytes=4 << 20, fixed_input_tokens=254),
            inventory=PreparedInventory(refill="on-demand", allocation="demand"),
        ),
        pllm.Deployment.local(root=args.root),
        pllm.ExecutionBudget(requests=3, max_input_tokens=254, max_new_tokens=2),
    )
    with pllm.serve_local(experiment) as roles, roles.client() as client:
        with client.conversation(
            max_state_bytes=4 << 20, max_output_tokens=2, temperature=0
        ) as chat:
            for prompt in ("Describe a tree.", "And its roots?", "One more detail."):
                result = chat.send(prompt)
                print(
                    json.dumps(
                        {
                            "output_tokens": result.usage.output_tokens,
                            "retained_state_bytes": chat.state_bytes,
                        }
                    )
                )
        assert chat.state_bytes == 0


if __name__ == "__main__":
    main()
