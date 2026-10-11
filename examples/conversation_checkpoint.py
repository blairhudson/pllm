"""Restore an encrypted, client-owned conversation with fresh provider material."""
from __future__ import annotations

import os
from pathlib import Path
import tempfile

import pllm
from pllm.preparation import PreparedInventory
from pllm.profiles import MaskedLinearCpu
from pllm.quantization import SymmetricPerRow
from pllm.state import ClientPrefixReuse, ConversationCheckpoint


def main():
    with tempfile.TemporaryDirectory(prefix="pllm-conversation-") as root:
        experiment = pllm.Experiment(
            "checkpoint-conversation", MaskedLinearCpu(pllm.Model.tiny(),
                quantization=SymmetricPerRow(causal_reduction="prefix_f32"),
                cache=ClientPrefixReuse(max_bytes=4 << 20, fixed_input_tokens=254),
                inventory=PreparedInventory(refill="on-demand", allocation="demand")),
            pllm.Deployment.local(root=root),
            pllm.ExecutionBudget(requests=3, max_input_tokens=254, max_new_tokens=2))
        key, path = os.urandom(32), Path(root) / "checkpoint"
        with pllm.serve_local(experiment) as roles:
            with roles.client() as client, client.conversation(
                max_state_bytes=4 << 20, max_output_tokens=2, temperature=0) as chat:
                chat.send("Describe a tree.")
                head = chat.checkpoint(path, key)
            # Keep key and trusted head in caller-controlled private storage.
            # Provider inventory and old response IDs are never restored.
            head = ConversationCheckpoint.from_spec(head.to_spec())
            with roles.client() as client, client.conversation(
                max_state_bytes=4 << 20, max_output_tokens=2, temperature=0) as chat:
                chat.restore(path, key, head)
                response = chat.send("And its roots?")
                print({"output_tokens": response.usage.output_tokens,
                       "retained_state_bytes": chat.state_bytes})


if __name__ == "__main__":
    main()
