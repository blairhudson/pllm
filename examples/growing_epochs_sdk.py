"""Fresh-wheel generated exact-state SDK/gateway validation, not a benchmark.

    python examples/growing_epochs_sdk.py --root target/growing-sdk-check
    python examples/growing_epochs_sdk.py --root target/growing-gateway-check --gateway

Run only after parent-coordinated wheel build. Sources are generated, not downloaded.
No timing, traffic saving, remote-work or peak-memory qualification is claimed.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from pllm import Deployment, ExecutionBudget, Experiment, Model
from pllm.preparation import PreparedInventory
from pllm.profiles import MaskedLinearCpu
from pllm.quantization import SymmetricPerRow
from pllm.runtime.servers import build_roles
from pllm.runtime.tiny_llama import create_tiny_llama_checkpoint
from pllm.state import ClientPrefixReuse


def experiment(root, model, *, growing):
    policy = {"decode_epoch_schedule": (4, 60)} if growing else {"decode_epoch_tokens": 64}
    return Experiment("growing-sdk", MaskedLinearCpu(Model.path(str(model), model_id="growing-sdk"),
        quantization=SymmetricPerRow(weight_bits=8, activation_bits=8, causal_reduction="prefix_f32"),
        inventory=PreparedInventory(refill="on-demand", allocation="demand", **policy),
        cache=ClientPrefixReuse(max_bytes=1 << 20, fixed_input_tokens=128)),
        Deployment.local(root=str(root)),
        ExecutionBudget(requests=4, max_input_tokens=128, max_new_tokens=64))


def run(root, *, gateway=False):
    model = create_tiny_llama_checkpoint(root / "model", model_type="qwen2", seed=179,
        hidden_size=32, intermediate_size=64, head_dim=8, num_hidden_layers=1,
        with_qkv_bias=True, vocab_size=258)
    states = []
    for name, growing in (("fixed", False), ("growing", True)):
        exp = experiment(root / name, model, growing=growing)
        with build_roles(exp, engine_threads=1) as roles, roles.client(
                bundle_cache_dir=root / f"cache-{name}") as client:
            request = dict(input="abc", max_output_tokens=64, temperature=0.8, top_p=0.9)
            if gateway:
                from fastapi.testclient import TestClient
                from pllm.runtime.sidecar import create_sidecar_app
                from pllm.runtime.types import Response

                with TestClient(create_sidecar_app(client=client, local_api_key="local")) as local:
                    reply = local.post("/v1/responses", headers={"Authorization": "Bearer local"},
                        json={"model": "growing-sdk", **request})
                    reply.raise_for_status()
                    response = Response.from_dict(reply.json())
            else:
                response = client.responses.create(**request)
            saved = client._core._transformer_conversations[response.id]
            values = [saved.next_logits, *(value[:row.length] for row in saved.snapshot.caches
                       for value in (row.key, row.value))]
            states.append((response.output_text, response.usage.to_dict(), saved.token_ids,
                saved.pending_token_ids, saved.snapshot.position,
                [value.view(np.uint32).copy() for value in values]))
            audit = client.privacy_audit
            assert audit.preparation_requests_during_online == 0
            assert audit.prepared_stage_rows_reserved == (
                audit.prepared_stage_rows_claimed + audit.prepared_stage_rows_burned)
            assert client._core._online_active == 0
            print({"policy": name, "output_tokens": response.usage.output_tokens,
                   "epochs": audit.decode_epochs_started, "pauses": audit.decode_epoch_pauses})
    assert states[0][:-1] == states[1][:-1]
    for control, candidate in zip(states[0][-1], states[1][-1], strict=True):
        np.testing.assert_array_equal(control, candidate)
    print({"exact_state": True, "gateway": gateway, "scope": "generated validation only"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--gateway", action="store_true")
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    run(args.root, gateway=args.gateway)


if __name__ == "__main__":
    main()
