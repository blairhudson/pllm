"""Generate locked party configs, then execute via separately started HTTP hosts.

Credentials are configured environment references, never written to these files.
This loopback example establishes functionality, not physical independence.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

from pllm import Deployment, ExecutionBudget, Experiment, Model, OpenAI
from pllm.compiler import plan
from pllm.deployment import NetworkSpec, PartySpec, discover, open_execution
from pllm.model_loader import resolve_model
from pllm.modeling import lower_model
from pllm.profiles import TwoOnlineOffsetCpu
from pllm.runtime.tiny_llama import create_tiny_llama_checkpoint
from pllm.search import PlanningPolicy, PlanningRequest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    if args.command == "prepare":
        root.mkdir(parents=True, exist_ok=True)
        checkpoint = create_tiny_llama_checkpoint(root / "checkpoint", hidden_size=8,
            intermediate_size=16, num_hidden_layers=1, num_attention_heads=2,
            num_key_value_heads=1, head_dim=4)
        network = NetworkSpec.from_file(Path(__file__).with_name("loopback.json"))
        model = Model.path(str(checkpoint), model_id="network-tiny")
        experiment = Experiment("installed-offset", TwoOnlineOffsetCpu(model),
                                Deployment.local(root="local://installed"), ExecutionBudget(1, 32, 2))
        source_lock = resolve_model(model).source_lock_digest
        (root / "network.json").write_bytes(network.canonical_bytes() + b"\n")
        for party in network.parties:
            spec = PartySpec(party.party_id, experiment, source_lock, ("worker_a", "worker_b"),
                             64 << 20, 64 << 20, peer_party_ids=("a", "b"))
            (root / f"party-{party.party_id}.json").write_bytes(spec.canonical_bytes() + b"\n")
        print("Locked checkpoint and public party/network specs written.")
        return
    network = NetworkSpec.from_file(root / "network.json")
    installed = PartySpec.from_file(root / "party-a.json")
    snapshot = discover(network)
    with (root / "checkpoint" / "config.json").open() as stream:
        model_plan = lower_model(json.load(stream), batch=1, max_input_tokens=32, max_new_tokens=2)
    experiment = replace(installed.experiment, name="selected-http-offset", deployment=Deployment.network(
        network_id=network.network_id, network_spec_digest=network.digest))
    request = PlanningRequest(model_plan, (experiment,), PlanningPolicy("client",
        time.time_ns() // 1_000_000, 4, 32, ("client_weight_bytes",), minimum_remote_mac_fraction=0.5),
        source_lock_digest=installed.source_lock_digest)
    decision = plan(request, snapshot=snapshot)
    (root / "decision.json").write_bytes(decision.canonical_bytes() + b"\n")
    with open_execution(decision, network=network) as lease:
        with OpenAI(execution=lease) as client:
            response = client.responses.create(input="Hi", max_output_tokens=2, temperature=0.0)
            print(response.output_text)
            print(f"Output tokens: {response.usage.output_tokens}; stage calls: {client.privacy_audit.inference_stage_calls}")


if __name__ == "__main__":
    main()
