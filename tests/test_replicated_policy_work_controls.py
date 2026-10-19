"""Installed-wheel generated control: observe repeatability, not numeric oracle."""
import json

from pllm import OpenAI
from test_replicated_policy_sdk import (
    combined_experiment, observe_publishers, observe_public_work, public_worker_counts,
    directed_peer_bytes, assert_publishers_retired, assert_subprocess_roles,
)


def test_identical_valid_private_inputs_public_work_control(tmp_path, monkeypatch):
    owners = observe_publishers(monkeypatch)
    work = observe_public_work(monkeypatch)
    counts = []
    with OpenAI(experiment=combined_experiment(tmp_path, rejecting=True), timeout=300) as sdk:
        topology = sdk._owned_topology
        assert_subprocess_roles(topology)
        for _ in range(2):
            assert sdk.responses.create(input="hello", max_output_tokens=1).output_text == "world"
            counts.append(public_worker_counts(sdk._core))
        first, second = work.values()
        assert first["admission"] == second["admission"]
        assert first["schedule"] == second["schedule"]
        first_bytes, second_bytes = directed_peer_bytes(first), directed_peer_bytes(second)
        second_counts = {role: {name: value - counts[0][role][name] for name, value in row.items()}
                         for role, row in counts[1].items()}
        print(json.dumps({"identical_valid_input": True, "first_bytes": first_bytes,
            "second_bytes": second_bytes, "first_counts": counts[0], "second_counts": second_counts,
            "bytes_equal": first_bytes == second_bytes, "counts_equal": counts[0] == second_counts}, sort_keys=True))
        assert len(sdk._core.records) == 2
        assert_publishers_retired(owners)
    assert all(not status.running for status in topology.statuses)
