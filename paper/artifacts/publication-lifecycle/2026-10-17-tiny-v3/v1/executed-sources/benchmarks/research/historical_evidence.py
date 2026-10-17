"""Frozen historical factories; replay creates new evidence, never old timings.

The authoritative reports contain complete serialized targets/Experiments. Hash
verification prevents a moving example or a newer build from replacing them.
Artifact paths and historical BuildRefs are deliberately not rewritten.
"""

import hashlib
import json
from copy import deepcopy
from pathlib import Path

SOURCES = {
    "research-mlp-tiling-qwen25-screen": "22b2008ca6a215c2fe8e7b0e5f0ad86c3f90e002d0097dd9b6e28ba8532d0c00",
    "research-mlp-tiling-qwen25-confirmation": "e865f74276cd9d275377a5bc9c3ed9c6bb3f179a44975717d209a638066c9658",
    "research-mlp-tiling-qwen25-confirmation-b": "a9ed73f0dea58a9e4a0360ad2e4e1509f4fc71b355c9b6fa7b3bf2aee1e7feaa",
    "research-mlp-tiling-qwen25-confirmation-c": "85770d1e7ce05478278b1b450703c795887233452730afbae5f5120cb6675a9e",
    "research-prepared-study-qwen25": "8d630bb02d51516785d4c7bde02211a148e735349618c256c5fdc06aabb5e20b",
    "research-prepared-study-qwen25-confirm": "9b54ee455e7307429d4cc1bf19c93f1c58ff3ef966c258ddebde2ec8c68c42db",
    "research-all-link-placement-sdk-confirmation": "9f81bc8c657c156d7f66fbcd90cb9ae9f708801ffba62c70f5365926a2812bf0",
    "epoch-prefix-sdk-qwen25-2026-10-06": "ccd5469f114bcfe7c5000a8debdc80c66586deb507c38d15399b6bfa31a05c62",
    "research-sota-combinations-qwen25": "5fb73b6883a0e42ac42b65b33a84cdfe0c6646d1e459f0df72639c38756ad54d",
}


def source_document(source):
    """Read only an explicitly frozen source, validating its original bytes."""
    expected = SOURCES[source]
    root = Path(__file__).resolve().parents[2]
    raw = (root / "docs" / "evidence" / f"{source}.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("historical source identity changed")
    return json.loads(raw)


def recorded_contract(source, name):
    """Frozen recipe/build/lifecycle metadata; never an assertion of replay success."""
    document = source_document(source)
    if document["schema"] == "pllm.epoch_prefix_sdk_benchmark.v1":
        workload, variant = name.split("/", 1)
        records = [record for record in document["results"]
                   if (record["workload"], record["variant"]) == (workload, variant)]
        if not records or name not in document["experiments"]:
            raise ValueError("unknown historical cell")
        return deepcopy({
            "experiment": document["experiments"][name], "build": document["build"],
            "source_lock_digest": document["source_lock_digest"],
            "measurement_sources": document["frozen_sources"], "scope": document["scope"],
            "workload_commitment": document["workload_commitment"],
            "lifecycle": {"workload": workload, "variant": variant,
                          "concurrent_requests": len(records[0]["client"]["responses"]),
                          "fresh_client_and_providers": True,
                          "response_statuses": [response["status"] for response in records[0]["client"]["responses"]]},
            "requires_original_measurement_driver": True,
        })
    records = [record for record in document["results"]
               if record["execution"]["case"]["name"] == name]
    if not records:
        raise ValueError("unknown historical case")
    spec = records[0]["execution"]["case"]
    if any(record["execution"]["case"] != spec for record in records):
        raise ValueError("historical case changed between replicates")
    return deepcopy({"case": spec, "sampling": document["suite"]["sampling"],
                     "cohort": document["cohort"],
                     "source_lock_digest": records[0]["model"]["source_lock_digest"],
                     "lifecycle": spec["target"]["parameters"]["scenario"],
                     "isolation": records[0]["execution"]["isolation"]})


def case(source, name):
    """Exact BenchmarkCase including historical BuildRef, target and budgets."""
    from pllm.metrics import BenchmarkCase

    return BenchmarkCase.from_spec(recorded_contract(source, name)["case"])


def target(source, name):
    """Target only; use case() to retain the required original BuildRef."""
    return case(source, name).target


def experiment(source, name):
    """Recipe only; recorded_contract() binds original build and driver lifecycle."""
    from pllm import Experiment

    return Experiment.from_spec(recorded_contract(source, name)["experiment"])
