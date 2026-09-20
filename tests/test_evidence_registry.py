from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

import pllm
from pllm.components import get, list_component_classes
from pllm.metrics import (
    Accuracy,
    Communication,
    Cost,
    Energy,
    Latency,
    Memory,
    Perplexity,
    Throughput,
)

ROOT = Path(__file__).resolve().parents[1]


def result_document(identity: str = "run-1") -> dict:
    environment = {"cpu": "fixture", "transport": "loopback"}
    return {
        "schema_version": "pllm.benchmark_result.v1",
        "id": identity,
        "created_at": "2026-09-19T12:00:00Z",
        "status": "completed",
        "scope": "deployment",
        "profile": "baseline.masked_linear_cpu",
        "component_ids": ["pllm/cpu", "pllm/masked-linear"],
        "model": {
            "id": "org/model",
            "checkpoint_digest": "1" * 64,
            "source_lock_digest": "2" * 64,
        },
        "plan_lock_digest": "3" * 64,
        "configuration_digest": "4" * 64,
        "workload_digest": "5" * 64,
        "privacy_cohort": "masked-linear",
        "numeric_cohort": "wrap32",
        "environment": {
            "digest": pllm.environment_digest(environment),
            "attributes": environment,
        },
        "warmups": 0,
        "repetitions": 1,
        "samples": [
            {
                "index": 0,
                "kind": "measurement",
                "status": "completed",
                "duration_seconds": 1.25,
                "artifact_digest": None,
            }
        ],
        "metrics": [
            {
                "id": "latency-full-median",
                "component": "pllm/latency",
                "parameters": {"statistic": "median", "phase": "full"},
                "value": 1.25,
                "unit": "seconds",
                "unit_detail": None,
                "statistic": "median",
                "phase": "full",
                "role": "client",
                "origin": "native_executed",
                "unavailable_reason": None,
                "sample_count": 1,
            },
            {
                "id": "throughput-token",
                "component": "pllm/throughput",
                "parameters": {"basis": "tokens"},
                "value": 8.0,
                "unit": "per_second",
                "unit_detail": "tokens_per_second",
                "statistic": "median",
                "phase": "online",
                "role": "client",
                "origin": "native_executed",
                "unavailable_reason": None,
                "sample_count": 1,
            },
            {
                "id": "energy-unavailable",
                "component": "pllm/energy",
                "parameters": {"source": "not_available"},
                "value": None,
                "unit": "joules",
                "unit_detail": None,
                "statistic": "measurement",
                "phase": "full",
                "role": None,
                "origin": "not_available",
                "unavailable_reason": "no external energy meter",
                "sample_count": 0,
            },
        ],
        "limitations": [
            "single-host loopback measurement",
            "energy was not measured",
        ],
    }


def test_metric_components_are_typed_registered_and_immutable() -> None:
    metrics = (
        Latency(),
        Throughput(),
        Communication(),
        Memory(),
        Energy(),
        Accuracy(dataset="fixture"),
        Perplexity(dataset="fixture"),
        Cost(),
    )
    assert len(list_component_classes()) == 29
    for metric in metrics:
        assert get(metric.component) is type(metric)
        assert metric.describe().category == "pllm/benchmark-metric"
        assert type(metric).from_params(metric.get_params()) == metric
    assert Latency().with_params(statistic="p95").params["statistic"] == "p95"
    with pytest.raises(pllm.ConfigurationError):
        Accuracy(dataset="")
    with pytest.raises(pllm.ConfigurationError):
        Cost(currency="usd")


def test_benchmark_result_is_schema_valid_canonical_and_digest_bound() -> None:
    document = result_document()
    schema = json.loads((ROOT / "schemas/benchmark-result.schema.json").read_text())
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(document)

    result = pllm.BenchmarkResult.from_dict(document)
    assert result.id == "run-1"
    assert len(result.digest) == 64
    assert result.to_dict()["component_ids"] == ["pllm/cpu", "pllm/masked-linear"]
    assert [metric["id"] for metric in result.to_dict()["metrics"]] == [
        "energy-unavailable",
        "latency-full-median",
        "throughput-token",
    ]
    assert result.canonical_bytes() == json.dumps(
        result.to_dict(),
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    with pytest.raises(TypeError):
        result.data["status"] = "failed"
    with pytest.raises(ValueError, match="canonical"):
        pllm.BenchmarkResult(json.dumps(result.to_dict()).encode())


@pytest.mark.parametrize(
    "change",
    [
        lambda value: value["environment"].update(digest="0" * 64),
        lambda value: value["metrics"][0].update(unit="bytes"),
        lambda value: value["metrics"][0].update(value=float("nan")),
        lambda value: value["metrics"][1].update(unit_detail="requests_per_second"),
        lambda value: value["metrics"][2].update(value=1.0),
        lambda value: value["samples"][0].update(duration_seconds=None),
        lambda value: value.update(repetitions=2),
        lambda value: value.update(limitations=[]),
        lambda value: value.update(unexpected=True),
    ],
)
def test_benchmark_result_rejects_semantic_mismatch(change) -> None:
    document = result_document()
    change(document)
    with pytest.raises((TypeError, ValueError)):
        pllm.BenchmarkResult.from_dict(document)


def test_registry_queries_exact_cohorts_and_preserves_negative_results() -> None:
    completed = pllm.BenchmarkResult.from_dict(result_document("completed"))
    negative_document = result_document("negative")
    negative_document.update(status="failed")
    negative_document["samples"][0].update(status="failed", duration_seconds=None)
    negative_document["limitations"].append("candidate failed before completion")
    negative = pllm.BenchmarkResult.from_dict(negative_document)
    registry = pllm.EvidenceRegistry([negative]).register(completed)

    assert registry.get("completed") is completed
    assert tuple(result.id for result in registry.list()) == ("completed", "negative")
    assert registry.query(status="failed") == (negative,)
    assert registry.query(profile="baseline.masked_linear_cpu") == (completed, negative)
    assert registry.query(component="pllm/cpu") == (completed, negative)
    assert registry.query(model="org/model") == (completed, negative)
    assert registry.query(metric="pllm/energy") == (completed, negative)
    assert registry.query(environment_digest=completed.to_dict()["environment"]["digest"]) == (
        completed,
        negative,
    )
    assert registry.register(completed) is registry

    conflict_document = result_document("completed")
    conflict_document["metrics"][0]["value"] = 2.0
    conflict = pllm.BenchmarkResult.from_dict(conflict_document)
    with pytest.raises(ValueError, match="conflicting"):
        registry.register(conflict)
    with pytest.raises(KeyError, match="not found"):
        registry.get("missing")


def test_not_available_result_requires_explicit_reason_and_no_samples() -> None:
    document = result_document("not-available")
    document.update(status="not_available", repetitions=0, samples=[])
    document["metrics"] = [document["metrics"][2]]
    result = pllm.BenchmarkResult.from_dict(document)
    assert result.to_dict()["metrics"][0]["unavailable_reason"] == "no external energy meter"
