from __future__ import annotations

import json
import dataclasses
from pathlib import Path

import pytest

import pllm


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "schemas" / "fixtures" / "compile-request.valid.json"


def test_public_compile_and_native_region_benchmark() -> None:
    plan = pllm.compile(FIXTURE.read_text())
    from pllm.plan import CompiledPlan

    assert isinstance(plan, CompiledPlan)
    assert pllm.CompiledPlan is CompiledPlan
    assert not hasattr(plan, "execute_wrap32")
    assert not hasattr(plan, "benchmark_wrap32")
    with pytest.raises(dataclasses.FrozenInstanceError):
        plan.plan_lock_digest = "changed"
    report = pllm.benchmark(
        plan,
        weights=bytes([1, 2, 3, 4, 5, 6]),
        input=b"".join(value.to_bytes(4, "little") for value in (7, 8, 9, 10, 11, 12)),
        id="tiny-region",
        privacy_cohort="masked-linear",
        numeric_cohort="wrap32",
        environment={"fixture": "compile-request.valid"},
        warmups=1,
        repetitions=3,
    )

    document = report.to_dict()
    assert report.schema_version == "pllm.benchmark_report.v1"
    assert report.canonical_bytes().startswith(b'{"claim_boundary":')
    assert len(document["samples"]) == 3
    assert document["scope"] == "region"
    assert document["plan_lock_digest"] == plan.plan_lock_digest
    assert document["claim_boundary"]["privacy"] == "not_claimed"
    with pytest.raises(TypeError):
        report.data["schema_version"] = "changed"


def test_native_assurance_includes_negative_controls() -> None:
    report = pllm.assure()
    document = report.to_dict()

    assert report.schema_version == "pllm.assurance_report.v1"
    assert "production runtime not attacked" in document["limitations"]
    findings = {finding["id"]: finding for finding in document["findings"]}
    assert findings["mask_reuse"]["outcome"] == "refuted_in_scope"
    assert findings["affine_label_reuse"]["outcome"] == "refuted_in_scope"
    assert findings["missing_truncation_carry"]["outcome"] == "refuted_in_scope"
    assert document["ideal_uniform_control"]["outcome"] == "proved_in_model"


def test_public_deployment_benchmark_is_canonical() -> None:
    request = {
        "options": {
            "id": "loopback-run",
            "plan_lock_digest": None,
            "privacy_cohort": "masked-linear",
            "numeric_cohort": "wrap32",
            "environment": {"transport": "loopback", "cpu": "fixture"},
        },
        "observations": [
            {
                "role": "client",
                "phase": "online",
                "origin": "native_executed",
                "metric": "latency",
                "unit": "seconds",
                "value": 1.25,
                "notes": "cold run",
                "evidence_paths": ["evidence/run.json"],
            },
            {
                "role": "preparation",
                "phase": "offline",
                "origin": "imported_archive",
                "metric": "throughput",
                "unit": "tokens_per_second",
                "value": 0,
            },
        ],
    }

    report = pllm.deployment_benchmark(request)
    document = report.to_dict()

    assert report.schema_version == "pllm.deployment_benchmark_report.v1"
    assert report.canonical_bytes() == json.dumps(
        document, allow_nan=False, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    assert len(document["measurements"]) == len(request["observations"])
    assert document["measurements"][0]["role"] == "client"
    assert document["measurements"][1]["origin"] == "imported_archive"
    assert document["measurements"][1]["value"] == 0


@pytest.mark.parametrize(
    "change",
    [
        lambda request: request.update({"unexpected": True}),
        lambda request: request["observations"][0].update({"role": "bad role"}),
        lambda request: request["observations"][0].update({"phase": "transition"}),
        lambda request: request["observations"][0].update({"value": -1}),
        lambda request: request["observations"][0].update({"origin": "analytic_estimate"}),
    ],
)
def test_deployment_benchmark_rejects_invalid_input(change) -> None:
    request = {
        "options": {
            "id": "run",
            "plan_lock_digest": None,
            "privacy_cohort": "privacy",
            "numeric_cohort": "numeric",
            "environment": {"cpu": "fixture"},
        },
        "observations": [
            {
                "role": "client",
                "phase": "online",
                "origin": "native_executed",
                "metric": "latency",
                "unit": "seconds",
                "value": 0,
            }
        ],
    }
    change(request)

    with pytest.raises(ValueError):
        pllm.deployment_benchmark(request)


def test_deployment_benchmark_requires_mapping() -> None:
    with pytest.raises(TypeError):
        pllm.deployment_benchmark([])


def test_compile_rejects_non_json_types() -> None:
    with pytest.raises(TypeError):
        pllm.compile(object())
    with pytest.raises(pllm.CompilationError):
        pllm.compile({"value": float("nan")})
