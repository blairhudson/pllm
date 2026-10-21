"""Streamed SDK admission follows its transport owners, not the eager reference."""

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from pllm import Model
from pllm.profiles import OpenWeightVerifiableCpu
from pllm.protocols import OpenWeightVerifiableOutsourcing
from pllm.runtime import benchmark_memory as memory, delegated_transport as transport
from pllm.runtime.delegated_decoder import decoder_contract

@pytest.fixture
def full_geometry():
    # Public configuration only. No checkpoint resolution or tensor-value reads.
    cfg = json.loads((Path(__file__).parent / "fixtures/qwen25_0_5b_config.json").read_text())
    contract = json.loads(decoder_contract(cfg, input_rows=2, decode_steps=1,
                                           capacity="decoder", streamed=True))
    source = SimpleNamespace(capacity="decoder", contract=contract,
        execution_payload_estimate_bytes=contract["execution_payload_estimate_bytes"])
    return cfg, source


def host(available):
    return memory.HostMemory(32 * memory.GiB, available, 0, 20 * memory.GiB,
        availability_source="macos_mach_dispatch_v2", pressure="normal",
        free=available, cached=0, purgeable=0, reusable=0)


@pytest.mark.parametrize("sharing", ["full", "seeded"])
def test_streamed_transport_and_host_admission_charge_same_owner(full_geometry, sharing):
    cfg, source = full_geometry
    pipeline = OpenWeightVerifiableCpu(Model.path("unused-public-source"),
        linear=OpenWeightVerifiableOutsourcing(capacity="decoder", sharing=sharing))
    estimate = memory.estimate_memory(cfg, pipeline, max_input_tokens=2, max_output_tokens=2)
    client = transport.request_memory_bytes(source)
    worker = transport.request_memory_bytes(source, worker=True)
    generic = json.loads(decoder_contract(cfg, input_rows=2, decode_steps=1,
                                          capacity="decoder"))
    # Generic eager/eight-window accounting remains available to reference APIs.
    resources = source.contract["transport_resources"]
    saved = (resources["generic_native_admitted_payload_bytes"]
             - resources["native_admitted_payload_bytes"])
    assert saved > 0
    assert client == generic["decoder_resources"]["client_payload_bytes"] - saved + (64 << 20)
    assert worker == generic["decoder_resources"]["worker_payload_bytes"] - saved + (64 << 20)
    assert resources["profile"] == "pllm.gkm.streamed-single-window.v1"
    assert resources["reserved_windows"] == 1
    assert resources["import_workspace_bytes"] == 131072
    charged = estimate["verification_allocation"]["native_allowance_by_role_bytes"]
    assert estimate["verification_allocation"]["native_transport_resources"] == resources
    assert charged == {"client": client - (64 << 20),
                       "worker_a": worker - (64 << 20),
                       "worker_b": worker - (64 << 20)}
    assert estimate["allocation_slack_percent"] == 25
    assert estimate["client_peak_bytes"] == ((256 * memory.MiB + client) * 125 + 99) // 100
    assert estimate["provider_peak_bytes"] == {
        name: ((256 * memory.MiB + worker) * 125 + 99) // 100 for name in ("worker_a", "worker_b")}

    required = estimate["native_total_peak_bytes"]
    accepted = memory.admit_memory(estimate, host(required))
    rejected = memory.admit_memory(estimate, host(required - 1))
    assert accepted["admitted"] and not rejected["admitted"]
    assert rejected["selected_backend"] is None
    expected = 3 * 256 * memory.MiB + client + 2 * worker
    native = accepted["candidates"]["native"]
    assert native["expected_peak_bytes"] == expected
    assert native["safety_margin_bytes"] == required - expected
    assert accepted["runtime_floor_bytes"] == native["safety_margin_bytes"] > 0
    assert rejected["runtime_floor_bytes"] == accepted["runtime_floor_bytes"]


def test_streamed_admission_preserves_wan_relay_owner_and_margin(full_geometry):
    cfg, _ = full_geometry
    pipeline = OpenWeightVerifiableCpu(Model.path("unused-public-source"),
        linear=OpenWeightVerifiableOutsourcing(capacity="decoder"))
    estimate = memory.estimate_memory(cfg, pipeline, max_input_tokens=2, max_output_tokens=2)
    plain = memory.admit_memory(estimate, host(16 * memory.GiB))
    wan = memory.admit_memory(estimate, host(16 * memory.GiB), enforced_wan=True)
    assert plain["admitted"] and wan["admitted"]
    p, w = (item["candidates"]["native"] for item in (plain, wan))
    assert w["required_host_bytes"] - p["required_host_bytes"] == 40 * memory.MiB
    assert w["expected_peak_bytes"] - p["expected_peak_bytes"] == 32 * memory.MiB
    assert wan["runtime_floor_bytes"] - plain["runtime_floor_bytes"] == 8 * memory.MiB


def test_streamed_admission_preserves_explicit_budget_and_pressure_gate(full_geometry):
    cfg, _ = full_geometry
    pipeline = OpenWeightVerifiableCpu(Model.path("unused-public-source"),
        linear=OpenWeightVerifiableOutsourcing(capacity="decoder"))
    estimate = memory.estimate_memory(cfg, pipeline, max_input_tokens=2, max_output_tokens=2)
    required = estimate["native_total_peak_bytes"]
    roomy = host(16 * memory.GiB)
    assert memory.admit_memory(estimate, roomy)["admitted"]
    capped = memory.admit_memory(estimate, roomy, memory_budget_bytes=required - 1)
    assert not capped["admitted"]
    assert capped["host"]["admission_budget_bytes"] == required - 1
    for pressure in ("warning", "critical"):
        blocked = memory.admit_memory(estimate, replace(roomy, pressure=pressure))
        assert not blocked["admitted"]
        assert f"macOS memory pressure is {pressure}" in blocked["candidates"]["native"]["reasons"]


@pytest.mark.parametrize("capacity", ["capsule", "decoder"])
@pytest.mark.parametrize("sharing", ["full", "seeded"])
@pytest.mark.parametrize("margin", [0, 25, 100])
def test_native_batch_geometry_quote_adds_owners_without_source_loading(capacity, sharing, margin):
    from tests.test_delegated_decoder import config

    cfg = config()
    options = {"capacity": capacity, "sharing": sharing}
    scalar = OpenWeightVerifiableCpu(Model.path("unused-public-source"),
        linear=OpenWeightVerifiableOutsourcing(**options))
    batch = OpenWeightVerifiableCpu(Model.path("unused-public-source"),
        linear=OpenWeightVerifiableOutsourcing(**options, execution="source_stage_batch_v1"))
    bounds = {"max_input_tokens": 2, "max_output_tokens": 2,
              "memory_safety_margin_percent": margin}
    base = memory.estimate_memory(cfg, scalar, **bounds)
    estimate = memory.estimate_memory(cfg, batch, **bounds)
    prices = estimate["verification_allocation"]
    assert prices["batch_python_client_peak_bytes"] > 0
    assert all(size > 0 for size in prices["batch_python_worker_peak_bytes"].values())
    assert prices["batch_runtime_backing_delta_bytes"] > 0
    assert estimate["client_peak_bytes"] > base["client_peak_bytes"]
    assert all(estimate["provider_peak_bytes"][role] > base["provider_peak_bytes"][role]
               for role in ("worker_a", "worker_b"))
    assert estimate["allocation_slack_percent"] == margin
    required = estimate["native_total_peak_bytes"]
    assert memory.admit_memory(estimate, host(required))["admitted"]
    assert not memory.admit_memory(estimate, host(required - 1))["admitted"]
