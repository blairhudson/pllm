"""Prepared protocol runs through authenticated party leases, with isolated inventories."""
from dataclasses import replace

import httpx
import pytest

from pllm import OpenAI
from pllm.deployment import discover, open_execution
from pllm.profiles import MaskedLinearCpu
from test_network_execution import http_hosts, selected, setup

pytestmark = pytest.mark.rust


@pytest.fixture
def prepared(setup):
    network, credentials, specs, model_plan, lock = setup
    pipeline = MaskedLinearCpu(specs[0].experiment.pipeline.model)
    installed = replace(specs[0].experiment, pipeline=pipeline)
    specs = tuple(replace(spec, experiment=installed, role_ids=(role,), max_sessions=2)
                  for spec, role in zip(specs, ("inference", "preparation"), strict=True))
    return network, credentials, specs, model_plan, lock


@pytest.mark.integration
def test_prepared_http_sdk_one_use_and_fresh_attempt(prepared, tmp_path):
    with http_hosts(prepared, tmp_path) as (network, credentials, _):
        result = selected(prepared, discover(network, credentials=credentials))
        assert {row["role_id"] for row in result.native_placement["roles"]} == {"client", "inference", "preparation"}
        for _ in range(2):
            with open_execution(result, network=network, credentials=credentials, ttl_seconds=30) as lease:
                with OpenAI(execution=lease) as client:
                    response = client.responses.create(input="Hi", max_output_tokens=2, temperature=0.0)
                    assert response.usage.output_tokens == 2
                    audit = client.privacy_audit.to_dict()
                    assert audit["inference_stage_calls"] > 0
                    assert audit["plaintext_prompt_bytes_sent"] == 0
                    assert audit["plaintext_token_ids_sent"] == 0
                connections = lease._topology._connections
                inference_url, bearer = connections["inference"]
                with httpx.Client(headers={"authorization": f"Bearer {bearer}"}) as http:
                    assert http.post(inference_url + "/v1/runtime/models/load", json={}).status_code == 403
                    assert http.get(inference_url + "/v1/models").status_code == 200
            assert httpx.get(inference_url + "/v1/models", headers={"authorization": f"Bearer {bearer}"}).status_code == 401
            result = selected(prepared, discover(network, credentials=credentials))


@pytest.mark.integration
def test_prepared_inventory_and_client_credentials_are_attempt_local(prepared, tmp_path):
    from contextlib import ExitStack

    with http_hosts(prepared, tmp_path) as (network, credentials, _), ExitStack() as stack:
        first = stack.enter_context(open_execution(
            selected(prepared, discover(network, credentials=credentials)),
            network=network, credentials=credentials, ttl_seconds=30))
        first_client = stack.enter_context(OpenAI(execution=first))
        response = first_client.responses.create(input="Hi", max_output_tokens=2, temperature=0.0)
        assert response.usage.output_tokens == 2
        second = stack.enter_context(open_execution(
            selected(prepared, discover(network, credentials=credentials)),
            network=network, credentials=credentials, ttl_seconds=30))
        url, first_key = first._topology._connections["inference"]
        _, second_key = second._topology._connections["inference"]
        assert first_key != second_key
        state = first._client._core._transformer_state(first._result.experiment.resolve().model)
        inventory_id = state.prepared_inventory.id
        endpoint = url + f"/v1/runtime/inventories/{inventory_id}"
        assert httpx.get(endpoint, headers={"authorization": f"Bearer {first_key}"}).status_code == 200
        assert httpx.get(endpoint, headers={"authorization": f"Bearer {second_key}"}).status_code == 404
        assert httpx.get(url + "/v1/models", headers={"authorization": f"Bearer {first_key}"}).status_code == 200
        second_client = stack.enter_context(OpenAI(execution=second))
        other = second_client.responses.create(input="Hi", max_output_tokens=2, temperature=0.0)
        assert other.output_text == response.output_text
        first.close()
        assert httpx.get(url + "/v1/models", headers={"authorization": f"Bearer {first_key}"}).status_code == 401
        assert httpx.get(url + "/v1/models", headers={"authorization": f"Bearer {second_key}"}).status_code == 200


@pytest.mark.integration
def test_prepared_network_ordinary_benchmark(prepared, tmp_path):
    from pllm.runtime.network_benchmark import run_network_benchmark

    with http_hosts(prepared, tmp_path) as (network, credentials, _):
        result = selected(prepared, discover(network, credentials=credentials))
        report = run_network_benchmark(network=network, result=result, credentials=credentials,
            prompt="Hi", max_output_tokens=2, warmups=0, repetitions=1, timeout_seconds=30,
            temperature=0.0, compare_feasible=True, capture_output_digest=False)
        assert report["checks"]["passed"]
