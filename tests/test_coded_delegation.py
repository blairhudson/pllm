"""SDK arithmetic, ownership accounting and admission for the RAA reference."""
import json
import struct

import pytest

from pllm import _native, Deployment, ExecutionBudget, Experiment, Model, Pipeline
from pllm.components import create_component
from pllm.configuration import ConfigurationError
from pllm.metrics import benchmark_operator, replay_operator_benchmark
from pllm.profiles import MaskedLinearCpu
from pllm.protocols import MaverickDelegatedLinear, ProtocolMethod


def test_component_geometry_replay_and_asymmetric_material_owners():
    c = MaverickDelegatedLinear()
    assert isinstance(c, ProtocolMethod)
    assert create_component(c.component, c.params) == c
    report = benchmark_operator(c, lanes=2, samples=2, geometry={"rows": 65, "columns": 64})
    assert report.configuration["schema"] == "pllm.operator_benchmark.v2"
    assert report.configuration["geometry"] == {"rows": 65, "columns": 64}
    assert replay_operator_benchmark(report.configuration).resources == report.resources
    assert report.quality.checked_outputs == report.quality.exact_integer_outputs == 260
    p = json.loads(_native.coded_delegation_projection(65, 64, 2))
    r = report.resources
    assert r.public_table_bytes_by_party == (p["persistent_client_payload_bytes"], 65 * 64)
    assert r.public_table_bytes_by_party[0] > 64 * 65 * 64
    assert r.raw_key_payload_bytes_per_party == (2 * 4 * (65 + 64), 0)
    assert r.peer_frame_bytes == 0  # Unimplemented codec cannot become measured bytes.
    assert r.client_entry_exit_payload_estimate_bytes == 2 * 4 * (65 + 64)
    for s in report.samples:
        assert s.online_seconds == s.evaluation_seconds + s.verification_seconds
        assert s.public_setup_seconds > 0 and s.issuance_seconds > 0
    assert report.to_dict()["whole_model_executable"] is False
    assert report.to_dict()["privacy_reviewed"] is False


@pytest.mark.parametrize("weight,input_value", [(-128, -128), (-128, 127), (127, 127), (0, -128)])
def test_centered_field_boundary_matches_independent_signed_oracle(weight, input_value):
    output, _ = _native.coded_delegation_benchmark(
        64, 64, 1, bytes(32), bytes([weight & 255]) * 4096, bytes([input_value & 255]) * 64,
    )
    assert struct.unpack("<64i", output) == (weight * input_value * 64,) * 64
    report = benchmark_operator(MaverickDelegatedLinear(), lanes=1, samples=1,
                                inputs={"weights": [weight] * 4096, "vectors": [input_value] * 64})
    assert report.input_source == "caller-owned"
    assert report.quality.exact_integer_outputs == 64
    assert '"vectors"' not in report.to_json()


def test_allocation_and_cpu_rejection_precede_payload_import(monkeypatch):
    for rows, cols, queries in ((896, 4864, 1), (9728, 896, 1), (1024, 1024, 256)):
        assert not json.loads(_native.coded_delegation_projection(rows, cols, queries))["admitted"]
        with pytest.raises(ValueError, match="256 MiB allocation or 64M"):
            _native.coded_delegation_benchmark(rows, cols, queries, b"", b"", b"")
    def forbidden(*args):
        pytest.fail("inadmissible geometry reached native issuance")
    monkeypatch.setattr(_native, "coded_delegation_benchmark", forbidden)
    with pytest.raises(ConfigurationError, match="256 MiB"):
        benchmark_operator(MaverickDelegatedLinear(), geometry={"rows": 9728, "columns": 896})


@pytest.mark.parametrize("seed", [None, 7, "", "0" * 63, "A" * 64, "x" * 64])
def test_public_seed_is_canonical_and_never_an_input_mask(seed):
    with pytest.raises(ConfigurationError):
        MaverickDelegatedLinear(public_code_seed=seed)


def test_changed_public_code_binds_configuration_but_preserves_numeric_contract():
    a = benchmark_operator(MaverickDelegatedLinear(), lanes=1, samples=1)
    b = benchmark_operator(MaverickDelegatedLinear(public_code_seed="ab" * 32), lanes=1, samples=1)
    assert a.configuration_digest != b.configuration_digest
    assert a.numeric_contract_digest == b.numeric_contract_digest
    assert a.resources == b.resources
    for geometry in ({"rows": 64}, {"rows": True, "columns": 64}, {"rows": 63, "columns": 64}):
        with pytest.raises(ConfigurationError):
            benchmark_operator(MaverickDelegatedLinear(), geometry=geometry)
    with pytest.raises(ConfigurationError):
        benchmark_operator(MaverickDelegatedLinear(), lanes=1,
                           inputs={"weights": [128] * 4096, "vectors": [0] * 64})


def test_benchmark_component_does_not_activate_decoder_execution():
    baseline = MaskedLinearCpu(model=Model.tiny())
    with pytest.raises(ConfigurationError):
        Experiment(name="unreviewed-maverick", pipeline=Pipeline(model=baseline.model,
            components={**baseline.components, "linear": MaverickDelegatedLinear()}),
            deployment=Deployment.local(root="local://maverick"),
            budget=ExecutionBudget(requests=1, max_input_tokens=2, max_new_tokens=1)).resolve()
