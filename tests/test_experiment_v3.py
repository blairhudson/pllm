"""Version boundary: explicit network records, unchanged v2 canonical identity."""

import copy
import hashlib
import json
from dataclasses import replace

import pytest

from pllm import Deployment, ExecutionBudget, Experiment, Model, _native
from pllm.configuration import ConfigurationError, canonical_bytes, loads_configuration
from pllm.profiles import TwoOnlineOffsetCpu

pytestmark = pytest.mark.rust


def local():
    return Experiment("v2", TwoOnlineOffsetCpu(Model.tiny()), Deployment.local(root="local://v2"),
                      ExecutionBudget(1, 8, 2))


def test_v2_bytes_digest_and_native_composition_preserved():
    experiment = local()
    expected = {"schema": "pllm.experiment.v2", "name": "v2",
                "pipeline": experiment.pipeline.to_spec(), "deployment": {"kind": "local", "root": "local://v2"},
                "budget": {"requests": 1, "max_input_tokens": 8, "max_new_tokens": 2}}
    blob = json.dumps(expected, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    assert canonical_bytes(experiment) == blob
    assert experiment.configuration_digest() == hashlib.sha256(b"pllm.configuration.v1\0" + blob).hexdigest()
    assert loads_configuration(blob.decode()).canonical_bytes() == blob
    native = _native.resolve_experiment(blob)
    assert native.configuration_digest == experiment.configuration_digest()


def test_v3_resolves_native_without_rewriting_intent_or_composition():
    v2 = local()
    v3 = replace(v2, deployment=Deployment.network(network_id="test", network_spec_digest="a" * 64))
    assert v3.to_spec()["schema"] == "pllm.experiment.v3"
    assert "root" not in v3.to_spec()["deployment"]
    assert v3.deployment.root is None
    assert Experiment.from_spec(v3.to_spec()) == v3
    native = _native.resolve_experiment(v3.canonical_bytes())
    assert native.configuration_digest == v3.configuration_digest()
    assert native.composition_digest == v2.resolve().composition_digest == v3.pipeline.digest()
    assert v3.configuration_digest() != v2.configuration_digest()


@pytest.mark.parametrize("mutation", [
    lambda d: d.update(schema="pllm.experiment.v2"),
    lambda d: d["deployment"].update(root="local://bad"),
    lambda d: d["deployment"].update(network_spec_digest="0" * 64),
    lambda d: d["deployment"].update(snapshot_digest="A" * 64),
    lambda d: d["deployment"].pop("snapshot_digest"),
    lambda d: d["deployment"].update(kind="local"),
    lambda d: d["deployment"].update(network_id=""),
    lambda d: d.update(extra=True),
])
def test_network_envelope_strict_python_and_native(mutation):
    experiment = replace(local(), deployment=Deployment.network(network_id="test", network_spec_digest="a" * 64))
    value = copy.deepcopy(experiment.to_spec())
    mutation(value)
    with pytest.raises((ConfigurationError, ValueError)):
        Experiment.from_spec(value)
    with pytest.raises(ValueError):
        _native.resolve_experiment(json.dumps(value).encode())


def test_v3_cannot_disguise_local_and_duplicate_native_fields_rejected():
    doc = local().to_spec()
    doc["schema"] = "pllm.experiment.v3"
    with pytest.raises(ValueError):
        _native.resolve_experiment(json.dumps(doc).encode())
    with pytest.raises(ConfigurationError):
        Experiment.from_spec(doc)
    v3 = replace(local(), deployment=Deployment.network(network_id="test", network_spec_digest="a" * 64))
    duplicate = v3.canonical_bytes().replace(b'"network_id":"test"', b'"network_id":"test","network_id":"other"')
    with pytest.raises(ValueError):
        _native.resolve_experiment(duplicate)
    from pllm.runtime.servers import TopologyError, build_roles

    with pytest.raises(TopologyError, match="live admission"):
        build_roles(v3)
