"""Container cleanup, numeric placement admission and Linux resource scope."""
import os

import pytest

from pllm import Deployment, ExecutionBudget, Experiment, Model
from pllm.profiles import MaskedLinearCpu, TwoOnlineOffsetCpu, VerifiedMaskedLinearCpu
from pllm.runtime import docker_roles
from pllm.runtime.servers import TopologyError, build_roles


def test_stop_timeout_still_forces_owned_container_cleanup(monkeypatch):
    calls = []
    def command(arguments, **kwargs):
        calls.append(arguments)
        if arguments[0] == "stop":
            raise TopologyError("stop timed out")
        return ""
    monkeypatch.setattr(docker_roles, "_docker", command)
    container = docker_roles._Container("owned-test")
    with pytest.raises(TopologyError):
        container.close()
    assert calls == [["stop", "--time", "5", "owned-test"], ["rm", "--force", "owned-test"]]
    assert container.closed
    container.close()
    assert len(calls) == 2


@pytest.mark.parametrize("created_on_daemon", [False, True])
def test_ambiguous_create_discovers_and_removes_only_owned_name(monkeypatch, created_on_daemon):
    calls = []
    def command(arguments, **kwargs):
        calls.append(arguments)
        return "owned-test" if arguments[0] == "ps" and created_on_daemon else ""
    monkeypatch.setattr(docker_roles, "_docker", command)
    container = docker_roles._Container("owned-test", created=False)
    container.close()
    assert calls[0] == ["ps", "--all", "--filter", "name=^owned-test$", "--format", "{{.Names}}"]
    assert calls[1:] == ([["stop", "--time", "5", "owned-test"], ["rm", "--force", "owned-test"]]
                         if created_on_daemon else [])
    assert container.closed


@pytest.mark.rust
def test_docker_backend_is_selected_without_changing_composition():
    topology = build_roles(Model.tiny(), docker=True)
    try:
        assert isinstance(topology, docker_roles.DockerTopology)
        assert topology._docker_blobs == ()
        assert topology._docker_hub is None
    finally:
        topology.close()


@pytest.mark.rust
@pytest.mark.integration
@pytest.mark.skipif(os.environ.get("PLLM_RUN_DOCKER") != "1", reason="opt-in Docker runtime")
@pytest.mark.parametrize("profile", [MaskedLinearCpu, TwoOnlineOffsetCpu, VerifiedMaskedLinearCpu])
@pytest.mark.parametrize("wan", [False, True])
def test_two_linux_roles_masked_request_samples_and_owned_cleanup(profile, wan, tmp_path):
    from pllm.deployment import WanConditions
    experiment = Experiment("docker-test", profile(Model.tiny(model_id="docker-tiny")),
                            Deployment.local(root=str(tmp_path)), ExecutionBudget(1, 32, 2))
    with build_roles(experiment, docker=True, wan=WanConditions() if wan else None,
                     docker_image=os.environ.get("PLLM_DOCKER_TEST_IMAGE")) as topology:
        assert topology.start() is topology
        with topology.client() as client:
            response = client.responses.create(input="Hi", max_output_tokens=2, temperature=0.0)
            assert response.usage.output_tokens == 2
            assert client.privacy_audit.to_dict()["plaintext_prompt_bytes_sent"] == 0
        samples = topology.resource_samples()
        assert samples["full_wire_bytes"] is None
        assert set(samples["roles"]) == ({"worker_a", "worker_b"} if profile is TwoOnlineOffsetCpu
                                        else {"inference", "preparation"})
        assert samples["directed_ip"]["bytes"] > 0
        assert samples["directed_ip"]["bytes"] == sum(row["bytes"] for row in samples["directed_ip"]["links"].values())
        for role in samples["roles"]:
            assert samples["directed_ip"]["links"][f"client->{role}"]["bytes"] > 0
            assert samples["directed_ip"]["links"][f"{role}->client"]["bytes"] > 0
            helper = f"{role}/router" if wan else role
            assert samples["measurement_helpers"][helper]["cpu_ns"] > 0
        if profile is not TwoOnlineOffsetCpu:
            assert samples["directed_ip"]["links"]["preparation->inference"]["bytes"] > 0
        for sample in samples["roles"].values():
            assert sample["cpu_ns"] > 0 and sample["memory_peak_bytes"] > 0
            assert any(interface["rx_bytes"] > 0 for interface in sample["interfaces"].values())
        names = tuple(topology._docker_names.values()) + tuple(item.name for item in topology._measurement_helpers.values())
        if wan:
            assert samples["wan_emulation"]["enforced"] is True
            assert samples["wan_emulation"]["conditions_digest"] == WanConditions().digest
            assert set(samples["wan_emulation"]["parties"]) == {"client", *samples["roles"]}
            names += tuple(topology._party_network.containers)
    for name in names:
        with pytest.raises(TopologyError):
            docker_roles._docker(["inspect", name, "--format", "{{.State.Status}}"])
