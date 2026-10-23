"""Mocked campaign-owner controls only; no models, children or memory evidence.

Both maintained and paged-prewarm factories retain their existing 64/8, 71-row,
W8A8/prefix_f32 contracts. Public preparation/service providers remain outside
client membership. Client-side startup, prewarm, paging and refill threads share
the root's RSS and whole isolated lifetime; WAN relay threads share that PID too.
"""
import ast
from pathlib import Path
import threading
from types import SimpleNamespace as NS

import pytest

from pllm import Model
from pllm.runtime import client_memory_scope as scopes, client_process_usage as usage
from pllm.runtime.native_wan import NativePartyNetwork, NativeWanTopology
from pllm.runtime.servers import LocalTopology


ROLES = ("client", "preparation", "inference")
PIN = "1cfa9a7208912126459214e8b04321603b3df60c"


@pytest.fixture
def owner_environment(monkeypatch):
    monkeypatch.setattr(scopes.os, "getpid", lambda: 101)
    monkeypatch.setattr(usage, "supported", lambda: True)  # Mocked Darwin record contract only.

    def forbidden(*args, **kwargs):
        pytest.fail("mocked owner controls must not launch a child or observe live birth")

    monkeypatch.setattr(usage, "_birth_identity", forbidden)
    monkeypatch.setattr(scopes.subprocess, "Popen", forbidden)


def _topology(source, *, wan=True):
    topology = (NativeWanTopology if wan else LocalTopology).__new__(
        NativeWanTopology if wan else LocalTopology)
    topology._process_lock = threading.Lock()
    topology._started, topology._closed = True, False
    topology._model, topology._privacy_mode = source, "public"
    topology._role_ids = ROLES[1:]
    # Owning topology declarations, not a discovered process tree or OS claims.
    topology._processes = {"preparation": NS(pid=201), "inference": NS(pid=301)}
    if wan:
        topology._native_network = NativePartyNetwork.__new__(NativePartyNetwork)
        topology._native_network.closed = False
        topology._native_network.thread = None
    return topology


def _runtime(topology):
    scope = scopes.ClientMemoryScope()
    scope.claim_sampling()
    scope.observe({"client": (101, 1., 100)}, span_seconds=0)
    return NS(_topology=topology, _benchmark_client_memory_owner=scope)


def _retire(topology):
    # Fixed closed-owner snapshots only. No close(), wait(), poll(), network or native work.
    topology._closed = True
    if type(topology) is NativeWanTopology:
        topology._native_network.closed = True


def _manifest(runtime, *, local=True):
    return scopes.client_lifetime_manifest(
        NS(client_memory_scope=runtime._benchmark_client_memory_owner), runtime._topology,
        ROLES, local_cpu_loopback=local)


def _finalize(runtime):
    scope = runtime._benchmark_client_memory_owner
    memory = scope.report()
    memory["lifetime_owner"] = _manifest(runtime)
    # Explicit fixture observation, never sampled RSS converted into OS HWM.
    memory["individual_client_hwm"] = {"bytes": 120,
        "source": "resource.getrusage(RUSAGE_SELF)",
        "window": "client_process_birth_through_report_endpoint",
        "coverage": "individual_process_only_not_client_group", "isolated_lifetime": None}
    report = {"checks": {"passed": True}, "client_owned_memory": memory}
    owner = usage.ClientUsagePopen.__new__(usage.ClientUsagePopen)
    owner._child_created = False
    owner.pid, owner.returncode = 101, 0
    owner._usage_identity = (101, 1.)
    owner._waitpid_lock = threading.Lock()
    owner._usage_scale, owner._usage_unavailable = 1, False
    owner._usage_final = (.01, .02, 9000)  # Provider-inflated family upper, not client HWM.
    usage.finalize_client_report(report, owner)
    return report


@pytest.mark.parametrize("arm", ["maintained", "paged-prepared"])
def test_existing_campaign_arms_bind_root_before_prewarm_and_qualify_only_after_retirement(owner_environment, arm):
    from examples.papers.low_memory_qwen3 import recipes

    experiment = recipes()[arm]  # Inert public factory, no checkpoint resolution.
    original = experiment.to_spec()
    assert experiment.budget.max_input_tokens == 64 and experiment.budget.max_new_tokens == 8
    assert experiment.pipeline.components["inventory"].params["rows"] == 71
    topology = _topology(experiment.pipeline.model)
    runtime = _runtime(topology)
    assert scopes.declare_builtin_client_lifetime(runtime)
    scope = runtime._benchmark_client_memory_owner
    assert scope._lifetime_declared and not scope._lifetime_closed_family
    assert not _manifest(runtime)["membership_complete"]
    # Root readings contain client-side preparation/paging work. Providers are
    # deliberately not handed to observe() as client members.
    scope.observe({"client": (101, 1., 110), "preparation": (201, 2., 9000)}, span_seconds=0)
    assert set(scope.report()["latest"]["members"]) == {"client"}
    _retire(topology)
    report = _finalize(runtime)
    memory = report["client_owned_memory"]
    assert memory["lifetime_owner"]["membership_complete"]
    assert set(memory["lifetime_owner"]["members"]) == {"client"}
    assert memory["lifetime_owner"]["root_birth_bound"]
    assert "_root_identity" not in memory["lifetime_owner"]
    assert "parent_source" in memory["lifetime_owner"]["outside_isolated_role"]
    assert usage.owned_family_interval(memory) == {"lower": 120, "upper": 9000}
    assert memory["physical_used_ram_peak_bytes"] is None
    assert "not_simultaneous_peak_bound" in memory["sampled_peak_rss_sum_kind"]
    # Declaration after sampling started does not retroactively qualify old samples.
    assert memory["sampling_coverage"] == "unknown_or_partial"
    assert experiment.to_spec() == original  # No factory, numeric, count or budget mutation.
    from pllm.runtime.process_usage import provider_lifetime_usage
    providers = provider_lifetime_usage(topology, ROLES, local_cpu_loopback=False, source_present=True)
    assert not providers["complete"] and providers["aggregate_peak_rss_bytes"] is None
    assert all(row["coverage"] == "unknown" for row in providers["roles"].values())


@pytest.mark.parametrize("source", [
    Model.hf("org/model", revision=PIN),  # Online source/helper path remains unknown.
    Model.hf("org/model", local_files_only=True),
    Model.hf("org/model", revision="main", local_files_only=True),
])
def test_unqualified_source_cannot_gain_owner_coverage_from_local_pid_or_wan_class(owner_environment, source):
    runtime = _runtime(_topology(source))
    assert not scopes.declare_builtin_client_lifetime(runtime)
    _retire(runtime._topology)
    assert not _manifest(runtime)["membership_complete"]
    assert usage.owned_family_interval(_finalize(runtime)["client_owned_memory"]) is None


@pytest.mark.parametrize("state", ["helper_pending", "sampling_sealed", "explicit_lifetime"])
def test_actual_creator_never_upgrades_or_overwrites_other_launcher_contract(owner_environment, state):
    runtime = _runtime(_topology(Model.hf("org/model", revision=PIN, local_files_only=True)))
    scope = runtime._benchmark_client_memory_owner
    if state == "helper_pending":
        scope.expect_unavailable_helper("client_helper_1")
    elif state == "sampling_sealed":
        scope.seal()
    else:
        scope.expect_unavailable_helper("client_helper_1", remote=True)
        scope.seal_lifetime_membership()
    assert not scopes.declare_builtin_client_lifetime(runtime)
    _retire(runtime._topology)
    assert not _manifest(runtime)["membership_complete"]
    assert usage.owned_family_interval(_finalize(runtime)["client_owned_memory"]) is None


@pytest.mark.parametrize("gap", ["open_relay", "live_relay", "missing_relay_retirement", "foreign_relay", "foreign_topology", "provider_graph", "remote"])
def test_known_local_graph_requires_retired_exact_owner_not_ancestry(owner_environment, gap):
    topology = _topology(Model.hf("org/model", revision=PIN, local_files_only=True))
    runtime = _runtime(topology)
    assert scopes.declare_builtin_client_lifetime(runtime)
    _retire(topology)
    if gap == "open_relay":
        topology._native_network.closed = False
    elif gap == "live_relay":
        thread = threading.Thread()  # No start, no OS thread.
        thread.is_alive = lambda: True
        topology._native_network.thread = thread
    elif gap == "foreign_relay":
        topology._native_network = NS(closed=True, thread=None)
    elif gap == "missing_relay_retirement":
        del topology._native_network.thread
    elif gap == "foreign_topology":
        class ForeignTopology(NativeWanTopology):
            pass
        runtime._topology = ForeignTopology.__new__(ForeignTopology)
    elif gap == "provider_graph":
        topology._processes.pop("preparation")
    assert not _manifest(runtime, local=gap != "remote")["membership_complete"]


def test_creator_and_report_wiring_leave_counts_legacy_windows_and_provider_policy_alone():
    # Source-only wiring checks complement functional mocked owner controls.
    directory = Path(scopes.__file__).parent
    tree = ast.parse((directory / "dashboard.py").read_text())
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    bind, = [node for node in calls if isinstance(node.func, ast.Name)
             and node.func.id == "declare_builtin_client_lifetime"]
    create, = [node for node in calls if ast.unparse(node.func) == "self._topology.client"]
    prewarm, = [node for node in calls if ast.unparse(node.func) == "self._background_call"
               and node.args and ast.unparse(node.args[0]) == "self._client.preprocess"]
    assert create.lineno < bind.lineno < prewarm.lineno
    tree = ast.parse((directory / "benchmark_cli.py").read_text())
    options = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in (
                "client_lifetime_manifest", "provider_lifetime_usage"):
            options[node.func.id] = next(ast.dump(kw.value) for kw in node.keywords
                                        if kw.arg == "local_cpu_loopback")
    for name, expression in (
        ("client_lifetime_manifest", "not docker and cpu_kernel"),
        ("provider_lifetime_usage", "not docker and not emulate_wan and cpu_kernel"),
    ):
        assert options[name] == ast.dump(ast.parse(expression, mode="eval").body)
