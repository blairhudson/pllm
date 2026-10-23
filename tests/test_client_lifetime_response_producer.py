"""Ordinary response producer, not a replacement runner or a memory campaign.

Default test doubles execution/OS usage; no model, native SDK or child runs.
When other heavy workloads are idle, opt in to the generated-checkpoint SDK test:
PLLM_RUN_CLIENT_LIFETIME_SDK=1 <wheel-python> -I -m pytest --import-mode=importlib
    -o pythonpath= --noconftest -q /absolute/path/to/test_client_lifetime_response_producer.py
The installed wheel supplies both controller and isolated worker; no source overlay.
Both cases use the same public suite. No downloaded/trained model or new metric.
"""
from copy import deepcopy
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import sys
import threading
from types import SimpleNamespace as NS

import pytest

import pllm
from pllm._benchmark_contract import digest
from pllm._benchmark_worker import Measurement, PROTOCOL
from pllm.evidence import EvidenceRegistry
from pllm.metrics import BenchmarkCase, BenchmarkSuite, BenchmarkTarget, BuildRef, Sampling, Scenario
from pllm.metrics import benchmark_runner as runner
from pllm.profiles import MaskedLinearCpu
from pllm.runtime import client_process_usage as usage
from pllm.runtime.client_memory_scope import ClientMemoryScope, client_lifetime_manifest


METRIC = "client.owned_lifetime_peak_rss_sum_interval"
pytestmark = pytest.mark.skipif(not usage.supported(), reason="qualified Darwin CPython owner required")


def _suite(checkpoint, build=None):
    experiment = pllm.Experiment("client-lifetime-response",
        MaskedLinearCpu(pllm.Model.path(str(checkpoint), model_id="client-lifetime-generated")),
        pllm.Deployment.local(root="local://client-lifetime-response"), pllm.ExecutionBudget(1, 128, 2))
    target = BenchmarkTarget.response(experiment, backend="native", scenario=Scenario.cold(max_output_tokens=2))
    return BenchmarkSuite((BenchmarkCase("generated", target, build or BuildRef(sys.executable)),),
        Sampling(processes=1, warmups=0, repetitions=1, threads=1, timeout_seconds=120, memory_bytes=2 << 30))


def _assert_scope(observed, *, exit_observed=True):
    from pllm._benchmark_evidence import _client_memory_observation
    row, = observed.to_dict()["results"]
    assert row["status"] == "completed", row["execution"]["failure_code"]
    report = row["execution"]["details"]["report"]
    scope = report["client_owned_memory"]
    manifest = scope["lifetime_owner"]
    assert manifest["membership_complete"] is True, manifest["reason"]
    assert manifest["root_birth_bound"] is True
    assert manifest["closed_client_owned_family"] is False
    assert set(manifest["members"]) == {"client"}  # Providers never join by ancestry.
    assert "_root_identity" not in manifest
    assert "unknown" in manifest["outside_isolated_role"]
    assert "controller" in manifest["outside_isolated_role"]
    assert "parent_source" in manifest["outside_isolated_role"]
    assert scope["physical_used_ram_peak_bytes"] is None
    assert scope["sampled_peak_rss_sum_kind"] == "non_atomic_same_pass_estimate_not_simultaneous_peak_bound"
    assert row["execution"]["isolation"]["controller_included"] is False
    legacy, = (m for m in row["metrics"] if m["id"] == "client.peak_rss")
    assert legacy["window"] == "process_birth_through_scenario"
    assert legacy["measurement_scope"] == "process_lifetime_high_water"
    assert legacy["value"] == scope["individual_client_hwm"]["bytes"]
    bounds = usage.owned_family_interval(scope)
    final = report["client_process_exit_usage"]
    interval = scope["lifetime_peak_rss_sum_interval"]
    assert EvidenceRegistry(observed.results).query(metric=METRIC)  # Discovery includes unknown reports.
    available = _client_memory_observation(row, METRIC)
    if not exit_observed:
        assert final["coverage"] == "unknown" and final["source"] is None
        assert all(final[field] is None for field in ("cpu_seconds", "peak_rss_bytes", "normal_exit"))
        assert bounds is None and not available
        assert interval["lower_bound_bytes"] is interval["upper_bound_bytes"] is None
        assert interval["reason"] == "client_exit_not_normal_or_unobserved"
        return report  # Accepted workload stays completed; optional metric missing.
    assert bounds is not None, interval
    assert final["coverage"] == "os_process_exit" and final["normal_exit"] is True
    assert final["source"] == "wait4" and final["raw_hwm_unit"] == "bytes"
    assert final["cpu_scope"] == "including_reaped_descendants"
    assert final["hwm_scope"] == "maximum_individual_in_reaped_family_not_self_or_group_sum"
    assert bounds == {"lower": scope["individual_client_hwm"]["bytes"], "upper": final["peak_rss_bytes"]}
    assert 0 < bounds["lower"] <= bounds["upper"]
    assert interval["source"] == "owner_bound_individual_prefix_lower_and_wait4_family_upper"
    assert interval["member_family_max_hwm_bytes"] == {"client": bounds["upper"]}
    assert interval["finalized_after_full_member_retirement"] is True
    assert available
    return report


def _assert_provider_boundaries(observed):
    row, = observed.to_dict()["results"]
    report = row["execution"]["details"]["report"]
    section = report["provider_lifetime_usage"]
    assert set(section["roles"]) == {"preparation", "inference"}
    assert section["aggregate_peak_rss_bytes"] is None
    assert "including reaped descendants" in section["cpu_scope"]
    assert "not simultaneous provider-tree peak" in section["peak_rss_scope"]
    assert any("Unreaped descendants remain outside" in text for text in section["limitations"])
    assert any("not proof of successful native/source/job retirement" in text for text in section["limitations"])
    assert section["complete"] == all(record["coverage"] == "os_process_exit" for record in section["roles"].values())
    metrics = {m["id"]: m for m in row["metrics"]}
    for role, record in section["roles"].items():
        for suffix, field, scope in (
            ("lifetime_cpu", "cpu_seconds", "wait4_user_system_including_reaped_descendants"),
            ("lifetime_max_individual_rss", "peak_rss_bytes", "os_maximum_individual_rss_not_simultaneous_provider_tree_peak"),
        ):
            metric = metrics[f"{role}.{suffix}"]
            assert metric["role"] == role and metric["window"] == "role_birth_through_exit"
            assert metric["unit"] == ("seconds" if field == "cpu_seconds" else "bytes")
            assert metric["measurement_scope"] == scope
            assert metric["value"] == record[field]
            if record["coverage"] == "os_process_exit":
                assert record["source"] == "wait4" and record[field] >= 0
            else:
                assert record["source"] is None and record[field] is None
                assert metric["origin"] == "not_available"
    # Family CPU overlaps provider CPU. Never replace or sum into legacy client CPU.
    assert "client.lifetime_cpu" not in metrics


def _cached_owner(cls, pid, values):
    """Fixed mocked cache only: no Popen constructor, launch, wait or PID lookup."""
    owner = cls.__new__(cls)
    owner._child_created = False
    owner.pid, owner.returncode = pid, 0
    owner._waitpid_lock = threading.Lock()
    owner._usage_scale, owner._usage_unavailable = 1, values is None
    owner._usage_final = values
    return owner


def _installed_wheel():
    dist = importlib.metadata.distribution("pllm.run")
    direct = json.loads(dist.read_text("direct_url.json") or "{}")
    assert not direct.get("dir_info", {}).get("editable", False), "non-editable installed wheel required"
    root = Path(pllm.__file__).resolve().parent
    assert root == Path(dist.locate_file("pllm")).resolve()
    native = importlib.util.find_spec("pllm._native")  # Find only; do not import/execute.
    assert native is not None and Path(native.origin).resolve().is_relative_to(root), "no external Root .so overlay"
    return root, native


@pytest.mark.parametrize("missing", [None, "client", "preparation"])
def test_mocked_ordinary_suite_preserves_source_class_and_finalizes_only_after_reap(monkeypatch, tmp_path, missing):
    """Real suite/target/adapter/serializer; transport, build and OS exits doubled."""
    from pllm.runtime import benchmark_cli as cli, benchmark_response_target as executor, servers
    from pllm.sources import BundleModel

    if sys.flags.isolated:
        _installed_wheel()  # Exercise the post-R package-resolution guard without a child/native run.

    def no_process(*args, **kwargs):
        pytest.fail("mocked control must not launch any process")

    monkeypatch.setattr(runner.subprocess, "Popen", no_process)
    monkeypatch.setitem(sys.modules, "pllm._native", None)
    suite = _suite(tmp_path / "not-materialized-mock-checkpoint")
    identity = suite.digest
    events = []
    monkeypatch.setattr(pllm.metrics, "benchmark_memory", lambda *a, **kw: {"admitted": True})
    import pllm._search_benchmark as boundary
    monkeypatch.setattr(boundary, "_boundary", lambda _: {"numeric": {}, "privacy": {}})

    def transport(**options):
        # Exercise the exact serialized source passed by ordinary execute_response.
        source = options["experiment"].pipeline.model
        assert type(source) is BundleModel and source.local_files_only
        topology = servers.LocalTopology.__new__(servers.LocalTopology)
        topology._process_lock = threading.Lock()
        topology._model = source
        topology._started, topology._closed = True, False
        topology._role_ids = ("preparation", "inference")
        topology._processes = {name: _cached_owner(usage.UsagePopen, 10001 + i,
            None if name == missing else (.002, .003, 9000 if i == 0 else 4000))
            for i, name in enumerate(topology._role_ids)}
        scope = ClientMemoryScope()
        scope.observe({"client": (*usage._birth_identity(os.getpid()), 30)}, span_seconds=0)
        sampler = NS(client_memory_scope=scope)
        roles = ("client", *topology._role_ids)
        assert not client_lifetime_manifest(sampler, topology, roles, local_cpu_loopback=True)["membership_complete"]
        events.append("mock_topology_closed")  # Not source/job-retirement evidence.
        topology._closed = True
        record = {"status": "completed", "cold": True, "warm": False,
            "model_fingerprint": "b" * 64, "max_output_tokens": 2,
            "tokens": {"authoritative": True, "input_tokens": 3, "output_tokens": 2},
            "generation": {"response_status": "completed", "output_text_digest": "c" * 64},
            "durations": {"full_seconds": .5, "online_seconds": .3, "ttft_seconds": .1,
                "tokens_per_second": 5, "preparation_seconds": .2},
            "privacy": {"plaintext_prompt_bytes_sent": 0, "plaintext_token_ids_sent": 0,
                "preparation_requests_during_online": 0}, "processes": {}}
        # Deliberately mocked report: avoid dashboard/client/native imports entirely.
        report = {"schema_version": "pllm.loopback_benchmark.v1", "checks": {"passed": True},
            "configuration": {"model_id": "client-lifetime-generated", "max_output_tokens": 2,
                "source_lock_digest": "a" * 64, "roles": list(roles), "provider_backend": "native",
                "sampling": {"temperature": 0}}, "runs": [record], "warmup_runs": []}
        report["client_process_memory"] = {"after": {"lifetime_peak_rss_bytes": 75}}
        report["client_owned_memory"] = scope.report(75)
        report["client_owned_memory"]["lifetime_owner"] = client_lifetime_manifest(
            sampler, topology, roles, local_cpu_loopback=True)
        from pllm.runtime.process_usage import provider_lifetime_usage
        report["provider_lifetime_usage"] = provider_lifetime_usage(
            topology, roles, local_cpu_loopback=True, source_present=True)
        report["process_cpu_accounting"] = {"cold_first_response_cpu_seconds_by_role": {"client": .001}}
        return report

    monkeypatch.setattr(cli, "run_loopback_benchmark", transport)
    attributes = {"scope": "explicit_execution_and_OS_usage_test_doubles"}
    monkeypatch.setattr(runner, "inspect_build", lambda _: {"attributes": attributes, "digest": digest(attributes)})

    def execute(case, sampling, inputs, salt, directory):
        m = Measurement({"sampling": sampling.to_spec(), "inputs": inputs,
            "salt": salt.hex(), "directory": str(directory)})
        m.admit = lambda required: m.admission.update(admitted=True, required_bytes=required,
            host_headroom_bytes=sampling.memory_bytes, scope="mock_only_not_resource_proof")
        executor.execute_response(case.target.to_spec()["parameters"], m)
        m.finalize_workload(case.target)
        payload = {"protocol": PROTOCOL, "status": "completed", "metrics": m.metrics,
            "samples": m.samples, "details": m.details, "fixture": m.fixture, "oracle": m.oracle,
            "comparison": m.comparison, "admission": m.admission, "environment": attributes,
            "limitations": ["All execution and OS usage mocked; no memory or model evidence."]}
        before = deepcopy(m.metrics)
        owner = _cached_owner(usage.ClientUsagePopen, os.getpid(),
            None if missing == "client" else (.01, .02, 9000))  # Family upper, NOT client peak.
        owner._usage_identity = tuple(m.details["report"]["client_owned_memory"]["lifetime_owner"]["_root_identity"])
        events.append("mock_original_parent_reap")
        usage.finalize_client_payload(payload, owner)
        assert m.metrics == before
        return payload

    monkeypatch.setattr(runner, "_execute", execute)
    observed = runner.benchmark_suite(suite, inputs={"generated": {"prompt": "Hi"}}, output=tmp_path / "mock.json")
    assert observed.to_dict()["suite"] == suite.to_spec() and suite.digest == identity
    report = _assert_scope(observed, exit_observed=missing != "client")
    _assert_provider_boundaries(observed)
    row, = observed.to_dict()["results"]
    cold, = (m for m in row["metrics"] if m["id"] == "client.cold_cpu")
    assert cold["value"] == .001 and cold["window"] == "benchmark_start_through_first_response"
    if missing != "client":
        assert report["client_process_exit_usage"]["cpu_seconds"] == .03
        assert report["client_owned_memory"]["lifetime_peak_rss_sum_interval"]["upper_bound_bytes"] == 9000
    assert events == ["mock_topology_closed", "mock_original_parent_reap"]


@pytest.mark.integration
@pytest.mark.skipif(os.environ.get("PLLM_RUN_CLIENT_LIFETIME_SDK") != "1", reason="queued until R releases native SDK slot")
def test_generated_sdk_provider_family_lifetime(monkeypatch, tmp_path):
    """Genuine suite isolation, SDK, provider launches, teardown and same-owner reap.

    No executor/transport/native/owner substitution. Only observe finalization's
    existing cached records; neither capture another wait nor sample RSS here.
    Source/job success is ordinary runtime checks, not an extra retirement proof.
    """
    from pllm.sources import _materialize_tiny_model

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    # Wheel only, even when pytest lives in a checkout. -I also protects worker
    # resolution; normal build inspection pins the exact observed implementation.
    assert sys.flags.isolated, "invoke installed wheel interpreter with -I, --import-mode=importlib and -o pythonpath="
    root, native = _installed_wheel()
    build = runner.inspect_build(BuildRef(sys.executable))
    from pllm._benchmark_build import file_digest
    assert build["attributes"]["native_sha256"] == file_digest(native.origin)
    for name in ("runtime/client_memory_scope.py", "runtime/client_process_usage.py",
                 "runtime/process_usage.py", "runtime/benchmark_cli.py", "runtime/benchmark_response_target.py",
                 "metrics/benchmark_runner.py"):
        expected = hashlib.sha256((root / name).read_bytes()).hexdigest()
        assert build["attributes"]["sdk_sources"].get(name) == expected, "controller and isolated SDK differ"
    checkpoint = _materialize_tiny_model(tmp_path / "generated-checkpoint")
    suite = _suite(checkpoint, BuildRef(sys.executable, expected_digest=build["digest"]))
    before = []  # One public, bounded phase observation; no PID/history archive.
    original = usage.finalize_client_payload

    def finalize(payload, process):
        assert process.returncode == 0
        assert usage.final_member_usage(process)["coverage"] == "os_process_exit"  # Cache only, no wait/poll.
        report = payload["details"]["report"]
        manifest = report["client_owned_memory"]["lifetime_owner"]
        assert manifest["membership_complete"], manifest["reason"]
        assert tuple(manifest["_root_identity"]) == process._usage_identity
        assert report["client_owned_memory"]["lifetime_peak_rss_sum_interval"]["upper_bound_bytes"] is None
        metrics = deepcopy(payload["metrics"])
        original(payload, process)
        assert payload["metrics"] == metrics
        before.append("original_owner_already_reaped_before_optional_report_finalization")

    monkeypatch.setattr(usage, "finalize_client_payload", finalize)
    observed = runner.benchmark_suite(suite, inputs={"generated": {"prompt": "Hi"}}, output=tmp_path / "sdk.json")
    report = _assert_scope(observed)
    assert before == ["original_owner_already_reaped_before_optional_report_finalization"]
    assert set(report["configuration"]["roles"]) == {"client", "preparation", "inference"}
    _assert_provider_boundaries(observed)
    for record in report["provider_lifetime_usage"]["roles"].values():
        # Provider peaks can inflate the client's family upper, never its lower.
        if record["coverage"] == "os_process_exit":
            assert report["client_process_exit_usage"]["peak_rss_bytes"] >= record["peak_rss_bytes"]
    assert report["checks"]["passed"]
    row, = observed.to_dict()["results"]
    assert row["execution"]["fixture"]["digest"] == report["configuration"]["source_lock_digest"]
