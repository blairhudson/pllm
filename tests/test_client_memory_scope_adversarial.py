"""Counterexamples for declared client RSS scope, not physical-memory accuracy.

Cached-owner doubles exercise lifetime algebra without launching a model/native
task. One bounded stdlib subprocess checks the real qualified wait4 boundary.
Reuse ordinary report/comparison fixtures; no alternate measurement pipeline.
"""
from copy import deepcopy
import json
import os
import subprocess
import sys
import threading
from types import SimpleNamespace as NS

import pytest

from pllm.runtime import client_process_usage as usage
from pllm.runtime.client_memory_scope import ClientMemoryScope, client_lifetime_manifest
from test_client_memory_comparison import INTERVAL, LOWER, SUM, interval, report, run


def _owner(pid, birth, peak):
    # An already-reaped owner cache, not a new waiter or a real process claim.
    owner = usage.UsageSpawnPopen.__new__(usage.UsageSpawnPopen)
    owner.pid, owner.returncode = pid, 0
    owner._usage_lock = threading.Lock()
    owner._usage_identity = (pid, birth)
    owner._usage_values = (.01, .02, peak)
    owner._usage_reason = None
    return owner


def _helper(pid=4242, birth=2., peak=60):
    process = usage.UsageSpawnProcess()
    process._popen = _owner(pid, birth, peak)
    return process


def _topology(providers=None):
    from pllm.runtime.servers import LocalTopology

    topology = LocalTopology.__new__(LocalTopology)
    topology._process_lock = threading.Lock()
    topology._closed = topology._started = True
    topology._processes = providers or {}
    topology._role_ids = tuple(topology._processes)
    return topology


def _finalized(prefix=40, family=100, *, helpers=None, closed=False):
    observed = report(prefix)
    scope = observed["client_owned_memory"]
    scope["lifetime_owner"] = {
        "schema": "pllm.client_lifetime_owner.v1", "membership_complete": True,
        "closed_client_owned_family": closed, "root_birth_bound": False,
        "_root_identity": (os.getpid(), 1.), "window": usage.WINDOW,
        "members": {"client": {}, **(helpers or {})}, "reason": None,
    }
    usage.finalize_client_report(observed, NS(_popen=_owner(os.getpid(), 1., family)))
    return observed


@pytest.mark.parametrize("closed", [False, True])
@pytest.mark.parametrize("simultaneous", [False, True])
def test_family_maxima_can_overlap_and_double_count_without_becoming_group_peak(closed, simultaneous):
    # Same HWM values admit two different simultaneous RSS peaks. Root's wait4
    # family max includes its reaped helper, so summing duplicates that maximum.
    trace = [(40, 60)] if simultaneous else [(40, 0), (0, 60)]
    true_peak = max(sum(row) for row in trace)
    helper = usage._observed(0, (.01, .02, 60))
    observed = _finalized(40, 60, helpers={"client_helper_1": helper}, closed=closed)
    scope = observed["client_owned_memory"]
    bounds = usage.owned_family_interval(scope)
    assert bounds == {"lower": 60 if closed else 40, "upper": 120}
    assert bounds["lower"] <= true_peak <= bounds["upper"]
    assert true_peak != bounds["upper"]
    assert scope["physical_used_ram_peak_bytes"] is None


def test_foreign_reaped_descendant_maximum_is_upper_only_not_client_lower():
    # Client root peaks at 40; an excluded provider/bookkeeping child peaks at
    # 9000 and is reaped by root. The family's 9000 is NOT a client lower bound.
    observed = _finalized(40, 9000)
    scope = observed["client_owned_memory"]
    assert usage.owned_family_interval(scope) == {"lower": 40, "upper": 9000}
    assert scope["lifetime_peak_rss_sum_interval"]["lower_bound_source"] == (
        "root_RUSAGE_SELF_prefix_within_isolated_lifetime")


def test_foreign_descendant_of_helper_cannot_supply_client_lower_bound_either():
    helper = usage._observed(0, (.01, .02, 9000))
    observed = _finalized(40, 9000, helpers={"client_helper_1": helper})
    assert usage.owned_family_interval(observed["client_owned_memory"]) == {
        "lower": 40, "upper": 18000}


@pytest.mark.parametrize("prefix", [None, -1, True, 101])
def test_foreign_family_without_valid_root_prefix_cannot_form_two_sided_interval(prefix):
    observed = _finalized()
    scope = observed["client_owned_memory"]
    scope["individual_client_hwm"]["bytes"] = prefix
    assert usage.owned_family_interval(scope) is None


@pytest.mark.parametrize("mutation", [
    {"window": "sampler_start_through_responses"},
    {"source": "resource.getrusage(RUSAGE_CHILDREN)"},
    {"coverage": "individual_process_only_not_client_group_wrong_owner"},
])
def test_prefix_must_be_self_only_and_within_the_same_isolated_lifetime(mutation):
    observed = _finalized()
    scope = observed["client_owned_memory"]
    scope["individual_client_hwm"].update(mutation)
    assert usage.owned_family_interval(scope) is None


@pytest.mark.parametrize("damage", ["not_reaped", "nonzero", "wrong_window", "wrong_units"])
def test_helper_final_coverage_cannot_be_replaced_by_last_rss_or_root_family_max(damage):
    helper = usage._observed(0, (.01, .02, 60))
    if damage == "not_reaped":
        helper = usage.unknown("not_reaped")
    elif damage == "nonzero":
        helper = usage._observed(7, (.01, .02, 60))
    else:
        helper["window" if damage == "wrong_window" else "raw_hwm_unit"] = "mismatched"
    observed = _finalized(40, 9000, helpers={"client_helper_1": helper})
    scope = observed["client_owned_memory"]
    assert scope["lifetime_peak_rss_sum_interval"]["upper_bound_bytes"] is None
    assert usage.owned_family_interval(scope) is None


def test_individual_hwm_sum_is_upper_even_when_each_peak_occurs_at_different_time():
    maxima = {"client": 40, "client_helper_1": 60}
    observed = interval(report(40), maxima)
    scope = observed["client_owned_memory"]
    scope["latest"]["members"]["client_helper_1"] = {"rss_bytes": 60, "reason": None}
    # Same-pass reads can be skewed; give the ordinary fixture consistent sums.
    scope["sampled_peak_rss_sum_bytes"] = scope["latest"]["same_pass_rss_sum_bytes"] = 100
    comparison = run(interval(report(1000), {"client": 1000}), observed).compare(
        "control", "candidate", INTERVAL)
    assert comparison["pairs"][0]["candidate"] == {"lower_bytes": 60, "upper_bytes": 100}
    assert max(sum(row) for row in [(40, 0), (0, 60)]) == 60 < 100


@pytest.mark.parametrize("overestimate", [False, True])
def test_nonatomic_pass_sum_can_be_on_either_side_of_true_simultaneous_peak(overestimate):
    helper = _helper()
    scope = ClientMemoryScope()
    scope.bind_helper("client_helper_1", helper, created_at=2.)
    scope.seal()
    trace = [(100, 0), (0, 100)] if overestimate else [(10, 10), (100, 100), (10, 10)]
    # Read root at first instant, helper at last instant, not atomically.
    scope.observe({"client": (os.getpid(), 1., trace[0][0]),
        "client_helper_1": (helper.pid, 2., trace[-1][1])}, span_seconds=.01)
    measured = scope.report()
    true_peak = max(sum(row) for row in trace)
    estimate = measured["sampled_peak_rss_sum_bytes"]
    assert (estimate > true_peak) if overestimate else (estimate < true_peak)
    assert measured["sampled_peak_rss_lower_bound_bytes"] <= true_peak
    assert measured["lifetime_peak_rss_sum_interval"]["upper_bound_bytes"] is None


@pytest.mark.parametrize("conflict", [False, True])
def test_provider_parentage_does_not_authorize_closed_client_family(conflict):
    helper = _helper()
    scope = ClientMemoryScope()
    if conflict:
        scope.bind_helper("client_helper_1", helper, created_at=2.)
    scope.seal_lifetime_membership(closed_family=True)
    scope.observe({"client": (os.getpid(), 1., 40)}, span_seconds=0)
    manifest = client_lifetime_manifest(NS(client_memory_scope=scope),
        _topology({"inference": helper}), ("client", "inference"), local_cpu_loopback=True)
    if conflict:
        assert manifest["membership_complete"] is False
    else:
        assert manifest["membership_complete"] is True
        assert manifest["closed_client_owned_family"] is False
        assert set(manifest["members"]) == {"client"}


def test_root_birth_mismatch_cannot_borrow_previous_process_final_hwm():
    observed = _finalized()
    manifest = observed["client_owned_memory"]["lifetime_owner"]
    manifest["_root_identity"] = (os.getpid(), 2.)
    usage.finalize_client_report(observed, NS(_popen=_owner(os.getpid(), 1., 9000)))
    assert usage.owned_family_interval(observed["client_owned_memory"]) is None


def test_reaped_helpers_cached_usage_is_not_replaced_by_foreign_pid_reuse():
    scope = ClientMemoryScope()
    helper = _helper(birth=2., peak=60)
    scope.bind_helper("client_helper_1", helper, created_at=2.)
    scope.seal_lifetime_membership(closed_family=True)
    # A foreign process can reuse a retired helper's PID. Do not sample that
    # outsider, but the original helper's cached final HWM remains usable.
    scope.observe({"client": (os.getpid(), 1., 40),
        "client_helper_1": (helper.pid, 3., 9000)}, span_seconds=0)
    assert scope.report()["latest"]["members"]["client_helper_1"]["reason"] == "member_identity_changed"
    manifest = client_lifetime_manifest(NS(client_memory_scope=scope), _topology(),
        ("client",), local_cpu_loopback=True)
    assert manifest["membership_complete"] is True
    observed = report(40)
    observed["client_owned_memory"] = scope.report(40)
    observed["client_owned_memory"]["lifetime_owner"] = manifest
    usage.finalize_client_report(observed, NS(_popen=_owner(os.getpid(), 1., 60)))
    bounds = usage.owned_family_interval(observed["client_owned_memory"])
    assert bounds == {"lower": 60, "upper": 120}
    assert observed["client_owned_memory"]["sampled_peak_rss_lower_bound_bytes"] == 40


def test_captured_helper_birth_cannot_be_rebound_to_reused_pid_generation():
    scope = ClientMemoryScope()
    helper = _helper(birth=2., peak=60)
    # Creation owner has birth 2, but a later psutil lookup could see a reused
    # PID with birth 3. It cannot turn the old cached 60 into the new member HWM.
    try:
        scope.bind_helper("client_helper_1", helper, created_at=3.)
    except ValueError:
        return  # Rejecting the contradictory binding immediately is sufficient.
    scope.seal_lifetime_membership(closed_family=True)
    for later_birth in (3., 2., 3.):
        scope.observe({"client": (os.getpid(), 1., 40),
            "client_helper_1": (helper.pid, later_birth, 9000)}, span_seconds=0)
        # Matching either declaration or old owner later cannot repair the
        # original contradictory binding or launder the foreign observation.
        assert scope.report()["sampled_peak_rss_lower_bound_bytes"] == 40
    manifest = client_lifetime_manifest(NS(client_memory_scope=scope), _topology(),
        ("client",), local_cpu_loopback=True)
    assert manifest["membership_complete"] is False, (
        "Cached owner birth 2 cannot finalize declared helper birth 3: its 60-byte "
        "HWM would make lifetime upper 120 smaller than sampled individual lower 9000")


@pytest.mark.parametrize("control,candidate,factor,established", [
    ((100, 9000), (1, 11), 10, False),
    ((100, 9000), (1, 10), 10, True),
    ((100, 9000), (1, 10), 11, False),
    ((0, 9000), (0, 0), 10, False),
    ((10 * ((1 << 53) + 1) - 1, 10 * ((1 << 53) + 1)),
     (1, (1 << 53) + 1), 10, False),
])
def test_interval_comparison_uses_unrounded_control_lower_and_candidate_upper(
        control, candidate, factor, established):
    observed = run(_finalized(*control), _finalized(*candidate))
    before = deepcopy(observed.to_dict())
    result = observed.compare("control", "candidate", INTERVAL, minimum_reduction=factor)
    assert result["status"] == "bounded"
    assert (result["reduction_status"] == "established_in_scope") is established
    assert result["pairs"][0]["candidate"]["upper_bytes"] == candidate[1]
    assert result["physical_peak"] is False
    assert "paired_control_over_candidate" not in result
    assert observed.to_dict() == before


@pytest.mark.parametrize("metric", [SUM, LOWER])
def test_large_sampled_ratios_still_cannot_establish_lifetime_reduction(metric):
    result = run(report(10000), report(1)).compare("control", "candidate", metric)
    assert result["reduction_status"] == "not_established"


@pytest.mark.skipif(not usage.supported(), reason="qualified Darwin owner only")
def test_real_reaped_foreign_descendant_hwm_never_becomes_client_lower_bound():
    descendant = """
import resource, signal
signal.alarm(5)
data = bytearray(48 << 20)
for index in range(0, len(data), 4096): data[index] = 1
print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, flush=True)
"""
    script = """
import json, resource, subprocess, sys
child = subprocess.Popen([sys.executable, '-I', '-c', CHILD], stdout=subprocess.PIPE)
try:
    out, _ = child.communicate(timeout=5)
    assert child.returncode == 0
finally:
    if child.poll() is None: child.kill()
    child.wait(timeout=5)
print(json.dumps([resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, int(out)]), flush=True)
""".replace("CHILD", repr(descendant))
    process = usage.ClientUsagePopen([sys.executable, "-I", "-c", script],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        out, err = process.communicate(timeout=10)
        assert process.returncode == 0, err
        prefix, foreign_peak = json.loads(out)
        assert 0 < prefix < foreign_peak  # No exact RSS value or physical-RAM claim.
        observed = report(prefix)
        observed["client_owned_memory"]["lifetime_owner"] = {
            "schema": "pllm.client_lifetime_owner.v1", "membership_complete": True,
            "closed_client_owned_family": False, "root_birth_bound": False,
            "_root_identity": process._usage_identity, "window": usage.WINDOW,
            "members": {"client": {}}, "reason": None,
        }
        usage.finalize_client_report(observed, process)
        bounds = usage.owned_family_interval(observed["client_owned_memory"])
        assert bounds["lower"] == prefix
        assert bounds["upper"] >= foreign_peak > bounds["lower"]
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=10)
        process.stdout.close()
        process.stderr.close()
    with pytest.raises(ChildProcessError):
        os.waitpid(process.pid, os.WNOHANG)
