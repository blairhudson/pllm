"""Queued DEVELOPMENT overlay producer/selector regression, not campaign evidence.

Existing generated-tiny suite, ordinary isolated SDK owner, current Python and
approved83 preloaded native. No runner/reaper patching, source copying, downloads,
published-build claim, 64/8 claim or measured reduction. Needs explicit free lane.
"""

from copy import deepcopy
import os
from pathlib import Path
import sys
import venv

import pytest


@pytest.mark.integration
@pytest.mark.skipif(os.environ.get("PLLM_RUN_CLIENT_LIFETIME_SDK") != "1",
    reason="queued: genuine generated SDK needs Parent lane grant")
def test_generated_sdk_qualified_endpoint_development_overlay(monkeypatch, tmp_path):
    """One actual run; reflexive comparison tests eligibility, not independent arms.

    Negative copies alter only ownership metadata, never measured RSS or actual
    terminal status. Native completion counters and terminal reason stay genuine.
    """
    import pllm
    from pllm.metrics import BenchmarkCase, BenchmarkRun, BenchmarkSuite, BenchmarkTarget, BuildRef, MetricSpec
    from pllm.metrics import benchmark_runner as runner
    from pllm.runtime import client_process_usage as usage
    from pllm.sources import _materialize_tiny_model
    from test_client_lifetime_response_producer import _assert_provider_boundaries, _assert_scope, _suite

    if not usage.supported():
        pytest.skip("Darwin CPython same-owner exit accounting required")
    source = Path(__file__).resolve().parents[1] / "python"
    approved = Path(sys.prefix).resolve()  # Explicitly supplied approved83 interpreter.
    assert sys.flags.isolated and sys.dont_write_bytecode
    assert Path(pllm.__file__).resolve() == source / "pllm/__init__.py"
    native = sys.modules.get("pllm._native")
    assert native is not None, "preload approved83 native BEFORE the development source overlay"
    assert Path(native.__file__).resolve().is_relative_to(approved)

    # Standard private venv, no pip/build and no wrapper executable. Site startup
    # reaches build inspection, -I SDK workers AND ordinary provider children.
    development = tmp_path / "development-env"
    venv.EnvBuilder(with_pip=False, symlinks=True).create(development)
    site = development / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    approved_site = Path(native.__file__).resolve().parents[1]
    startup = ("import sys,site; sys.dont_write_bytecode=True; "
        f"site.addsitedir({str(approved_site)!r}); import pllm._native; "
        "[sys.modules.pop(k) for k in list(sys.modules) "
        "if k == 'pllm' or k.startswith('pllm.') and k != 'pllm._native']; "
        f"sys.path.insert(0,{str(source)!r})\n")
    (site / "development_source_overlay.pth").write_text(startup)
    python = development / "bin/python"
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    build = runner.inspect_build(BuildRef(str(python)))
    from pllm._benchmark_build import file_digest
    assert build["attributes"]["native_sha256"] == file_digest(native.__file__)
    for name in ("_benchmark_evidence.py", "runtime/dashboard.py", "runtime/benchmark_cli.py"):
        assert build["attributes"]["sdk_sources"][name] == file_digest(source / "pllm" / name)
    checkpoint = _materialize_tiny_model(tmp_path / "generated-checkpoint")
    base = _suite(checkpoint, BuildRef(str(python), expected_digest=build["digest"]))
    case, = base.cases
    # Existing byte-chat fixture: "Hi" gave authoritative 22/2 in prior real SDK
    # smoke. Do not force EOS, change weights or rewrite token-limit termination.
    declaration = case.target.to_spec()
    declaration["parameters"].update(state_audit=True, expected_input_tokens=22, expected_output_tokens=2)
    target = BenchmarkTarget.from_spec(declaration)
    suite = BenchmarkSuite((BenchmarkCase(case.name, target, case.build),), base.sampling)
    output = tmp_path / "development-endpoint-sdk.json"
    runner.benchmark_suite(suite, inputs={case.name: {"prompt": "Hi"}}, output=output)
    observed = BenchmarkRun.load(output)
    report = _assert_scope(observed)
    _assert_provider_boundaries(observed)
    measured, = report["runs"]
    assert measured["generation"]["response_status"] == "incomplete"
    assert measured["generation"]["incomplete_details"] == {"reason": "max_output_tokens"}
    assert measured["privacy"]["generation_completed_responses"] == 1
    assert measured["privacy"]["generation_output_tokens"] == measured["privacy"]["generation_requested_output_tokens"] == 2
    selector = MetricSpec("client.peak_rss", qualification="root_only_report_endpoint")
    saved = observed.to_dict()
    for gap in ("provider_is_not_client", "unknown_helper_coverage", "missing_final_barrier", "missing_terminal_reason"):
        damaged = deepcopy(saved)
        report = damaged["results"][0]["execution"]["details"]["report"]
        owner = report["client_owned_memory"]["lifetime_owner"]
        if gap == "provider_is_not_client":
            owner["members"]["preparation"] = deepcopy(report["provider_lifetime_usage"]["roles"]["preparation"])
        elif gap == "unknown_helper_coverage":
            owner.update(membership_complete=False, reason="undeclared_client_helpers")
        elif gap == "missing_final_barrier":
            report["runs"][0]["privacy"].pop("generation_completed_responses")
        else:
            report["runs"][0]["generation"].pop("incomplete_details")
        assert BenchmarkRun(damaged).compare(case.name, case.name, selector)["status"] == "unranked"
    compared = observed.compare(case.name, case.name, selector)
    assert compared["status"] == "comparable_in_scope", compared
    assert compared["control_median"] > 0 and compared["paired_control_over_candidate"] == [1]
    assert compared["window"] == "client_process_birth_through_report_endpoint"
    assert compared["physical_peak"] is False and compared["whole_lifetime"] is False
    assert compared["reduction_status"] == "not_established"
