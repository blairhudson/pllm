"""Document current shared-quality rejection handling without running a model.

These are control-flow/lifetime doubles only, not memory admission or native
acceptance claims. A checked-range rejection currently aborts rather than
producing an explicit failed quality case; the CLI emits QUALITY_FAILED and
doesn't write a partial report. Keep that limitation visible until shared quality
failure retention is implemented centrally.
"""
from contextlib import contextmanager
import json
from types import SimpleNamespace

import pytest

from pllm.configuration import Experiment
from pllm.runtime import benchmark_memory, reference_benchmark, replicated_transport, servers
from test_replicated_execution_policy import experiment_for


@pytest.mark.parametrize("accepted_before_failure", [0, 1])
def test_shared_quality_rejection_aborts_cohort_and_closes_roles_and_guard(monkeypatch, accepted_before_failure):
    experiment = experiment_for(weights="public", range_policy="checked", lookup="tiled",
        selection="tiled", peer_transport="streams").with_params(budget__requests=4)
    source = SimpleNamespace(tokenizer=lambda: SimpleNamespace(encode=lambda *args, **kwargs: (3,)),
                             tokenizer_descriptor={"add_bos_token": False}, contract={"vocabulary": 5})
    events, requests = [], []

    class Watchdog:
        def __init__(self, *args):
            pass

        def start(self):
            events.append("guard_started")

        def check(self):
            pass

        def close(self):
            events.append("guard_closed")

    @contextmanager
    def build(*args, **kwargs):
        events.append("roles_started")
        try:
            yield SimpleNamespace(worker_connections=lambda: {"authenticated": "unit-only"})
        finally:
            events.append("roles_closed")

    def execute(*args, **kwargs):
        requests.append(args[2])
        if len(requests) > accepted_before_failure:
            raise ValueError("replicated decoder failed protected range checks")
        return SimpleNamespace(selected=(4,))

    monkeypatch.setattr(benchmark_memory, "benchmark_memory", lambda *args, **kwargs: {"admitted": True})
    monkeypatch.setattr(benchmark_memory, "MemoryWatchdog", Watchdog)
    monkeypatch.setattr(servers, "build_roles", build)
    monkeypatch.setattr(replicated_transport, "execute_request", execute)
    with pytest.raises(ValueError, match="failed protected range checks"):
        reference_benchmark._replicated_candidate(experiment, ("first", "second", "third"), source, 0, None)
    assert len(requests) == accepted_before_failure + 1  # later cases aren't silently evaluated/dropped
    assert events == ["guard_started", "roles_started", "roles_closed", "guard_closed"]


def test_quality_cli_rejection_is_quality_failed_without_partial_report(tmp_path, monkeypatch):
    from pllm._cli.app import _benchmark_quality
    from pllm._cli.errors import RuntimeFailure
    from pllm._cli import targets
    from pllm.profiles import resolve_runtime_composition

    experiment = experiment_for(weights="public", range_policy="checked", selection="tiled", peer_transport="streams")
    monkeypatch.setattr(Experiment, "resolve", lambda item: resolve_runtime_composition(item.pipeline))
    monkeypatch.setattr(targets, "resolve_target", lambda *args, **kwargs: SimpleNamespace(configuration=experiment))

    def reject(*args, **kwargs):
        raise ValueError("replicated decoder failed protected range checks")

    monkeypatch.setattr(reference_benchmark, "run_reference_benchmark", reject)
    prompts, output = tmp_path / "prompts.json", tmp_path / "quality.json"
    prompts.write_text(json.dumps(["a frozen public test prompt"]))
    args = SimpleNamespace(experiment=["unit-only"], top_k=5, memory_safety_margin_percent=25,
        factory=None, trust_python=False, prompts_file=prompts, output=output, force=False,
        reference_storage="streamed", decode_steps=0)
    with pytest.raises(RuntimeFailure) as failure:
        _benchmark_quality(args, "json", True, False)
    assert failure.value.code == "QUALITY_FAILED"
    assert not output.exists()
