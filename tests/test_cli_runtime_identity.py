"""Progress identity follows the shared composition, including failed startup."""
import json

import pytest

from pllm.cli import main
from pllm.profiles import OpenWeightVerifiableCpu, PumaThreePartyCpu
from pllm.runtime import benchmark_cli
from pllm.runtime.benchmark_memory import BenchmarkMemoryError
from test_benchmark_timeouts import cli_options, declaration, native_capability as native_capability


@pytest.mark.parametrize("kind,delivery", [
    ("fss", "buffered_final_ids"), ("gkm", "buffered_until_verified"),
    ("replicated", "buffered_final_ids"),
])
@pytest.mark.parametrize("mode", ["dry-run", "failed-progress"])
def test_cli_runtime_identity_survives_memory_rejection(tmp_path, native_capability,
                                                       monkeypatch, capsys, kind, delivery, mode):
    # Same misleading name for every composition: names/model IDs cannot select
    # a runtime. Native capability is a double; source inspection is forbidden.
    experiment = declaration(tmp_path, name="qwen-fusefss")
    if kind != "fss":
        constructor = OpenWeightVerifiableCpu if kind == "gkm" else PumaThreePartyCpu
        experiment = experiment.with_params(pipeline=constructor(experiment.pipeline.model))
    target, output = tmp_path / "experiment.json", tmp_path / "progress.json"
    target.write_bytes(experiment.canonical_bytes())
    calls = []

    def assert_identity(document):
        entry, = document["configuration"]["experiments"]
        assert entry["name"] == "qwen-fusefss"
        assert entry["runtime_kind"] == kind
        assert entry["output_delivery"] == delivery
        assert entry["generation_policy"] == "fixed_length_greedy"
        assert document["configuration"]["inventory_policy"] == "none"

    def fail(**kwargs):
        calls.append(kwargs)
        assert kwargs["experiment"] == experiment
        running = json.loads(output.read_text())
        assert running["schema_version"] == "pllm.benchmark_progress.v1"
        assert running["status"] == "running"
        assert_identity(running)
        raise BenchmarkMemoryError("bounded simulated disk rejection")

    monkeypatch.setattr(benchmark_cli, "run_loopback_benchmark", fail)
    arguments = [*cli_options(3600), "--experiment", str(target), "--output", str(output)]
    if mode == "dry-run":
        main(["--dry-run", *arguments])
        assert_identity(json.loads(capsys.readouterr().out)["data"])
        assert not calls and not output.exists()
    else:
        with pytest.raises(SystemExit) as error:
            main(arguments)
        assert error.value.code == 6
        assert len(calls) == 1
        failed = json.loads(output.read_text())
        assert failed["status"] == "failed" and not failed["complete"]
        assert failed["failure"] == {"type": "BenchmarkMemoryError"}
        assert failed["candidates"][0]["report"] is None
        assert_identity(failed)
