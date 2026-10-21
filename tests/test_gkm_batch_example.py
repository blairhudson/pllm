"""Batch study reuses ordinary response targets and keeps sharing controls distinct."""

from examples.benchmarks import gkm_prefill_batch
from pllm import Model
from pllm.configuration import Experiment
from pllm.metrics import BenchmarkSuite

from test_delegated_source import write_source_fixture


def test_batch_suite_has_matching_scalar_control_for_each_sharing(tmp_path):
    write_source_fixture(tmp_path)
    selected = gkm_prefill_batch.suite(
        model=Model.path(str(tmp_path)), processes=1, input_tokens=3, output_tokens=2)
    assert [case.name for case in selected.cases] == [
        "gkm-full", "gkm-full-batch", "gkm-seeded", "gkm-seeded-batch"]
    assert BenchmarkSuite.from_spec(selected.to_spec()) == selected
    assert selected.sampling.processes == 1
    for scalar, batch in (selected.cases[:2], selected.cases[2:]):
        assert scalar.build == batch.build
        control_spec = scalar.target.to_spec()["parameters"]
        batch_spec = batch.target.to_spec()["parameters"]
        control = Experiment.from_spec(control_spec["model"]["experiment"])
        candidate = Experiment.from_spec(batch_spec["model"]["experiment"])
        assert "execution" not in control.pipeline.components["linear"].params
        assert candidate.pipeline.components["linear"].params["execution"] == "source_stage_batch_v1"
        assert candidate.with_params(
            pipeline__components__linear__execution=None).canonical_bytes() == control.canonical_bytes()
        assert batch_spec["state_audit"] is True
        for key in control_spec.keys() - {"model"}:
            assert batch_spec[key] == control_spec[key]
