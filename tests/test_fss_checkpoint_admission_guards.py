"""Cheap source-import lifetime guards; bridge doubles prove no numeric acceptance."""

import threading

import pytest

from pllm.runtime import fss_source as source

from test_fss_source import admit, bridge as bridge, write_checkpoint


def test_checkpoint_qualification_uses_ordinary_phase_and_preserves_rejection(monkeypatch):
    import test_fss_checkpoint_admission as checkpoint
    from pllm.runtime import benchmark_memory as memory

    experiment = object()
    report = {
        "admitted": True,
        "admission_scope": "source_qualification_only",
        "full_topology_admitted": False,
    }
    calls = []

    def qualify(candidate, **kwargs):
        calls.append((candidate, kwargs))
        return report

    monkeypatch.setattr(memory, "execution_memory_admission", qualify)
    assert checkpoint._source_qualification_report(experiment) is report
    assert calls == [(experiment, {"backend": "native"})]
    # Existing pressure/memory/disk rejection must propagate, not become a skip
    # or a reduced-margin retry. This test creates neither sources nor tapes.
    failure = memory.BenchmarkMemoryError("qualification rejected")

    def rejected(candidate, **kwargs):
        raise failure

    monkeypatch.setattr(memory, "execution_memory_admission", rejected)
    with pytest.raises(memory.BenchmarkMemoryError) as observed:
        checkpoint._source_qualification_report(experiment)
    assert observed.value is failure


@pytest.mark.parametrize(
    "scope,full",
    [
        ("complete_topology_before_children", True),
        ("source_qualification_only", True),
        (None, False),
    ],
)
def test_checkpoint_qualification_rejects_wrong_phase(monkeypatch, scope, full):
    import test_fss_checkpoint_admission as checkpoint
    from pllm.runtime import benchmark_memory as memory

    monkeypatch.setattr(
        memory,
        "execution_memory_admission",
        lambda *args, **kwargs: {
            "admitted": True,
            "admission_scope": scope,
            "full_topology_admitted": full,
        },
    )
    with pytest.raises(memory.BenchmarkMemoryError, match="source-only qualification"):
        checkpoint._source_qualification_report(object())


@pytest.mark.parametrize("initial_rows", [1, 4])
def test_cancel_after_tensor_seal_retires_published_program(
    tmp_path, monkeypatch, bridge, initial_rows
):
    """Cancellation after allocation must close both program and pending builder."""
    write_checkpoint(tmp_path)
    admitted = admit(
        bridge,
        tmp_path,
        rows=initial_rows,
        outputs=1,
        material_delivery="batch-v1",
        online_schedule="tensor-v1",
    )
    cancelled = threading.Event()
    builder_type = source._native.FssSourceBuilder
    seal_tensor = builder_type.seal_tensor

    def seal_then_cancel(builder):
        program = seal_tensor(builder)
        assert not builder.closed and not program.closed
        cancelled.set()
        return program

    monkeypatch.setattr(builder_type, "seal_tensor", seal_then_cancel)
    with pytest.raises(source.FssSourceError, match="cancelled"):
        source.load_role_source(
            bridge.experiment,
            source=admitted,
            expected_pins=admitted.snapshot_pins,
            input_rows=initial_rows,
            outputs=1,
            cancelled=cancelled.is_set,
        )
    owner = bridge.owners[-1]
    assert owner.online_schedule == "tensor-v1"
    assert owner.frames and owner.closed
    assert owner.program is not None and owner.program.closed
