"""Cheap source-import lifetime guards; bridge doubles prove no numeric acceptance."""

import threading

import pytest

from pllm.runtime import fss_source as source

from test_fss_source import admit, bridge as bridge, write_checkpoint


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
