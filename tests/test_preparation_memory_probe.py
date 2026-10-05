import asyncio
from dataclasses import replace
from pathlib import Path
import runpy
from unittest.mock import patch

import numpy as np
import pytest

from pllm import Model
from pllm.model_loader import resolve_model
from pllm.runtime.preparation_protocol import PreparationRequest
from pllm.runtime.tiny_llama import create_tiny_llama_checkpoint
from pllm.runtime.transformer_engine import MaskedTransformerEngine, TransformerEngineError

probe = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/probe_preparation_memory.py"))
check_parity, measure, paged_loader = (probe[name] for name in ("check_parity", "measure", "paged_loader"))


def test_paged_preparation_full_stage_frames_match_and_cohort_rejects_drift(tmp_path):
    root = create_tiny_llama_checkpoint(tmp_path / "model")
    model = Model.path(str(root), model_id="preparation-memory-probe")
    samples = [asyncio.run(measure(model, mode, 3, 4, tmp_path, b"p" * 32))
               for mode in ("resident", "paged")]
    check_parity(samples)
    assert samples[0]["resident_weight_bytes"] == samples[0]["logical_weight_bytes"] > 0
    assert samples[1]["resident_weight_bytes"] == 0
    assert samples[1]["private_snapshot_disk_bytes"] == samples[0]["logical_weight_bytes"]
    assert samples[1]["correction_body_bytes"] > 0
    with pytest.raises(ValueError, match="correction_digest"):
        check_parity([samples[0], samples[1] | {"correction_digest": "0" * 64}])


def test_paged_preparation_releases_arrays_and_keeps_weight_admission(tmp_path):
    root = create_tiny_llama_checkpoint(tmp_path / "model")
    manifest = resolve_model(Model.path(str(root), model_id="preparation-memory-probe")).manifest
    engine = MaskedTransformerEngine(threads=1, weight_bits=8, activation_bits=8,
        weight_residency="provider", prepared_output_encoding="row_residues")
    snapshots = []
    with patch.object(engine, "_load_stage", paged_loader(engine, tmp_path, snapshots)):
        asyncio.run(engine.load(manifest))
    try:
        assert snapshots and not list(tmp_path.glob("stage-*.i8"))
        assert all(stage.quantized_weight is None for stage in engine.models[manifest.id].stages.values())
        stage = next(s for s in engine.models[manifest.id].stages.values() if s.compiled_weight is not None)
        request = PreparationRequest("01" * 16, "test-session", manifest.id,
            manifest.metadata["body_fingerprint"], stage.spec.id, "0" * 64, 1,
            stage.spec.in_features, stage.spec.out_features, 8, 8, stage.signed_output_bound,
            stage.seeded_profile.ring, stage.seeded_profile.modulus, stage.seeded_profile.wire_bits, b"s" * 32)
        with pytest.raises(TransformerEngineError, match="weight commitment"):
            asyncio.run(engine.prepare_seeded_stage(request))
        valid = replace(request, weight_digest=stage.weight_digest)
        correction = asyncio.run(engine.prepare_seeded_stage(valid))
        assert correction.rows == 1 and correction.output_residue_bits == stage.output_residue_bits
        with pytest.raises(TransformerEngineError, match="not resident"):
            engine.client_bundle(manifest.id)
    finally:
        asyncio.run(engine.unload(manifest.id))
        for snapshot in snapshots:
            snapshot.close()
    with pytest.raises(ValueError, match="closed"):
        snapshots[0].clear(np.zeros((1, stage.spec.in_features), dtype=np.int8))


def test_probe_closes_snapshots_on_issuance_failure(tmp_path, monkeypatch):
    root = create_tiny_llama_checkpoint(tmp_path / "model")
    from pllm.native import PagedGEMM

    closed = []
    original = PagedGEMM.close

    def close(self):
        closed.append(self)
        original(self)

    async def fail(*args):
        raise RuntimeError("issuance failed")

    monkeypatch.setattr(PagedGEMM, "close", close)
    monkeypatch.setattr(MaskedTransformerEngine, "prepare_seeded_stage", fail)
    with pytest.raises(RuntimeError, match="issuance failed"):
        asyncio.run(measure(Model.path(str(root)), "paged", 3, 4, tmp_path, b"p" * 32))
    assert closed
    assert not list(tmp_path.glob("stage-*.i8"))
