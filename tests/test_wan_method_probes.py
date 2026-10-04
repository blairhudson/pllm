import numpy as np
import pytest

from pllm import _native
from pllm.metrics import (ArtifactPlaneProbe, BatchedPrivateLookupProbe, PreparedResidueProbe,
                          ProjectedResharingProbe, OrthogonalActivationProbe)
from test_state_reuse_probes import fixture


def test_projected_resharing_exactness_and_directed_body_count():
    result = ProjectedResharingProbe().run()
    a, b = result["configurations"]
    assert a["exact_field_outputs"] and b["exact_field_outputs"]
    assert a["peer_frame_bytes"] == 6 * (46 + 64 * 4)
    assert b["peer_frame_bytes"] == 6 * (46 + 16 * 4)
    assert not result["whole_decoder_executable"]


def test_prepared_widths_keep_masked_roundtrip_exact():
    w = np.array([[0, 0, 0], [-128, 127, 1], [1, 1, 1]], dtype=np.int8)
    result = PreparedResidueProbe(rows=7).run(w)
    a, native, b = result["configurations"]
    assert all(row["exact_integer_output"] for row in (a, native, b))
    assert native["input_masked_body_bytes"] == a["input_masked_body_bytes"]
    assert b["offline_correction_body_bytes"] < a["offline_correction_body_bytes"]
    assert b["online_output_body_bytes"] < a["online_output_body_bytes"]
    assert b["input_masked_body_bytes"] == a["input_masked_body_bytes"]
    assert result["public_width_bits_range"][0] == 1


@pytest.mark.parametrize("bits", [16, 24, 32])
def test_native_prepared_masking_matches_integer_oracle_and_chunks(bits, monkeypatch):
    from pllm.runtime import native
    rng = np.random.default_rng(bits)
    x = rng.integers(-128, 128, (7, 35), dtype=np.int8)
    mask = rng.integers(0, 1 << bits, x.shape, dtype=np.uint32)
    masked = native.mask_prepared_input(x, mask, bits)
    np.testing.assert_array_equal(masked, (x.astype(np.int64) - mask.astype(np.int64)) % (1 << bits))
    np.testing.assert_array_equal(native.unmask_prepared_output(masked, mask, bits), x)
    monkeypatch.setattr(native, "_PREPARED_CHUNK_ELEMENTS", 13)
    np.testing.assert_array_equal(native.mask_prepared_input(x, mask, bits), masked)
    np.testing.assert_array_equal(native.unmask_prepared_output(masked, mask, bits), x)
    if bits < 32:
        forged = mask.copy()
        forged[2, 8] = 1 << bits
        with pytest.raises(ValueError, match="ring"):
            native.mask_prepared_input(x, forged, bits)
        with pytest.raises(ValueError, match="ring"):
            native.unmask_prepared_output(masked, forged, bits)
    with pytest.raises(ValueError):
        _native.prepared_center_output(b"\0", b"\0", bits)
    with pytest.raises(ValueError):
        _native.prepared_mask_input(b"a", b"\0", bits)


def test_public_byte_codec_and_private_batch_do_not_change_results():
    report = ArtifactPlaneProbe(repetitions=1).run(bytes(range(256)) * 257)
    assert all(r["exact_raw_hash"] for r in report["configurations"])
    assert report["online_stage_body_savings"] == 0
    report = BatchedPrivateLookupProbe(records=17, batch_size=7).run()
    a, b = report["configurations"]
    assert a["exact_records"] and b["exact_records"]
    assert a["query_and_reply_body_bytes"] == b["query_and_reply_body_bytes"]
    server = _native.PrivatePageServer(bytes(range(128)), 8, 0)
    a, _, _ = _native.private_page_issue(*server.descriptor(), 3)
    with pytest.raises(ValueError):
        server.evaluate_batch([a, b"malformed"])
    assert len(server.evaluate_batch([a])) == 1
    with pytest.raises(ValueError):
        server.evaluate(a)


def test_orthogonal_probe_keeps_unrotated_eight_bit_control_and_checks_source(tmp_path):
    compiled, remote, *_ = fixture(tmp_path / "model", "qwen2")
    probe = OrthogonalActivationProbe(activation_bits=8, block=8, decode_steps=1)
    result = probe.run(compiled, remote, remote.weight_matrices, [[3, 8, 7]])
    assert result["configurations"][0]["all_checked_selections_match"]
    assert result["configurations"][0]["worst_absolute_logit_error"] == 0
    assert not result["whole_decoder_executable"]
    weights = dict(remote.weight_matrices)
    key = next(s.stage_id for s in compiled.stage_bindings if s.layer_index is not None)
    weights[key] = weights[key].copy()
    weights[key][0, 0] ^= np.int8(1)
    with pytest.raises(ValueError, match="commitment"):
        probe.run(compiled, remote, weights, [[3, 8, 7]])
