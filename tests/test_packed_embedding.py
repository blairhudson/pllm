"""Encrypted one-hot algebra, hidden carry conversion and bounded SDK ownership."""
import json
from pathlib import Path
import struct

import pytest

from pllm import _native
from pllm.components import create_component, get
from pllm.configuration import ConfigurationError
from pllm.metrics import benchmark_operator, replay_operator_benchmark
from pllm.protocols import PrivateEmbeddingLookup, ProtocolMethod


@pytest.mark.parametrize("precision", ["int8", "int4", "mixed_4_3_3"])
@pytest.mark.parametrize("rows,columns", [(1, 1), (17, 7), (65, 64)])
def test_encrypted_lookup_and_full_conversion_match_independent_row_oracle(precision, rows, columns):
    pytest.importorskip("_sealapi_cpp")
    report = benchmark_operator(PrivateEmbeddingLookup(vocabulary_size=rows, precision=precision),
                                lanes=columns, samples=2)
    assert report.quality.exact_integer_outputs == 2 * columns
    assert report.resources.combined_raw_key_payload_bytes > 0
    assert report.resources.public_table_bytes_by_party[0] == 0
    assert report.resources.public_table_bytes_by_party[1] > rows * columns
    assert report.resources.public_table_bytes_per_party == 0
    assert report.resources.client_entry_exit_payload_estimate_bytes > 0
    assert report.samples[0].peer_frame_bytes_by_phase[0] == report.resources.peer_frame_bytes_by_direction
    assert report.samples[0].public_setup_seconds > 0
    assert report.samples[1].public_setup_seconds == 0
    assert report.to_dict()["full_wire_bytes"] is None
    assert not report.to_dict()["whole_model_executable"]


def test_fixed_geometry_replay_and_zero_table_have_constant_frames():
    pytest.importorskip("_sealapi_cpp")
    component = PrivateEmbeddingLookup(vocabulary_size=19)
    report = benchmark_operator(component, lanes=7, samples=1)
    replay = replay_operator_benchmark(report.configuration, inputs={"table": [0] * (19 * 7), "token": [18]})
    assert replay.resources == report.resources
    assert replay.input_source == "caller-owned"
    assert '"inputs"' not in replay.to_json()
    assert '"outputs"' not in replay.to_json()
    assert '"token"' not in replay.to_json()
    assert isinstance(component, ProtocolMethod)
    assert get(component.component) is PrivateEmbeddingLookup
    assert create_component(component.component, component.params) == component


def test_conversion_corrects_cross_slot_borrows_and_sign_extends_before_resharing():
    codes = [-8, -4, 3]
    word = (codes[0] & 15) | ((codes[1] & 7) << 5) | ((codes[2] & 7) << 9)
    for mask in [0, 1, 31, 32, 511, 4095, 8191]:
        raw, metadata = _native.packed_embedding_convert("mixed_4_3_3", (word - mask) % 8192, mask)
        assert struct.unpack("<3i", raw) == tuple(codes)
        assert json.loads(metadata)["and_gates"] > 0
    with pytest.raises(ValueError): _native.packed_embedding_convert("mixed_4_3_3", 8192, 0)


def test_public_geometry_and_input_bounds_reject_before_model_or_material_allocation(monkeypatch):
    monkeypatch.setattr(_native, "packed_embedding_table", lambda *args: pytest.fail("unexpected packing"))
    for params in ({"vocabulary_size": True}, {"vocabulary_size": 4097}, {"vocabulary_size": 0}, {"precision": "int3"}):
        with pytest.raises(ConfigurationError): PrivateEmbeddingLookup(**params)
    with pytest.raises(ConfigurationError, match="128 MiB"):
        benchmark_operator(PrivateEmbeddingLookup(vocabulary_size=4096, precision="int8"), lanes=256)
    for table, token in [([0] * 6, [3]), ([0] * 5, [0]), ([0] * 6, [True]),
                         ([0, 4] * 3, [0]), ([0] * 6, [0, 1]), ([0.0] * 6, [0])]:
        with pytest.raises(ConfigurationError):
            benchmark_operator(PrivateEmbeddingLookup(vocabulary_size=3), lanes=2, inputs={"table": table, "token": token})


def test_bounded_seal_codec_rejects_length_and_padding_before_backend_load(tmp_path):
    from pllm.metrics.packed_embedding import _load_frame
    class NoLoad:
        def load(self, *args): pytest.fail("malformed frame reached SEAL")
    for frame in (b"", bytes(32), struct.pack("<Q", 25) + bytes(24), struct.pack("<Q", 1) + b"a" + bytes(22) + b"b"):
        with pytest.raises(ValueError): _load_frame(NoLoad(), None, frame, tmp_path / "unused", 32)
    assert not (tmp_path / "unused").exists()


def test_native_geometry_and_code_bounds_reject():
    for rows, columns, layout, codes in [(0, 1, "int8", b""), (4097, 1, "int8", bytes(4097)),
        (1, 257, "int8", bytes(257)), (1, 1, "unknown", b"\0"), (1, 1, "int4", b"\x08"),
        (2, 1, "int8", b"\0")]:
        with pytest.raises(ValueError): _native.packed_embedding_table(rows, columns, layout, codes)


def test_citation_is_bound_to_the_checked_primary_source():
    from pllm.metrics.packed_embedding import _SOURCE
    library = json.loads((Path(__file__).resolve().parents[1] / "docs/data/research/paper-library.json").read_text())
    paper = next(p for p in library["papers"] if p["id"] == _SOURCE.paper_id)
    assert paper["sha256"] == _SOURCE.source_sha256


def test_channel_quantizer_matches_independent_rounding_and_saturation_oracle():
    import numpy as np
    weights = np.asarray([[i / 2, i / 8, i * 2] for i in range(-300, 301)], dtype="<f4")
    scales = np.asarray([1, 0.25, 4], dtype="<f4")
    widths = bytes([3, 4, 8])
    actual = np.frombuffer(_native.packed_embedding_quantize(weights.tobytes(), widths, scales.tobytes()), dtype="i1").reshape(weights.shape)
    expected = np.clip(np.rint(weights.astype(np.float64) / scales), -np.asarray([3, 7, 127]), [3, 7, 127])
    assert np.array_equal(actual, expected)
    stats = json.loads(_native.packed_embedding_statistics(3, weights.tobytes()))
    assert stats["maxima"] == np.max(np.abs(weights), axis=0).tolist()
    assert stats["sums"] == np.sum(np.abs(weights).astype(np.float64), axis=0).tolist()
    for invalid in (b"", b"x", np.asarray([np.nan], dtype="<f4").tobytes()):
        with pytest.raises(ValueError): _native.packed_embedding_statistics(1, invalid)


def test_operator_configuration_cannot_activate_unimplemented_decoder_placement():
    from pllm import Deployment, ExecutionBudget, Experiment, Model, Pipeline
    from pllm.profiles import MaskedLinearCpu

    baseline = MaskedLinearCpu(model=Model.tiny())
    method = PrivateEmbeddingLookup()
    experiment = Experiment(
        name="embedding-operator-boundary",
        pipeline=Pipeline(model=baseline.model, components={**baseline.components, "protocol": method}),
        deployment=Deployment.local(root="local://embedding-operator-boundary"),
        budget=ExecutionBudget(requests=1, max_input_tokens=2, max_new_tokens=1),
    )
    replay = Experiment.from_spec(json.loads(experiment.canonical_bytes()))
    assert replay.pipeline.components["protocol"] == method
    with pytest.raises(ConfigurationError):
        replay.resolve()
