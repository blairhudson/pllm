"""Exact numeric and binding gates for the second client-offload reference round."""
import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
spec = importlib.util.spec_from_file_location("client_offload_numeric", SCRIPTS / "client_offload_numeric.py")
numeric = importlib.util.module_from_spec(spec)
spec.loader.exec_module(numeric)


def contract():
    return numeric.rope_contract("a" * 64, {
        "coefficient_profile": "pllm.numeric.rope.q30.libm.v1",
        "input_layout": "batch_heads_sequence_feature", "output_layout": "batch_heads_sequence_feature",
        "pairing": "split_half", "position_policy": "sequential_absolute", "tail_policy": "unchanged",
        "rotary_dimensions": 8, "theta": 1000000,
    }, 24)


def test_public_rope_matches_private_prefill_decode_and_partial_tail_bits(tmp_path):
    from pllm.runtime.semantic_executor import SemanticDecoderRuntime

    binding = contract()
    artifact = tmp_path / "rope.bin"
    sha = numeric.export_rope(artifact, binding)
    candidate = numeric.PublicRope(artifact, sha, binding)
    assert not candidate.table.flags.writeable
    rng = np.random.default_rng(10)
    for positions in (np.arange(16), np.asarray([16]), np.asarray([23, 0, 8])):
        x = rng.standard_normal((1, 2, len(positions), 12), dtype=np.float32)
        original = x.copy()
        expected = SemanticDecoderRuntime._rotary(x, positions, binding["attributes"])
        assert np.array_equal(candidate.apply(x, positions).view("u4"), expected.view("u4"))
        assert np.array_equal(x, original)


def test_public_rope_trusted_source_plan_numeric_and_capacity_rejections(tmp_path):
    binding = contract()
    artifact = tmp_path / "rope.bin"
    sha = numeric.export_rope(artifact, binding)
    with pytest.raises(ValueError, match="identity"):
        numeric.PublicRope(artifact, "0" * 64, binding)
    with pytest.raises(ValueError, match="binding"):
        numeric.PublicRope(artifact, sha, binding | {"plan_digest": "b" * 64})
    with pytest.raises(ValueError, match="contract"):
        numeric.PublicRope(artifact, sha, binding | {"numpy": "another-arithmetic-build"})
    candidate = numeric.PublicRope(artifact, sha, binding)
    x = np.zeros((1, 1, 1, 8), np.float32)
    for positions in (np.asarray([-1]), np.asarray([24]), np.asarray([0.5])):
        with pytest.raises(ValueError, match="public contract"):
            candidate.apply(x, positions)
    changed = bytearray(artifact.read_bytes())
    changed[-1] ^= 1
    artifact.write_bytes(changed)
    with pytest.raises(ValueError, match="identity"):
        numeric.PublicRope(artifact, sha, binding)
    with pytest.raises(ValueError, match="contract"):
        numeric.rope_contract("a" * 64, binding["attributes"] | {"frequency_scaling": {}}, 24)


@pytest.mark.parametrize("value", [np.float32(0), np.float32(-1), np.float32(np.inf),
                                  np.float32(np.nan), np.float32(np.nextafter(np.float32(0), np.float32(1)))])
def test_private_sqrt_rejects_outside_declared_positive_normal_domain(value):
    with pytest.raises(ValueError, match="positive normal"):
        numeric.sqrt_index(value)


def test_private_sqrt_table_exact_bits_and_fresh_oblivious_queries(tmp_path):
    from pllm import _native

    path = tmp_path / "sqrt.f32"
    sha = numeric.export_sqrt(path)
    assert path.stat().st_size == 64 << 20
    assert numeric.digest(path) == sha
    data = path.read_bytes()
    table = np.frombuffer(data, "<f4")
    rng = np.random.default_rng(441)
    values = rng.integers(0x00800000, 0x7f800000, 2048, dtype=np.uint32).view(np.float32)
    for x in values:
        index, power = numeric.sqrt_index(x)
        actual = numeric.sqrt_restore(table[index].tobytes(), power)
        assert actual.view("u4") == np.sqrt(x).view("u4")
        assert actual == np.float32(math.sqrt(float(x)))
    servers = [_native.PrivatePageServer(data, 256, party) for party in (0, 1)]
    descriptor = servers[0].descriptor()
    lengths = []
    keys = []
    for x in (np.float32(1e-6), np.float32(3), np.float32(3)):
        index, power = numeric.sqrt_index(x)
        a, b, decoder = _native.private_page_issue(*descriptor, index // 64)
        left, right = servers[0].evaluate(a), servers[1].evaluate(b)
        page = decoder.decode(left, right)
        assert numeric.sqrt_restore(page[index % 64 * 4:index % 64 * 4 + 4], power) == np.sqrt(x)
        lengths.append(tuple(map(len, (a, b, left, right))))
        keys.append((a, b))
        with pytest.raises(ValueError, match="consumed"):
            decoder.decode(left, right)
        with pytest.raises(ValueError):
            servers[0].evaluate(a)
    assert lengths[0] == lengths[1] == lengths[2]
    assert keys[1][0] != keys[2][0] and keys[1][1] != keys[2][1]


def test_public_norm_folding_exposes_changed_quantized_numeric_contract():
    rng = np.random.default_rng(771)
    weights = rng.standard_normal((16, 32), dtype=np.float32)
    gamma = rng.uniform(0.1, 3, 32).astype(np.float32)
    x = rng.standard_normal((4, 32), dtype=np.float32)
    result = numeric.folding_probe(weights, gamma, x)
    assert result["w8a8_changed_elements"] > 0
    assert result["w8a8_changed_input_codes"] > 0
    assert len(set(result["original_and_folded_stage_sha256"])) == 2
    # Identity gamma is an admitted control, not an unconditional rejection oracle.
    identity = numeric.folding_probe(weights, np.ones(32, np.float32), x)
    assert identity["w8a8_changed_elements"] == 0
    assert identity["w8a8_changed_input_codes"] == 0


def test_scaled_record_lookup_composes_with_exact_two_worker_head():
    from pllm import _native
    from pllm.native import MaskedGEMM
    from pllm.runtime.quantization import dequantize_matmul, quantize_weight_per_row

    rng = np.random.default_rng(440)
    quantized = quantize_weight_per_row(rng.standard_normal((17, 8), dtype=np.float32), bits=8)
    records = np.empty((17, 12), dtype=np.uint8)
    records[:, :8] = quantized.values.view(np.uint8)
    records[:, 8:] = np.frombuffer(quantized.scales.astype("<f4").tobytes(), np.uint8).reshape(-1, 4)
    servers = [_native.PrivatePageServer(records.tobytes(), 12, i) for i in (0, 1)]
    for index in (0, 8, 16):
        a, b, decoder = _native.private_page_issue(*servers[0].descriptor(), index)
        result = decoder.decode(servers[0].evaluate(a), servers[1].evaluate(b))
        actual = np.frombuffer(result[:8], np.int8).astype(np.float32) * np.frombuffer(result[8:], "<f4")[0]
        assert np.array_equal(actual.view("u4"), quantized.dequantize()[index].view("u4"))
    x = np.asarray([[-127, 127, 0, -1, 1, 32, -32, 100]], np.int8)
    first = rng.integers(0, 1 << 32, x.shape, dtype=np.uint32)
    second = np.subtract(x.astype(np.uint32), first, dtype=np.uint32)
    kernel = MaskedGEMM(threads=1).compile(quantized.values)
    result = np.add(kernel.wrap32(first), kernel.wrap32(second), dtype=np.uint32).view(np.int32)
    assert np.array_equal(result, x.astype(np.int64) @ quantized.values.astype(np.int64).T)
    assert np.array_equal(dequantize_matmul(result, np.float32(0.017), quantized.scales),
                          dequantize_matmul(kernel.clear(x), np.float32(0.017), quantized.scales))
