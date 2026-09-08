import numpy as np

from pllm.runtime.native import MaskedGEMM
from pllm.runtime.quantization import (
    centered_residues,
    choose_plain_modulus,
    choose_wire_bits,
    dequantize_matmul,
    quantize_activation_per_row,
    quantize_weight_per_row,
)
from pllm.runtime.stage_protocol import pack_residues, unpack_residues


def test_native_prime_modular_gemm_matches_reference():
    rng = np.random.default_rng(7)
    weight = rng.integers(-7, 8, size=(37, 51), dtype=np.int8)
    inputs = rng.integers(0, 65537, size=(6, 51), dtype=np.uint32)
    kernel = MaskedGEMM()
    actual = kernel.modular(weight, inputs, 65537)
    expected = (inputs.astype(np.int64) @ weight.astype(np.int64).T) % 65537
    assert np.array_equal(actual, expected)


def test_w4a4_quantized_matmul_round_trip():
    rng = np.random.default_rng(11)
    weight = rng.normal(size=(23, 19)).astype(np.float32)
    activation = rng.normal(size=(4, 19)).astype(np.float32)
    qw = quantize_weight_per_row(weight)
    qa = quantize_activation_per_row(activation)
    accum = qa.values.astype(np.int32) @ qw.values.astype(np.int32).T
    output = dequantize_matmul(accum, qa.scales, qw.scales, output_shape=(4, 23))
    reference = activation @ weight.T
    assert np.mean((output - reference) ** 2) < 0.25


def test_stage_modulus_and_u24_codec():
    modulus = choose_plain_modulus(1536)
    assert modulus == 786433
    assert choose_plain_modulus(11008, weight_bits=8, activation_bits=8) == 536_690_689
    assert choose_wire_bits(modulus) == 24
    values = np.array([[0, 1, modulus - 1, 700000]], dtype=np.uint32)
    payload = pack_residues(values, 24)
    assert len(payload) == values.size * 3
    assert np.array_equal(unpack_residues(payload, values.shape, 24), values)
    assert np.array_equal(centered_residues(np.array([modulus - 1]), modulus), np.array([-1]))


def test_native_prime_modular_gemm_matches_reference_across_service_batches():
    rng = np.random.default_rng(7001)
    kernel = MaskedGEMM(threads=5)
    for modulus in (40_961, 65_537, 786_433, 6_537_217):
        for batch in (1, 3, 4, 16, 32):
            weight = rng.integers(-7, 8, size=(73, 129), dtype=np.int8)
            inputs = rng.integers(0, modulus, size=(batch, 129), dtype=np.uint32)
            actual = kernel.modular(weight, inputs, modulus)
            expected = (inputs.astype(np.int64) @ weight.astype(np.int64).T) % modulus
            assert np.array_equal(actual, expected)


def test_native_wrap32_matches_exact_ring_reference():
    rng = np.random.default_rng(7002)
    kernel = MaskedGEMM(threads=5)
    weight = rng.integers(-7, 8, size=(131, 257), dtype=np.int8)
    inputs = rng.integers(0, 2**32, size=(16, 257), dtype=np.uint32)
    actual = kernel.wrap32(weight, inputs)
    expected = (inputs.astype(np.int64) @ weight.astype(np.int64).T).astype(np.uint32)
    assert np.array_equal(actual, expected)
