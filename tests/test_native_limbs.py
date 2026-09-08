import numpy as np

from pllm.runtime.native import MaskedGEMM


def test_limb12_kernel_matches_modular_reference_across_shapes():
    rng = np.random.default_rng(808)
    kernel = MaskedGEMM(threads=2)
    for in_features, out_features, batch, modulus in [
        (1536, 257, 1, 786433),
        (1536, 257, 16, 786433),
        (12288, 73, 4, 2056193),
        (5003, 19, 3, 786433),
    ]:
        weight = rng.integers(-7, 8, size=(out_features, in_features), dtype=np.int8)
        values = rng.integers(0, modulus, size=(batch, in_features), dtype=np.uint32)
        expected = (values.astype(np.int64) @ weight.astype(np.int64).T) % modulus
        actual = kernel.modular_limb12(weight, values, modulus)
        assert np.array_equal(actual, expected.astype(np.uint32))


def test_auto_kernel_is_exact_for_large_gemma_shapes():
    rng = np.random.default_rng(809)
    kernel = MaskedGEMM(threads=2)
    weight = rng.integers(-7, 8, size=(1024, 12288), dtype=np.int8)
    values = rng.integers(0, 2056193, size=(8, 12288), dtype=np.uint32)
    expected = kernel.modular_limb12(weight, values, 2056193)
    actual = kernel.modular(weight, values, 2056193)
    assert np.array_equal(actual, expected)
