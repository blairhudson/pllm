"""Installed-wheel exact native API checks. Run serially after rebuilding wheel."""
import numpy as np
import pytest

from pllm import _native
from pllm.runtime.native import MaskedGEMM


@pytest.mark.parametrize("width", [1, 3, 4, 15, 16, 17, 31, 32, 33, 127, 896])
@pytest.mark.parametrize("simd", [False, True])
def test_native_byte_dot_complete_integer_output(width, simd):
    weights = np.resize(np.array([-128, 127, 0, -1, 1], np.int8), (7, width))
    values = np.array([0, 127, 128, 255, 256, 0x7fffffff, 0x80000000,
                       0xffffff80, 0xffffffff], np.uint32)
    inputs = np.resize(values, (1, width))
    matrix = MaskedGEMM(threads=1, simd=simd).compile(weights)
    expected = np.asarray((inputs.astype(object) @ weights.astype(object).T) % 2**32, np.uint32)
    np.testing.assert_array_equal(matrix.wrap32_byte_dot(inputs), expected)
    np.testing.assert_array_equal(matrix.wrap32(inputs), expected)
    np.testing.assert_array_equal(matrix.weight_view(), weights)


def test_native_byte_dot_rejects_malformed_bytes_and_input_rows():
    executor = _native.Executor(1, False)
    matrix = _native.Matrix(bytes([128]) * 17, 1, 17)
    for data in (b"", b"x", bytes(16 * 4), bytes(18 * 4), bytes(34 * 4), bytes(4 * 131072)):
        with pytest.raises(ValueError):
            executor.wrap32_byte_dot(matrix, data)
