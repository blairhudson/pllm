"""Metal integer-stage parity against the shipping native kernel."""

import importlib.util
import platform

import numpy as np
import pytest

from pllm.native import MaskedGEMM, MetalGEMM
from pllm.runtime.native import NativeKernelError


pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin"
    or platform.machine() != "arm64"
    or importlib.util.find_spec("mlx") is None,
    reason="Apple Silicon and pllm.run[metal] are required",
)


@pytest.mark.parametrize("shape,batch", [((3, 7), 0), ((3, 7), 1), ((129, 257), 13), ((1024, 1024), 2)])
def test_metal_integer_stage_matches_neon_and_owns_weights(shape, batch):
    rng = np.random.default_rng(51)
    weights = rng.integers(-128, 128, size=shape, dtype=np.int8)
    old = weights.copy()
    metal = MetalGEMM().compile(weights)
    neon = MaskedGEMM().compile(old)
    weights.fill(0)

    clear = rng.integers(-128, 128, size=(batch, shape[1]), dtype=np.int8)
    ring = rng.integers(0, 2**32, size=(batch, shape[1]), dtype=np.uint32)
    if batch:
        ring[0, 0] = 2**32 - 1
        clear[0, 0] = -128
    assert np.array_equal(metal.clear(clear), neon.clear(clear))
    assert np.array_equal(metal.wrap32(ring), neon.wrap32(ring))
    for modulus in (1 << 16, 1 << 24):
        reduced = ring % modulus
        assert np.array_equal(metal.modular(reduced, modulus), neon.modular(reduced, modulus))


def test_metal_rejects_unsupported_domains_before_gpu_dispatch():
    metal = MetalGEMM().compile(np.ones((2, 3), dtype=np.int8))
    with pytest.raises(NativeKernelError, match="expected weights"):
        metal.wrap32(np.ones((2, 4), dtype=np.uint32))
    with pytest.raises(NativeKernelError, match="values"):
        metal.clear(np.array([[0, 256, 0]]))
    with pytest.raises(NativeKernelError, match="declared u16/u24/u32 ring"):
        metal.modular(np.ones((2, 3), dtype=np.uint32), 2_013_265_921)
    with pytest.raises(NativeKernelError, match="512 MiB"):
        metal.wrap32(np.broadcast_to(np.array([1], dtype=np.uint32), (70_000_000, 3)))
