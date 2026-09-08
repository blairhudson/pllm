import json
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

from pllm.runtime.safetensors_store import SafeTensorStore


def _pack_uint4(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.uint32)
    assert values.ndim == 2 and values.shape[1] % 8 == 0
    groups = values.reshape(values.shape[0], -1, 8)
    shifts = (np.arange(8, dtype=np.uint32) * 4).reshape(1, 1, 8)
    return np.sum(groups << shifts, axis=-1, dtype=np.uint32)


def test_native_mlx_affine_int4_dequantization(tmp_path: Path):
    root = tmp_path / "mlx"
    root.mkdir()
    (root / "config.json").write_text(json.dumps({
        "model_type": "llama",
        "quantization": {"bits": 4, "group_size": 4},
    }))
    q = np.asarray([
        [0, 1, 2, 3, 4, 5, 6, 7],
        [8, 9, 10, 11, 12, 13, 14, 15],
    ], dtype=np.uint32)
    scales = np.asarray([[0.5, 0.25], [0.125, 0.0625]], dtype=np.float32)
    biases = np.asarray([[-1.0, -0.5], [-0.25, -0.125]], dtype=np.float32)
    save_file({
        "layers.0.proj.weight": _pack_uint4(q),
        "layers.0.proj.scales": scales,
        "layers.0.proj.biases": biases,
    }, root / "model.safetensors")

    actual = SafeTensorStore(root).get_linear(("layers.0.proj.weight",))
    expected = q.astype(np.float32) * np.repeat(scales, 4, axis=1) + np.repeat(biases, 4, axis=1)
    assert np.array_equal(actual, expected)


def test_native_mlx_symmetric_int4_implicit_midpoint(tmp_path: Path):
    root = tmp_path / "mlx"
    root.mkdir()
    (root / "config.json").write_text(json.dumps({
        "quantization": {"bits": 4, "group_size": 8},
    }))
    q = np.arange(16, dtype=np.uint32).reshape(2, 8)
    scales = np.asarray([[0.5], [0.25]], dtype=np.float32)
    save_file({
        "proj.weight": _pack_uint4(q),
        "proj.scales": scales,
    }, root / "model.safetensors")
    actual = SafeTensorStore(root).get_linear(("proj.weight",))
    expected = (q.astype(np.float32) - 8.0) * scales
    assert np.array_equal(actual, expected)
