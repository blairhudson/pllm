import hashlib
import struct
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from pllm.native import MaskedGEMM, PagedGEMM


@pytest.mark.parametrize("compression", ["raw", "zlib"])
@pytest.mark.parametrize("simd", [False, True])
def test_paged_exact_operations_and_local_gather(tmp_path, compression, simd):
    rng = np.random.default_rng(721)
    weights = rng.integers(-128, 128, (137, 67), dtype=np.int8)
    weights[16:32] = 0
    path = tmp_path / "weights.pllm"
    digest = PagedGEMM.export(weights, path, page_rows=16, compression=compression)
    reference = MaskedGEMM(threads=1, simd=simd).compile(weights)
    signed = rng.integers(-128, 128, (3, 67), dtype=np.int8)
    masked = rng.integers(0, 2**32, (3, 67), dtype=np.uint32)
    with PagedGEMM(path, digest, threads=1, simd=simd) as matrix:
        assert matrix.weight_digest == hashlib.sha256(weights.tobytes()).hexdigest()
        assert matrix.artifact_bytes == path.stat().st_size
        assert matrix.transient_weight_bytes <= 3 * 16 * 67 + 1
        assert matrix.metadata_bytes < weights.nbytes
        np.testing.assert_array_equal(
            matrix.clear(signed), signed.astype(np.int64) @ weights.astype(np.int64).T
        )
        np.testing.assert_array_equal(matrix.wrap32(masked), reference.wrap32(masked))
        np.testing.assert_array_equal(
            matrix.modular(masked % 65521, 65521), reference.modular(masked % 65521, 65521)
        )
        np.testing.assert_array_equal(
            matrix.gather_rows([136, 16, 0, 136]), weights[[136, 16, 0, 136]]
        )
        # Source mutation cannot affect the admitted private snapshot.
        path.write_bytes(b"changed after snapshot")
        with ThreadPoolExecutor(max_workers=3) as pool:
            results = list(pool.map(lambda _: matrix.clear(signed), range(3)))
        for result in results:
            np.testing.assert_array_equal(result, reference.clear(signed))
        with pytest.raises(AttributeError, match="immutable"):
            matrix._matrix = None
        with pytest.raises(ValueError):
            matrix.gather_rows([-1])
        with pytest.raises(ValueError):
            matrix.gather_rows([137])
        with pytest.raises(ValueError):
            matrix.clear(np.zeros((1, 66), np.int8))
        with pytest.raises(ValueError):
            matrix.wrap32(np.full((1, 67), -1))
    with pytest.raises(ValueError, match="closed"):
        matrix.clear(signed)
    matrix.close()


@pytest.mark.parametrize(
    "mutation",
    ["reserved", "count", "offset", "codec", "decoded", "trailing", "weight_hash", "zlib"],
)
def test_paged_rejects_malformed_committed_artifacts(tmp_path, mutation):
    path = tmp_path / "weights"
    PagedGEMM.export(np.ones((64, 64), np.int8), path, page_rows=16, compression="zlib")
    data = bytearray(path.read_bytes())
    if mutation == "reserved":
        data[56] = 1
    elif mutation == "count":
        struct.pack_into("<I", data, 20, 65537)
    elif mutation == "offset":
        struct.pack_into("<Q", data, 64, 0)
    elif mutation == "codec":
        data[80] = 2
    elif mutation == "decoded":
        struct.pack_into("<I", data, 76, 2**30)
    elif mutation == "trailing":
        data.append(0)
    elif mutation == "weight_hash":
        data[24] ^= 1
    elif mutation == "zlib":
        data[-1] ^= 0x80
    path.write_bytes(data)
    # Even a caller committing the malformed bytes cannot bypass structure,
    # decompression, decoded-weight identity or resource admission.
    with pytest.raises(ValueError):
        PagedGEMM(path, hashlib.sha256(data).hexdigest())


def test_paged_preserves_existing_files_and_digest_admission(tmp_path):
    path = tmp_path / "weights"
    weights = np.zeros((16, 16), np.int8)
    digest = PagedGEMM.export(weights, path)
    original = path.read_bytes()
    with pytest.raises(ValueError):
        PagedGEMM.export(weights, path)
    assert path.read_bytes() == original
    with pytest.raises(ValueError, match="digest"):
        PagedGEMM(path, "0" * 64)
    with pytest.raises(ValueError, match="canonical"):
        PagedGEMM(path, digest.upper())
    with pytest.raises(ValueError):
        PagedGEMM.export(weights, tmp_path / "bad", page_rows=2**30)
    assert not (tmp_path / "bad").exists()
    with pytest.raises(ValueError):
        PagedGEMM(path, digest, threads=True)


def test_paged_is_native_required_and_not_pickleable(tmp_path, monkeypatch):
    import pickle
    import pllm.runtime.paged as implementation

    path = tmp_path / "weights"
    digest = PagedGEMM.export(np.ones((16, 16), np.int8), path)
    with PagedGEMM(path, digest) as matrix:
        with pytest.raises(TypeError):
            pickle.dumps(matrix)
    monkeypatch.setattr(implementation, "extension", lambda: None)
    with pytest.raises(ValueError, match="requires the native"):
        PagedGEMM(path, digest)


def test_paged_wide_low_magnitude_clear_preserves_resident_admission(tmp_path):
    weights = np.ones((1, 140000), dtype=np.int8)
    path = tmp_path / "wide"
    digest = PagedGEMM.export(weights, path, page_rows=1)
    with PagedGEMM(path, digest) as matrix:
        result = matrix.clear(np.full((1, 140000), -128, dtype=np.int8))
        assert int(result[0, 0]) == -128 * 140000


def test_paged_relative_export_and_nonfile_rejection(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    digest = PagedGEMM.export(np.eye(2, dtype=np.int8), "local.pllm")
    with PagedGEMM("local.pllm", digest) as matrix:
        np.testing.assert_array_equal(matrix.clear(np.ones((1, 2), np.int8)), [[1, 1]])
        with pytest.raises(AttributeError):
            matrix.shape = (1, 4)
    with pytest.raises(ValueError, match="regular file"):
        PagedGEMM(tmp_path, "0" * 64)
