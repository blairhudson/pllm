from __future__ import annotations

import gc
import hashlib

import msgpack
import numpy as np
import pytest

from pllm.runtime.bundle_document import BundleDocument, CHUNK_BYTES
from pllm.runtime.bundle_compression import encode_bundle_frames
from pllm.runtime.native import MaskedGEMM


@pytest.mark.parametrize("size", [0, 255, 256, 65535, 65536, 2 * CHUNK_BYTES + 19])
def test_canonical_messagepack_and_fixed_compressed_frames(size):
    value = {"v": 2, "metadata": [None, True, -33, 2**40, 0.125, "λ"],
             "data": bytes(index % 251 for index in range(size)), "last": b"after"}
    expected = msgpack.packb(value, use_bin_type=True)
    document = BundleDocument(value)
    assert document.to_bytes() == expected
    assert b"".join(document.chunks()) == expected
    assert document.descriptor["sha256"] == hashlib.sha256(expected).hexdigest()
    assert document.descriptor["size"] == len(expected)
    assert list(encode_bundle_frames(document)) == list(encode_bundle_frames(expected))
    assert all(len(chunk) == CHUNK_BYTES for chunk in list(document.chunks())[:-1])


def test_native_owner_shared_and_pinned_after_kernel_release():
    matrix = MaskedGEMM(threads=1).compile(np.arange(20, dtype=np.int8).reshape(4, 5))
    source = matrix.weight_view()
    document = BundleDocument({"v": 2, "data": memoryview(source), "shape": [4, 5]})
    assert np.shares_memory(source, np.frombuffer(document._value["data"], dtype=np.int8))
    expected = document.to_bytes()
    del matrix, source
    gc.collect()
    assert document.to_bytes() == expected


def test_readonly_facade_over_mutable_storage_is_snapshotted():
    array = np.arange(4, dtype=np.int8)
    value = {"v": 2, "data": memoryview(array).toreadonly(), "shape": [4]}
    document = BundleDocument(value)
    expected = document.to_bytes()
    array[:] = 0
    value["shape"].append(9)
    assert document.to_bytes() == expected
    assert document._value["data"].tobytes() == bytes(range(4))
    assert document._value["shape"] == [4]


def test_mutable_buffers_and_cyclic_documents_rejected():
    with pytest.raises(ValueError, match="immutable"):
        BundleDocument({"v": 2, "data": memoryview(bytearray(4))})
    value = {"v": 2}
    value["cycle"] = value
    with pytest.raises(ValueError, match="node/depth"):
        BundleDocument(value)
