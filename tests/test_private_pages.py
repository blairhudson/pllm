import copy
import json

import numpy as np
import pytest

from pllm import _native
from pllm.metrics import PrivatePageLookupProbe


@pytest.mark.parametrize("leaf_bits", [1, 128])
@pytest.mark.parametrize("records,width", [(1, 1), (3, 900), (33, 67), (128, 32), (129, 17), (257, 9)])
def test_native_public_pages_match_independent_bytes(records, width, leaf_bits):
    data = np.random.default_rng(417).integers(0, 256, records * width, dtype=np.uint8).tobytes()
    a, b = (_native.PrivatePageServer(data, width, party) for party in (0, 1))
    lengths = set()
    for index in range(records):
        ka, kb, decoder = _native.private_page_issue(*a.descriptor(), index, leaf_bits=leaf_bits)
        ra, rb = a.evaluate(ka), b.evaluate(kb)
        assert decoder.decode(ra, rb) == data[index * width : (index + 1) * width]
        lengths.add((len(ka), len(kb), len(ra), len(rb)))
        with pytest.raises(ValueError, match="consumed"):
            decoder.decode(ra, rb)
        with pytest.raises(ValueError, match="replay"):
            a.evaluate(ka)
    assert len(lengths) == 1


def test_decoder_burns_before_python_conversion_and_cross_query_mix():
    a, b = (_native.PrivatePageServer(bytes(range(120)), 10, p) for p in (0, 1))
    ka, kb, decoder = _native.private_page_issue(*a.descriptor(), 3)
    ra, rb = a.evaluate(ka), b.evaluate(kb)
    with pytest.raises(TypeError):
        decoder.decode(object(), rb)
    with pytest.raises(ValueError, match="consumed"):
        decoder.decode(ra, rb)
    ka, kb, decoder = _native.private_page_issue(*a.descriptor(), 4)
    fresh = a.evaluate(ka)
    with pytest.raises(ValueError, match="binding"):
        decoder.decode(fresh, rb)
    with pytest.raises(ValueError, match="consumed"):
        decoder.decode(fresh, b.evaluate(kb))


def test_source_party_cancellation_and_opaque_handles():
    a = _native.PrivatePageServer(bytes(100), 10, 0)
    other = _native.PrivatePageServer(bytes([1]) * 100, 10, 0)
    ka, kb, decoder = _native.private_page_issue(*a.descriptor(), 2)
    with pytest.raises(ValueError, match="party"):
        a.evaluate(kb)
    with pytest.raises(ValueError, match="source"):
        other.evaluate(ka)
    with pytest.raises(TypeError):
        copy.deepcopy(a)
    with pytest.raises(TypeError):
        copy.copy(decoder)
    a.cancel(ka)
    with pytest.raises(ValueError, match="replay"):
        a.evaluate(ka)
    decoder.cancel()
    with pytest.raises(ValueError, match="consumed"):
        decoder.decode(b"", b"")


@pytest.mark.parametrize("pages", [1, 2, 8, 32])
def test_sdk_page_packing_and_sanitized_evidence(pages):
    probe = PrivatePageLookupProbe(records=33, record_bytes=16, page_records=pages)
    report = probe.run()
    assert report["exact_byte_parity"]
    assert report["client_online_table_bytes_required"] == 0
    assert report["public_table_scan_bytes_per_query_both_workers"] >= 33 * 16 * 2
    assert report["whole_decoder_executable"] is False
    assert report["peak_client_memory_bytes"] is None
    serialized = json.dumps(report)
    assert "query_id" not in serialized and "token_ids" not in serialized


@pytest.mark.parametrize(
    "params",
    [
        {"records": True},
        {"records": 0},
        {"page_records": 3},
        {"records": 262144, "record_bytes": 65536},
        {"record_bytes": 65536, "page_records": 2},
    ],
)
def test_probe_preflights_allocation(params):
    with pytest.raises(ValueError):
        PrivatePageLookupProbe(**params)


def _point_share(key, index):
    """Independent point evaluator using OpenSSL AES, not native tree traversal."""
    import hashlib
    import struct
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

    magic, party, records, width = struct.unpack("<8sBII", key[:17])
    packed = magic == b"PLLMPQ02"
    assert magic in (b"PLLMPQ01", b"PLLMPQ02")
    group_bits = 128 if packed else 1
    depth = ((records + group_bits - 1) // group_bits - 1).bit_length()
    domain = hashlib.sha256(
        (b"pllm/private-page-dpf/aes128/v2\0" if packed else b"pllm/private-page-dpf/aes128/v1\0")
        + key[17:49] + key[49:65] + struct.pack("<QQ", records, width)
    ).digest()[:16]
    seed, control = key[65:81], party

    def aes(seed, blocks):
        encryptor = Cipher(algorithms.AES(seed), modes.ECB()).encryptor()
        return encryptor.update(blocks) + encryptor.finalize()

    for level in range(depth):
        blocks = []
        for i in range(3):
            block = bytearray(domain)
            block[0] ^= 0xa1
            block[1] ^= level
            block[2] ^= i
            blocks.append(block)
        value = aes(seed, b"".join(blocks))
        direction = ((index // group_bits) >> (depth - level - 1)) & 1
        next_seed = value[direction * 16:(direction + 1) * 16]
        next_control = (value[32] >> direction) & 1
        if control:
            offset = 81 + level * 17
            next_seed = bytes(a ^ b for a, b in zip(next_seed, key[offset:offset + 16], strict=True))
            next_control ^= (key[offset + 16] >> direction) & 1
        seed, control = next_seed, next_control
    block = bytearray(domain)
    block[0] ^= 0xb2
    value = aes(seed, bytes(block))
    bit = index % group_bits
    correction = key[81 + 17 * depth:]
    return ((value[bit // 8] ^ (correction[bit // 8] if control else 0)) >> (bit % 8)) & 1


@pytest.mark.parametrize("leaf_bits", [1, 128])
@pytest.mark.parametrize("records", [1, 127, 128, 129, 257])
def test_openssl_point_oracle_and_native_all_selection_bits(records, leaf_bits):
    # A complete identity table reveals every selector bit in each reply share.
    table = np.eye(records, dtype=np.uint8).tobytes()
    a, b = (_native.PrivatePageServer(table, records, p) for p in (0, 1))
    indices = sorted({0, records // 2, records - 1})
    for index in indices:
        ka, kb, decoder = _native.private_page_issue(*a.descriptor(), index, leaf_bits=leaf_bits)
        replies = [a.evaluate(ka), b.evaluate(kb)]
        for key, reply in zip((ka, kb), replies, strict=True):
            assert reply[65:] == bytes(_point_share(key, row) for row in range(records))
        assert decoder.decode(*replies) == table[index * records:(index + 1) * records]


def test_packed_layout_admission_namespace_and_failure_retirement():
    a, b = (_native.PrivatePageServer(bytes(range(255)) * 9, 9, p) for p in (0, 1))
    for leaf_bits in (0, 2, 64, 129):
        with pytest.raises(ValueError, match="leaf_bits"):
            _native.private_page_issue(*a.descriptor(), 0, leaf_bits=leaf_bits)
    ka, kb, decoder = _native.private_page_issue(*a.descriptor(), 128, leaf_bits=128)
    assert len(ka) == len(kb) == 114  # 65 header + 16 root + 17 correction + 16 final.
    with pytest.raises(ValueError):
        a.evaluate(b"PLLMPQ01" + ka[8:])
    ra, rb = a.evaluate(ka), b.evaluate(kb)
    with pytest.raises(ValueError, match="binding"):
        decoder.decode(b"PLLMPR01" + ra[8:], rb)
    with pytest.raises(ValueError, match="consumed"):
        decoder.decode(ra, rb)
    ka, kb, decoder = _native.private_page_issue(*a.descriptor(), 254, leaf_bits=128)
    ra, rb = a.evaluate(ka), b.evaluate(kb)
    with pytest.raises(TypeError):
        decoder.decode(object(), rb)
    with pytest.raises(ValueError, match="consumed"):
        decoder.decode(ra, rb)
