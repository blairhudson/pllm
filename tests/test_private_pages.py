import copy
import json

import numpy as np
import pytest

from pllm import _native
from pllm.metrics import PrivatePageLookupProbe


@pytest.mark.parametrize("records,width", [(1, 1), (3, 900), (33, 67), (128, 32)])
def test_native_public_pages_match_independent_bytes(records, width):
    data = np.random.default_rng(417).integers(0, 256, records * width, dtype=np.uint8).tobytes()
    a, b = (_native.PrivatePageServer(data, width, party) for party in (0, 1))
    lengths = set()
    for index in range(records):
        ka, kb, decoder = _native.private_page_issue(*a.descriptor(), index)
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
