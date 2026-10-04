import pytest

from pllm.metrics import PreparedDuplexProbe
from pllm.metrics.prepared_duplex import _finish


def test_native_chunking_keeps_integer_outputs_and_charges_every_frame():
    report = PreparedDuplexProbe(rows=9, repetitions=1, chunk_rows=(1, 4)).run()
    control = report["configurations"][0]
    assert control["chunk_rows"] == 9 and control["chunks"] == 1
    for row in report["configurations"]:
        assert row["exact_integer_output"]
        assert row["request_body_bytes"] + row["response_body_bytes"] >= control["request_body_bytes"] + control["response_body_bytes"]
    assert report["whole_decoder_executable"] is False


def test_duplex_schedule_has_independent_directions_and_one_round_trip():
    assert _finish([(100, 100, 0)], 100, 100, 0.5) == 3
    assert _finish([(50, 50, 0), (50, 50, 0)], 100, 100, 0.5) == 2.5
    assert _finish([(50, 50, 1), (50, 50, 1)], 100, 100, 0.5) == 4


@pytest.mark.parametrize("options", [{"rows": 129}, {"upload_mbps": 0}, {"latency_ms": float("nan")}, {"chunk_rows": (1, 1)}])
def test_duplex_resource_bounds(options):
    with pytest.raises(ValueError):
        PreparedDuplexProbe(**options)
