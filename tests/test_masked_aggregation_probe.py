import numpy as np
import pytest

from pllm import _native
from pllm.metrics import MaskedAggregationProbe


def test_relay_moves_download_to_peer_without_claiming_all_link_saving():
    rng = np.random.default_rng(913)
    a = rng.integers(0, 2**32, size=(7, 31), dtype=np.uint32)
    b = rng.integers(0, 2**32, size=a.shape, dtype=np.uint32)
    report = MaskedAggregationProbe(repetitions=1).run(a, b, bytes(range(2, 33)))
    assert report["exact_integer_parity"]
    assert report["relay_client_download_bytes"] * 2 == report["direct_client_download_bytes"]
    assert report["relay_all_link_minimum_bytes"] > report["direct_all_link_bytes"]
    assert not report["all_link_gate_passed"]


@pytest.mark.parametrize("role", [0, 1, 2])
def test_every_role_burns_before_bad_argument_conversion(role):
    worker, relay, client = _native.output_aggregation_issue(bytes(32), b"\x10", 1)
    fn, args = ((worker.apply, (object(),)), (relay.combine, (object(), b"")),
                (client.finish, (object(),)))[role]
    with pytest.raises(TypeError):
        fn(*args)
    with pytest.raises(ValueError, match="consumed"):
        fn(*args)


def test_cross_issuance_frame_burns_receiver():
    worker, relay, _ = _native.output_aggregation_issue(bytes(32), b"\x10", 1)
    _, _, receiver = _native.output_aggregation_issue(bytes(32), b"\x10", 1)
    frame = relay.combine(bytes(4), worker.apply(bytes(4)))
    with pytest.raises(ValueError, match="context"):
        receiver.finish(frame)
    with pytest.raises(ValueError, match="consumed"):
        receiver.finish(frame)
