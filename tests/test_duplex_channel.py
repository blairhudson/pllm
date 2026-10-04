import queue
import threading

import pytest

from pllm.runtime.client import _Channel


class Socket:
    def __init__(self, fail=False):
        self.frames = queue.Queue()
        self.receiving = threading.Event()
        self.closed = False
        self.fail = fail

    def send(self, payload):
        if payload == b"b":
            assert self.receiving.wait(2), "sender blocked receiver progress"
        if self.fail:
            raise RuntimeError("send failed")
        self.frames.put(payload)

    def recv(self, timeout):
        self.receiving.set()
        result = self.frames.get(timeout=2)
        if result is None:
            raise RuntimeError("socket closed")
        return result

    def close(self):
        self.closed = True
        self.frames.put(None)


def test_duplex_channel_keeps_both_directions_moving_and_reuses_one_sender():
    channel = _Channel(None, "http://localhost", "key", "session", "websocket")
    channel.socket = Socket()
    try:
        assert channel.exchange_many([b"a", b"b", b"c"]) == [b"a", b"b", b"c"]
        pool = channel._send_pool
        assert channel.exchange_many([b"d", b"e"]) == [b"d", b"e"]
        assert channel._send_pool is pool
    finally:
        channel.close()
    assert not any(t.name.startswith("pllm-stage-send") for t in threading.enumerate())


def test_duplex_sender_failure_closes_before_join_and_releases_receiver():
    channel = _Channel(None, "http://localhost", "key", "session", "websocket")
    socket = channel.socket = Socket(fail=True)
    with pytest.raises(RuntimeError, match="closed|failed"):
        channel.exchange_many([b"a", b"b"])
    assert socket.closed and channel._send_pool is None


def test_duplex_has_no_http_retry_or_unbounded_frame_list():
    channel = _Channel(None, "http://localhost", "key", "session", "http")
    with pytest.raises(Exception, match="frame budget"):
        channel.exchange_many([b"one-use"])
    channel.mode = "websocket"
    with pytest.raises(Exception, match="frame budget"):
        channel.exchange_many([b"x"] * 129)


def test_duplex_bounds_total_response_bytes_not_only_individual_frames():
    channel = _Channel(None, "http://localhost", "key", "session", "websocket")
    socket = channel.socket = Socket()
    large = bytes(9 << 20)
    socket.recv = lambda timeout: large
    with pytest.raises(Exception, match="result-body budget"):
        channel.exchange_many([b"a", b"a"])
    assert socket.closed and channel._send_pool is None
