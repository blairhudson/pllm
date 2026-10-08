"""Response bodies retire independently of Python's cyclic-GC schedule."""
import gc
import weakref

import httpx
import pytest

from pllm.runtime.http_response import owned_response


@pytest.mark.parametrize("streamed", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_response_and_request_retire_without_gc(streamed, fail):
    refs = []

    def consume(client):
        request = client.build_request("POST", "http://local/stage", content=b"q" * 65536)
        response = client.send(request, stream=streamed)
        refs.append((weakref.ref(response), weakref.ref(request)))
        with owned_response(response):
            if fail:
                raise ValueError("stage failed")
            body = response.read()
            assert response.status_code == 200
            assert body == b"r" * 65536
        # Closing/detaching does not invalidate content or response metadata.
        assert response.content == body
        assert response.elapsed.total_seconds() >= 0
        return body

    was_enabled = gc.isenabled()
    gc.disable()
    try:
        with httpx.Client(transport=httpx.MockTransport(
            lambda request: httpx.Response(200, stream=httpx.ByteStream(b"r" * 65536))
        )) as client:
            for _ in range(16):
                if fail:
                    with pytest.raises(ValueError, match="stage failed"):
                        consume(client)
                else:
                    assert consume(client) == b"r" * 65536
        assert all(response() is None and request() is None for response, request in refs)
    finally:
        if was_enabled:
            gc.enable()


def test_real_stream_is_closed_once_even_on_interruption():
    class Stream(httpx.SyncByteStream):
        closes = 0

        def __iter__(self):
            yield b"partial"
            raise AssertionError("must not read the remainder on cancellation")

        def close(self):
            self.closes += 1

    stream = Stream()
    with httpx.Client(transport=httpx.MockTransport(
        lambda request: httpx.Response(200, stream=stream)
    )) as client:
        response = client.send(client.build_request("GET", "http://local/stage"), stream=True)
        with pytest.raises(KeyboardInterrupt):
            with owned_response(response):
                assert next(response.iter_raw()) == b"partial"
                raise KeyboardInterrupt()
        assert response.is_closed
        response.close()
        assert stream.closes == 1
