import hashlib
import json

import numpy as np
import pytest

from pllm.runtime.benchmark_sealed_state import StreamingSealedStateFixture
from pllm.runtime.benchmark_state_targets import SealedStateFixture, _consume


@pytest.mark.parametrize("tokens", [3, 4097])
def test_streaming_gcm_interoperates_and_preserves_attention(tmp_path, tokens):
    layout = [{"layer": 0, "kind": kind, "shape": [1, 2, tokens, 64]} for kind in ("key", "value")]
    key = b"q" * 32
    values = [np.random.default_rng(i).standard_normal(row["shape"], dtype=np.float32)
              for i, row in enumerate(layout)]
    for issuer, reader in ((StreamingSealedStateFixture, SealedStateFixture),
                           (SealedStateFixture, StreamingSealedStateFixture)):
        root = tmp_path / issuer.__name__
        store = issuer(root, key, "source", layout)
        head = store.commit(values, 0)
        restored = reader(root, key, "source", layout)
        for actual, expected in zip(restored.read(head), values, strict=True):
            np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
        assert _consume(restored.read(head), True) == _consume(values, True)
        with pytest.raises(ValueError, match="injected"):
            store.commit([v + 1 for v in values], 1, fail_after=0)
        assert _consume(restored.read(head)) == _consume(values)
        assert not list(root.glob("pending-*"))


def test_streaming_gcm_authenticates_before_exposing_plaintext(tmp_path):
    from cryptography.exceptions import InvalidTag
    layout = [{"layer": 0, "kind": "key", "shape": [1, 1, 3, 4]}]
    store = StreamingSealedStateFixture(tmp_path, b"k" * 32, "source", layout)
    head = store.commit([np.ones((1, 1, 3, 4), np.float32)], 0)
    path = tmp_path / "0" / "0"
    damaged = bytearray(path.read_bytes())
    damaged[13] ^= 1
    path.write_bytes(damaged)
    # Even a caller supplying a new trusted hash cannot forge the AEAD tag.
    manifest = json.loads((tmp_path / "0" / "manifest").read_bytes())
    manifest["hashes"][0] = hashlib.sha256(damaged).hexdigest()
    raw = json.dumps(manifest).encode()
    (tmp_path / "0" / "manifest").write_bytes(raw)
    with pytest.raises(InvalidTag):
        next(store.read({"revision": 0, "manifest": hashlib.sha256(raw).hexdigest()}))
    path.write_bytes(damaged[:-1])
    with pytest.raises(ValueError, match="frame"):
        next(store.read({"revision": 0, "manifest": hashlib.sha256(raw).hexdigest()}))
