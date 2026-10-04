import struct

import numpy as np
import pytest

from pllm import _native
from pllm.metrics import ArtifactEntropyProbe


def decode_reference(frame):
    """Independent Python integer oracle; absent from measured native execution."""
    assert frame[:8] == b"PLLRANS1"
    size = int.from_bytes(frame[8:12], "little")
    if frame[12] == 0:
        return frame[13:]
    frequencies = struct.unpack("<256H", frame[13:525])
    lookup = []
    for symbol, frequency in enumerate(frequencies):
        lookup.extend([(symbol, frequency, len(lookup))] * frequency)
    states = list(struct.unpack("<4I", frame[525:541]))
    cursor, output = 541, bytearray()
    for index in range(size):
        lane = index % 4
        state = states[lane]
        symbol, frequency, start = lookup[state % 4096]
        state = frequency * (state // 4096) + state % 4096 - start
        while state < 8388608:
            state = state * 256 + frame[cursor]
            cursor += 1
        output.append(symbol)
        states[lane] = state
    assert cursor == len(frame) and states == [8388608] * 4
    return bytes(output)


@pytest.mark.parametrize("kind", ["constant", "gaussian", "uniform", "rare"])
def test_entropy_matches_independent_oracle_and_raw_hash(kind):
    rng = np.random.default_rng(819)
    payload = {"constant": b"q" * 4099,
               "gaussian": np.clip(np.rint(rng.normal(0, 19, 32771)), -127, 127).astype(np.int8).tobytes(),
               "uniform": rng.integers(0, 256, 16385, dtype=np.uint8).tobytes(),
               "rare": b"\0" * 65537 + bytes(range(256))}[kind]
    frame = _native.public_entropy_encode(payload)
    assert _native.public_entropy_decode(frame, len(payload)) == decode_reference(frame) == payload
    with pytest.raises(ValueError):
        _native.public_entropy_decode(frame + b"\0", len(payload))


def test_entropy_probe_reports_exact_bounded_codec_costs():
    result = ArtifactEntropyProbe(repetitions=1).run(b"x" * 100000)
    assert result["schema"] == "pllm.artifact_entropy_probe.v1"
    assert all(row["exact_raw_hash"] for row in result["configurations"])
    assert {row["encoding"] for row in result["configurations"]} == {"rans", "zlib"}
    assert result["peak_client_memory_bytes"] is None
