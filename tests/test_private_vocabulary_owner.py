"""One immutable native matrix serves exact head products and private lookup."""
import hashlib

import numpy as np
import pytest

from pllm import _native
from pllm.native import MaskedGEMM


def test_matrix_alias_lookup_context_and_retirement():
    values = np.random.default_rng(181).integers(-128, 128, (129, 19), dtype=np.int8)
    scales = np.linspace(0.01, 0.15, len(values), dtype="<f4")
    engine = MaskedGEMM(threads=1, build=False)
    assert engine.native
    matrix = engine.compile(values)
    view = matrix.weight_view()
    assert not view.flags.writeable
    roots = [_native.PrivatePageServer.from_matrix(matrix._matrix, scales.tobytes(), p) for p in (0, 1)]
    records = b"".join(row.tobytes() + scale.tobytes() for row, scale in zip(values, scales, strict=True))
    raw = _native.PrivatePageServer(records, 23, 0)
    assert roots[0].descriptor() == roots[1].descriptor() == raw.descriptor()
    context = hashlib.sha256(b"source-plan-session-epoch-fixture").digest()
    servers = [root.bind_context(context) for root in roots]
    digest = hashlib.sha256(b"pllm/private-page-context/v1\0" + raw.descriptor()[0] + context).digest()
    assert servers[0].descriptor() == servers[1].descriptor() == (digest, 129, 23)
    with pytest.raises(ValueError, match="context replayed"):
        roots[0].bind_context(context)
    wrong = roots[0].bind_context(bytes(32))
    # The shared immutable storage retains ordinary head behavior.
    x = np.random.default_rng(93).integers(0, 2**32, (1, 19), dtype=np.uint32)
    oracle = np.asarray((x.astype(object) @ values.astype(object).T) % (1 << 32), dtype=np.uint32)
    np.testing.assert_array_equal(matrix.wrap32(x), oracle)
    del values, scales, matrix, engine, roots, raw
    # Lookup children and NumPy view each outlive the original matrix wrapper.
    np.testing.assert_array_equal(view[0], np.frombuffer(records[:19], np.int8))
    queries = [_native.private_page_issue(*servers[0].descriptor(), index, leaf_bits=128) for index in (0, 127, 128)]
    with pytest.raises(ValueError, match="source"):
        wrong.evaluate(queries[0][0])
    replies = [server.evaluate_batch([q[p] for q in queries]) for p, server in enumerate(servers)]
    for i, (index, query) in enumerate(zip((0, 127, 128), queries, strict=True)):
        assert query[2].decode(replies[0][i], replies[1][i]) == records[index * 23:(index + 1) * 23]
    with pytest.raises(ValueError, match="replay"):
        servers[0].evaluate(queries[0][0])


@pytest.mark.parametrize("scales", [b"", b"abc", np.float32(0).tobytes(), np.float32(-1).tobytes(), np.float32(np.nan).tobytes()])
def test_vocab_scales_and_context_reject(scales):
    matrix = _native.Matrix(bytes(19), 1, 19)
    with pytest.raises(ValueError, match="scales"):
        _native.PrivatePageServer.from_matrix(matrix, scales, 0)
    server = _native.PrivatePageServer.from_matrix(matrix, np.float32(1).tobytes(), 0)
    with pytest.raises(ValueError, match="32 bytes"):
        server.bind_context(bytes(31))
    # Rejected malformed context does not consume an unrelated valid context.
    server.bind_context(bytes(32))
