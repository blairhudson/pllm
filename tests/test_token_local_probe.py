import numpy as np
import pytest

from pllm import _native
from pllm.metrics import TokenLocalProjectionProbe
from pllm.metrics.token_local import _token_local_stages
from test_state_reuse_probes import fixture


@pytest.mark.parametrize("family", ["qwen2", "qwen3"])
def test_token_local_discovery_and_memo_preserve_whole_decoder(tmp_path, family):
    compiled, remote, *_ = fixture(tmp_path / family, family)
    selected = _token_local_stages(compiled)
    assert len(selected) == 1
    assert selected[0].layer_index == 0
    report = TokenLocalProjectionProbe(cache_bytes=8192, decode_steps=3).run(
        compiled, remote, remote.weight_matrices, [[3, 8, 3, 7], [5, 8, 7]])
    assert report["exact_logits_and_all_kv"] == [True, True]
    assert report["cache_hits"] >= 2
    assert report["accounted_cache_bytes"] <= 8192
    assert not report["conditional_remote_skip_privacy_admitted"]
    assert not report["tenfold_body_gate_passed"]
    weights = dict(remote.weight_matrices)
    weights[selected[0].stage_id] = weights[selected[0].stage_id].copy()
    weights[selected[0].stage_id][0, 0] ^= np.int8(1)
    with pytest.raises(ValueError, match="commitment"):
        TokenLocalProjectionProbe().run(compiled, remote, weights, [[3]])


def test_native_memo_is_matrix_bound_and_eviction_does_not_change_outputs():
    weights = np.array([[1, 2, -1], [2, 3, 4]], dtype=np.int8)
    memo = _native._IntegerRowMemo(weights.tobytes(), 2, 3, 267)
    inputs = np.array([[1, 2, 3], [4, 5, 6], [1, 2, 3]], dtype=np.int8)
    actual = np.frombuffer(memo.evaluate(inputs.tobytes(), 3), dtype="<i4").reshape(3, 2)
    assert np.array_equal(actual, inputs.astype(np.int64) @ weights.astype(np.int64).T)
    assert memo.stats() == (0, 3, 1, 267)
    memo.clear()
    assert memo.stats()[-1] == 0
