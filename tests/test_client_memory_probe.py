from pathlib import Path

import numpy as np
import pytest

from pllm import Model, lower_model
from pllm.profiles import MaskedLinearCpu
from pllm.runtime import decoder_memory
from pllm.runtime.transformer_client import LayerCache
from pllm.sources import _tiny_model_config


def test_kv_probe_restores_matching_execution_and_admission_even_on_failure(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    from scripts.probe_client_response_memory import allocation_control

    plan = lower_model(_tiny_model_config(), batch=1, max_input_tokens=16, max_new_tokens=8)
    schedule = plan.runtime_schedule(MaskedLinearCpu(Model.tiny()))
    append, estimate = LayerCache.append, decoder_memory.decoder_memory
    current = estimate(plan, schedule)
    row = np.ones((16, 2, 8), np.float32)
    with pytest.raises(RuntimeError, match="interrupted"):
        with allocation_control("kv-capacity", "legacy"):
            cache = LayerCache()
            cache.append(row, row, maximum_sequence=23)
            assert cache.key.shape[0] == 64
            historical = decoder_memory.decoder_memory(plan, schedule)
            assert historical["state_bytes"] * 23 == current["state_bytes"] * 64
            assert historical["working_bytes"] == current["working_bytes"]
            raise RuntimeError("interrupted")
    assert LayerCache.append is append and decoder_memory.decoder_memory is estimate
    with allocation_control("kv-capacity", "scoped"):
        cache = LayerCache()
        cache.append(row, row, maximum_sequence=23)
        assert cache.key.shape[0] == 23
        assert decoder_memory.decoder_memory(plan, schedule) == current
