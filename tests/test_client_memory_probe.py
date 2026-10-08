from pathlib import Path
import ast
import gc
import weakref

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


@pytest.mark.parametrize("control", ["tokenizer-source", "kv-capacity"])
def test_ablation_retires_source_parsing_owners_before_either_response(monkeypatch, control):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    from scripts import probe_client_response_memory as probe

    parse, references = ast.parse, []

    def tracked(*args, **kwargs):
        tree = parse(*args, **kwargs)
        references.extend(weakref.ref(node) for node in ast.walk(tree)
                          if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef)))
        return tree

    monkeypatch.setattr(ast, "parse", tracked)
    enabled = gc.isenabled()
    gc.disable()
    try:
        counts = []
        for mode in ("legacy", "scoped"):
            references.clear()
            with probe.allocation_control(control, mode):
                assert references and all(reference() is None for reference in references)
                counts.append(len(references))
        assert counts[0] == counts[1]
    finally:
        if enabled:
            gc.enable()
