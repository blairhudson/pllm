"""Phase ownership/import controls, not model memory or quality measurements.

Reuse ordinary synthetic quality/scoring fixtures. One isolated exec verifies
real JSON-tokenizer/import boundaries with both reference frameworks forbidden.
"""
import builtins
from contextlib import nullcontext
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest

from pllm.runtime import reference_benchmark as benchmark
from pllm.runtime import streamed_reference as streamed
from test_fss_quality import fss_controls as fss_controls, experiment as fss_experiment
from test_paper_quality_cli import lane as cli_lane, quality as quality
from test_quality_outcomes import quality_controls as quality_controls, experiment


ROOT = Path(__file__).resolve().parents[1]


def observe_imports(monkeypatch):
    events, original = [], builtins.__import__

    def importing(name, *args, **kwargs):
        if name in ("torch", "transformers"):
            events.append(name)
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", importing)
    return events


@pytest.mark.parametrize("storage", ["streamed", "resident"])
def test_full_seeded_keep_oracles_pins_and_thread_settings_with_phase_order(cli_lane, monkeypatch, storage):
    events = observe_imports(monkeypatch)
    torch = sys.modules["torch"]
    threads = Mock(return_value=3)
    monkeypatch.setattr(torch, "get_num_threads", threads)
    accepted = cli_lane.accepted

    def execute(*args, **kwargs):
        if storage == "streamed":
            assert not events and threads.call_count == 0
        else:
            assert events[:2] == ["torch", "transformers"] and threads.call_count == 1
        return accepted(*args, **kwargs)

    cli_lane.execute.side_effect = execute
    if storage == "resident":
        # Keep the real reference guard/scorer, substitute only tensor loading.
        monkeypatch.setattr(cli_lane.benchmark, "_load_reference_trajectories", lambda *args:
            ([[cli_lane.vector.copy()] * 4 for _ in range(12)], None))
    report = cli_lane.benchmark.run_reference_benchmark(
        [cli_lane.quality.experiment(), cli_lane.quality.seeded()], cli_lane.prompts,
        decode_steps=3, reference_storage=storage)
    assert cli_lane.execute.call_count == 24
    assert report["cohort"]["same_token_decode"]["teacher_forcing"] == "first-candidate-greedy"
    assert all(row["sample_count"] == 12 and row["same_token_decode"]["sample_count"] == 36
               and row["top1_agreement"] == 1 for row in report["candidates"])
    assert report["environment"]["reference_threads"] == 3 and threads.call_count == 1
    assert report["environment"]["torch"] == "synthetic-test-only"
    assert report["environment"]["python_reference"] == "transformers/synthetic-test-only"
    assert cli_lane.source_reader.call_count == 2  # Actual pre/post source guards.
    assert [item for item in cli_lane.pin_events if isinstance(item, tuple)] == [
        (kind, sharing, phase) for sharing in ("full", "seeded")
        for _ in range(12) for phase in ("before", "after") for kind in ("candidate",)]
    cli_lane.forbidden.assert_not_called()


@pytest.mark.parametrize("kind", ["gkm", "puma"])
def test_native_admission_and_execution_precede_streamed_imports(quality_controls, monkeypatch, kind):
    from pllm.runtime import benchmark_memory as memory

    events = observe_imports(monkeypatch)
    original = memory.benchmark_memory

    def price(*args, **kwargs):
        assert not events
        return original(*args, **kwargs)

    monkeypatch.setattr(memory, "benchmark_memory", price)
    reference = benchmark._reference_trajectories

    def load(*args):
        assert events[:2] == ["torch", "transformers"]
        assert quality_controls.lifecycle[-2:] == ["roles_close", "guard_close"]
        return reference(*args)

    monkeypatch.setattr(benchmark, "_reference_trajectories", load)
    report = benchmark.run_reference_benchmark([experiment(kind)], ["pass"], reference_storage="streamed")
    assert report["candidates"][0]["top1_agreement"] == 1
    assert quality_controls.calls == [(3,)]
    assert quality_controls.lifecycle == ["guard_start", "roles_start", "roles_close", "guard_close"]


def test_fss_source_qualification_and_final_ids_precede_streamed_imports(fss_controls, monkeypatch):
    events = observe_imports(monkeypatch)
    reference = benchmark._reference_trajectories

    def load(*args):
        assert fss_controls.calls and fss_controls.phases
        assert events[:2] == ["torch", "transformers"]
        assert fss_controls.shared.lifecycle[-2:] == ["roles_close", "guard_close"]
        return reference(*args)

    monkeypatch.setattr(benchmark, "_reference_trajectories", load)
    candidate = fss_experiment("fusefss.dense-rope-checked-ranges-ring50-q12.v1").with_params(
        pipeline__linear__capacity="decoder")
    report = benchmark.run_reference_benchmark([candidate], ["pass"], reference_storage="streamed")
    assert report["candidates"][0]["top1_agreement"] == 1
    assert report["candidates"][0]["candidate_logits_available"] is False


def test_local_clear_supported_candidate_needs_no_reference_framework(quality_controls, monkeypatch):
    from test_reference_benchmark import _experiment
    from pllm import modeling
    import numpy as np

    events = observe_imports(monkeypatch)
    vector = np.array([0, 0, 0, 0, 1], dtype=np.float32)
    closed = []
    session = NS(prefill_ids=lambda ids: vector, close=lambda: closed.append(True))
    compiled = NS(runtime=lambda remote: NS(encode_prompt=lambda prompt: (3,)),
                  session=lambda remote: session, validate=lambda: None)

    class Engine:
        _public_equalization_profile = None

        def __init__(self, **kwargs):
            assert not events

        async def load(self, manifest):
            assert not events

        def client_bundle_document(self, model_id):
            return NS(_value={})

    original_read = Path.read_bytes
    monkeypatch.setattr(Path, "read_bytes", lambda path:
        b"{}" if str(path) == "/unit-only-source/config.json" else original_read(path))
    monkeypatch.setattr(benchmark, "MaskedTransformerEngine", Engine)
    monkeypatch.setattr(benchmark, "ClientBundle", NS(_from_document=lambda value: value))
    monkeypatch.setattr(benchmark, "compile_runtime_model", lambda *args, **kwargs: compiled)
    monkeypatch.setattr(benchmark, "_candidate_remote", lambda *args: None)
    monkeypatch.setattr(modeling, "lower_model", lambda *args, **kwargs: None)
    reference = benchmark._reference_trajectories

    def load(*args):
        assert closed == [True] and events[:2] == ["torch", "transformers"]
        return reference(*args)

    monkeypatch.setattr(benchmark, "_reference_trajectories", load)
    report = benchmark.run_reference_benchmark([_experiment("synthetic-clear", 8)], ["pass"],
                                              reference_storage="streamed")
    assert report["candidates"][0]["top1_agreement"] == 1


@pytest.mark.parametrize("mode", ["abort", "record"])
def test_candidate_failure_never_imports_reference_or_loses_guard_note(cli_lane, monkeypatch, mode):
    from pllm.runtime import benchmark_memory as memory
    from pllm.runtime.quality_outcomes import read_report

    events = observe_imports(monkeypatch)
    message = "benchmark stopped: macOS memory pressure is warning"
    guard = NS(error=message, failure_observation={"pressure": "warning", "available_bytes": 12,
               "swap_used_bytes": 3}, start=lambda: None, close=lambda: None, check=lambda: None)
    monkeypatch.setattr(memory, "MemoryWatchdog", lambda *args: guard)
    error = sys.modules["pllm.runtime.client"].ProtocolError("synthetic rejected admission")
    cli_lane.execute.side_effect = error
    options = {"outcomes": mode}
    if mode == "record":
        options.update(cohort_key=b"k" * 32, cohort_key_id="lazy-import-test",
                       continuations=[[2, 2, 2]] * 12)
    if mode == "abort":
        with pytest.raises(type(error)) as caught:
            cli_lane.benchmark.run_reference_benchmark([cli_lane.quality.experiment()], cli_lane.prompts,
                reference_storage="streamed", decode_steps=3, **options)
        assert caught.value is error and message in error.__notes__[0]
    else:
        report = cli_lane.benchmark.run_reference_benchmark([cli_lane.quality.experiment()], cli_lane.prompts,
            reference_storage="streamed", decode_steps=3, **options)
        read_report(json.dumps(report))
        assert report["reference_status"] == "unavailable"
        assert report["candidates"][0]["all_positions"]["sample_count"] == 0
        assert error.__notes__[0] in report["limitations"]
        assert all(report["environment"][key] is None for key in ("torch", "python_reference", "reference_threads"))
    assert not events
    cli_lane.streamed.assert_not_called()
    cli_lane.topology.__exit__.assert_called_once()


def test_presence_probe_is_not_framework_initialization(monkeypatch):
    for name in ("torch", "transformers"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    seen = []
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: seen.append(name) or object())
    events = observe_imports(monkeypatch)
    assert benchmark._reference_dependencies(initialize=False) == (None, None)
    assert seen == ["torch", "transformers"] and not events


def test_missing_presence_still_rejects_before_candidate_execution(cli_lane, monkeypatch):
    for name in ("torch", "transformers"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    with pytest.raises(benchmark.ReferenceBenchmarkError, match="requires torch and transformers"):
        benchmark.run_reference_benchmark([cli_lane.quality.experiment()], cli_lane.prompts,
                                          reference_storage="streamed")
    cli_lane.execute.assert_not_called()


def test_native_memory_rejection_before_framework_init_uses_unchanged_quote(cli_lane, monkeypatch):
    from pllm.runtime import benchmark_memory as memory

    events = observe_imports(monkeypatch)
    error = memory.BenchmarkMemoryError("synthetic unchanged admission refusal")
    reject = Mock(side_effect=error)
    monkeypatch.setattr(memory, "require_admission", reject)
    candidate = cli_lane.quality.experiment()
    with pytest.raises(memory.BenchmarkMemoryError) as caught:
        benchmark.run_reference_benchmark([candidate], cli_lane.prompts,
                                          reference_storage="streamed")
    assert caught.value is error and not events
    memory.benchmark_memory.assert_called_once_with(candidate, backend="native", memory_safety_margin_percent=25)
    cli_lane.roles.assert_not_called()
    cli_lane.execute.assert_not_called()


@pytest.mark.parametrize("mode", ["abort", "record"])
def test_broken_library_import_is_deferred_without_scoring_candidates(cli_lane, monkeypatch, mode):
    original, error = builtins.__import__, ImportError("synthetic binary import failure")

    def importing(name, *args, **kwargs):
        if name == "torch":
            assert cli_lane.execute.call_count == 12
            assert cli_lane.topology.__exit__.call_count == 1
            raise error
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", importing)
    kwargs = {"reference_storage": "streamed", "decode_steps": 3, "outcomes": mode}
    if mode == "record":
        kwargs.update(cohort_key=b"k" * 32, cohort_key_id="lazy-binary-test", continuations=[[2, 2, 2]] * 12)
        report = benchmark.run_reference_benchmark([cli_lane.quality.experiment()], cli_lane.prompts, **kwargs)
        assert report["reference_status"] == "unavailable" and report["terminal_state"] == "execution_failed"
        row = report["candidates"][0]["all_positions"]
        assert row["coverage"]["accepted"] == 48 and row["sample_count"] == 0
        assert row["top1_agreement"] is row["accepted_agreement"]["top1_agreement"] is None
    else:
        with pytest.raises(benchmark.ReferenceBenchmarkError, match="requires torch and transformers") as caught:
            benchmark.run_reference_benchmark([cli_lane.quality.experiment()], cli_lane.prompts, **kwargs)
        assert caught.value.__cause__ is error
    cli_lane.streamed.assert_not_called()


def test_streamed_host_admission_after_meta_setup_before_weight_hooks(monkeypatch):
    from pllm.runtime import benchmark_memory as memory

    events = []
    parameter = NS(shape=(2, 2), is_meta=True, ndim=2, numel=lambda: 4)
    model = NS(_parameters={"weight": parameter}, eval=lambda: model,
        named_modules=lambda: [("", model)], named_buffers=lambda **kwargs: [],
        named_parameters=lambda **kwargs: [("weight", parameter)])
    model.register_forward_pre_hook = model.register_forward_hook = Mock(
        side_effect=AssertionError("weight loader installed before admission"))

    def make_model(*args, **kwargs):
        events.append("meta_setup")
        assert kwargs["dtype"] == "fake-float32" and kwargs["attn_implementation"] == "eager"
        return model

    torch = NS(device=lambda name: nullcontext(), float32="fake-float32")
    config = NS(vocab_size=3, hidden_size=2, intermediate_size=2, num_attention_heads=1,
                num_hidden_layers=1, num_key_value_heads=1, head_dim=2)
    transformers = NS(AutoConfig=NS(from_pretrained=lambda *args, **kwargs: config),
                      AutoModelForCausalLM=NS(from_config=make_model))
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    store = NS(keys=["weight"], _locations={}, tensor_shape=lambda key: (2, 2),
               tensor_dtype=lambda key: "F32", clear_cache=lambda: events.append("store_closed"))
    monkeypatch.setattr(streamed, "SafeTensorStore", lambda path: store)

    def host():
        assert events == ["meta_setup"]
        events.append("host_gate")
        return NS(available=0, reserve=0)

    monkeypatch.setattr(memory, "host_memory", host)
    with pytest.raises(streamed.StreamedReferenceError, match="post-reserve host headroom"):
        streamed.StreamedFloat32Reference("synthetic-only", memory_safety_margin_percent=25)
    assert events == ["meta_setup", "host_gate", "store_closed"]
    model.register_forward_pre_hook.assert_not_called()


def test_real_optional_tokenizer_and_import_graph_need_neither_reference_framework():
    script = r'''
import importlib.abc, sys
from pathlib import Path
from types import SimpleNamespace as NS
class NoReference(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('torch', 'transformers'):
            raise AssertionError('candidate imports reference framework: ' + fullname)
sys.meta_path.insert(0, NoReference())
import pllm, pllm.runtime, pllm._native
root = Path(sys.argv[1])
pllm.__path__.insert(0, str(root / 'python/pllm'))
pllm.runtime.__path__.insert(0, str(root / 'python/pllm/runtime'))
import pllm.cli
from pllm.runtime import reference_benchmark as quality, delegated_source, replicated_source, fss_source
from pllm.runtime import benchmark_memory, servers, delegated_transport, replicated_transport, fss_transport
from pllm.runtime.transformer_client import ClientBundle
from tokenizers import Tokenizer, models, pre_tokenizers
tokenizer = Tokenizer(models.WordLevel({'[UNK]': 0, 'bos': 1, 'hello': 2}, unk_token='[UNK]'))
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
raw = tokenizer.to_str().encode()
descriptor = {'bos_token_id': 1, 'eos_token_id': 0, 'add_bos_token': True, 'model': raw, 'kind': 'tokenizer_json'}
owner = NS(tokenizer_bytes=raw, tokenizer_descriptor=descriptor, cfg={'bos_token_id': 1, 'eos_token_id': 0})
for cls in (delegated_source.AdmittedSource, replicated_source.ReplicatedSource):
    owner.tokenizer = lambda cls=cls: cls.tokenizer(owner)
    assert quality._source_ids(owner, ('hello',)) == ((1, 2),)
assert quality._source_ids(owner, ('hello',), runtime='compiled_fss_v1') == ((1, 2),)
assert ClientBundle.tokenizer(owner).encode('hello', add_bos=True) == [1, 2]
assert not any(name.split('.')[0] in ('torch', 'transformers') for name in sys.modules)
print('candidate import graph and real tokenizers: no reference framework')
'''
    completed = subprocess.run([sys.executable, "-I", "-c", script, str(ROOT)],
                               capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0, completed.stderr
    assert "no reference framework" in completed.stdout
