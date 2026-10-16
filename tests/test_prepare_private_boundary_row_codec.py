"""Publisher admission regressions; all source, engine and certificate work mocked."""
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from pllm import Model
from scripts import prepare_private_boundary_row_codec as publisher


def test_offline_clone_preserves_original_source_spec():
    original = Model.hf("synthetic/not-downloaded", revision="a" * 40)
    frozen = original.to_spec()
    clone = publisher.offline_model(original)
    assert clone.local_files_only is True
    assert original.to_spec() == frozen
    assert clone.to_spec() == {**frozen, "local_files_only": True}


@pytest.mark.parametrize("kind", ["tiny", "ollama", "gguf", "mlx"])
def test_materializing_or_nonlocal_kinds_rejected_without_resolution(kind):
    with pytest.raises(ValueError, match="prepositioned"):
        publisher.offline_model(SimpleNamespace(kind=kind))


@pytest.mark.parametrize("field", ["path", "source_lock_digest", "checkpoint_digest"])
def test_source_identity_requires_complete_lock(field):
    source = SimpleNamespace(path="synthetic", source_lock_digest="a" * 64,
                             checkpoint_digest="b" * 64)
    setattr(source, field, None)
    with pytest.raises(ValueError, match="complete local source lock"):
        publisher.source_identity(source)


def test_parent_first_resolution_is_offline_before_guard(monkeypatch, tmp_path):
    from pllm import Experiment, model_loader

    model = Model.hf("synthetic/not-downloaded", revision="a" * 40)
    frozen = model.to_spec()
    monkeypatch.setattr(Experiment, "from_spec", lambda _:
                        SimpleNamespace(pipeline=SimpleNamespace(model=model)))
    monkeypatch.setattr(publisher.study, "bounded_json", lambda *a: {})
    monkeypatch.setattr(publisher.study, "hashes", lambda *a: {})
    writes = Mock()
    monkeypatch.setattr(publisher, "atomic_json", writes)
    guard = Mock(side_effect=AssertionError("measurement lock must not be acquired"))
    monkeypatch.setitem(sys.modules, "frontier_probe_support", SimpleNamespace(fresh_workers=guard))
    def resolve(choice):
        assert choice.local_files_only is True
        assert model.to_spec() == frozen
        raise RuntimeError("mock stop before source access")
    resolve_mock = Mock(side_effect=resolve)
    monkeypatch.setattr(model_loader, "resolve_model", resolve_mock)
    monkeypatch.setattr(sys, "argv", ["prepare_private_boundary_row_codec.py", "--tiny",
        "--control", "mock-control.json", "--output", str(tmp_path / "publication")])
    with pytest.raises(RuntimeError, match="mock stop before source access"):
        publisher.main()
    resolve_mock.assert_called_once()
    guard.assert_not_called()
    assert writes.call_args.args[1]["complete"] is False
    assert writes.call_args.args[1]["failure_type"] == "RuntimeError"


@pytest.fixture
def mocked_worker(monkeypatch, tmp_path):
    """Exercise worker control flow without constructing any actual engine/tensor."""
    from pllm import Experiment, Pipeline
    from pllm import model_loader
    from pllm.roles import PrivateLookupAndFullHead
    from pllm.runtime import model_binding, private_boundary_certificate
    from pllm.runtime import private_boundary_loader, transformer_client, transformer_engine
    import pllm

    (tmp_path / "config.json").write_text("{}")
    model = Model.hf("synthetic/not-downloaded", revision="a" * 40)
    components = {"boundary": SimpleNamespace(params={"head_dispatch": "parallel",
        "head_output_encoding": "uniform_residues"}),
        "delivery": SimpleNamespace(params={"storage": "memory"})}
    pipeline = SimpleNamespace(model=model, components=components, profile="mock",
                               digest=lambda: "c" * 64)
    first = Mock(pipeline=pipeline, budget=SimpleNamespace(max_input_tokens=128, max_new_tokens=2))
    first.to_spec.return_value = {"frozen": "control"}
    candidate = Mock(pipeline=pipeline)
    candidate.to_spec.return_value = {"frozen": "candidate"}
    offline_experiment = object()
    def with_params(**kwargs):
        if "pipeline__model" in kwargs:
            assert kwargs["pipeline__model"].local_files_only
            return offline_experiment
        return candidate
    first.with_params.side_effect = with_params
    monkeypatch.setattr(Experiment, "from_spec", lambda _: first)
    monkeypatch.setattr(Pipeline, "from_profile", lambda *a, **k: pipeline)
    source = SimpleNamespace(path=tmp_path, source_lock_digest="a" * 64,
        checkpoint_digest="b" * 64, manifest=SimpleNamespace(id="synthetic"))
    resolve = Mock(return_value=source)
    monkeypatch.setattr(model_loader, "resolve_model", resolve)
    monkeypatch.setattr(publisher, "source_price", lambda _: 100)
    pin = {"path": str(tmp_path / "head.cert"), "digest": "d" * 64, "size_bytes": 100}
    publish = Mock(return_value=pin)
    monkeypatch.setattr(PrivateLookupAndFullHead, "publish_head_certificate", publish)
    descriptor = SimpleNamespace(composition_digest="c" * 64, schedule_digest="e" * 64,
                                 digest="f" * 64)
    monkeypatch.setattr(private_boundary_loader, "build_private_boundary_contract",
                        lambda *a: (descriptor, None))
    monkeypatch.setattr(pllm, "lower_model", lambda *a, **k:
                        SimpleNamespace(runtime_schedule=lambda _: object()))
    compiled = SimpleNamespace(private_boundary_descriptor=lambda: descriptor,
                               to_spec=lambda: {"body_fingerprint": "1" * 64})
    monkeypatch.setattr(model_binding, "compile_runtime_model", lambda *a, **k: compiled)
    monkeypatch.setattr(transformer_client.ClientBundle, "_from_document", lambda *a, **k: object())
    metadata = {"source": {"source_digest": "a" * 64, "checkpoint_digest": "b" * 64}}
    certificate = Mock(return_value=(bytes([19, 20]), metadata))
    monkeypatch.setattr(private_boundary_certificate, "import_certificate", certificate)
    monkeypatch.setattr(publisher.study, "validate_pair_specs", Mock())
    writes = Mock()
    monkeypatch.setattr(publisher, "atomic_json", writes)
    events = []
    class Engine:
        def __init__(self, **kwargs):
            events.append("construct")
            head = SimpleNamespace(compiled_weight=SimpleNamespace(weight_view=lambda: object()))
            self.models = {"synthetic": SimpleNamespace(stages={"lm_head": head}, tokenizer={})}
        async def load(self, manifest):
            events.append("load")
        def client_bundle(self, model_id):
            return b"\x80"  # Empty MessagePack map, not a model bundle.
        async def unload(self, model_id):
            events.append("unload")
    monkeypatch.setattr(transformer_engine, "MaskedTransformerEngine", Engine)
    payload = {"controls": [{}], "tiny": True, "approved_head": None,
               "memory_bytes": 100, "output": str(tmp_path)}
    return SimpleNamespace(payload=payload, source=source, resolve=resolve, publish=publish,
        compiled=compiled, metadata=metadata, events=events, writes=writes,
        certificate=certificate, model=model)


@pytest.mark.parametrize("reason", ["capacity", "approved_source", "publication_source"])
def test_rejection_precedes_full_engine_construction(mocked_worker, reason):
    state = mocked_worker
    if reason == "capacity":
        state.payload["memory_bytes"] = 99
        message = "reserved capacity"
    elif reason == "approved_source":
        state.payload.update(tiny=False, approved_head={"certificate": {
            "source_lock_sha256": "wrong", "checkpoint_sha256": "b" * 64}})
        message = "approved checkpoint before import"
    else:
        changed = SimpleNamespace(path=state.source.path, source_lock_digest="changed",
                                  checkpoint_digest="b" * 64)
        state.resolve.side_effect = [state.source, changed]
        message = "changed during certificate publication"
    with pytest.raises(ValueError, match=message):
        publisher.worker(state.payload)
    assert state.events == []
    if reason != "publication_source":
        state.publish.assert_not_called()
    assert all(call.args[0].local_files_only for call in state.resolve.call_args_list)


@pytest.mark.parametrize("reason", ["descriptor", "certificate_source", "final_source"])
def test_late_binding_failures_unload_without_success(mocked_worker, reason):
    state = mocked_worker
    if reason == "descriptor":
        state.compiled.private_boundary_descriptor = lambda: object()
        message = "descriptors differ"
    elif reason == "certificate_source":
        state.metadata["source"]["checkpoint_digest"] = "wrong"
        message = "publisher source and admitted source differ"
    else:
        changed = SimpleNamespace(path=state.source.path, source_lock_digest="changed",
                                  checkpoint_digest="b" * 64)
        state.resolve.side_effect = [state.source, state.source, changed]
        message = "changed during independent source binding"
    with pytest.raises(ValueError, match=message):
        publisher.worker(state.payload)
    assert state.events == ["construct", "load", "unload"]
    assert not any(call.args[0].name == "certificate-accounting.json"
                   for call in state.writes.call_args_list)


def test_success_revalidates_source_and_preserves_original_model(mocked_worker):
    state = mocked_worker
    original = state.model.to_spec()
    result = publisher.worker(state.payload)
    assert result["complete"] is True
    assert state.resolve.call_count == 3
    assert all(call.args[0].local_files_only for call in state.resolve.call_args_list)
    assert state.model.to_spec() == original
    assert state.events == ["construct", "load", "unload"]
    assert result["independent_source_binding"][0]["payload_bytes_per_role"] == 5
