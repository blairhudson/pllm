"""Weights-free ownership and bounded public metadata; no tensor/native execution."""
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_delegated_client import harness as harness

from pllm.runtime.responses import (
    ResponsesError, _owned_prompt_source, _prepare_prompt, _prompt_request,
    _prompt_source, _render_prompt, normalize_input, prompt_text,
)


class Tokenizer:
    def __init__(self):
        self.calls = []
        self.closed = 0

    def encode(self, text, *, add_bos):
        self.calls.append((text, add_bos))
        return ([99] if add_bos else []) + list(text.encode())

    def close(self):
        self.closed += 1


def source(template=None):
    model = SimpleNamespace(to_spec=lambda: {"kind": "test", "source": "fixture"})
    return SimpleNamespace(model=model, config_source_bytes=b'{"vocab_size":256}', source_lock_digest="1" * 64,
        tokenizer_descriptor={"chat_template": template, "add_bos_token": False})


def body():
    return {"model": "fixture", "input": "hello", "max_output_tokens": 2}


def prepare(s=None, tokenizer=None, request=None):
    s = s or source()
    tokenizer = tokenizer or Tokenizer()
    return _prepare_prompt(body() if request is None else request,
        descriptor=s.tokenizer_descriptor, tokenizer=tokenizer,
        source=_owned_prompt_source(s)), tokenizer, s


def test_fallback_is_original_admission_not_legacy_role_prefix():
    rows = normalize_input("hello")
    assert prompt_text(rows) == "user: hello"
    assert _render_prompt(rows, {}) == "user: hello\nassistant:"
    assert _render_prompt(rows, {}, add_generation_prompt=False) == "user: hello"


def test_template_system_generation_and_bos_are_preserved():
    template = "{{ bos_token }}SYSTEM|{{ messages[0].content }}|{% if add_generation_prompt %}ASSISTANT{% endif %}"
    descriptor = {"chat_template": template, "bos_token": "B", "add_bos_token": True}
    tokenizer = Tokenizer()
    prepared = _prepare_prompt(body(), descriptor=descriptor, tokenizer=tokenizer,
        source=_prompt_source(source().model, {}, descriptor))
    assert prepared._payload.rendered == "BSYSTEM|hello|ASSISTANT"
    assert prepared._payload.ids[0] == 99
    assert tokenizer.calls == [("BSYSTEM|hello|ASSISTANT", True)]


@pytest.mark.parametrize("template", ["{% broken %}", "{{ missing.value }}", "{{ raise_exception('no') }}"])
def test_present_broken_template_refuses_before_encode(template):
    tokenizer = Tokenizer()
    with pytest.raises(ResponsesError, match="template failed"):
        prepare(source(template), tokenizer)
    assert tokenizer.calls == []


def test_private_payload_immutable_encode_once_consume_once():
    prepared, tokenizer, s = prepare()
    assert len(tokenizer.calls) == 1
    with pytest.raises(AttributeError):
        prepared._payload.ids = ()
    actual, ids = prepared.consume(body(), _owned_prompt_source(s))
    assert actual is tokenizer and ids == tuple(b"user: hello\nassistant:")
    with pytest.raises(ResponsesError, match="consumed"):
        prepared.consume(body(), _owned_prompt_source(s))
    assert len(tokenizer.calls) == 1
    prepared.close()
    assert tokenizer.closed == 1


@pytest.mark.parametrize("change", ["input", "outputs", "template", "config", "model", "descriptor"])
def test_request_source_and_template_mismatch_burns(change):
    prepared, _, s = prepare()
    request = body()
    if change == "input": request["input"] = "changed"
    if change == "outputs": request["max_output_tokens"] = 1
    if change == "template": s.tokenizer_descriptor["chat_template"] = "changed"
    if change == "config": s.config_source_bytes = b'{"vocab_size":255}'
    if change == "model": s.model = SimpleNamespace(to_spec=lambda: {"source": "other"})
    if change == "descriptor": s.tokenizer_descriptor["add_bos_token"] = True
    with pytest.raises(ResponsesError, match="association changed"):
        prepared.consume(request, _owned_prompt_source(s))
    with pytest.raises(ResponsesError, match="consumed"):
        prepared.consume(body(), prepared._payload.source)
    prepared.close()


def test_claim_and_cancel_before_execution_retire_owner():
    prepared, tokenizer, _ = prepare()
    assert prepared.claim() is prepared
    prepared.close_unclaimed()
    assert tokenizer.closed == 0
    with pytest.raises(ResponsesError, match="attached"):
        prepared.claim()
    prepared.close()
    assert tokenizer.closed == 1 and prepared._payload is None


def test_failure_to_close_keeps_original_owner_for_retry():
    prepared, tokenizer, _ = prepare()
    def failed(): raise RuntimeError("busy")
    tokenizer.close = failed
    with pytest.raises(RuntimeError, match="busy"): prepared.close()
    assert prepared._payload.tokenizer is tokenizer
    tokenizer.close = lambda: None
    prepared.close()
    assert prepared._payload is None


def test_request_defaults_do_not_modify_caller():
    request = body()
    key = _prompt_request(request)
    assert "temperature" not in request
    assert key == _prompt_request({**request, "temperature": 0, "top_p": 1})


def test_encoder_cannot_rebind_mutated_request_after_render():
    request = body()
    tokenizer = Tokenizer()
    original = tokenizer.encode
    def changed(text, *, add_bos):
        request["input"] = "changed"
        return original(text, add_bos=add_bos)
    tokenizer.encode = changed
    prepared, _, s = prepare(tokenizer=tokenizer, request=request)
    assert prepared._payload.rendered == "user: hello\nassistant:"
    with pytest.raises(ResponsesError, match="association"):
        prepared.consume(request, _owned_prompt_source(s))
    prepared.close()


def test_actual_cached_qwen_public_tokenizer_control():
    path = os.environ.get("PLLM_TEST_QWEN25_TOKENIZER_DIR")
    if path is None:
        pytest.skip("actual Qwen cache diagnostic requires explicit path; not fixture evidence")
    from tokenizers import Tokenizer as PublicTokenizer
    def read(name, limit):
        with (Path(path) / name).open("rb") as stream: data = stream.read(limit + 1)
        assert len(data) <= limit
        return data
    options = json.loads(read("tokenizer_config.json", 65536))
    raw = read("tokenizer.json", (8 << 20) - 1)
    tokenizer = PublicTokenizer.from_buffer(raw)
    rows = normalize_input("hello")
    rendered = _render_prompt(rows, options)
    ids = tokenizer.encode(rendered).ids
    legacy = tokenizer.encode(prompt_text(rows)).ids
    assert options["add_bos_token"] is False
    assert ids == [151644,8948,198,2610,525,1207,16948,11,3465,553,54364,14817,13,
                   1446,525,264,10950,17847,13,151645,198,151644,872,198,14990,
                   151645,198,151644,77091,198]
    assert legacy == [872,25,23811]
    assert len(ids) == 30 and len(legacy) == 3
    print(f"actual cached public tokenizer path={path}; bytes={len(raw)}; ChatML=30 legacy=3; no model execution")


def _real_prompt_metadata(tmp_path, monkeypatch, kind, *, missing_bos=False,
                          template=None, manifest_template=None, model_id=None):
    """Real source tokenizer loaders/SDK types, PUBLIC metadata only.

    This is not native source admission: the checked seat is a metadata-layer
    fixture. No Safetensors/index is written/read and no source contract/native
    arithmetic is constructed. Transport is forbidden throughout these tests.
    """
    import hashlib
    from dataclasses import replace
    from tokenizers import Tokenizer as PublicTokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import WhitespaceSplit

    from pllm import Deployment, ExecutionBudget, Model, _native
    from pllm.configuration import Experiment
    from pllm.profiles import OpenWeightVerifiableCpu
    from pllm.runtime import delegated_source, delegated_transport, fss_source, fss_transport
    from pllm.runtime.delegated_client import DelegatedClient
    from pllm.runtime.fss_client import FssClient
    from pllm.runtime.safetensors_store import SafeTensorStore
    from test_fss_source import ROPE, experiment as fss_experiment

    def forbidden(*args, **kwargs):
        pytest.fail("prompt metadata reached payload import/native arithmetic/transport")

    for name in ("FssSourceBuilder", "FssClientRequest", "FssRoleRequest",
                 "GkmSourceBuilder", "GkmClientRequest", "ReplicatedSourcePublisher"):
        monkeypatch.setattr(_native, name, forbidden, raising=False)
    monkeypatch.setattr(SafeTensorStore, "get", forbidden)
    monkeypatch.setattr(fss_source, "load_role_source", forbidden)
    monkeypatch.setattr(delegated_source.AdmittedSource, "load_values", forbidden)
    monkeypatch.setattr(delegated_source.AdmittedSource, "load_transport", forbidden)
    monkeypatch.setattr(fss_transport, "execute_request", forbidden)
    monkeypatch.setattr(delegated_transport, "execute_request", forbidden)

    cfg = {"model_type": "qwen3", "num_hidden_layers": 1, "hidden_size": 4,
           "intermediate_size": 8, "num_attention_heads": 1, "num_key_value_heads": 1,
           "head_dim": 4, "max_position_embeddings": 64, "vocab_size": 32,
           "hidden_act": "silu", "rms_norm_eps": 1e-6, "rope_theta": 10000,
           "tie_word_embeddings": True, "bos_token_id": 0, "eos_token_id": 0}
    if missing_bos:
        del cfg["bos_token_id"]
    raw_config = json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode()
    (tmp_path / "config.json").write_bytes(raw_config)
    vocab = {"<bos>": 0, "<eos>": 1, "<unk>": 2, "SYS": 3, "hello": 4,
             "ASSISTANT": 5, "user:": 6, "assistant:": 7}
    vocab.update({f"t{i}": i for i in range(8, 32)})
    native_tokenizer = PublicTokenizer(WordLevel(vocab, unk_token="<unk>"))
    native_tokenizer.pre_tokenizer = WhitespaceSplit()
    raw_tokenizer = native_tokenizer.to_str().encode()
    (tmp_path / "tokenizer.json").write_bytes(raw_tokenizer)
    options = {"model_max_length": 64, "add_bos_token": True,
               "bos_token": "<bos>", "eos_token": "<eos>"}
    if template is not None:
        options["chat_template"] = template
    (tmp_path / "tokenizer_config.json").write_text(json.dumps(options))

    if kind == "fss":
        # Registered prefix capacity 16 includes the three feedback rows.
        experiment = fss_experiment(tmp_path, profile=ROPE, rows=13, outputs=4)
        cls = FssClient
    else:
        experiment = Experiment("gkm-prompt-metadata",
            OpenWeightVerifiableCpu(Model.path(str(tmp_path), model_id=model_id)),
            Deployment.local(root="local://prompt-metadata"),
            ExecutionBudget(requests=1, max_input_tokens=16, max_new_tokens=4))
        cls = DelegatedClient
    manifest = delegated_source._manifest(tmp_path, experiment.pipeline.model, cfg, ())
    manifest = replace(manifest, chat_template=manifest_template, created_at=0.0)
    contract = {"vocabulary": 32, "input_rows": 16, "decode_steps": 3,
                 "maximum_sequence": 20, "source_float32_bytes": 64}
    if kind == "fss":
        contract.update(input_rows=13, maximum_sequence=16)
    encoded_contract = json.dumps(contract, sort_keys=True).encode()
    digest = hashlib.sha256(raw_tokenizer).hexdigest()
    if kind == "fss":
        loaded_tokenizer, descriptor = fss_source._tokenizer(tmp_path, cfg, contract)
        assert loaded_tokenizer == raw_tokenizer
        owned = fss_source.FrozenFssSource(
            model=experiment.pipeline.model, path=tmp_path,
            experiment_bytes=experiment.canonical_bytes(), config_bytes=raw_config,
            config_source_bytes=raw_config, contract_bytes=encoded_contract,
            checkpoint_digest="2" * 64, source_lock_digest="1" * 64,
            tokenizer_digest=digest, tokenizer_bytes=raw_tokenizer,
            _lock=b"{}", _tokenizer_options=descriptor, _identities=(), _specs=())
    else:
        descriptor = delegated_source._tokenizer(tmp_path, manifest, cfg, contract)
        assert descriptor.pop("model") == raw_tokenizer
        owned = delegated_source.AdmittedSource(
            model=experiment.pipeline.model, path=tmp_path, config_bytes=raw_config,
            config_source_bytes=raw_config, contract_bytes=encoded_contract,
            checkpoint_digest="2" * 64, source_lock_digest="1" * 64,
            tokenizer_bytes=raw_tokenizer, tokenizer_digest=digest,
            _source_lock_bytes=b"{}", _manifest_bytes=json.dumps(manifest.to_dict()).encode(),
            _tokenizer_options_bytes=json.dumps(descriptor).encode(),
            _identities=(), _tensor_specs=(), experiment_bytes=experiment.canonical_bytes())
    connections = {role: (f"http://{role.replace('_', '-')}.invalid/v1", "metadata-only")
                   for role in cls.role_names}
    core = cls(experiment, connections, source=owned)
    core._source_checked = True  # Metadata fixture, NEVER a native/source admission claim.
    return experiment, manifest, cfg, core, owned


def _source_prompt_lease(experiment, manifest, cfg, core, owned):
    from pllm.deployment import execution
    from pllm.runtime.transformer_engine import MaskedTransformerEngine

    # Exercise the real generic metadata loader too: it deliberately keeps its
    # historical zero/default behavior, and is not the execution source descriptor.
    descriptor = MaskedTransformerEngine._load_tokenizer_descriptor(owned.path, manifest, cfg)
    view = execution._TokenizerView(descriptor, cfg, manifest, original_config=cfg,
                                   source_lock_digest=owned.source_lock_digest)
    result = SimpleNamespace(experiment=experiment, snapshot=None, validate=lambda **kw: None)
    lease = execution.ExecutionLease(result, SimpleNamespace(backend="local", snapshot=None),
        lambda: 0, SimpleNamespace(close=lambda: None), view, SimpleNamespace(),
        _key=execution._LEASE_KEY)
    lease._client = SimpleNamespace(_core=core)
    return lease


def _observe_source_tokenizer(monkeypatch, owned):
    original = type(owned).tokenizer
    observed = []

    class Observed:
        def __init__(self, inner):
            self.inner, self.calls, self.closed = inner, [], 0

        def encode(self, text, *, add_bos):
            self.calls.append((text, add_bos))
            return self.inner.encode(text, add_bos=add_bos)

        def decode(self, ids):
            return self.inner.decode(ids)

        def close(self):
            self.closed += 1
            self.inner = None

    def make(source):
        value = Observed(original(source))
        observed.append(value)
        return value

    monkeypatch.setattr(type(owned), "tokenizer", make)
    return observed


@pytest.mark.parametrize("kind", ["gkm", "fss"])
@pytest.mark.parametrize("case", ["zero_ids", "missing_bos", "manifest_template_fallback"])
def test_prepared_real_source_descriptor_matches_lease(tmp_path, monkeypatch, kind, case):
    exp, manifest, cfg, core, owned = _real_prompt_metadata(tmp_path, monkeypatch, kind,
        missing_bos=case == "missing_bos",
        manifest_template="SYS {{ messages[0].content }}" if case == "manifest_template_fallback" else None)
    observed = _observe_source_tokenizer(monkeypatch, owned)
    lease = _source_prompt_lease(exp, manifest, cfg, core, owned)
    request = {"input": "hello", "max_output_tokens": 4}
    prepared = lease._admit_request(request)
    try:
        assert prepared._payload.source == _owned_prompt_source(owned)
        expected_bos = 2 if kind == "gkm" and case == "missing_bos" else 0
        assert owned.tokenizer_descriptor["bos_token_id"] == expected_bos
        assert owned.tokenizer_descriptor["eos_token_id"] == 0
        assert prepared._payload.ids[0] == expected_bos
        if case == "manifest_template_fallback":
            assert prepared._payload.rendered == ("SYS hello" if kind == "gkm"
                else "user: hello\nassistant:")
        # Direct callers use the SAME real factory/renderer, not a test descriptor.
        with core.tokenizer_scope():
            _, direct = core._encode(core._admit(dict(request)), owned)
        assert tuple(direct) == prepared._payload.ids
        actual, ids = prepared.consume(request, _owned_prompt_source(owned))
        assert actual is observed[0] and ids == tuple(direct)
        assert len(observed[0].calls) == 1
    finally:
        prepared.close()
        core.close()
    assert observed[0].closed == 1


@pytest.mark.parametrize("template,rendered,expected_ids", [
    ("", "", (0,)),
    (None, "SYS hello", (0, 3, 4)),
])
def test_gkm_empty_source_template_overrides_manifest(tmp_path, monkeypatch,
                                                     template, rendered, expected_ids):
    from jinja2.sandbox import SandboxedEnvironment

    manifest_template = "SYS {{ messages[0].content }}"
    exp, manifest, cfg, core, owned = _real_prompt_metadata(
        tmp_path, monkeypatch, "gkm", template=template, manifest_template=manifest_template)
    observed = _observe_source_tokenizer(monkeypatch, owned)
    rendered_templates = []
    original = SandboxedEnvironment.from_string

    def from_string(self, selected, *args, **kwargs):
        rendered_templates.append(selected)
        return original(self, selected, *args, **kwargs)

    monkeypatch.setattr(SandboxedEnvironment, "from_string", from_string)
    descriptor = owned.tokenizer_descriptor
    selected = manifest_template if template is None else template
    assert descriptor["chat_template"] == selected
    assert descriptor["bos_token_id"] == 0 and descriptor["add_bos_token"] is True
    association = _owned_prompt_source(owned)
    assert _prompt_source(owned.model, cfg, descriptor, template=manifest_template,
        source_lock_digest=owned.source_lock_digest) == association
    assert _render_prompt(normalize_input("hello"), descriptor,
        template=manifest_template) == rendered
    lease = _source_prompt_lease(exp, manifest, cfg, core, owned)
    request = {"input": "hello", "max_output_tokens": 4}
    prepared = lease._admit_request(request)
    try:
        assert prepared._payload.source == association
        assert prepared._payload.rendered == rendered
        assert prepared._payload.ids == expected_ids
        with core.tokenizer_scope():
            direct_tokenizer, direct = core._encode(core._admit(dict(request)), owned)
        assert direct_tokenizer is observed[1] and tuple(direct) == expected_ids
        actual, ids = prepared.consume(request, association)
        assert actual is observed[0] and ids == expected_ids
        assert len(observed) == 2
        assert [item.calls for item in observed] == [[(rendered, True)], [(rendered, True)]]
        assert rendered_templates == [selected, selected, selected]
    finally:
        prepared.close()
        core.close()
    assert [item.closed for item in observed] == [1, 1]


@pytest.mark.parametrize("ending", ["execution_boundary", "wrong_source", "wrong_template", "cancel"])
def test_fss_prepared_ids_encode_once_and_retire(tmp_path, monkeypatch, ending):
    from dataclasses import replace
    from pllm.runtime.client import ProtocolError
    from pllm.runtime.delegated_client import DelegatedClient
    from pllm.runtime.fss_client import FssClient
    from pllm.runtime.responses import _prepared_prompt_request

    template = "{{ bos_token }} {% if messages[0].role != 'system' %}SYS {% endif %}{{ messages[0].content }} {% if add_generation_prompt %}ASSISTANT{% endif %}"
    exp, manifest, cfg, core, owned = _real_prompt_metadata(
        tmp_path, monkeypatch, "fss", template=template)
    observed = _observe_source_tokenizer(monkeypatch, owned)
    lease = _source_prompt_lease(exp, manifest, cfg, core, owned)
    request = {"input": "hello", "max_output_tokens": 4}
    prepared = lease._admit_request(request)
    assert prepared._payload.rendered == "<bos> SYS hello ASSISTANT"
    assert prepared._payload.ids == (0, 0, 3, 4, 5)
    assert FssClient._produce is DelegatedClient._produce
    assert FssClient._encode is DelegatedClient._encode
    reached = []

    class ExecutionBoundary(Exception):
        pass

    def execute(source, ids, count, attempt):
        reached.append((source, ids, count))
        assert source is owned and ids is prepared._payload.ids and count == 4
        raise ExecutionBoundary("no transport/native execution permitted")

    monkeypatch.setattr(core, "_execute", execute)
    if ending == "wrong_source":
        core._source = replace(owned, source_lock_digest="3" * 64)
    elif ending == "wrong_template":
        options = json.loads(owned._tokenizer_options)
        options["chat_template"] = "other {{ messages[0].content }}"
        core._source = replace(owned, _tokenizer_options=json.dumps(options).encode())
    token = _prepared_prompt_request.set(prepared)
    try:
        attempt = core.events(dict(request))
    finally:
        _prepared_prompt_request.reset(token)
    try:
        if ending == "cancel":
            attempt.cancel()
        expected = ExecutionBoundary if ending == "execution_boundary" else (
            ProtocolError if ending == "cancel" else ValueError)
        with pytest.raises(expected):
            next(attempt)
        assert bool(reached) is (ending == "execution_boundary")
    finally:
        attempt.close()
        core.close()
    assert prepared._payload is None and observed[0].closed == 1
    assert len(observed) == 1 and len(observed[0].calls) == 1
    assert core._active is None
    with pytest.raises(ResponsesError, match="consumed or retired"):
        prepared.consume(request, _owned_prompt_source(owned))


def test_fss_rendered_prefix_over_limit_refuses_before_import(tmp_path, monkeypatch):
    from pllm.deployment.network import NetworkError

    exp, manifest, cfg, core, owned = _real_prompt_metadata(
        tmp_path, monkeypatch, "fss", template=" ".join(["hello"] * 17))
    observed = _observe_source_tokenizer(monkeypatch, owned)
    lease = _source_prompt_lease(exp, manifest, cfg, core, owned)
    with pytest.raises(NetworkError, match="input exceeds selected budget"):
        lease._admit_request({"input": "hello", "max_output_tokens": 4})
    assert observed[0].calls == [(" ".join(["hello"] * 17), True)]
    assert observed[0].closed == 1 and lease._requests == 0
    assert not list(tmp_path.glob("*.safetensors"))
    core.close()


@pytest.mark.parametrize("change", ["config", "model", "tokenizer_bytes", "source_lock"])
def test_real_source_lease_association_refuses_before_encode(tmp_path, monkeypatch, change):
    from dataclasses import replace
    from pllm import Model
    from pllm.deployment.network import NetworkError

    exp, manifest, cfg, core, owned = _real_prompt_metadata(tmp_path, monkeypatch, "fss")
    observed = _observe_source_tokenizer(monkeypatch, owned)
    lease = _source_prompt_lease(exp, manifest, cfg, core, owned)
    if change == "config":
        lease._tokenizer.original_config = {**cfg, "max_position_embeddings": 32}
    elif change == "model":
        core._source = replace(owned, model=Model.path(str(tmp_path / "foreign")))
    elif change == "tokenizer_bytes":
        lease._tokenizer.tokenizer_descriptor["model"] = b"{}"
    else:
        lease._tokenizer.source_lock_digest = "3" * 64
    with pytest.raises(NetworkError, match="selected source metadata"):
        lease._admit_request({"input": "hello", "max_output_tokens": 4})
    assert observed == [] and lease._requests == 0
    core.close()


def _cold_metadata_client(tmp_path, monkeypatch, kind, **kwargs):
    from pllm import Model
    from pllm.configuration import Deployment, ExecutionBudget, Experiment
    from pllm.profiles import PumaThreePartyCpu
    from pllm.runtime.replicated_client import ReplicatedClient

    exp, manifest, cfg, core, owned = _real_prompt_metadata(
        tmp_path, monkeypatch, "fss" if kind == "fss" else "gkm", **kwargs)
    if kind == "puma":
        core.close()
        exp = Experiment("puma-prompt-metadata", PumaThreePartyCpu(owned.model),
            Deployment.local(root="local://prompt-metadata"),
            ExecutionBudget(requests=1, max_input_tokens=16, max_new_tokens=4))
        core = ReplicatedClient(exp, {r: (f"http://{r.replace('_', '-')}.invalid/v1", "metadata")
                                     for r in ReplicatedClient.role_names})
    core._source = None
    core._source_checked = False
    return exp, manifest, cfg, core, owned


def _forbid_full_source_work(monkeypatch):
    from pllm import _native, model_loader
    from pllm.runtime import delegated_source, fss_source, replicated_source
    from pllm.runtime import delegated_client

    def forbidden(*args, **kwargs):
        pytest.fail("cold prompt crossed into full resolve/hash/inventory/native source admission")

    monkeypatch.setattr(model_loader, "resolve_model", forbidden)
    monkeypatch.setattr(model_loader, "resolve_huggingface_source", forbidden, raising=False)
    for module in (delegated_source, fss_source, replicated_source):
        for name in ("_files", "_store", "_locked", "admit_source", "resolve_client_source"):
            if hasattr(module, name):
                monkeypatch.setattr(module, name, forbidden)
    monkeypatch.setattr(delegated_client, "_admit_source", forbidden)
    for name in ("gkm_source_contract", "fss_source_contract", "replicated_source_contract"):
        monkeypatch.setattr(_native, name, forbidden, raising=False)


@pytest.mark.parametrize("kind", ["gkm", "fss", "puma"])
def test_cold_rendered_30_refuses_before_all_full_source_work(tmp_path, monkeypatch, kind):
    from pllm.runtime.http_gateway import GatewayError
    from pllm.runtime import tokenizer_admission

    exp, manifest, cfg, core, owned = _cold_metadata_client(tmp_path, monkeypatch, kind,
        template=" ".join(["hello"] * 29))
    encoded = []
    original = tokenizer_admission._PromptTokenizer.encode

    def observe(self, text, *, add_bos):
        ids = original(self, text, add_bos=add_bos)
        encoded.append(tuple(ids))
        return ids

    monkeypatch.setattr(tokenizer_admission._PromptTokenizer, "encode", observe)
    _forbid_full_source_work(monkeypatch)
    try:
        attempt = core.events({"input": "hello", "max_output_tokens": 4})
        with pytest.raises(GatewayError, match="input exceeds selected budget"):
            next(attempt)
        assert len(encoded) == 1 and len(encoded[0]) == 30
        assert core._source is None and not core._source_checked and core._active is None
        assert not list(tmp_path.glob("*.safetensors"))
    finally:
        core.close()


@pytest.mark.parametrize("kind", ["gkm", "fss", "puma"])
@pytest.mark.parametrize("missing_bos", [False, True])
def test_cold_snapshot_descriptor_preserves_factory_defaults(tmp_path, monkeypatch, kind, missing_bos):
    from pllm.runtime import tokenizer_admission

    exp, manifest, cfg, core, owned = _cold_metadata_client(
        tmp_path, monkeypatch, kind, missing_bos=missing_bos)
    _forbid_full_source_work(monkeypatch)
    snapshot = tokenizer_admission._capture_prompt_metadata(exp.pipeline.model,
        policy=core._prompt_descriptor_policy, capacity=core._capacity, input_rows=16, outputs=4)
    assert snapshot.tokenizer_descriptor == owned.tokenizer_descriptor
    expected = 2 if missing_bos and kind != "fss" else 0
    assert snapshot.tokenizer_descriptor["bos_token_id"] == expected
    assert snapshot.tokenizer_descriptor["eos_token_id"] == 0
    core.close()


@pytest.mark.parametrize("change", ["config", "tokenizer", "options", "request", "source_lock"])
def test_cold_prepared_one_way_binding_rejects_changes_and_burns(tmp_path, monkeypatch, change):
    from pllm.runtime.responses import ResponsesError

    exp, manifest, cfg, core, owned = _cold_metadata_client(tmp_path, monkeypatch, "fss")
    lease = _source_prompt_lease(exp, manifest, cfg, core, owned)
    _forbid_full_source_work(monkeypatch)
    body = {"input": "hello", "max_output_tokens": 4}
    prepared = lease._admit_request(body)
    assert prepared._payload.pending_source is not None
    if change in {"config", "tokenizer", "options"}:
        name = {"config": "config.json", "tokenizer": "tokenizer.json", "options": "tokenizer_config.json"}[change]
        with (tmp_path / name).open("ab") as stream:
            stream.write(b" ")  # Even equivalent JSON mutation invalidates original identity.
    elif change == "request":
        body["input"] = "changed"
    else:
        from dataclasses import replace
        owned = replace(owned, source_lock_digest="4" * 64)
    with pytest.raises(ValueError):
        prepared._bind_owned_source(body, owned)
    with pytest.raises(ResponsesError, match="consumed or retired"):
        prepared._bind_owned_source({"input": "hello", "max_output_tokens": 4}, owned)
    prepared.close()
    assert prepared._payload is None
    core.close()


@pytest.mark.parametrize("name", ["config.json", "tokenizer.json"])
def test_cold_snapshot_missing_required_metadata_never_resolves(tmp_path, monkeypatch, name):
    from pllm.runtime import tokenizer_admission

    exp, _, _, core, _ = _cold_metadata_client(tmp_path, monkeypatch, "fss")
    (tmp_path / name).unlink()
    _forbid_full_source_work(monkeypatch)
    with pytest.raises(ValueError, match="missing prompt metadata"):
        tokenizer_admission._capture_prompt_metadata(exp.pipeline.model,
            policy=core._prompt_descriptor_policy, capacity=core._capacity, input_rows=16, outputs=4)
    core.close()


@pytest.mark.parametrize("name,limit", [("config.json", 16 << 10),
    ("tokenizer.json", 256 << 10), ("tokenizer_config.json", 16 << 10)])
def test_cold_snapshot_oversize_metadata_never_resolves(tmp_path, monkeypatch, name, limit):
    from pllm.runtime import tokenizer_admission
    exp, _, _, core, _ = _cold_metadata_client(tmp_path, monkeypatch, "fss")
    (tmp_path / name).write_bytes(b" " * (limit + 1))
    _forbid_full_source_work(monkeypatch)
    with pytest.raises(ValueError, match="oversize/empty prompt metadata"):
        tokenizer_admission._capture_prompt_metadata(exp.pipeline.model,
            policy=core._prompt_descriptor_policy, capacity=core._capacity, input_rows=16, outputs=4)
    core.close()


def _pending_lifecycle_fixture(tmp_path, monkeypatch, *, backend="local", fail=None,
                               phase_hook=None, requests=2):
    """PURE control simulation; never provider/native admission qualification."""
    import hashlib
    from dataclasses import replace
    from pllm.deployment import execution
    from pllm.runtime import tokenizer_admission
    from pllm.runtime.responses import _prepared_prompt_request

    exp, manifest, cfg, core, owned = _real_prompt_metadata(tmp_path, monkeypatch, "fss")
    exp = replace(exp, budget=replace(exp.budget, requests=requests))
    core._declaration, core._experiment_budget = exp, exp.budget
    snapshot = tokenizer_admission._capture_prompt_metadata(exp.pipeline.model,
        policy=core._prompt_descriptor_policy, capacity=core._capacity,
        input_rows=16, outputs=4, expected_source_lock=owned.source_lock_digest)
    view = execution._TokenizerView._from_prompt_snapshot(snapshot)
    view._prompt_memory_bytes = 1 << 20
    phases, retired, closed_tokenizers = [], [], []
    clock = [0]

    def validate(**kwargs):
        if kwargs.get("evaluated_at_ms", 0) > 0:
            raise execution.NetworkError("planning validity expired")
        placement = {}
        digest = hashlib.sha256(b"pllm.network_placement.validated.v1\0" + execution.canonical(placement)).hexdigest()
        return {"placement_digest": digest}

    result = SimpleNamespace(experiment=exp, snapshot=None, validate=validate, digest="5" * 64)
    network = SimpleNamespace(backend=backend, snapshot=None, digest="6" * 64)
    original = tokenizer_admission._PromptTokenizer.close

    def close_tokenizer(value):
        closed_tokenizers.append(value)
        original(value)

    monkeypatch.setattr(tokenizer_admission._PromptTokenizer, "close", close_tokenizer)

    class Controls:
        cleanup_complete = False

        def close(self):
            retired.append("topology")
            self.cleanup_complete = True

    class Resource:
        def create(self, **kwargs):
            from pllm.runtime.client import ResponseStream
            prepared = _prepared_prompt_request.get()
            prepared.claim()
            if kwargs.get("stream"):
                def events():
                    try:
                        prepared.consume(kwargs, _owned_prompt_source(owned))
                        yield SimpleNamespace(type="response.created", response={"id": "simulation"})
                        yield SimpleNamespace(type="response.completed", response={"id": "simulation"})
                    finally:
                        prepared.close()
                return ResponseStream(events(), close=prepared.close)
            try:
                _, ids = prepared.consume(kwargs, _owned_prompt_source(owned))
                return SimpleNamespace(ids=ids)  # Not a native accepted result.
            finally:
                prepared.close()

    def activate(lease, prepared, body):
        phases.append("resolve")
        if phase_hook is not None:
            phase_hook(lease, "resolve")
        lease._check_activation_cancelled()
        if fail == "resolve":
            raise ValueError("simulated resolve failure")
        prepared._bind_owned_source(body, owned)
        lease._source_owner = owned
        lease._topology = Controls()
        for phase in ("reserve", "arm"):
            phases.append(phase)
            if phase_hook is not None:
                phase_hook(lease, phase)
            lease._check_activation_cancelled()
            if fail == phase:
                raise ValueError(f"simulated {phase} failure")
        lease._client = SimpleNamespace(_core=core, responses=Resource(), close=core.close)
        return execution.ExecutionBinding(result.digest, network.digest, execution.canonical(validate()))

    lease = execution.ExecutionLease(result, network, lambda: clock[0], None, view, None,
        _key=execution._LEASE_KEY, _pending_factory=activate)
    return lease, core, phases, retired, closed_tokenizers, clock


def test_pending_binding_unavailable_and_close_has_zero_activation(tmp_path, monkeypatch):
    from pllm.deployment.network import NetworkError
    lease, core, phases, retired, _, _ = _pending_lifecycle_fixture(tmp_path, monkeypatch)
    with pytest.raises(NetworkError, match="binding is unavailable"):
        _ = lease.binding
    lease.close()
    assert phases == retired == [] and lease._state == "closed"
    core.close()


def test_pending_oversize_30_never_activates_and_burns(tmp_path, monkeypatch):
    from pllm.deployment import execution
    from pllm.runtime import tokenizer_admission
    lease, core, phases, retired, closed, _ = _pending_lifecycle_fixture(tmp_path, monkeypatch)
    options = json.loads((tmp_path / "tokenizer_config.json").read_text())
    options["chat_template"] = " ".join(["hello"] * 29)
    (tmp_path / "tokenizer_config.json").write_text(json.dumps(options))
    # This is fresh pending capture, not mutation of an already prepared request.
    snapshot = tokenizer_admission._capture_prompt_metadata(lease._result.experiment.pipeline.model,
        policy=core._prompt_descriptor_policy, capacity=core._capacity, input_rows=16, outputs=4)
    lease._tokenizer = execution._TokenizerView._from_prompt_snapshot(snapshot)
    lease._tokenizer._prompt_memory_bytes = 1 << 20
    with pytest.raises(execution.NetworkError, match="input exceeds selected budget"):
        lease._construct_client().responses.create(input="hello", max_output_tokens=4)
    assert phases == retired == [] and lease._state == "closed"
    assert len(closed) == 1
    core.close()


def test_pending_metadata_changes_before_binding_cleans_without_topology(tmp_path, monkeypatch):
    def mutate(lease, phase):
        if phase == "resolve":
            with (tmp_path / "tokenizer_config.json").open("ab") as stream:
                stream.write(b" ")
    lease, core, phases, retired, _, _ = _pending_lifecycle_fixture(tmp_path, monkeypatch, phase_hook=mutate)
    with pytest.raises(ValueError, match="metadata changed"):
        lease._construct_client().responses.create(input="hello", max_output_tokens=4)
    assert phases == ["resolve"] and retired == [] and lease._state == "closed"
    core.close()


def test_pending_async_stream_retains_request_until_cancel_retirement(tmp_path, monkeypatch):
    import asyncio
    from pllm.runtime.client import AsyncResponsesResource
    lease, _, _, retired, closed, _ = _pending_lifecycle_fixture(tmp_path, monkeypatch)
    resource = AsyncResponsesResource(lease._construct_client().responses)

    async def run():
        stream = await resource.create(input="hello", max_output_tokens=4, stream=True)
        event = await anext(stream)
        assert event.type == "response.created"
        assert lease._request_owner is not None and lease._request_busy and closed == []
        await stream.aclose()

    asyncio.run(run())
    assert not lease._request_busy and lease._request_owner is None and len(closed) == 1
    lease.close()
    assert retired == ["topology"]


def test_pending_two_local_requests_encode_each_and_activate_once(tmp_path, monkeypatch):
    lease, _, phases, retired, closed, _ = _pending_lifecycle_fixture(tmp_path, monkeypatch)
    client = lease._construct_client()
    first = client.responses.create(input="hello", max_output_tokens=4)
    second = client.responses.create(input="hello", max_output_tokens=4)
    assert first.ids == second.ids
    assert phases == ["resolve", "reserve", "arm"] and lease._requests == 2
    assert len(closed) == 1  # Cold owner; second request uses retained factory tokenizer.
    assert lease.binding is not None and lease._state == "active"
    lease.close()
    assert retired == ["topology"] and lease._cleanup_complete


def test_pending_http_second_attempt_refuses_before_new_activation(tmp_path, monkeypatch):
    from pllm.deployment.network import NetworkError
    lease, _, phases, _, _, _ = _pending_lifecycle_fixture(tmp_path, monkeypatch, backend="http")
    client = lease._construct_client()
    client.responses.create(input="hello", max_output_tokens=4)
    with pytest.raises(NetworkError, match="one admitted attempt"):
        client.responses.create(input="hello", max_output_tokens=4)
    assert phases == ["resolve", "reserve", "arm"] and lease._requests == 1
    lease.close()


def test_pending_expiry_before_request_burns_without_activation(tmp_path, monkeypatch):
    from pllm.deployment.network import NetworkError
    lease, core, phases, retired, _, clock = _pending_lifecycle_fixture(tmp_path, monkeypatch)
    client = lease._construct_client()
    clock[0] = 1
    with pytest.raises(NetworkError, match="expired"):
        client.responses.create(input="hello", max_output_tokens=4)
    assert phases == retired == [] and lease._state == "closed"
    core.close()


@pytest.mark.parametrize("phase", ["resolve", "reserve", "arm"])
@pytest.mark.parametrize("cancel", [False, True])
def test_pending_failure_or_cancel_burns_and_cleans_partial_controls(tmp_path, monkeypatch, phase, cancel):
    from pllm.deployment.network import NetworkError

    def at_phase(lease, current):
        if cancel and current == phase:
            lease._activation_cancel.cancel()

    lease, core, phases, retired, closed, _ = _pending_lifecycle_fixture(tmp_path, monkeypatch,
        fail=None if cancel else phase, phase_hook=at_phase)
    with pytest.raises(NetworkError if cancel else ValueError):
        lease._construct_client().responses.create(input="hello", max_output_tokens=4)
    assert phases[-1] == phase and lease._state == "closed"
    assert len(closed) == 1 and not lease._request_busy
    assert bool(retired) is (phase != "resolve")
    with pytest.raises(NetworkError):
        _ = lease.binding
    core.close()


def test_pending_competing_request_does_not_cancel_first_activation(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    import threading
    from pllm.deployment.network import NetworkError
    entered, resume = threading.Event(), threading.Event()

    def block(lease, phase):
        if phase == "resolve":
            entered.set()
            assert resume.wait(3)

    lease, _, phases, _, _, _ = _pending_lifecycle_fixture(tmp_path, monkeypatch, phase_hook=block)
    client = lease._construct_client()
    with ThreadPoolExecutor(max_workers=1) as worker:
        first = worker.submit(client.responses.create, input="hello", max_output_tokens=4)
        try:
            assert entered.wait(3)
            with pytest.raises(NetworkError, match="already owns"):
                client.responses.create(input="hello", max_output_tokens=4)
            assert lease._state == "activating" and not lease.closed
        finally:
            resume.set()
        assert first.result(timeout=3).ids
    assert phases == ["resolve", "reserve", "arm"]
    lease.close()


def test_pending_async_cancellation_drains_late_activation_worker(tmp_path, monkeypatch):
    import asyncio
    import threading
    from pllm.runtime.client import AsyncResponsesResource
    from pllm.runtime.prepared_queue import PreparedCancellation
    entered, resume, cancelled = threading.Event(), threading.Event(), threading.Event()
    original_cancel = PreparedCancellation.cancel

    def mark_cancel(owner):
        original_cancel(owner)
        cancelled.set()

    monkeypatch.setattr(PreparedCancellation, "cancel", mark_cancel)

    def block(lease, phase):
        if phase == "reserve":
            entered.set()
            assert resume.wait(3)

    lease, core, phases, retired, _, _ = _pending_lifecycle_fixture(tmp_path, monkeypatch, phase_hook=block)
    resource = AsyncResponsesResource(lease._construct_client().responses)

    async def run():
        task = asyncio.create_task(resource.create(input="hello", max_output_tokens=4))
        assert await asyncio.wait_for(asyncio.to_thread(entered.wait), 3)
        task.cancel()
        assert await asyncio.wait_for(asyncio.to_thread(cancelled.wait), 3)
        resume.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run())
    assert lease._activation_thread is None and lease._activation_event.is_set()
    assert lease._state == "closed" and retired == ["topology"]
    assert phases == ["resolve", "reserve"]
    core.close()


def test_existing_eager_lease_still_constructs_client_without_deferred_activation(tmp_path, monkeypatch):
    from pllm.deployment import execution
    from pllm.runtime.client import _protected_client_type
    pending, core, _, _, _, _ = _pending_lifecycle_fixture(tmp_path, monkeypatch)
    calls = []
    pending._result.request = SimpleNamespace(policy=SimpleNamespace(max_client_native_threads=None))
    resource = SimpleNamespace(create=lambda **kwargs: None)
    client = SimpleNamespace(_core=core, responses=resource, close=core.close)
    topology = SimpleNamespace(client=lambda: (calls.append("client") or client),
                               close=lambda: calls.append("close"), cleanup_complete=True)
    eager = execution.ExecutionLease(pending._result, pending._network, pending._clock,
        topology, pending._tokenizer, SimpleNamespace(), _key=execution._LEASE_KEY)
    assert _protected_client_type(SimpleNamespace(client_runtime="legacy-generic")) is None
    assert eager._construct_client() is client and calls == ["client"]
    assert eager.binding is eager._binding and not eager._deferred
    eager.close()
    pending.close()
    assert calls == ["client", "close"]


def _pending_result_stub(exp, source_lock, *, snapshot=None):
    """Provisional planning metadata fixture; replay/validate edges are doubled.

    Not a real planning decision or native/source admission proof. The public
    open/lease state machine and original metadata/count paths remain real.
    """
    from pllm.deployment.execution import canonical
    from pllm.plan import PlanningResult

    if snapshot is None:
        snapshot_spec = {"network_id": "pending-fixture", "observed_at_ms": 0,
                         "expires_at_ms": 1000}
        snapshot = SimpleNamespace(**snapshot_spec,
            canonical_bytes=lambda: canonical(snapshot_spec), to_spec=lambda: dict(snapshot_spec))
    else:
        snapshot_spec = snapshot.to_spec()
    request_spec = {"source_lock_digest": source_lock, "model_plan": {"digest": "3" * 64},
                    "policy": {"max_observation_age_ms": 1000}}
    request = SimpleNamespace(source_lock_digest=source_lock,
        canonical_bytes=lambda: canonical(request_spec),
        model_plan=SimpleNamespace(digest="3" * 64, to_dict=lambda: {"transformations": []}),
        policy=SimpleNamespace(max_client_native_threads=1, max_observation_age_ms=1000,
                               allow_incomplete_execution=False, client_peak_memory_limit_bytes=None))
    document = {"schema": "pllm.planning_result.v1", "request": request_spec,
                "snapshot": snapshot_spec, "result_digest": "5" * 64,
                "status": "feasible", "coverage": {"exhaustive": True},
                "selection": {"configuration_digest": exp.configuration_digest(),
                    # Planning double price for bounded metadata ownership, NOT
                    # a source/body/runtime memory quote or native validation.
                    "costs": {"client_incremental_memory_bytes": 1 << 20},
                    "native_placement": {"roles": []}}}
    result = PlanningResult(canonical(document), request, snapshot, exp)
    return result, snapshot


def _real_activation_closure_fixture(tmp_path, monkeypatch, kind="gkm", *, backend="local",
                                     **metadata_options):
    """Production activation closure; replay/native/source/topology edges doubled.

    Generated public metadata, no native/source authentication qualification,
    provider sockets, checkpoint store, tensor/index/header or arithmetic. This
    tests ordering/lifecycle, NOT genuine replay or native source authentication.
    In particular this fixture never replaces _open_protected_pending or its
    nested activate function, unlike the earlier control-only simulation.
    """
    import hashlib
    from pllm.deployment import execution
    from pllm.runtime import servers
    from pllm.runtime.responses import _prepared_prompt_request

    exp, _, cfg, core, owned = _cold_metadata_client(tmp_path, monkeypatch, kind, **metadata_options)
    cls = type(core)
    state = {"calls": [], "planning_calls": [], "timeline": [], "clock": 0,
             "topologies": [], "owned": owned,
             "hook": None, "release_refused": False, "after_publish_expiry": False}
    placement = {"placement_digest": hashlib.sha256(
        b"pllm.network_placement.validated.v1\0" + execution.canonical({})).hexdigest()}

    def clock():
        lease = state.get("lease")
        if state["after_publish_expiry"] and lease is not None and lease._state == "active":
            return 5000
        return state["clock"]

    def validate(**kwargs):
        state["planning_calls"].append("validate")
        state["timeline"].append("validate")
        if kwargs.get("evaluated_at_ms", 0) >= 1000:
            raise execution.NetworkError("planning validity expired")
        return placement

    result, snapshot = _pending_result_stub(exp, owned.source_lock_digest)
    state["result"] = result
    network_spec = {"backend": backend, "network_id": "pending-fixture"}
    network = SimpleNamespace(backend=backend, snapshot=snapshot, digest="6" * 64,
                              to_spec=lambda: dict(network_spec))
    state["network"] = network

    def replay(_cls, document):
        state["planning_calls"].append("replay")
        state["timeline"].append("replay")
        assert execution.canonical(document) == result.canonical_bytes()
        if state.get("replay_error"):
            raise execution.NetworkError("simulated replay refusal")
        return state.get("replayed", result)

    def lower(config, **kwargs):
        state["planning_calls"].append("lower")
        state["timeline"].append("lower")
        assert config == cfg
        assert kwargs == {"batch": 1, "max_input_tokens": exp.budget.max_input_tokens,
                          "max_new_tokens": exp.budget.max_new_tokens}
        return SimpleNamespace(digest=state.get("lowered_digest", "3" * 64))

    from pllm import modeling
    monkeypatch.setattr(execution.PlanningResult, "from_spec", classmethod(replay))
    monkeypatch.setattr(execution.PlanningResult, "validate", lambda _self, **kw: validate(**kw))
    monkeypatch.setattr(modeling, "lower_model", lower)

    def phase(name):
        state["calls"].append(name)
        state["timeline"].append(name)
        if state["hook"] is not None:
            state["hook"](name)

    def resolve(source_cls, declaration):
        assert source_cls is cls and declaration is exp
        phase("resolve")
        return state["owned"]

    def ensure(value):
        assert value is core and value._source is state["owned"]
        phase("ensure")  # Original native/source check boundary, simulated.
        value._source_checked = True
        return value._source

    class Resource:
        def create(self, **kwargs):
            phase("execute")
            prepared = _prepared_prompt_request.get()
            prepared.claim()
            try:
                _, ids = prepared.consume(kwargs, _owned_prompt_source(state["owned"]))
                if "response" in state:
                    return state["response"](kwargs, ids)
                return SimpleNamespace(simulated=True)  # No candidate token/result oracle.
            finally:
                prepared.close()

    def close_client():
        phase("client-close")
        core.close()

    client = SimpleNamespace(_core=core, responses=Resource(), close=close_client)

    class Topology:
        def __init__(self, *args):
            lease = state["lease"]
            assert lease._request_owner._payload.pending_source is None
            assert lease._source_owner is state["owned"]
            phase("construct")
            self.cleanup_complete = False
            self.rows = []
            self.attempt = "7" * 64
            self.contract = SimpleNamespace(digest="8" * 64)
            self.attempt_digest = hashlib.sha256(b"pllm.execution_attempt.v2\0" + execution.canonical({
                "attempt_id": self.attempt, "execution_digest": self.contract.digest,
                "network_digest": network.digest})).hexdigest()
            self.expires_at_ms = 5000
            state["topologies"].append(self)

        def start(self, *args):
            self.rows.append("owned-reservation")
            phase("reserve")
            phase("arm")
            return placement

        def client(self):
            phase("client")
            return client

        def close(self):
            phase("release")
            if state["release_refused"]:
                raise execution.NetworkError("simulated release unavailable")
            self.rows.clear()
            self.cleanup_complete = True

    # Original factory dispatch and activation closure remain in production code.
    # Planning/lowering, source authentication and topology I/O are test doubles;
    # their order does not mint genuine execution authority in this unit test.
    monkeypatch.setattr(cls, "_resolve_prompt_source", classmethod(resolve))
    monkeypatch.setattr(cls, "_ensure_source", ensure)
    monkeypatch.setattr(servers, "build_roles", lambda declaration: Topology())
    monkeypatch.setattr(execution, "_LiveTopology", Topology)
    lease = execution._open_protected_pending(result, network, None, clock, 5, cfg, cls)
    state["lease"] = lease
    return lease, core, state


@pytest.mark.parametrize("kind", ["gkm", "fss", "puma"])
def test_real_activation_closure_authenticates_before_topology_once(tmp_path, monkeypatch, kind):
    lease, core, state = _real_activation_closure_fixture(tmp_path, monkeypatch, kind)
    try:
        assert state["calls"] == [] and lease._state == "pending"
        value = lease._construct_client().responses.create(input="hello", max_output_tokens=4)
        assert value.simulated and lease._state == "active"
        assert state["calls"][:7] == ["resolve", "construct", "reserve", "arm",
                                     "client", "ensure", "execute"]
        assert lease.binding is lease._binding
    finally:
        lease.close()
        core.close()
    assert all(t.cleanup_complete and not t.rows for t in state["topologies"])


@pytest.mark.parametrize("change", ["metadata", "source"])
def test_real_activation_closure_mismatch_never_starts_topology(tmp_path, monkeypatch, change):
    from dataclasses import replace
    lease, core, state = _real_activation_closure_fixture(tmp_path, monkeypatch)
    if change == "metadata":
        with (tmp_path / "tokenizer_config.json").open("ab") as stream:
            stream.write(b" ")
    else:
        state["owned"] = replace(state["owned"], config_source_bytes=state["owned"].config_source_bytes + b" ")
    try:
        with pytest.raises(ValueError):
            lease._construct_client().responses.create(input="hello", max_output_tokens=4)
        assert not state["topologies"] and "construct" not in state["calls"]
        assert lease.closed and lease._cleanup_complete and lease._binding is None
        assert state["calls"] == ([] if change == "metadata" else ["resolve"])
    finally:
        lease.close()
        core.close()


@pytest.mark.parametrize("backend", ["local", "http"])
@pytest.mark.parametrize("phase", ["ensure", "client"])
@pytest.mark.parametrize("release_refused", [False, True])
def test_real_activation_closure_late_expiry_cleans_before_publication(
        tmp_path, monkeypatch, backend, phase, release_refused):
    from pllm.deployment.network import NetworkError
    lease, core, state = _real_activation_closure_fixture(tmp_path, monkeypatch, backend=backend)
    state["release_refused"] = release_refused

    def expire(name):
        if name == phase:
            state["clock"] = 5000 if backend == "http" else 1000
            with pytest.raises(NetworkError, match="binding.*unavailable"):
                _ = lease.binding

    state["hook"] = expire
    try:
        with pytest.raises(NetworkError, match="expired"):
            lease._construct_client().responses.create(input="hello", max_output_tokens=4)
        assert lease._binding is None and "execute" not in state["calls"]
        assert lease.closed
        with pytest.raises(NetworkError, match="binding.*unavailable"):
            _ = lease.binding
        assert state["topologies"] and "release" in state["calls"]
        if release_refused:
            assert lease._state == "closing" and not lease._cleanup_complete
            assert lease._topology is state["topologies"][0]
            assert lease._topology.rows and lease._pending_metadata_owner is not None
        else:
            assert lease._state == "closed" and lease._cleanup_complete
            assert not state["topologies"][0].rows
    finally:
        state["release_refused"] = False
        lease.close()
        core.close()
    assert lease._cleanup_complete and not state["topologies"][0].rows


def test_real_activation_closure_immediate_postpublication_expiry_releases_active_owner(tmp_path, monkeypatch):
    from pllm.deployment.network import NetworkError
    lease, core, state = _real_activation_closure_fixture(tmp_path, monkeypatch, backend="http")
    state["after_publish_expiry"] = True
    try:
        with pytest.raises(NetworkError, match="expired"):
            lease._construct_client().responses.create(input="hello", max_output_tokens=4)
        assert "execute" not in state["calls"] and lease.closed and lease._cleanup_complete
        assert all(not t.rows for t in state["topologies"])
    finally:
        lease.close()
        core.close()


def test_real_activation_closure_pending_close_has_no_source_or_role_work(tmp_path, monkeypatch):
    lease, core, state = _real_activation_closure_fixture(tmp_path, monkeypatch)
    lease.close()
    assert state["calls"] == [] and not state["topologies"]
    assert lease.closed and lease._cleanup_complete
    core.close()


@pytest.mark.parametrize("phase", ["resolve", "ensure"])
def test_real_activation_closure_cancel_before_or_after_start_drains_owner(tmp_path, monkeypatch, phase):
    from pllm.deployment.network import NetworkError
    lease, core, state = _real_activation_closure_fixture(tmp_path, monkeypatch)

    def cancel(name):
        if name == phase:
            lease.close()  # Same activation thread may signal, not self-join.
            assert lease._state == "closing" and not lease._activation_event.is_set()
            if phase == "ensure":
                assert lease._topology is state["topologies"][0]
                assert lease._topology.rows and not lease._cleanup_complete

    state["hook"] = cancel
    try:
        with pytest.raises(NetworkError, match="cancelled"):
            lease._construct_client().responses.create(input="hello", max_output_tokens=4)
        assert lease._binding is None and "execute" not in state["calls"]
        assert lease._activation_event.is_set() and lease._cleanup_complete
        assert all(not t.rows for t in state["topologies"])
        if phase == "resolve":
            assert not state["topologies"]
    finally:
        lease.close()
        core.close()


@pytest.mark.parametrize("symlink", [False, True])
def test_prompt_named_reader_accepts_regular_or_cached_file_symlink(tmp_path, symlink):
    from pllm.runtime.tokenizer_admission import _read_named
    target = tmp_path / "config.json"
    if symlink:
        (tmp_path / "metadata-blob").write_bytes(b"{}")
        target.symlink_to("metadata-blob")
    else:
        target.write_bytes(b"{}")
    raw, identity = _read_named(tmp_path, "config.json", 16)
    assert raw == b"{}" and identity is not None


def test_prompt_named_reader_fifo_swap_refuses_without_writer_or_timeout(tmp_path, monkeypatch):
    import os
    from pathlib import Path
    from pllm.runtime.tokenizer_admission import _read_named, _PromptMetadataError
    if not hasattr(os, "mkfifo") or not hasattr(os, "O_NONBLOCK"):
        pytest.skip("FIFO race regression requires nonblocking FIFO platform")
    path = tmp_path / "config.json"
    path.write_bytes(b"{}")
    original = os.open
    opened = []

    def swap(name, flags, *args, **kwargs):
        if Path(name) == path:
            # Assert BEFORE creating FIFO or invoking OS open: a regression in
            # flags fails deterministically, never hangs waiting for a writer.
            assert flags & os.O_NONBLOCK
            path.unlink()
            os.mkfifo(path)
            opened.append(True)
        return original(name, flags, *args, **kwargs)

    def forbidden_read(*args, **kwargs):
        pytest.fail("nonregular opened descriptor reached buffered read")

    monkeypatch.setattr(os, "open", swap)
    monkeypatch.setattr(os, "fdopen", forbidden_read)
    with pytest.raises(_PromptMetadataError, match="changed/nonregular"):
        _read_named(tmp_path, "config.json", 16)
    assert opened == [True]  # No thread, writer, sleep or timeout used.


@pytest.mark.parametrize("change", ["inode", "content"])
def test_prompt_named_reader_replacement_and_content_change_refuse(tmp_path, monkeypatch, change):
    import os
    from pathlib import Path
    from pllm.runtime.tokenizer_admission import _read_named, _PromptMetadataError
    path = tmp_path / "config.json"
    path.write_bytes(b"{}")
    if change == "inode":
        original = os.open

        def swapped(name, flags, *args, **kwargs):
            if Path(name) == path:
                replacement = tmp_path / "replacement"
                replacement.write_bytes(b"{}")
                replacement.replace(path)
            return original(name, flags, *args, **kwargs)

        monkeypatch.setattr(os, "open", swapped)
    else:
        original = os.fdopen

        def changed(descriptor, *args, **kwargs):
            path.write_bytes(b"{}x")
            return original(descriptor, *args, **kwargs)

        monkeypatch.setattr(os, "fdopen", changed)
    with pytest.raises(_PromptMetadataError, match="changed"):
        _read_named(tmp_path, "config.json", 16)


@pytest.mark.parametrize("raw", [b"{" + b" " * 16, b"not JSON"])
def test_prompt_named_reader_bounded_oversize_or_invalid_json(tmp_path, raw):
    from pllm.runtime.tokenizer_admission import _read_named, _json_metadata, _PromptMetadataError
    (tmp_path / "config.json").write_bytes(raw)
    with pytest.raises(_PromptMetadataError):
        value, _ = _read_named(tmp_path, "config.json", 16)
        _json_metadata(value, error=_PromptMetadataError, label="config")


def test_real_pending_metadata_capture_fifo_swap_has_zero_activation(tmp_path, monkeypatch):
    import os
    from pathlib import Path
    from pllm.deployment import execution
    from pllm.runtime.tokenizer_admission import _PromptMetadataError
    if not hasattr(os, "mkfifo") or not hasattr(os, "O_NONBLOCK"):
        pytest.skip("FIFO race regression requires nonblocking FIFO platform")
    # Retain the actual pending-open closure factory; swap the config only after
    # its lstat/stat check, before open. There is no writer or timeout success.
    exp, _, cfg, core, _ = _cold_metadata_client(tmp_path, monkeypatch, "fss")
    cls = type(core)
    calls = []

    def forbidden(*args, **kwargs):
        calls.append(True)
        pytest.fail("metadata FIFO reached source activation")

    monkeypatch.setattr(cls, "_resolve_prompt_source", classmethod(forbidden))
    path = tmp_path / "config.json"
    original = os.open

    def swap(name, flags, *args, **kwargs):
        if Path(name) == path:
            assert flags & os.O_NONBLOCK
            path.unlink()
            os.mkfifo(path)
        return original(name, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", swap)
    result = SimpleNamespace(experiment=exp, request=SimpleNamespace(source_lock_digest=None))
    try:
        with pytest.raises(_PromptMetadataError, match="changed/nonregular"):
            execution._open_protected_pending(result, SimpleNamespace(backend="local"),
                None, lambda: 0, 5, cfg, cls)
        assert calls == []
    finally:
        core.close()


@pytest.mark.parametrize("kind", ["gkm", "puma", "fss"])
def test_public_pending_open_facade_oversize_has_zero_replay_lower_or_source(
        tmp_path, monkeypatch, kind):
    from pllm import _native, model_loader, modeling
    from pllm.deployment import execution

    # 29 literal public words + the existing BOS = 30 IDs, original budget 16.
    old, core, state = _real_activation_closure_fixture(
        tmp_path, monkeypatch, kind, template=" ".join(["hello"] * 29))
    old.close()
    forbidden_calls = []

    def forbidden(*args, **kwargs):
        forbidden_calls.append(True)
        pytest.fail("pre-count path reached planning/native/source/topology")

    monkeypatch.setattr(execution, "NetworkSpec", type(state["network"]))  # Typed metadata double.
    monkeypatch.setattr(execution.PlanningResult, "from_spec", classmethod(forbidden))
    monkeypatch.setattr(execution.PlanningResult, "validate", forbidden)
    monkeypatch.setattr(modeling, "lower_model", forbidden)
    monkeypatch.setattr(_native, "lower_model", forbidden)
    monkeypatch.setattr(_native, "resolve_experiment", forbidden)
    for name in ("gkm_decoder_contract", "fss_decoder_contract", "replicated_decoder_contract"):
        if hasattr(_native, name):
            monkeypatch.setattr(_native, name, forbidden)
    monkeypatch.setattr(model_loader, "resolve_model", forbidden)
    monkeypatch.setattr(type(core), "_resolve_prompt_source", classmethod(forbidden))
    monkeypatch.setattr(type(core), "_ensure_source", forbidden)
    # An injected unchecked owner is NOT allowed to turn metadata into native
    # contract authority before count. Original checked-owner borrowing differs.
    core._source = state["owned"]
    core._source_checked = False
    lease = execution.open_execution(state["result"], network=state["network"], clock=lambda: 0)
    state["lease"] = lease
    try:
        client = lease._construct_client()
        assert lease._state == "pending" and not forbidden_calls
        with pytest.raises(execution.NetworkError, match="input exceeds"):
            client.responses.create(input="hello", max_output_tokens=4)
        assert not forbidden_calls and state["planning_calls"] == []
        assert not state["topologies"] and lease._binding is None
        assert lease.closed and lease._cleanup_complete
    finally:
        lease.close()
        core.close()


@pytest.mark.parametrize("kind", ["gkm", "puma", "fss"])
def test_pending_count_then_original_replay_lower_auth_validate_reserve_order(
        tmp_path, monkeypatch, kind):
    lease, core, state = _real_activation_closure_fixture(tmp_path, monkeypatch, kind)
    request_view_type = type(lease._tokenizer)
    original_tokenizer = request_view_type.tokenizer
    request_views = []

    class Counted:
        def __init__(self, inner):
            self.inner = inner

        def encode(self, text, **kwargs):
            state["timeline"].append("count")
            return self.inner.encode(text, **kwargs)

        def close(self):
            close = getattr(self.inner, "close", None)
            if close is not None:
                close()

    def tokenize_request_view(view):
        request_views.append(id(view))  # Observe identity without retaining metadata custody.
        return Counted(original_tokenizer(view))

    monkeypatch.setattr(request_view_type, "tokenizer", tokenize_request_view)
    try:
        assert lease._construct_client().responses.create(input="hello", max_output_tokens=4).simulated
        timeline = state["timeline"]
        assert timeline[:6] == ["count", "replay", "validate", "lower", "resolve", "validate"]
        assert timeline.index("lower") < timeline.index("construct") < timeline.index("reserve")
        assert timeline.index("reserve") < timeline.index("arm") < timeline.index("execute")
        assert timeline.count("count") == timeline.count("replay") == timeline.count("lower") == 1
        assert len(request_views) == 1 and request_views[0] != id(lease._tokenizer)
        assert lease._result.canonical_bytes() == state["result"].canonical_bytes()
    finally:
        lease.close()
        core.close()


@pytest.mark.parametrize("failure", ["replay", "config", "expiry", "identity"])
def test_pending_original_planning_failures_never_reserve(tmp_path, monkeypatch, failure):
    from pllm.deployment import execution
    lease, core, state = _real_activation_closure_fixture(tmp_path, monkeypatch)
    if failure == "replay":
        state["replay_error"] = True
    elif failure == "config":
        state["lowered_digest"] = "4" * 64
    elif failure == "expiry":
        state["clock"] = 1000
    else:
        from dataclasses import replace
        document = state["result"].to_spec()
        document["result_digest"] = "4" * 64
        lease._result = replace(state["result"], _document=execution.canonical(document))
    try:
        with pytest.raises(execution.NetworkError):
            lease._construct_client().responses.create(input="hello", max_output_tokens=4)
        assert not state["topologies"] and "reserve" not in state["calls"]
        assert "resolve" not in state["calls"] and lease._binding is None
    finally:
        lease.close()
        core.close()


def test_active_deferred_budget_refusal_has_zero_native_revalidation(tmp_path, monkeypatch):
    from pllm.deployment import execution
    lease, core, state = _real_activation_closure_fixture(tmp_path, monkeypatch)
    from pllm import _native

    def no_resolution(*args, **kwargs):
        pytest.fail("repeat-request admission reached native resolution before count")

    monkeypatch.setattr(_native, "resolve_experiment", no_resolution)
    try:
        # Construct the real facade but test only the exclusion/check boundary;
        # original budget is one. No extra attempt/activation is manufactured.
        lease._construct_client()
        assert state["planning_calls"] == []
        lease._reserve_request()
        lease._release_request()
        assert state["planning_calls"] == []
        lease._state = "active"
        lease._requests = lease._result.experiment.budget.requests
        with pytest.raises(execution.NetworkError, match="budget exhausted"):
            lease._admit_request({"input": "hello", "max_output_tokens": 4})
        assert state["planning_calls"] == []  # No native revalidation before count/refusal.
    finally:
        lease.close()
        core.close()


@pytest.fixture
def pending_actual_delivery(harness, monkeypatch):
    """Real protected core/attempt and accept/abort methods; transport is doubled.

    Activation here is a lifecycle double, NOT planning/source authentication.
    Those production edges are covered separately above with recorded stubs.
    """
    from pllm.deployment import execution
    from pllm.runtime.client import AsyncResponsesResource
    h = harness
    h.core._source = h.source
    h.core._source_checked = True  # Already-owned metadata fixture, not native authority.
    accepted, aborted = [], []
    accept = h.core._accept_decode_response
    abort = h.core._abort_decode_response

    def accept_actual(response_id):
        accepted.append((h.core, response_id))
        return accept(response_id)

    def abort_actual(response_id, **kwargs):
        aborted.append((h.core, response_id))
        return abort(response_id, **kwargs)

    monkeypatch.setattr(h.core, "_accept_decode_response", accept_actual)
    monkeypatch.setattr(h.core, "_abort_decode_response", abort_actual)
    result = SimpleNamespace(experiment=h.experiment, snapshot=None, validate=lambda **kw: None)

    class MetadataLease(execution.ExecutionLease):
        def _admit_request(self, request):
            request.setdefault("model", h.core.default_model)
            request.setdefault("max_output_tokens", h.experiment.budget.max_new_tokens)
            prepared = h.core._prepare_request_prompt(request)
            self._requests += 1
            return prepared

    client = SimpleNamespace(_core=h.core, responses=h.responses, close=h.core.close)

    def activate(lease, prepared, request):
        lease._client = client
        return SimpleNamespace()  # Local lifecycle double, no execution binding proof.

    lease = MetadataLease(result, SimpleNamespace(backend="local", snapshot=None), lambda: 0,
        SimpleNamespace(close=lambda: None), None, None, _key=execution._LEASE_KEY,
        _pending_factory=activate)
    resource = execution._LeaseResponses(lease, None)
    value = SimpleNamespace(h=h, lease=lease, resource=resource,
        async_resource=AsyncResponsesResource(resource), client=client,
        accepted=accepted, aborted=aborted)
    yield value
    lease._client = client  # Restore the original owner if a test changed only the lookup slot.
    lease.close()


def test_pending_delivery_accessor_is_nonactivating(pending_actual_delivery):
    from pllm.deployment.execution import NetworkError
    p = pending_actual_delivery
    with pytest.raises(NetworkError, match="before activation"):
        _ = p.resource._delivery_owner
    assert p.lease._state == "pending" and p.lease._client is None
    assert p.h.source.encodings == p.h.calls == []


@pytest.mark.asyncio
async def test_pending_async_completion_uses_actual_postcreate_owner(pending_actual_delivery):
    p = pending_actual_delivery
    response = await p.async_resource.create(input="hello")
    assert p.resource._delivery_owner is p.h.core
    assert p.accepted == [(p.h.core, response.id)] and not p.aborted
    assert len(p.h.source.encodings) == 1 and p.h.core._active is None


@pytest.mark.asyncio
async def test_active_deferred_second_count_precedes_native_revalidation(pending_actual_delivery):
    p = pending_actual_delivery
    observed = []
    p.lease._result.validate = lambda **kw: observed.append(len(p.h.source.encodings))
    await p.async_resource.create(input="hello")
    await p.async_resource.create(input="hello")
    assert observed == [1, 2]
    assert len(p.h.source.encodings) == len(p.accepted) == 2
    assert not p.aborted and p.h.core._active is None


@pytest.mark.asyncio
async def test_pending_async_terminal_retains_owner_not_mutable_lease_lookup(pending_actual_delivery):
    p = pending_actual_delivery
    stream = await p.async_resource.create(input="hello", stream=True)
    assert p.resource._delivery_owner is p.h.core
    first = await stream.__anext__()
    assert first.type == "response.created"
    # Not an authority replacement: the stream must retain the original core.
    p.lease._client = SimpleNamespace(_core=object())
    try:
        events = [first, *[event async for event in stream]]
        assert events[-1].type == "response.completed"
        assert p.accepted == [(p.h.core, events[-1].response["id"])]
        assert not p.aborted and len(p.h.source.encodings) == 1
    finally:
        await stream.aclose()
        p.lease._client = p.client


@pytest.mark.asyncio
async def test_pending_async_cancel_after_completed_worker_aborts_actual_owner(
        pending_actual_delivery, monkeypatch):
    import asyncio
    import threading
    p = pending_actual_delivery
    completed, release = threading.Event(), threading.Event()
    original = p.h.responses.create

    def blocked(**request):
        response = original(**request)
        completed.set()
        assert release.wait(2), "test must release completed worker"
        return response

    monkeypatch.setattr(p.h.responses, "create", blocked)
    task = asyncio.create_task(p.async_resource.create(input="hello"))
    try:
        assert await asyncio.to_thread(completed.wait, 2)
        task.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not p.accepted and len(p.aborted) == 1
        assert p.aborted[0][0] is p.h.core
        assert p.aborted[0][1] not in p.h.core.records
        assert p.h.core._active is None and len(p.h.source.encodings) == 1
    finally:
        release.set()
        if not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass


@pytest.mark.asyncio
async def test_pending_async_cancel_after_event_aborts_captured_owner(pending_actual_delivery):
    p = pending_actual_delivery
    stream = await p.async_resource.create(input="hello", stream=True)
    event = await stream.__anext__()
    response_id = event.response["id"]
    p.lease._client = SimpleNamespace(_core=object())
    try:
        await stream.aclose()
        assert p.aborted == [(p.h.core, response_id)] and not p.accepted
        assert response_id not in p.h.core.records and p.h.core._active is None
    finally:
        p.lease._client = p.client


@pytest.mark.asyncio
async def test_pending_async_unstarted_close_retires_without_response_id(pending_actual_delivery):
    p = pending_actual_delivery
    stream = await p.async_resource.create(input="hello", stream=True)
    assert len(p.h.source.encodings) == 1 and not p.h.calls
    await stream.aclose()
    assert p.accepted == p.aborted == []
    assert p.h.core._active is None and p.h.source.tokenizers_closed == 1


@pytest.mark.asyncio
async def test_pending_async_activation_cancel_drains_partial_without_response_id(tmp_path, monkeypatch):
    import asyncio
    import threading
    from pllm.runtime.client import AsyncResponsesResource

    lease, core, state = _real_activation_closure_fixture(tmp_path, monkeypatch)
    entered, release = threading.Event(), threading.Event()

    def pause(phase):
        if phase == "reserve":
            entered.set()
            assert release.wait(2), "test must release partial activation"

    def no_response_id(*args, **kwargs):
        pytest.fail("pre-response failure must not invent delivery authority/ID")

    state["hook"] = pause
    monkeypatch.setattr(core, "_accept_decode_response", no_response_id)
    monkeypatch.setattr(core, "_abort_decode_response", no_response_id)
    resource = AsyncResponsesResource(lease._construct_client().responses)
    task = asyncio.create_task(resource.create(input="hello", max_output_tokens=4))
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        task.cancel()
        cancellation = lease._activation_cancel
        assert await asyncio.to_thread(cancellation.event.wait, 2)
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert lease._cleanup_complete and lease._activation_event.is_set()
        assert not lease._request_busy and lease._binding is None
        assert all(t.cleanup_complete and not t.rows for t in state["topologies"])
        assert "execute" not in state["calls"]
    finally:
        release.set()
        if not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        lease.close()
        core.close()


@pytest.mark.parametrize("kind", ["puma", "fss"])
def test_valid_count_still_requires_original_native_resolution(tmp_path, monkeypatch, kind):
    from pllm import _native
    lease, core, state = _real_activation_closure_fixture(tmp_path, monkeypatch, kind)
    original = _native.resolve_experiment
    calls = []

    def observe(*args, **kwargs):
        assert lease._requests == 1 and lease._request_owner is not None
        assert 0 < len(lease._request_owner._payload.ids) <= lease._result.experiment.budget.max_input_tokens
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(_native, "resolve_experiment", observe)
    try:
        client = lease._construct_client()
        assert calls == []
        assert client.responses.create(input="hello", max_output_tokens=4).simulated
        assert calls  # Native composition resolution, not native source authentication.
    finally:
        lease.close()
        core.close()


@pytest.mark.parametrize("kind", ["gkm", "puma"])
@pytest.mark.parametrize("model_id", [None, "explicit-prompt-identity"])
def test_pending_model_identity_matches_original_resolution_after_count(
        tmp_path, monkeypatch, kind, model_id):
    from pllm import _native
    from pllm.configuration import Experiment
    from pllm.deployment import execution

    lease, core, state = _real_activation_closure_fixture(
        tmp_path, monkeypatch, kind, model_id=model_id)
    # Independent conventions from ExperimentProfile, not candidate/helper output.
    expected = model_id if model_id is not None else (
        str(tmp_path) if kind == "gkm" else tmp_path.resolve().name)
    full_resolution, native_resolution = [], []
    original_resolve = Experiment.resolve
    original_native = _native.resolve_experiment

    def observe_full(value):
        assert lease._requests == 1 and lease._state == "activating"
        full_resolution.append(value)
        return original_resolve(value)

    def guard_native(*args, **kwargs):
        # Stays armed before count; only original post-count resolution may pass.
        assert lease._requests == 1 and lease._request_owner is not None
        native_resolution.append(True)
        return original_native(*args, **kwargs)

    monkeypatch.setattr(Experiment, "resolve", observe_full)
    monkeypatch.setattr(_native, "resolve_experiment", guard_native)
    try:
        assert execution._provisional_model_id(lease._result.experiment) == expected
        assert lease.default_model == expected
        client = lease._construct_client()
        assert client._core is lease and client._core.default_model == expected
        assert full_resolution == native_resolution == []
        assert client.responses.create(input="hello", max_output_tokens=4).simulated
        assert full_resolution == [lease._result.experiment]
        if kind == "puma":
            assert native_resolution
        else:
            assert not native_resolution  # Proprietary resolution is originally Python-only.
        assert lease._request_owner is None and lease._requests == 1
    finally:
        lease.close()
        core.close()


@pytest.mark.parametrize("threads", [1, 2])
def test_client_local_real_thread_binding_remains_required(threads):
    from pllm.deployment import execution
    from pllm.runtime.client import RuntimeClient

    # Metadata state for the REAL binder. No engine/model is constructed.
    state = SimpleNamespace(_transformer_states={}, _native_worker_limit=None,
                            _local_engine=SimpleNamespace(kernel=SimpleNamespace(threads=threads)))
    core = SimpleNamespace(_bind_native_worker_limit=lambda limit:
                          RuntimeClient._bind_native_worker_limit(state, limit))
    client = SimpleNamespace(_core=core, responses=SimpleNamespace(core=core), close=lambda: None)
    profile = SimpleNamespace(client_runtime="compiled_client_local_v1")
    result = SimpleNamespace(snapshot=None, validate=lambda **kw: None,
        experiment=SimpleNamespace(resolve=lambda: profile),
        request=SimpleNamespace(policy=SimpleNamespace(max_client_native_threads=1)))
    topology = SimpleNamespace(client=lambda: client, close=lambda: None, cleanup_complete=True)
    lease = execution.ExecutionLease(result, SimpleNamespace(backend="local", snapshot=None),
        lambda: 0, topology, None, None, _key=execution._LEASE_KEY)
    try:
        if threads == 1:
            assert lease._construct_client() is client
            assert state._native_worker_limit == 1
        else:
            with pytest.raises(ValueError, match="model loading exceeds"):
                lease._construct_client()
            assert state._native_worker_limit is None
    finally:
        lease.close()


def test_fss_16_input_four_outputs_retains_native_capability_refusal(tmp_path, monkeypatch):
    from pllm import _native
    from test_fss_source import ROPE

    # Preserve the existing native invalid-geometry control, not a claim that
    # Experiment.resolve alone validates decoder prefix capacity. This authored
    # constructor-refusal case needs its own explicit future native grant; it is
    # NOT included in a no-native metadata/parse gate.
    original_builder = _native.FssSourceBuilder
    exp, _, config, core, _ = _cold_metadata_client(tmp_path, monkeypatch, "fss")
    try:
        assert exp.budget.max_input_tokens == 13 and exp.budget.max_new_tokens == 4
        # Separate invalid-capability control, NOT the positive oversize fixture.
        try:
            invalid = original_builder(json.dumps(config, sort_keys=True).encode(), 16, 4, ROPE)
        except ValueError:
            pass  # Constructor refused; no payload append or protocol operation.
        else:
            invalid.close()
            pytest.fail("16+3 prefix unexpectedly admitted by the capsule geometry")
    finally:
        core.close()
