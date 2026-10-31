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


@pytest.mark.parametrize("template", [False, 0, [], {}])
@pytest.mark.parametrize("descriptor", [{}, {"chat_template": None}])
def test_malformed_fallback_template_refuses(descriptor, template):
    with pytest.raises(ResponsesError, match="template failed"):
        _render_prompt(normalize_input("hello"), descriptor, template=template)


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


def _count_cleanup_fixture(monkeypatch, kind="gkm", case="success", failures=1):
    import threading
    from contextvars import ContextVar
    from pllm.deployment import resources
    from pllm.runtime import responses, tokenizer_admission
    from pllm.runtime.delegated_client import DelegatedClient
    from pllm.runtime.fss_client import FssClient
    from pllm.runtime.replicated_client import ReplicatedClient

    cls = {"gkm": DelegatedClient, "fss": FssClient, "puma": ReplicatedClient}[kind]
    core = cls.__new__(cls)
    core._lock = threading.RLock()
    core._closing = False
    core._active = core._compact_cleanup = core._failed_request_cleanup = None
    core._count_preparation = None
    core._fss_failure_observer = core._result_hook = None
    core._tokenizer_scopes = set()
    core._tokenizer_owner = ContextVar("count_cleanup_tokenizer", default=None)
    core._capacity = "capsule"
    core._source, core._source_checked = None, False
    core.records = {"sentinel": "preserve until count retires"}
    core.default_model = "fixture"
    core._experiment_budget = SimpleNamespace(max_input_tokens=16, max_new_tokens=4)
    s = source("hello")
    core._declaration = SimpleNamespace(pipeline=SimpleNamespace(model=s.model))
    descriptor = {**s.tokenizer_descriptor, "vocab_size": 256}
    control = SimpleNamespace(failures=failures, close_hook=None, encode_hook=None)
    tokenizer = Tokenizer()
    attempts, reservations, originals, cleanup_errors = [], [], [], []
    original_error = (ResponsesError("count render failed", param="input")
                      if case == "render" else ValueError("count encode failed"))
    original_error.add_note("original detail")

    def encode(text, *, add_bos):
        tokenizer.calls.append((text, add_bos))
        if control.encode_hook is not None:
            control.encode_hook()
        if case == "encode":
            raise original_error
        return [4] * (17 if case == "over_budget" else 2)

    def close():
        import sys
        slot = core._count_preparation
        attempts.append((slot, slot.owner, slot.owner._metadata_owner, tokenizer))
        originals.append(sys.exception())
        if control.close_hook is not None:
            control.close_hook()
        if control.failures:
            control.failures -= 1
            error = RuntimeError("count tokenizer busy")
            cleanup_errors.append(error)
            raise error
        tokenizer.closed += 1

    tokenizer.encode, tokenizer.close = encode, close
    snapshot = SimpleNamespace(config={}, tokenizer_descriptor=descriptor,
        expected_source_lock=s.source_lock_digest, tokenizer=lambda: tokenizer,
        association=lambda: _prompt_source(s.model, {}, descriptor))
    monkeypatch.setattr(tokenizer_admission, "_capture_prompt_metadata", lambda *a, **k: snapshot)
    if case == "render":
        def render(*args, **kwargs):
            raise original_error
        monkeypatch.setattr(responses, "_render_prompt", render)

    # Use the existing module ledger/reservation, fake only host observation.
    capacity = resources._capacity
    baseline = dict(capacity._owners)
    monkeypatch.setattr(capacity, "observe", lambda: SimpleNamespace(
        headroom_bytes=(1 << 40) - sum(capacity._owners.values())))
    original_reserve = resources.reserve_incremental_memory

    def reserve(amount):
        assert core._count_preparation.state == "preparing"
        assert core._count_preparation.owner._metadata_owner is not None
        reservation = original_reserve(amount)
        reservations.append(reservation)
        return reservation

    monkeypatch.setattr(resources, "reserve_incremental_memory", reserve)
    _forbid_full_source_work(monkeypatch)
    return SimpleNamespace(core=core, tokenizer=tokenizer, control=control,
        attempts=attempts, reservations=reservations, capacity=capacity, baseline=baseline,
        originals=originals, original_error=original_error, cleanup_errors=cleanup_errors)


@pytest.mark.parametrize("kind", ["gkm", "fss", "puma"])
@pytest.mark.parametrize("case", ["success", "over_budget", "render", "encode"])
@pytest.mark.parametrize("failures", [1, 3])
def test_count_cleanup_retains_actual_owner_and_reservation(monkeypatch, kind, case, failures):
    from pllm.runtime.client import ProtocolError
    from pllm.runtime.http_gateway import GatewayError

    f = _count_cleanup_fixture(monkeypatch, kind, case, failures)
    try:
        expected = {"success": RuntimeError, "over_budget": GatewayError,
                    "render": GatewayError, "encode": ValueError}[case]
        with pytest.raises(expected) as caught:
            f.core._response_input_tokens("fixture", "hello")
        if case == "success":
            assert caught.value is f.cleanup_errors[0]
        elif case == "render":
            assert caught.value.__cause__ is f.original_error is f.originals[0]
            assert str(caught.value) == str(f.original_error)
            assert caught.value.param == "input"
        else:
            assert caught.value is f.originals[0]
            if case == "encode":
                assert caught.value is f.original_error
        primary = caught.value.__cause__ if case == "render" else caught.value
        notes = list(getattr(primary, "__notes__", ()))
        if case != "success":
            assert notes[-1] == "token count cleanup retained owner: RuntimeError"
            if case in {"render", "encode"}:
                assert notes[0] == "original detail"
        slot = f.core._count_preparation
        owner, reservation = slot.owner, f.reservations[0]
        metadata = owner._metadata_owner
        assert slot.state == "pending" and slot.reservation is reservation
        assert owner._tokenizer is f.tokenizer and metadata is not None
        assert (owner._payload is None) == (case in {"render", "encode"})
        assert reservation._failed == (owner,) and reservation.pending_retirement
        assert reservation._token.retained is reservation
        assert f.capacity._owners[reservation._token] == reservation._amount == 1 << 20
        assert f.core.records and f.tokenizer.closed == 0
        def assert_rejected():
            attempts = len(f.attempts)
            for admit in (
                    lambda: f.core._response_input_tokens("fixture", "hello"),
                    lambda: f.core.events({"model": "fixture", "input": "hello"}),
                    lambda: f.core.tokenizer_scope().__enter__()):
                with pytest.raises(ProtocolError, match="pending token count|closed"):
                    admit()
            assert len(f.reservations) == 1 and len(f.attempts) == attempts
            assert f.core._active is None and not f.core._tokenizer_scopes
            assert f.core._count_preparation is slot and slot.owner is owner
            assert slot.reservation is reservation and owner._metadata_owner is metadata
            assert reservation._failed == (owner,) and reservation.pending_retirement
            assert reservation._token.retained is reservation
            assert f.capacity._owners[reservation._token] == 1 << 20

        assert_rejected()
        for _ in range(failures - 1):
            with pytest.raises(RuntimeError, match="count tokenizer busy"):
                f.core.close()
            assert f.core._count_preparation is slot and slot.state == "pending"
            assert slot.owner is owner and owner._metadata_owner is metadata
            assert reservation._failed == (owner,) and reservation.pending_retirement
            assert reservation._token.retained is reservation
            assert f.capacity._owners[reservation._token] == 1 << 20
            assert f.core.records and f.tokenizer.closed == 0
            assert_rejected()
        f.core.close()
        assert f.core._count_preparation is None and f.core._closing
        assert owner._payload is owner._tokenizer is owner._metadata_owner is None
        assert not reservation.pending_retirement and reservation._exited
        assert reservation._token not in f.capacity._owners
        assert f.capacity._owners == f.baseline
        assert not f.core.records and f.tokenizer.closed == 1
        assert len(f.attempts) == failures + 1
        assert all(a[0] is slot and a[1] is owner and a[2] is metadata
                   and a[3] is f.tokenizer for a in f.attempts)
        assert getattr(primary, "__notes__", []) == notes
        calls = len(f.attempts)
        f.core.close()
        assert len(f.attempts) == calls
    finally:
        f.control.failures = 0
        f.core.close()


@pytest.mark.parametrize("occupied", ["active", "scope", "compact", "request_cleanup", "closing"])
def test_count_rejects_existing_owners_before_metadata(monkeypatch, occupied):
    from pllm.runtime.client import ProtocolError

    f = _count_cleanup_fixture(monkeypatch, failures=0)
    if occupied == "active":
        f.core._active = object()
    elif occupied == "scope":
        f.core._tokenizer_scopes.add(object())
    elif occupied == "compact":
        f.core._compact_cleanup = SimpleNamespace(closed=False)
    elif occupied == "request_cleanup":
        f.core._failed_request_cleanup = SimpleNamespace(pending_retirement=True)
    else:
        f.core._closing = True
    try:
        with pytest.raises(ProtocolError):
            f.core._response_input_tokens("fixture", "hello")
        assert f.core._count_preparation is None
        assert f.reservations == f.attempts == f.tokenizer.calls == []
        assert f.capacity._owners == f.baseline
    finally:
        f.core._active = f.core._compact_cleanup = f.core._failed_request_cleanup = None
        f.core._tokenizer_scopes.clear()
        f.core.close()


def test_count_metadata_failure_retires_empty_preparation(monkeypatch):
    from pllm.runtime import tokenizer_admission

    f = _count_cleanup_fixture(monkeypatch, failures=0)
    error = ValueError("count metadata unavailable")

    def capture(*args, **kwargs):
        slot = f.core._count_preparation
        assert slot.state == "preparing" and slot.owner._payload is None
        assert slot.owner._tokenizer is None and slot.owner._metadata_owner is not None
        assert slot.reservation is f.reservations[0]
        raise error

    monkeypatch.setattr(tokenizer_admission, "_capture_prompt_metadata", capture)
    try:
        with pytest.raises(ValueError) as caught:
            f.core._response_input_tokens("fixture", "hello")
        assert caught.value is error and not getattr(error, "__notes__", ())
        assert f.core._count_preparation is None and f.attempts == []
        assert f.reservations[0]._exited and f.capacity._owners == f.baseline
    finally:
        f.core.close()


@pytest.mark.parametrize("phase", ["preparing", "draining", "retry_draining"])
def test_count_close_exclusion_uses_bounded_gates(monkeypatch, phase):
    import threading
    from pllm.runtime.client import ProtocolError

    f = _count_cleanup_fixture(monkeypatch, failures=1 if phase == "retry_draining" else 0)
    entered, release, done = threading.Event(), threading.Event(), threading.Event()
    errors, counts = [], []
    if phase == "retry_draining":
        with pytest.raises(RuntimeError, match="count tokenizer busy"):
            f.core._response_input_tokens("fixture", "hello")
        original_slot = f.core._count_preparation

    def gate():
        entered.set()
        assert release.wait(3), "count cleanup gate was not released"

    if phase == "preparing":
        f.control.encode_hook = gate
    else:
        f.control.close_hook = gate

    def run():
        try:
            if phase == "retry_draining":
                f.core.close()
            else:
                counts.append(f.core._response_input_tokens("fixture", "hello"))
        except BaseException as error:
            errors.append(error)
        finally:
            done.set()

    thread = threading.Thread(target=run, daemon=True)
    try:
        thread.start()
        assert entered.wait(3), "count worker did not enter gate"
        assert f.core._lock.acquire(timeout=3), "count retirement held core lock"
        try:
            slot = f.core._count_preparation
            assert slot.state == ("preparing" if phase == "preparing" else "draining")
            if phase == "retry_draining":
                assert slot is original_slot
            owner, reservation = slot.owner, slot.reservation
            assert owner._tokenizer is f.tokenizer and owner._metadata_owner is not None
        finally:
            f.core._lock.release()
        with pytest.raises(ProtocolError, match="pending token count|closed"):
            f.core._response_input_tokens("fixture", "hello")
        with pytest.raises(ProtocolError, match="pending token count|closed"):
            f.core.events({"model": "fixture", "input": "hello"})
        with pytest.raises(ProtocolError, match="busy; retry close"):
            f.core.close()
        assert f.core._closing is (phase == "retry_draining")
        assert f.core._count_preparation is slot
        assert slot.owner is owner and f.core.records and not done.is_set()
        assert f.capacity._owners[reservation._token] == 1 << 20
        assert f.tokenizer.closed == 0 and len(f.reservations) == 1
        release.set()
        thread.join(3)
        assert not thread.is_alive(), "count worker did not drain"
        assert errors == []
        assert f.core._count_preparation is None and f.tokenizer.closed == 1
        if phase != "retry_draining":
            # Real admission after count success, not merely an open flag.
            with f.core.tokenizer_scope():
                assert len(f.core._tokenizer_scopes) == 1
            attempt = f.core.events({"model": "fixture", "input": "hello"})
            try:
                assert f.core._active is attempt
            finally:
                attempt.close()
            assert f.core._active is None and not f.core._closing
    finally:
        release.set()
        if thread.ident is not None:
            thread.join(3)
        f.control.close_hook = f.control.encode_hook = None
        f.control.failures = 0
        if not thread.is_alive():
            f.core.close()
    assert not thread.is_alive(), "count worker did not drain"
    assert errors == []
    assert counts == ([] if phase == "retry_draining" else [2])
    assert f.core._count_preparation is None and f.tokenizer.closed == 1
    assert f.capacity._owners == f.baseline and not f.core.records


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


def _pending_over_budget_cleanup_fixture(monkeypatch, *, failures):
    """Fake tokenizer; real pending preparation/admission/retirement, no activation."""
    import sys
    from pllm import _native, model_loader, modeling
    from pllm.deployment import execution, resources
    from pllm.runtime import responses, servers

    _bounded_custody_edges(monkeypatch)
    events, prepared_owners, close_attempts, admission_errors, forbidden_calls = [], [], [], [], []
    control = SimpleNamespace(failures=failures, allowance_live=False)
    tokenizer = Tokenizer()
    original_close = tokenizer.close

    def close_tokenizer():
        events.append("tokenizer.close")
        assert control.allowance_live
        if control.failures:
            control.failures -= 1
            raise RuntimeError("tokenizer busy")
        original_close()
        events.append("tokenizer.retired")

    tokenizer.close = close_tokenizer

    capacity = resources._capacity
    monkeypatch.setattr(capacity, "observe", lambda: SimpleNamespace(headroom_bytes=1 << 40))
    original_enter = resources._MemoryReservation.__enter__
    original_exit = resources._MemoryReservation.__exit__

    def enter(reservation):
        assert not control.allowance_live
        observation = original_enter(reservation)
        control.allowance_live = True
        events.append("allowance.enter")
        return observation

    def exit(reservation, *exc):
        assert tokenizer.closed == 1
        result = original_exit(reservation, *exc)
        assert reservation._token not in capacity._owners
        events.append("allowance.release")
        control.allowance_live = False
        return result

    def forbidden(*args, **kwargs):
        forbidden_calls.append((args, kwargs))
        pytest.fail("over-budget pending request reached activation/source/factory/native work")

    _forbid_full_source_work(monkeypatch)
    monkeypatch.setattr(model_loader, "resolve_model", forbidden)
    monkeypatch.setattr(modeling, "lower_model", forbidden)
    monkeypatch.setattr(servers, "build_roles", forbidden)
    for name in ("FssSourceBuilder", "FssClientRequest", "FssRoleRequest",
                 "GkmSourceBuilder", "GkmClientRequest", "ReplicatedSourcePublisher"):
        monkeypatch.setattr(_native, name, forbidden, raising=False)
    monkeypatch.setattr(execution.ExecutionLease, "_activate_pending", forbidden)
    monkeypatch.setattr(execution, "_provisional_model_id", lambda experiment: "fixture")
    monkeypatch.setattr(resources._MemoryReservation, "__enter__", enter)
    monkeypatch.setattr(resources._MemoryReservation, "__exit__", exit)

    s = source()
    descriptor = s.tokenizer_descriptor
    snapshot = SimpleNamespace(config={}, tokenizer_descriptor=descriptor,
        expected_source_lock=s.source_lock_digest, check_unchanged=lambda: None,
        association=lambda: _prompt_source(s.model, {}, descriptor),
        tokenizer=lambda: tokenizer)
    view = execution._TokenizerView._from_prompt_snapshot(snapshot)
    view._prompt_memory_bytes = 1 << 20
    budget = SimpleNamespace(requests=2, max_input_tokens=16, max_new_tokens=4)
    result = SimpleNamespace(experiment=SimpleNamespace(budget=budget), validate=forbidden)

    def close_metadata():
        assert tokenizer.closed == 1 and not control.allowance_live
        events.append("metadata.release")

    metadata_owner = SimpleNamespace(close=close_metadata)
    lease = execution.ExecutionLease(result, SimpleNamespace(backend="local"),
        lambda: 0, None, view, None, _key=execution._LEASE_KEY,
        _pending_factory=forbidden, _metadata_owner=metadata_owner)
    original_prepare = responses._prepare_prompt
    original_prepared_close = responses._PreparedPromptUse.close

    def prepare(request, **kwargs):
        assert request["max_output_tokens"] == budget.max_new_tokens == 4
        prepared = original_prepare(request, **kwargs)
        assert len(prepared._payload.ids) == 22 > budget.max_input_tokens == 16
        prepared_owners.append(prepared)
        return prepared

    def close_prepared(prepared):
        close_attempts.append((prepared, prepared._payload.tokenizer,
            lease._request_owner, lease._request_busy, prepared._metadata_owner))
        if len(close_attempts) == 1:
            admission_errors.append(sys.exception())
        return original_prepared_close(prepared)

    monkeypatch.setattr(responses, "_prepare_prompt", prepare)
    monkeypatch.setattr(responses._PreparedPromptUse, "close", close_prepared)
    return SimpleNamespace(lease=lease, resource=lease._construct_client().responses,
        tokenizer=tokenizer, control=control, events=events, prepared=prepared_owners,
        attempts=close_attempts, errors=admission_errors, forbidden=forbidden_calls,
        metadata_owner=metadata_owner, budget=budget)


def _assert_pending_budget_refusal(fixture, caught):
    from pllm.deployment.network import NetworkError

    assert type(caught.value) is NetworkError
    assert caught.value is fixture.errors[0]
    assert str(caught.value) == "request input exceeds selected budget"
    assert getattr(caught.value, "param", None) is None
    assert fixture.budget.max_input_tokens == 16 and fixture.budget.max_new_tokens == 4
    assert fixture.lease._requests == 0
    assert fixture.tokenizer.calls == [("user: hello\nassistant:", False)]
    assert len(fixture.prepared) == 1 and fixture.forbidden == []


def test_pending_budget_refusal_retries_same_owner_after_one_close_failure(monkeypatch):
    from pllm.deployment.network import NetworkError

    f = _pending_over_budget_cleanup_fixture(monkeypatch, failures=1)
    with pytest.raises(NetworkError) as caught:
        f.resource.create(input="hello")
    _assert_pending_budget_refusal(f, caught)
    prepared = f.prepared[0]
    assert caught.value.__notes__ == ["admission cleanup retained owner: RuntimeError"]
    assert len(f.attempts) == 1
    assert f.lease._request_owner is prepared and f.lease._request_busy
    assert prepared._payload.tokenizer is f.tokenizer
    assert prepared._metadata_owner is f.attempts[0][4]
    assert f.control.allowance_live and not f.lease._cleanup_complete
    assert f.events == ["allowance.enter", "tokenizer.close"]
    f.lease.close()  # One refusal attempt; only explicit close retries it.
    assert len(f.attempts) == 2
    assert all(attempt[0] is prepared and attempt[1] is f.tokenizer for attempt in f.attempts)
    assert f.attempts[0][2] is prepared and f.attempts[0][3]
    assert f.attempts[1][2] is prepared and f.attempts[1][3]
    assert f.attempts[0][4] is f.attempts[1][4] and f.attempts[1][4] is not None
    assert prepared._payload is None and prepared._metadata_owner is None
    assert f.lease._request_owner is None and not f.lease._request_busy
    assert f.lease._pending_metadata_owner is None and f.lease._cleanup_complete
    assert f.lease.closed and f.lease._state == "closed"
    assert f.events == ["allowance.enter", "tokenizer.close", "tokenizer.close",
        "tokenizer.retired", "allowance.release", "metadata.release"]


def test_pending_budget_refusal_retains_owner_until_explicit_close(monkeypatch):
    from pllm.deployment.network import NetworkError

    f = _pending_over_budget_cleanup_fixture(monkeypatch, failures=2)
    with pytest.raises(NetworkError) as caught:
        f.resource.create(input="hello")
    _assert_pending_budget_refusal(f, caught)
    prepared = f.prepared[0]
    assert caught.value.__notes__ == ["admission cleanup retained owner: RuntimeError"]
    assert f.lease._request_owner is prepared and f.lease._request_busy
    assert prepared._payload.tokenizer is f.tokenizer
    assert prepared._metadata_owner is f.attempts[0][4] and prepared._metadata_owner is not None
    assert f.lease._pending_metadata_owner is f.metadata_owner
    assert f.control.allowance_live and not f.lease._cleanup_complete
    assert f.lease.closed and f.lease._state == "closing"
    assert f.events == ["allowance.enter", "tokenizer.close"]
    with pytest.raises(NetworkError, match="execution lease is closed"):
        f.resource.create(input="hello")
    assert len(f.attempts) == 1 and len(f.tokenizer.calls) == 1
    assert f.lease._request_owner is prepared and f.lease._request_busy
    assert f.control.allowance_live and f.lease._pending_metadata_owner is f.metadata_owner
    with pytest.raises(RuntimeError, match="tokenizer busy"):
        f.lease.close()  # Second genuine fault; no automatic retry consumed it.
    assert len(f.attempts) == 2 and f.control.failures == 0
    assert f.lease._request_owner is prepared and f.lease._request_busy
    assert f.control.allowance_live and f.lease._pending_metadata_owner is f.metadata_owner
    f.lease.close()
    assert len(f.attempts) == 3
    assert all(attempt[0] is prepared and attempt[1] is f.tokenizer for attempt in f.attempts)
    assert all(attempt[2] is prepared and attempt[3] for attempt in f.attempts[1:])
    assert prepared._payload is None and prepared._metadata_owner is None
    assert f.lease._request_owner is None and f.lease._pending_metadata_owner is None
    assert not f.control.allowance_live and f.lease._cleanup_complete
    assert f.lease.closed and f.lease._state == "closed"
    assert f.events == ["allowance.enter", "tokenizer.close", "tokenizer.close",
        "tokenizer.close", "tokenizer.retired", "allowance.release", "metadata.release"]
    assert f.forbidden == [] and len(f.tokenizer.calls) == 1


@pytest.mark.parametrize("close_failures", [0, 2])
def test_pending_budget_refusal_concurrent_close_serializes_retirement(monkeypatch, close_failures):
    import threading
    from pllm.deployment.network import NetworkError

    f = _pending_over_budget_cleanup_fixture(monkeypatch, failures=2)
    with pytest.raises(NetworkError) as caught:
        f.resource.create(input="hello")
    _assert_pending_budget_refusal(f, caught)
    notes = ["admission cleanup retained owner: RuntimeError"]
    assert caught.value.__notes__ == notes
    prepared = f.prepared[0]
    reservation = prepared._metadata_owner
    manager, token, amount = reservation._manager, reservation._token, reservation._amount
    capacity = reservation._capacity
    f.control.failures = close_failures
    entered, release, second_attempt, second_entered = (threading.Event() for _ in range(4))
    original_close = f.tokenizer.close

    def blocked_close():
        if not entered.is_set():
            entered.set()
            assert release.wait(3), "owner close was not released"
        original_close()

    monkeypatch.setattr(f.tokenizer, "close", blocked_close)
    errors = [None, None]
    done = [threading.Event(), threading.Event()]

    def close(index):
        try:
            f.lease.close()
        except BaseException as error:
            errors[index] = error
        finally:
            done[index].set()

    threads = [threading.Thread(target=close, args=(index,)) for index in range(2)]
    started = []
    try:
        _start_custody_worker(threads[0], started)
        assert entered.wait(3), "first closer did not reach owner"
        generation = f.lease._operation_event
        wait = generation.wait

        def observed_wait(timeout=None):
            second_attempt.set()  # Actual wait on the captured retirement.
            assert wait(3 if timeout is None else timeout)
            second_entered.set()
            return True

        monkeypatch.setattr(generation, "wait", observed_wait)
        _start_custody_worker(threads[1], started)
        assert second_attempt.wait(3), "second closer did not wait on retirement generation"
        assert not second_entered.is_set() and not any(event.is_set() for event in done)
        assert len(f.attempts) == 2 and f.tokenizer.closed == 0
        assert f.lease._lock.acquire(timeout=3), "closer held lifecycle lock while waiting"
        try:
            assert f.lease._request_owner is prepared and f.lease._request_busy
            assert f.lease.closed and f.lease._state == "closing"
            assert not f.lease._cleanup_complete
            assert f.control.allowance_live and f.lease._pending_metadata_owner is f.metadata_owner
        finally:
            f.lease._lock.release()
    finally:
        _assert_custody_workers_joined(started, releases=(release.set,))
    assert all(not thread.is_alive() for thread in threads)
    assert all(event.is_set() for event in done) and second_entered.is_set()
    assert caught.value is f.errors[0] and caught.value.__notes__ == notes
    if close_failures:
        assert all(type(error) is RuntimeError and str(error) == "tokenizer busy" for error in errors)
        assert errors[0] is errors[1] is generation._error
        assert len(f.attempts) == 2 and f.tokenizer.closed == 0
        assert f.lease._request_owner is prepared and f.lease._request_busy
        assert prepared._payload.tokenizer is f.tokenizer and prepared._metadata_owner is not None
        assert f.control.allowance_live and f.lease._pending_metadata_owner is f.metadata_owner
        assert f.lease.closed and f.lease._state == "closing" and not f.lease._cleanup_complete
        assert prepared._metadata_owner is reservation and capacity._owners[token] == amount
    else:
        assert errors == [None, None] and generation._error is None
        assert prepared._payload is None and prepared._metadata_owner is None
        assert token not in capacity._owners and f.lease._cleanup_complete
    # The main thread must not synchronously wait on retirement while withholding
    # its release. Probe closed admission only AFTER both actual closers exited,
    # and before any genuinely subsequent explicit recovery of the failed owner.
    attempt_count, calls, events = len(f.attempts), list(f.tokenizer.calls), list(f.events)
    request_owner, metadata_owner = f.lease._request_owner, f.lease._pending_metadata_owner
    prepared_metadata, charge = prepared._metadata_owner, capacity._owners.get(token)
    closed = f.tokenizer.closed
    with pytest.raises(NetworkError, match="execution lease is closed"):
        f.resource.create(input="hello")
    assert len(f.attempts) == attempt_count and f.tokenizer.calls == calls and f.events == events
    assert f.tokenizer.closed == closed and capacity._owners.get(token) == charge
    assert f.lease._request_owner is request_owner and f.lease._pending_metadata_owner is metadata_owner
    assert prepared._metadata_owner is prepared_metadata
    assert reservation._manager is manager and reservation._token is token and reservation._amount == amount
    assert caught.value is f.errors[0] and caught.value.__notes__ == notes
    if close_failures:
        assert errors[0] is errors[1] is generation._error
        assert prepared._metadata_owner is reservation and capacity._owners[token] == amount
        with pytest.raises(RuntimeError, match="tokenizer busy"):
            f.lease.close()  # Only this explicit call consumes fault two.
        assert len(f.attempts) == 3 and f.lease._request_owner is prepared
        assert prepared._metadata_owner is reservation and capacity._owners[token] == amount
        f.lease.close()
    assert len(f.attempts) == 2 + close_failures
    assert all(attempt[0] is prepared and attempt[1] is f.tokenizer for attempt in f.attempts)
    assert prepared._payload is None and prepared._metadata_owner is None
    assert f.tokenizer.closed == 1 and not f.control.allowance_live
    assert f.events.count("tokenizer.retired") == 1
    assert f.events.count("allowance.release") == f.events.count("metadata.release") == 1
    assert f.lease._request_owner is None and not f.lease._request_busy
    assert f.lease._pending_metadata_owner is None and f.lease._cleanup_complete
    assert token not in capacity._owners and reservation._manager is manager
    assert f.lease.closed and f.lease._state == "closed"
    assert f.forbidden == [] and len(f.tokenizer.calls) == 1
    assert caught.value.__notes__ == notes
    events, attempt_count = list(f.events), len(f.attempts)
    f.lease.close()
    assert f.events == events and len(f.attempts) == attempt_count


def test_pending_budget_refusal_successful_cleanup_unchanged(monkeypatch):
    from pllm.deployment.network import NetworkError

    f = _pending_over_budget_cleanup_fixture(monkeypatch, failures=0)
    with pytest.raises(NetworkError) as caught:
        f.resource.create(input="hello")
    _assert_pending_budget_refusal(f, caught)
    assert getattr(caught.value, "__notes__", []) == []
    assert len(f.attempts) == 1 and f.attempts[0][0] is f.prepared[0]
    assert f.attempts[0][1] is f.tokenizer and f.attempts[0][2] is f.prepared[0]
    assert f.prepared[0]._payload is None and f.prepared[0]._metadata_owner is None
    assert f.lease._request_owner is None and not f.lease._request_busy
    assert f.lease._pending_metadata_owner is None and f.lease._cleanup_complete
    assert f.lease.closed and f.lease._state == "closed"
    assert f.events == ["allowance.enter", "tokenizer.close", "tokenizer.retired",
        "allowance.release", "metadata.release"]


@pytest.mark.parametrize("stage,cleanup", [
    ("constructor", None), ("constructor", "frames"),
    ("partial_wrapper", None), ("partial_wrapper", "frames"),
    ("render", None), ("render", "frames"), ("render", "close"),
    ("encode", None), ("encode", "frames"), ("encode", "close"),
    ("nested_cause", None), ("nested_cause", "frames"), ("nested_cause", "close"),
])
def test_pending_construction_unwind_destroys_payload_before_refund(monkeypatch, stage, cleanup):
    """Real constructor ordering and lease drain; only native tokenizer/host are fake.

    No source authentication, activation, native execution or memory sizing proof.
    No native constructor qualification; partial_wrapper injects failure after
    real initialization, not a naturally failing wrapper constructor.
    Arbitrary error args/attrs and custom callbacks/finalizers are out of scope;
    fixture finalizers capture only the event list and a string, never payloads.
    Caller keeps the original error alive through retirement and explicit retry.
    """
    import sys
    import weakref
    from types import ModuleType
    from pllm.deployment import execution, resources
    from pllm.runtime import delegated_protocol, responses, tokenizer_admission
    from pllm.runtime.transformer_client import TokenizersJSONTokenizer

    events, references, owners, allowances, reservations, originals = [], [], [], [], [], []
    published, published_allowances, payload_references, topology = [], [], [], []
    remaining = [2 if cleanup else 0]  # Admission, then EXPLICIT failed recovery.
    original_error = ValueError("original tokenizer failure")
    original_error.add_note("original detail")
    capacity = resources._capacity
    baseline = dict(capacity._owners)
    monkeypatch.setattr(capacity, "observe", lambda: SimpleNamespace(
        headroom_bytes=(1 << 40) - sum(capacity._owners.values())))

    class FramePayload:
        pass

    def cause_leaf():
        payload = FramePayload()  # Sole payload owner: this inactive cause frame.
        payload_references.append(weakref.ref(payload))
        raise ValueError("nested cause")

    def raise_outer():
        try:
            cause_leaf()
        except ValueError as cause:
            raise original_error from cause

    class Native:
        def __init__(self):
            references.append(weakref.ref(self))
            events.append("native.created")

        @classmethod
        def from_buffer(cls, raw):
            assert raw == b"{}"
            inner = cls()
            if stage == "constructor":
                raise original_error
            return inner

        def encode(self, text):
            events.append("native.encode")
            assert stage in {"encode", "nested_cause"}
            if stage == "nested_cause":
                raise_outer()
            raise original_error

        def __del__(self):
            events.append("native.destroyed")

    tokenizers = ModuleType("tokenizers")
    tokenizers.Tokenizer = Native
    monkeypatch.setitem(sys.modules, "tokenizers", tokenizers)
    original_init = tokenizer_admission._PromptTokenizer.__init__

    def injected_init(self, inner):
        original_init(self, inner)
        assert type(self) is tokenizer_admission._PromptTokenizer
        assert type(self._inner) is TokenizersJSONTokenizer
        assert self._inner is inner and self._inner.inner is references[0]()
        references.append(weakref.ref(self))
        weakref.finalize(self, events.append, "tokenizer.destroyed")
        assert type(lease._request_owner) is responses._PreparedPromptUse
        assert lease._request_owner is published[0] and lease._request_busy
        assert lease._request_owner._metadata_owner is published_allowances[0]
        assert lease._request_owner._tokenizer is None
        assert capacity._owners[reservations[0]._token] == 1 << 20
        # INJECTED initializer failure AFTER real initialization. Native factory
        # returned normally; both real wrappers own native before outer adoption.
        raise original_error

    if stage == "partial_wrapper":
        monkeypatch.setattr(tokenizer_admission._PromptTokenizer, "__init__", injected_init)
    original_close = tokenizer_admission._PromptTokenizer.close

    def close_tokenizer(tokenizer):
        events.append("tokenizer.close")
        assert capacity._owners[reservations[0]._token] == 1 << 20
        if cleanup == "close" and remaining[0]:
            remaining[0] -= 1
            raise RuntimeError("tokenizer busy")
        original_close(tokenizer)

    monkeypatch.setattr(tokenizer_admission._PromptTokenizer, "close", close_tokenizer)
    original_adopt = responses._PreparedPromptUse._adopt_tokenizer

    def adopt(owner, tokenizer):
        assert stage != "partial_wrapper"  # Injected failure never reaches adoption.
        original_adopt(owner, tokenizer)
        references.append(weakref.ref(tokenizer))
        weakref.finalize(tokenizer, events.append, "tokenizer.destroyed")

    monkeypatch.setattr(responses._PreparedPromptUse, "_adopt_tokenizer", adopt)
    original_clear = delegated_protocol.clear_failure_frames

    def clear(error):
        assert error is originals[0]
        if stage == "nested_cause" and not topology:
            # Capture exceptions/topology before cleanup, never frame locals.
            assert error is original_error
            cause = error.__cause__
            assert type(cause) is ValueError and cause.args == ("nested cause",)
            assert error.__context__ is cause and error.__suppress_context__
            assert cause.__cause__ is cause.__context__ is None
            topology.extend((current, current.__cause__, current.__context__,
                             current.__suppress_context__) for current in (error, cause))
        if cleanup == "frames" and remaining[0]:
            remaining[0] -= 1
            raise RuntimeError("frame clearing busy")
        original_clear(error)

    monkeypatch.setattr(delegated_protocol, "clear_failure_frames", clear)
    original_prepared_close = responses._PreparedPromptUse.close

    def close_prepared(owner):
        if not owners:
            owners.append(owner)
            allowances.append(owner._metadata_owner)
            originals.append(owner._failure)
        assert owner is owners[0] is lease._request_owner
        assert owner is published[0]
        assert owner._metadata_owner is allowances[0]
        assert owner._metadata_owner is published_allowances[0]
        assert owner._failure is originals[0]
        assert owner._payload is None  # Failure precedes immutable publication.
        return original_prepared_close(owner)

    monkeypatch.setattr(responses._PreparedPromptUse, "close", close_prepared)
    original_reserve = resources.reserve_incremental_memory

    def reserve(amount):
        assert not reservations  # No replacement owner/reservation on failure.
        assert lease._request_busy and lease._request_owner._payload is None
        published.append(lease._request_owner)
        assert lease._request_owner._metadata_owner is None
        reservation = original_reserve(amount)
        reservations.append(reservation)
        published_allowances.append(reservation)
        return reservation

    monkeypatch.setattr(resources, "reserve_incremental_memory", reserve)
    original_exit = resources._MemoryReservation.__exit__

    def retire(reservation, *exc):
        assert reservation is reservations[0]
        assert capacity._owners[reservation._token] == 1 << 20
        assert not reservation._exited
        assert events.count("native.destroyed") == 1
        if stage != "constructor":
            assert events.count("tokenizer.destroyed") == 1
        assert references and all(reference() is None for reference in references)
        if stage == "nested_cause":
            assert len(payload_references) == 1 and payload_references[0]() is None
            assert len(topology) == 2 and topology[0][0] is original_error
            inspected = set()
            for error, cause, context, suppressed in topology:
                assert error.__cause__ is cause and error.__context__ is context
                assert error.__suppress_context__ is suppressed
                tb = error.__traceback__
                while tb is not None:
                    name = tb.tb_frame.f_code.co_name
                    if name in {"cause_leaf", "raise_outer", "encode", "_prepare_prompt",
                                "_prepare_pending_prompt"}:
                        inspected.add(name)
                        assert tb.tb_frame.f_locals == {}
                    tb = tb.tb_next
            assert {"cause_leaf", "raise_outer", "encode", "_prepare_prompt",
                    "_prepare_pending_prompt"} <= inspected
            # Outer-only traceback clearing leaves cause_leaf's payload alive.
        events.append("refund")
        return original_exit(reservation, *exc)

    monkeypatch.setattr(resources._MemoryReservation, "__exit__", retire)

    def forbidden(*args, **kwargs):
        pytest.fail("construction failure reached activation/source work")

    _forbid_full_source_work(monkeypatch)
    monkeypatch.setattr(execution, "_provisional_model_id", lambda experiment: "fixture")
    owned = source()
    descriptor = dict(owned.tokenizer_descriptor, bos_token_id=0, eos_token_id=0,
        chat_template="{{ missing }}" if stage == "render" else "hello")
    snapshot = SimpleNamespace(config={}, tokenizer_descriptor=descriptor,
        tokenizer_bytes=b"{}", expected_source_lock=None, check_unchanged=lambda: None,
        association=lambda: _prompt_source(owned.model, {}, descriptor))
    # Actual snapshot factory: from_buffer -> TokenizersJSONTokenizer -> _PromptTokenizer.
    snapshot.tokenizer = lambda: tokenizer_admission._PromptMetadataSnapshot.tokenizer(snapshot)
    view = execution._TokenizerView._from_prompt_snapshot(snapshot)
    view._prompt_memory_bytes = 1 << 20
    budget = SimpleNamespace(requests=2, max_input_tokens=16, max_new_tokens=4)
    result = SimpleNamespace(experiment=SimpleNamespace(budget=budget), validate=forbidden)
    lease = execution.ExecutionLease(result, SimpleNamespace(backend="local"), lambda: 0,
        None, view, None, _key=execution._LEASE_KEY, _pending_factory=forbidden)
    resource = lease._construct_client().responses
    try:
        with pytest.raises(ResponsesError if stage == "render" else ValueError) as caught:
            resource.create(input="hello")
        escaped = caught.value
        assert escaped is originals[0]
        if stage == "render":
            assert str(escaped) == "source chat template failed" and escaped.__cause__ is not None
            assert "native.encode" not in events
        else:
            assert escaped is original_error and escaped.__notes__[0] == "original detail"
        owner, allowance, reservation = owners[0], allowances[0], reservations[0]
        assert lease._requests == 0 and len(reservations) == 1
        assert len(references) == (1 if stage == "constructor" else 2)
        if cleanup:
            assert remaining[0] == 1
            assert lease._request_owner is owner and lease._request_busy
            assert owner._failure is escaped and owner._metadata_owner is allowance
            assert (owner._tokenizer is None) == (stage in {"constructor", "partial_wrapper"})
            assert capacity._owners[reservation._token] == 1 << 20
            assert references[0]() is not None and "refund" not in events
            if stage != "constructor":
                assert references[1]() is not None  # Wrapper/native retained separately from payload.
            if stage == "nested_cause":
                assert len(payload_references) == 1
                if cleanup == "frames":
                    assert payload_references[0]() is not None
                else:
                    # Frame cleanup succeeded before tokenizer close failed.
                    assert payload_references[0]() is None
            assert lease.closed and lease._state == "closing" and not lease._cleanup_complete
            assert escaped.__notes__[-1:] == ["admission cleanup retained owner: RuntimeError"]
            with pytest.raises(execution.NetworkError, match="execution lease is closed"):
                resource.create(input="hello")
            assert len(reservations) == 1 and lease._request_owner is owner
            with pytest.raises(RuntimeError, match="frame clearing busy|tokenizer busy"):
                lease.close()
            assert remaining[0] == 0 and lease._request_owner is owner
            assert owner._failure is escaped and owner._metadata_owner is allowance
            assert capacity._owners[reservation._token] == 1 << 20
            assert references[0]() is not None and "refund" not in events
            lease.close()  # Explicit same-owner success, never a replacement.
        assert owner._failure is owner._tokenizer is owner._payload is owner._metadata_owner is None
        assert reservation._exited and capacity._owners == baseline
        assert lease._request_owner is None and not lease._request_busy
        assert lease._cleanup_complete and lease._state == "closed"
        assert references[0]() is None
        assert events.count("refund") == events.count("native.destroyed") == 1
        assert events.index("native.destroyed") < events.index("refund")
        if stage != "constructor":
            assert events.index("tokenizer.destroyed") < events.index("refund")
        # Inspect the escaped original after cleanup; helper/constructor frames
        # must no longer carry the actual tokenizer, wrapper or native payload.
        tb = escaped.__traceback__
        while tb is not None:
            if tb.tb_frame.f_code.co_name in {"_prepare_pending_prompt", "_prepare_prompt",
                                             "from_buffer", "encode", "injected_init"}:
                assert not tb.tb_frame.f_locals
            tb = tb.tb_next
        before = list(events)
        lease.close()
        assert events == before and len(reservations) == 1
    finally:
        remaining[0] = 0
        primary = sys.exception()
        try:
            lease.close()
        except BaseException as cleanup_error:
            if primary is None:
                raise
            primary.add_note(f"test teardown failed: {type(cleanup_error).__name__}")


def _request_metadata_repair_fixture(monkeypatch):
    """Real request seat and byte ledger; controlled telemetry/tokenizer only."""
    import threading
    from pllm.deployment import resources
    from pllm.runtime import tokenizer_admission
    from pllm.runtime.delegated_client import DelegatedClient, _Attempt

    _bounded_custody_edges(monkeypatch)

    capacity = resources._ClientCapacity()
    monkeypatch.setattr(capacity, "observe", lambda: SimpleNamespace(headroom_bytes=1 << 40))
    reservations = []

    def reserve(amount):
        reservation = resources._MemoryReservation(capacity, amount)
        reservations.append(reservation)
        return reservation

    monkeypatch.setattr(resources, "reserve_incremental_memory", reserve)
    core = DelegatedClient.__new__(DelegatedClient)
    core._lock = threading.RLock()
    core._closing = core._source_checked = False
    core._source = core._count_preparation = core._failed_request_cleanup = None
    core._capacity = "capsule"
    core._experiment_budget = SimpleNamespace(max_input_tokens=64, max_new_tokens=2)
    owned = source()
    core._declaration = SimpleNamespace(pipeline=SimpleNamespace(model=owned.model))
    tokenizer = Tokenizer()
    descriptor = dict(owned.tokenizer_descriptor, vocab_size=256)
    snapshot = SimpleNamespace(config={}, tokenizer_descriptor=descriptor,
        expected_source_lock=None, check_unchanged=lambda: None,
        tokenizer=lambda: tokenizer, association=lambda: _prompt_source(owned.model, {}, descriptor))
    monkeypatch.setattr(tokenizer_admission, "_capture_prompt_metadata", lambda *a, **k: snapshot)
    attempt = _Attempt(core, body())
    core._active = attempt
    return SimpleNamespace(core=core, attempt=attempt, tokenizer=tokenizer,
        capacity=capacity, reservations=reservations, snapshot=snapshot)


@pytest.mark.parametrize("fault", [RuntimeError, KeyboardInterrupt])
def test_request_metadata_retry_refund_and_tokenizer_dependency(monkeypatch, fault):
    from pllm.deployment import resources
    f = _request_metadata_repair_fixture(monkeypatch)
    owner = f.core._prepare_request_prompt(body(), attempt=f.attempt)
    reservation = f.reservations[0]
    assert owner is f.attempt._prepared_prompt and owner._metadata_owner is reservation
    assert owner._claimed and len(f.reservations) == 1
    original = resources._MemoryReservation.__exit__
    calls = []

    def exit(current, *exc):
        calls.append(current)
        assert f.tokenizer.closed == 1 and owner._tokenizer is owner._payload is None
        assert f.capacity._owners[current._token] == 1 << 20
        if len(calls) <= 2:
            raise fault("refund busy")
        return original(current, *exc)

    monkeypatch.setattr(resources._MemoryReservation, "__exit__", exit)
    for _ in range(2):
        with pytest.raises(fault, match="refund busy"):
            f.attempt.close()
        assert f.attempt._prepared_prompt is owner and not f.attempt.closed
        assert not owner._closed and owner._metadata_owner is reservation
        assert f.capacity._owners[reservation._token] == 1 << 20
        with pytest.raises(ResponsesError, match="incomplete or retired"):
            f.core._prepare_request_prompt(body(), attempt=f.attempt)
    f.attempt.close()
    assert owner._closed and owner._metadata_owner is None and reservation._closed
    assert not f.capacity._owners and f.tokenizer.closed == 1
    owner.close()
    reservation.close()
    f.attempt.close()
    assert calls == [reservation] * 3 and len(f.reservations) == 1


@pytest.mark.parametrize("stage", ["constructor", "handoff", "render", "complete"])
@pytest.mark.parametrize("fault", [RuntimeError, KeyboardInterrupt])
def test_request_preparation_failure_keeps_original_attempt_root(monkeypatch, stage, fault):
    from pllm.deployment import resources
    from pllm.runtime import responses
    f = _request_metadata_repair_fixture(monkeypatch)
    primary = fault("preparation failed")

    def fail(*args, **kwargs):
        assert f.attempt._prepared_prompt._metadata_owner is f.reservations[0]
        raise primary

    if stage == "constructor":
        f.snapshot.tokenizer = fail
    elif stage == "handoff":
        original_adopt = responses._PreparedPromptUse._adopt_tokenizer
        def adopt(owner, tokenizer):
            original_adopt(owner, tokenizer)
            fail()
        monkeypatch.setattr(responses._PreparedPromptUse, "_adopt_tokenizer", adopt)
    elif stage == "render":
        monkeypatch.setattr(responses, "_render_prompt", fail)
    else:
        monkeypatch.setattr(responses._PreparedPromptUse, "_complete_payload", fail)
    with pytest.raises(fault) as caught:
        f.core._prepare_request_prompt(body(), attempt=f.attempt)
    owner = f.attempt._prepared_prompt
    assert caught.value is owner._failure is primary
    assert owner._metadata_owner is f.reservations[0] and not owner._claimed
    assert len(f.reservations) == 1 and f.capacity._owners
    original_exit = resources._MemoryReservation.__exit__
    remaining = [2]

    def exit(reservation, *exc):
        if remaining[0]:
            remaining[0] -= 1
            raise RuntimeError("refund busy")
        return original_exit(reservation, *exc)

    monkeypatch.setattr(resources._MemoryReservation, "__exit__", exit)
    for _ in range(2):
        with pytest.raises(RuntimeError, match="refund busy"):
            f.attempt.close()
        assert f.attempt._prepared_prompt is owner and owner._metadata_owner is f.reservations[0]
        assert not owner._closed and f.capacity._owners
        with pytest.raises(ResponsesError, match="incomplete or retired"):
            f.core._prepare_request_prompt(body(), attempt=f.attempt)
    f.attempt.close()
    assert not f.capacity._owners and owner._closed and f.attempt.closed
    assert f.tokenizer.closed == (0 if stage == "constructor" else 1)


@pytest.mark.parametrize("charged", [False, True])
def test_request_reservation_enter_failure_tracks_actual_charge(monkeypatch, charged):
    from pllm.deployment import resources
    from pllm.runtime.responses import _PreparedPromptUse
    capacity = resources._ClientCapacity()
    owner = _PreparedPromptUse._empty()  # Caller-rooted before reserve helper.
    primary = KeyboardInterrupt("enter rejected")
    reservations = []

    def observe():
        if not charged:
            raise primary
        return SimpleNamespace(headroom_bytes=1 << 40)

    monkeypatch.setattr(capacity, "observe", observe)

    def reserve(amount):
        reservation = resources._MemoryReservation(capacity, amount)
        if charged:
            actual = reservation._manager
            class ChargedEnterFault:
                def __enter__(self):
                    assert owner._metadata_owner is reservation
                    actual.__enter__()
                    raise primary
                def __exit__(self, *exc):
                    return actual.__exit__(*exc)
            reservation._manager = ChargedEnterFault()
        reservations.append(reservation)
        return reservation

    monkeypatch.setattr(resources, "reserve_incremental_memory", reserve)
    with pytest.raises(KeyboardInterrupt) as caught:
        owner._reserve_metadata(123)
    reservation = reservations[0]
    assert caught.value is primary and owner._metadata_owner is reservation
    assert reservation._acquisition_started and reservation._acquisition_failed
    assert reservation._acquired is charged
    assert (reservation._token in capacity._owners) is charged
    owner._failure = primary
    owner.close()
    assert owner._closed and reservation._closed and not capacity._owners
    owner.close()


def test_consumed_hold_bytes_exit_never_fakes_refund(monkeypatch):
    import threading
    from pllm.deployment import resources
    from pllm.runtime.responses import _PreparedPromptUse
    capacity = resources._ClientCapacity()
    monkeypatch.setattr(capacity, "observe", lambda: SimpleNamespace(headroom_bytes=1 << 40))
    monkeypatch.setattr(resources, "reserve_incremental_memory",
        lambda amount: resources._MemoryReservation(capacity, amount))
    owner = _PreparedPromptUse._empty()
    owner._reserve_metadata(123)
    reservation = owner._metadata_owner
    lock = threading.RLock()

    class ExitLockFault:
        def __enter__(self):
            raise KeyboardInterrupt("ledger release unavailable")
        def __exit__(self, *exc):
            return False

    # Fault INSIDE the real hold_bytes finally consumes its generator. No fake
    # retry backend, no outside-ledger refund, no promised eventual recovery.
    capacity.lock = ExitLockFault()
    with pytest.raises(KeyboardInterrupt, match="ledger release unavailable"):
        owner.close()
    capacity.lock = lock
    assert not reservation._exited and capacity._owners[reservation._token] == 123
    for _ in range(3):
        with pytest.raises(resources.NetworkError, match="MEMORY_RETIREMENT_INCOMPLETE"):
            owner.close()
        assert owner._metadata_owner is reservation and not owner._closed
        assert reservation._exited and not reservation._closed
        assert capacity._owners[reservation._token] == 123
    # Isolated test ledger intentionally remains unresolved. Production close
    # cannot repair arbitrary partial backend side effects by inventing credit.


def test_request_metadata_retained_normal_exit_is_incomplete(monkeypatch):
    from pllm.deployment import resources
    from pllm.runtime.responses import _PreparedPromptUse
    capacity = resources._ClientCapacity()
    telemetry = SimpleNamespace(headroom_bytes=1 << 40)
    monkeypatch.setattr(capacity, "observe", lambda: telemetry)
    monkeypatch.setattr(resources, "_capacity", capacity)
    owner = _PreparedPromptUse._empty()
    owner._reserve_metadata(64 << 10)
    reservation = owner._metadata_owner
    assert type(reservation) is resources._MemoryReservation and reservation is not telemetry
    retired = []
    payload = SimpleNamespace(close=lambda: retired.append(True))
    reservation.retain_failed((payload,))
    for _ in range(2):
        with pytest.raises(resources.NetworkError, match="MEMORY_RETIREMENT_INCOMPLETE"):
            owner.close()
        assert not reservation._exited and not reservation._closed and not owner._closed
        assert owner._metadata_owner is reservation and reservation.pending_retirement
        assert capacity._owners[reservation._token] == 64 << 10
    reservation.drain_failed()  # Existing explicit count/failed-retirement protocol.
    assert retired == [True] and reservation._exited and not capacity._owners
    owner.close()
    owner.close()
    assert owner._closed and owner._metadata_owner is None and reservation._closed


def test_request_requires_caller_root_before_any_acquisition(monkeypatch):
    f = _request_metadata_repair_fixture(monkeypatch)
    with pytest.raises(ResponsesError, match="caller-rooted owner"):
        f.core._prepare_request_prompt(body())
    assert not f.reservations and not f.capacity._owners and not f.tokenizer.calls


def _request_repair_lease(monkeypatch, *, pending):
    from pllm.deployment import execution
    f = _request_metadata_repair_fixture(monkeypatch)
    monkeypatch.setattr(execution, "_provisional_model_id", lambda experiment: "fixture")
    budget = SimpleNamespace(requests=2, max_input_tokens=64, max_new_tokens=2)
    experiment = SimpleNamespace(budget=budget, resolve=lambda: SimpleNamespace(model="fixture"))
    result = SimpleNamespace(experiment=experiment, snapshot=None, validate=lambda **kw: None)
    events = []
    topology = SimpleNamespace(cleanup_complete=False)
    def close_topology():
        if not topology.cleanup_complete:
            events.append("topology")
            topology.cleanup_complete = True
    topology.close = close_topology
    view = execution._TokenizerView._from_prompt_snapshot(f.snapshot) if pending else None
    if view is not None:
        view._prompt_memory_bytes = 1 << 20
    lease = execution.ExecutionLease(result, SimpleNamespace(backend="local", snapshot=None),
        lambda: 0, topology, view, None, _key=execution._LEASE_KEY,
        _pending_factory=(lambda *args: pytest.fail("unexpected activation")) if pending else None)
    if not pending:
        lease._client = SimpleNamespace(_core=f.core, close=lambda: events.append("client"))
    f.lease, f.events = lease, events
    return f


@pytest.mark.parametrize("pending", [False, True])
def test_lease_admission_metadata_failure_sticky_and_dependency_order(monkeypatch, pending):
    from pllm.deployment import resources
    from pllm.runtime import responses
    f = _request_repair_lease(monkeypatch, pending=pending)
    owners = []
    primary = KeyboardInterrupt("payload completion failed")

    def complete(owner, payload):
        owners.append(owner)
        assert f.lease._request_owner is owner and f.lease._request_busy
        assert owner._metadata_owner is f.reservations[0]
        assert f.lease._lock.acquire(blocking=False)
        f.lease._lock.release()
        raise primary

    monkeypatch.setattr(responses._PreparedPromptUse, "_complete_payload", complete)
    original = resources._MemoryReservation.__exit__
    failures = [2]

    def exit(reservation, *exc):
        assert f.lease._lock.acquire(blocking=False)
        f.lease._lock.release()
        assert f.lease._state == "closing" and f.lease._request_busy
        assert f.tokenizer.closed == 1
        if failures[0]:
            failures[0] -= 1
            raise RuntimeError("refund busy")
        return original(reservation, *exc)

    monkeypatch.setattr(resources._MemoryReservation, "__exit__", exit)
    with pytest.raises(KeyboardInterrupt) as caught:
        f.lease._admit_request(body())
    assert caught.value is primary and len(owners) == len(f.reservations) == 1
    owner = owners[0]
    with pytest.raises(RuntimeError, match="refund busy"):
        f.lease.close()
    assert f.events.count("topology") == 1  # HEAD legacy attempts topology even on failure.
    assert f.lease._request_owner is owner and owner._failure is primary
    assert owner._metadata_owner is f.reservations[0] and not owner._closed
    assert f.capacity._owners and not f.lease._cleanup_complete
    with pytest.raises(resources.NetworkError, match="lease is closed"):
        f.lease._admit_request(body())
    assert len(owners) == len(f.reservations) == 1
    f.lease.close()
    assert owner._closed and not f.capacity._owners and f.tokenizer.closed == 1
    assert f.lease._request_owner is None and not f.lease._request_busy
    assert f.lease._cleanup_complete and f.events.count("topology") == 1
    assert f.events == (["topology"] if pending else ["client", "topology", "client"])
    f.lease.close()
    assert f.events.count("topology") == 1


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("failure", ["cancel", "producer", "none"])
@pytest.mark.asyncio
async def test_admitted_unclaimed_delivery_failure_keeps_lease_root(monkeypatch, asynchronous, failure):
    import asyncio
    from pllm.deployment import execution, resources
    from pllm.runtime.responses import _prepared_prompt_request
    f = _request_repair_lease(monkeypatch, pending=False)
    owners = []
    primary = (asyncio.CancelledError("cancelled") if failure == "cancel"
               else RuntimeError("producer failed"))

    def produce(**kwargs):
        owner = _prepared_prompt_request.get()
        owners.append(owner)
        assert f.lease._request_owner is owner and not owner._claimed
        if failure != "none":
            raise primary
        return SimpleNamespace()  # Unclaimed delivery, no native acceptance.

    async def async_produce(**kwargs):
        return produce(**kwargs)

    resource = execution._LeaseResponses(f.lease,
        SimpleNamespace(create=async_produce if asynchronous else produce))
    original = resources._MemoryReservation.__exit__
    remaining = [1]

    def exit(reservation, *exc):
        assert f.tokenizer.closed == 1
        if remaining[0]:
            remaining[0] -= 1
            raise KeyboardInterrupt("refund interrupted")
        return original(reservation, *exc)

    monkeypatch.setattr(resources._MemoryReservation, "__exit__", exit)
    expected = KeyboardInterrupt if failure == "none" else type(primary)
    with pytest.raises(expected) as caught:
        if asynchronous:
            await resource.create(**body())
        else:
            resource.create(**body())
    if failure != "none":
        assert caught.value is primary
        assert primary.__notes__ == ["prepared cleanup retained owner: KeyboardInterrupt"]
    owner = owners[0]
    assert f.lease._request_owner is owner and f.lease._request_busy
    assert f.lease.closed and f.lease._state == "closing"
    assert owner._metadata_owner is f.reservations[0] and not owner._closed
    assert f.capacity._owners and f.tokenizer.closed == 1
    f.lease.close()
    assert owner._closed and not f.capacity._owners and f.lease._cleanup_complete
    assert f.events == ["client", "topology"] and f.tokenizer.closed == 1


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
            from pllm.runtime.responses import _PreparedPromptUse
            request.setdefault("model", h.core.default_model)
            request.setdefault("max_output_tokens", h.experiment.budget.max_new_tokens)
            prepared = _PreparedPromptUse._empty()
            self._request_owner = prepared
            prepared = h.core._prepare_request_prompt(request, _preparation_owner=prepared)
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


@pytest.mark.parametrize("lifecycle", ["repeated_entry", "preentry_close", "preentry_exit"])
def test_memory_reservation_rejects_entry_after_lifecycle_started(monkeypatch, lifecycle):
    from pllm.deployment import resources

    capacity = resources._ClientCapacity()
    monkeypatch.setattr(capacity, "observe", lambda: SimpleNamespace(headroom_bytes=1 << 40))
    reservation = resources._MemoryReservation(capacity, 123)
    actual = reservation._manager
    calls = []

    class ObservedManager:
        def __enter__(self):
            calls.append("enter")
            return actual.__enter__()

        def __exit__(self, *exc):
            calls.append("exit")
            return actual.__exit__(*exc)

    manager = reservation._manager = ObservedManager()
    token = reservation._token
    if lifecycle == "repeated_entry":
        reservation.__enter__()
        assert capacity._owners == {token: 123}
    elif lifecycle == "preentry_close":
        reservation.close()
    else:
        reservation.__exit__(None, None, None)
    before = list(calls)
    ledger = dict(capacity._owners)
    for _ in range(2):
        with pytest.raises(resources.NetworkError, match="CLIENT_MEMORY_ACQUISITION_UNAVAILABLE"):
            reservation.__enter__()
        assert calls == before and capacity._owners == ledger
        assert reservation._manager is manager and reservation._token is token
    reservation.close()
    reservation.close()
    reservation.__exit__(None, None, None)
    assert token not in capacity._owners and not capacity._owners
    assert calls == (["enter", "exit"] if lifecycle == "repeated_entry" else [])
    assert reservation._exited and reservation._closed


@pytest.mark.parametrize("charged", [False, True])
def test_memory_reservation_failed_entry_cannot_reacquire(monkeypatch, charged):
    from pllm.deployment import resources

    capacity = resources._ClientCapacity()
    primary = RuntimeError("entry failed")
    calls = []

    def observe():
        if not charged:
            raise primary
        return SimpleNamespace(headroom_bytes=1 << 40)

    monkeypatch.setattr(capacity, "observe", observe)
    reservation = resources._MemoryReservation(capacity, 123)
    actual = reservation._manager

    class FailedEntryManager:
        def __enter__(self):
            calls.append("enter")
            actual.__enter__()
            raise primary  # Charged fault leaves the actual generator open.

        def __exit__(self, *exc):
            calls.append("exit")
            return actual.__exit__(*exc)

    manager = reservation._manager = FailedEntryManager()
    token = reservation._token
    with pytest.raises(RuntimeError) as caught:
        reservation.__enter__()
    assert caught.value is primary and reservation._acquisition_failed
    assert reservation._acquired is charged
    assert capacity._owners == ({token: 123} if charged else {})
    for closed in (False, True):
        if closed:
            reservation.close()
        before = list(calls)
        ledger = dict(capacity._owners)
        with pytest.raises(resources.NetworkError, match="CLIENT_MEMORY_ACQUISITION_UNAVAILABLE"):
            reservation.__enter__()
        assert calls == before and capacity._owners == ledger
        assert reservation._manager is manager and reservation._token is token
    reservation.close()
    reservation.__exit__(None, None, None)
    assert token not in capacity._owners and not capacity._owners
    assert calls == ["enter", "exit"] and reservation._closed and reservation._exited


def _start_custody_worker(worker, workers):
    # Register BEFORE start: a raised start may nevertheless have launched work.
    record = {"worker": worker, "returned": False, "start_error": None,
              "join_error": None, "observation_error": None, "unresolved": False}
    workers.append(record)
    try:
        worker.start()
    except BaseException as error:
        record["start_error"] = error
        raise
    else:
        record["returned"] = True


def _assert_custody_workers_joined(workers, *, releases=(), timeout=3, primary=None):
    """Attempt ALL teardown edges; return explicit retirement eligibility.

    Start-raised is unresolved even if ident is None/alive is False. Join or
    observation failures stay sticky. Doubled witnesses below test orchestration,
    NOT physical drain. No guarantee against arbitrary partial Python startup.
    """
    import sys
    primary = sys.exception() if primary is None else primary
    eligible, failures = True, []
    for release in releases:
        try:
            release()
        except BaseException as error:
            failures.append(("release", error))
            eligible = False
    for index, record in enumerate(workers):
        worker = record["worker"]
        if not record["returned"]:
            error = record["start_error"]
            if error is None:
                error = AssertionError("start did not return; launch/termination unresolved")
            failures.append((f"worker[{index}] uncertain start", error))
            eligible = False
        # Every possibly-started worker gets its own join and observations,
        # regardless of any earlier release/join/observation interruption.
        try:
            result = worker.join(timeout=timeout)
            if result is not None:
                raise AssertionError("malformed join result")
        except BaseException as error:
            if record["join_error"] is None:
                record["join_error"] = error
        ident = alive = None
        try:
            ident = worker.ident
            if type(ident) is not int:
                raise AssertionError("thread identity unavailable after join")
        except BaseException as error:
            if record["observation_error"] is None:
                record["observation_error"] = error
        try:
            alive = worker.is_alive()
            if type(alive) is not bool:
                raise AssertionError("malformed liveness observation")
        except BaseException as error:
            if record["observation_error"] is None:
                record["observation_error"] = error
        for phase in ("join_error", "observation_error"):
            error = record[phase]
            if error is not None:
                failures.append((f"worker[{index}] {phase}", error))
                eligible = False
        if alive is not False:
            record["unresolved"] = True
        if record["unresolved"]:
            failures.append((f"worker[{index}] unresolved",
                AssertionError("worker live/unknown or prior unresolved observation")))
            eligible = False
    if failures:
        selected = primary if primary is not None else failures[0][1]
        for phase, error in failures:
            try:
                selected.add_note(f"custody teardown {phase}: {type(error).__name__}")
            except BaseException:
                # Annotation is not control flow: eligibility remains false and
                # the already-selected primary must not be replaced.
                pass
        if primary is None:
            raise selected
    return eligible


def _bounded_custody_edges(monkeypatch):
    """Observe real blocking edges; timeout raises, NEVER grants an acquisition.

    Narrow test isolation for intentional lock/generation regressions. No fake
    completion, refund or producer identity; no production timing guarantee.
    """
    import threading
    from pllm.deployment import execution, resources
    from pllm.runtime import delegated_client, responses

    class Lock:
        def __init__(self, actual):
            self.actual = actual

        def acquire(self, blocking=True, timeout=-1):
            if not blocking:
                return self.actual.acquire(False)
            if not self.actual.acquire(True, 1 if timeout < 0 else min(timeout, 1)):
                raise TimeoutError("actual custody mutex did not acquire")
            return True

        def release(self):
            self.actual.release()

        def __enter__(self):
            self.acquire()
            return self

        def __exit__(self, *exc):
            self.release()

    class Event(threading.Event):
        def wait(self, timeout=None):
            result = super().wait(1 if timeout is None else timeout)
            if timeout is None and not result:
                raise TimeoutError("actual custody generation did not complete")
            return result

    class Edges:
        def Event(self):
            return Event()

        def Lock(self):
            return Lock(threading.Lock())

        def RLock(self):
            return Lock(threading.RLock())

        def __getattr__(self, name):
            return getattr(threading, name)

    for module in (execution, resources, responses, delegated_client):
        monkeypatch.setattr(module, "threading", Edges())
    return Lock


def _gkm_custody_gate_fixture(tmp_path, monkeypatch, *, kind="gkm"):
    """Production pending factory/admission and _Attempt, with I/O edges doubled.

    Real byte ledger, metadata tokenizer, ResponsesResource and async ownership.
    Native/source authentication, topology and producer output are NOT qualified.
    """
    from pllm.deployment import resources
    from pllm.runtime import servers
    from pllm.runtime.client import ResponsesResource
    from pllm.runtime.delegated_client import DelegatedClient

    observe_lock = _bounded_custody_edges(monkeypatch)
    capacity = resources._ClientCapacity()
    monkeypatch.setattr(capacity, "observe", lambda: SimpleNamespace(headroom_bytes=1 << 40))

    def reserve(amount):
        return resources._MemoryReservation(capacity, amount)

    monkeypatch.setattr(resources, "reserve_incremental_memory", reserve)
    lease, core, state = _real_activation_closure_fixture(tmp_path, monkeypatch, kind)
    core._lock = observe_lock(core._lock)
    events = core.events

    def observed_events(body):
        attempt = events(body)
        attempt.lock = observe_lock(attempt.lock)
        return attempt

    monkeypatch.setattr(core, "events", observed_events)
    assert lease._expected_protected_type is type(core)
    if kind == "gkm":
        assert type(core) is DelegatedClient
    # Keep the actual protected activation closure and its published client.
    # Only replace its metadata-only endpoint with the real response adapter.
    build = servers.build_roles

    def build_with_responses(declaration):
        topology = build(declaration)
        original = topology.client

        def client():
            value = original()
            value.responses = ResponsesResource(core)
            return value

        topology.client = client
        return topology

    monkeypatch.setattr(servers, "build_roles", build_with_responses)
    return SimpleNamespace(lease=lease, core=core, state=state,
        capacity=capacity)


def _gkm_joined_call(function, *, timeout=3, releases=()):
    """Joined non-daemon call; fixture bounds actual lock/generation edges."""
    import sys
    import threading
    result, errors = [], []

    def run():
        try:
            result.append(function())
        except BaseException as error:
            errors.append(error)

    worker = threading.Thread(target=run)
    started = []
    eligible = False
    primary = None
    try:
        _start_custody_worker(worker, started)
        if not _assert_custody_workers_joined(started, timeout=timeout):
            primary = sys.exception()
            if primary is not None:
                raise primary
            raise AssertionError("custody unresolved; retirement forbidden")
    finally:
        primary = sys.exception()
        eligible = _assert_custody_workers_joined(started,
            releases=releases, timeout=timeout, primary=primary)
    if not eligible:
        if primary is not None:
            raise primary
        raise AssertionError("custody unresolved; retirement forbidden")
    return result, errors


@pytest.mark.parametrize("failed_refund", [False, True])
def test_gkm_gate_preparation_close_reentry_keeps_original_root(tmp_path, monkeypatch, failed_refund):
    from pllm.deployment import resources
    from pllm.runtime import tokenizer_admission
    f = _gkm_custody_gate_fixture(tmp_path, monkeypatch)
    facade = f.lease._construct_client()
    captured, exits = [], []
    allow_refund = False
    encode = tokenizer_admission._PromptTokenizer.encode
    exit = resources._MemoryReservation.__exit__

    def reenter(value, text, **kwargs):
        owner = f.lease._request_owner
        reservation = owner._metadata_owner
        captured.append((owner, reservation))
        assert f.lease._operation == "preparing" and f.lease._request_busy
        f.lease.close()
        assert f.lease._request_owner is owner and not owner._closed
        assert reservation._token in f.capacity._owners
        assert f.lease._topology is None and not f.lease._cleanup_complete
        return encode(value, text, **kwargs)

    def refund(reservation, *exc):
        if captured and reservation is captured[0][1]:
            exits.append(reservation)
            if failed_refund and not allow_refund:
                raise KeyboardInterrupt("first metadata refund")
        return exit(reservation, *exc)

    monkeypatch.setattr(tokenizer_admission._PromptTokenizer, "encode", reenter)
    monkeypatch.setattr(resources._MemoryReservation, "__exit__", refund)
    _, errors = _gkm_joined_call(lambda: facade.responses.create(input="hello", max_output_tokens=4))
    assert len(errors) == 1 and "closed" in str(errors[0])
    owner, reservation = captured[0]
    assert f.lease._closed and f.lease._operation == "idle"
    assert f.lease._operation_event.is_set() and not f.state["topologies"]
    if failed_refund:
        assert f.lease._request_owner is owner and f.lease._request_busy
        assert reservation._token in f.capacity._owners and not owner._closed
        assert any("cleanup retained" in note for note in errors[0].__notes__)
    allow_refund = True
    _, cleanup = _gkm_joined_call(f.lease.close)
    assert not cleanup and f.lease._cleanup_complete and not f.capacity._owners
    assert owner._closed and reservation._closed
    assert exits == [reservation] * (2 if failed_refund else 1)


@pytest.mark.parametrize("failed_refund", [False, True])
def test_gkm_gate_cleanup_close_reentry_is_not_recursive(tmp_path, monkeypatch, failed_refund):
    from pllm.deployment import resources
    f = _gkm_custody_gate_fixture(tmp_path, monkeypatch)
    stream = f.lease._construct_client().responses.create(input="hello", max_output_tokens=4, stream=True)
    attempt = f.core._active
    owner, topology, client = attempt._prepared_prompt, f.lease._topology, f.lease._client
    reservation = owner._metadata_owner
    close = type(owner._tokenizer).close
    exit = resources._MemoryReservation.__exit__
    callbacks, exits = [], []
    allow_refund = False

    def cleanup(value):
        callbacks.append(value)
        assert len(callbacks) == 1
        f.lease.close()  # Nonexecuting _Attempt.close is itself still unwinding.
        assert f.lease._client is client and f.lease._topology is topology
        assert f.lease._request_owner is owner and not attempt._completion.is_set()
        assert reservation._token in f.capacity._owners
        return close(value)

    def refund(current, *exc):
        if current is reservation:
            exits.append(current)
            if failed_refund and not allow_refund:
                raise RuntimeError("first refund")
        return exit(current, *exc)

    monkeypatch.setattr(type(owner._tokenizer), "close", cleanup)
    monkeypatch.setattr(resources._MemoryReservation, "__exit__", refund)
    _, errors = _gkm_joined_call(stream.close)
    assert bool(errors) is failed_refund
    if failed_refund:
        assert f.lease._drain_attempt is attempt and f.lease._topology is topology
    else:
        assert f.lease._cleanup_complete  # Successful outer boundary serviced its ticket.
    if failed_refund:
        assert attempt._prepared_prompt is owner and not attempt._completion.is_set()
        assert reservation._token in f.capacity._owners
    allow_refund = True
    _, errors = _gkm_joined_call(f.lease.close)
    assert not errors and attempt._completion.is_set() and attempt._retired
    assert f.lease._cleanup_complete and not f.capacity._owners
    assert len(callbacks) == 1 and exits == [reservation] * (2 if failed_refund else 1)


@pytest.mark.parametrize("active_cleared", [False, True])
def test_gkm_gate_original_attempt_not_active_pointer_proves_drain(tmp_path, monkeypatch, active_cleared):
    import threading
    from pllm.runtime.delegated_client import _Attempt
    f = _gkm_custody_gate_fixture(tmp_path, monkeypatch)
    reached, release = threading.Event(), threading.Event()
    captured = []

    def inspect(attempt):
        owner = f.lease._request_owner
        client, topology = f.lease._client, f.lease._topology
        captured.append((attempt, owner, client, topology))
        f.lease.close()
        assert f.lease._drain_attempt is attempt
        assert f.lease._client is client and f.lease._topology is topology
        assert f.lease._request_owner is owner and not f.lease._cleanup_complete
        assert not attempt._completion.is_set() and attempt.running_thread == threading.get_ident()
        if active_cleared:
            assert f.core._active is None and attempt._retired
        else:
            assert owner._metadata_owner._token in f.capacity._owners
        reached.set()
        assert release.wait(timeout=3)

    def produce(attempt):
        if not active_cleared:
            inspect(attempt)
        yield SimpleNamespace(type="response.completed" if active_cleared else "response.output_text.delta",
            response={"id": "gate-fixture"})

    if active_cleared:
        retire = _Attempt._retire

        def retirement(attempt, **kwargs):
            retire(attempt, **kwargs)
            if kwargs["delivered"]:
                inspect(attempt)  # _active cleared, but __next__.finally not run.

        monkeypatch.setattr(_Attempt, "_retire", retirement)
    monkeypatch.setattr(f.core, "_produce", produce)
    stream = f.lease._construct_client().responses.create(input="hello", max_output_tokens=4, stream=True)
    errors = []

    def advance():
        try:
            next(stream)
        except BaseException as error:
            errors.append(error)

    worker = threading.Thread(target=advance)
    started = []
    try:
        _start_custody_worker(worker, started)
        assert reached.wait(timeout=3)
        assert not captured[0][0]._completion.is_set() and not f.lease._cleanup_complete
    finally:
        _assert_custody_workers_joined(started, releases=(release.set,))
    assert not worker.is_alive() and not errors
    _, errors = _gkm_joined_call(f.lease.close)
    assert not errors and captured[0][0]._completion.is_set()
    assert f.lease._cleanup_complete and not f.capacity._owners
    stream.close()


@pytest.mark.parametrize("terminal", [False, True])
def test_fss_legacy_production_cleanup_exception_precedence(tmp_path, monkeypatch, terminal):
    f, stream, attempt, owner = _fss_custody_stream(tmp_path, monkeypatch)
    production = RuntimeError("HEAD production primary")
    cleanup = KeyboardInterrupt("HEAD cleanup replaces production")
    close_tokenizer = type(owner._tokenizer).close
    calls = []

    def close(value):
        calls.append(value)
        if terminal and len(calls) == 1:
            raise production
        raise cleanup

    def produce(original):
        raise production

    monkeypatch.setattr(type(owner._tokenizer), "close", close)
    if not terminal:
        monkeypatch.setattr(f.core, "_produce", produce)
    with pytest.raises(KeyboardInterrupt) as caught:
        next(stream.iterator)
    assert caught.value is cleanup
    assert calls == [owner._tokenizer] * (2 if terminal else 1)
    assert cleanup.__context__ is production
    assert attempt._prepared_prompt is f.lease._request_owner is owner
    assert owner._metadata_owner._token in f.capacity._owners
    monkeypatch.setattr(type(owner._tokenizer), "close", close_tokenizer)
    f.lease.close()
    assert f.lease._cleanup_complete and not f.capacity._owners
    stream.close()


@pytest.mark.parametrize("deferred", [False, True])
@pytest.mark.parametrize("failed", [(), ("client",), ("request",), ("topology",),
    ("client", "topology"), ("request", "topology")])
def test_legacy_lease_cleanup_order_and_exception_matrix(monkeypatch, deferred, failed):
    from pllm.deployment import execution

    f = _request_repair_lease(monkeypatch, pending=deferred)
    events = []
    faults = {name: RuntimeError(name) for name in failed}
    recovering = False

    def phase(name):
        assert f.lease._lock.acquire(blocking=False)
        f.lease._lock.release()
        events.append(name)
        f.lease.close()  # Same owner must defer, not recursively retire.
        if not recovering and name in faults:
            raise faults[name]

    client = SimpleNamespace(close=lambda: phase("client"))
    owner = SimpleNamespace(close=lambda: phase("request"))
    topology = SimpleNamespace(cleanup_complete=True, close=lambda: phase("topology"))
    f.lease._client, f.lease._request_owner, f.lease._topology = client, owner, topology
    f.lease._request_busy = True
    if failed:
        expected = (faults.get("topology") if not deferred else None)
        expected = expected or faults.get("client") or faults.get("request") or faults["topology"]
        with pytest.raises(RuntimeError) as caught:
            f.lease.close()
        assert caught.value is expected
        assert not f.lease._cleanup_complete and f.lease.closed
    else:
        f.lease.close()
        assert f.lease._cleanup_complete
    assert events == (["client", "topology"] if "client" in failed
                      else ["client", "request", "topology"])
    assert f.lease._operation == "idle" and f.lease._operation_event.is_set()
    recovering = True
    f.lease.close()
    assert f.lease._cleanup_complete
    retired = list(events)
    f.lease.close()
    assert events == retired


@pytest.mark.parametrize("kind", ["fss", "puma"])
@pytest.mark.parametrize("phase", ["validation", "enter", "constructor", "render"])
@pytest.mark.parametrize("failed_refund", [False, True])
def test_legacy_pending_preparation_reentry_original_charged_root(
        tmp_path, monkeypatch, kind, phase, failed_refund):
    from pllm.deployment import resources
    from pllm.runtime import http_gateway, tokenizer_admission

    f = _gkm_custody_gate_fixture(tmp_path, monkeypatch, kind=kind)
    facade = f.lease._construct_client()
    observed, exits = [], []
    validate = http_gateway.validate_response_body
    enter = resources._MemoryReservation.__enter__
    exit = resources._MemoryReservation.__exit__
    tokenizer = tokenizer_admission._PromptMetadataSnapshot.tokenizer
    encode = tokenizer_admission._PromptTokenizer.encode
    fault = RuntimeError("explicit refund retry required")
    recovering = False

    def reenter():
        if observed:
            return
        owner = f.lease._request_owner
        assert owner is not None and f.lease._operation == "preparing"
        reservation = owner._metadata_owner
        if phase == "validation":
            assert reservation is None  # Empty root is not admitted backing.
        else:
            assert reservation._token in f.capacity._owners
        observed.append((owner, reservation))
        f.lease.close()
        assert f.lease._request_owner is owner and not f.lease._cleanup_complete

    def validate_body(value):
        if phase == "validation":
            reenter()
        return validate(value)

    def acquire(value):
        result = enter(value)
        if phase == "enter":
            reenter()
        return result

    def construct(snapshot):
        if phase == "constructor":
            reenter()
        return tokenizer(snapshot)

    def render(value, text, **kwargs):
        if phase == "render":
            reenter()
        return encode(value, text, **kwargs)

    def refund(value, *exc):
        if observed and value is observed[0][1]:
            exits.append(value)
            if failed_refund and not recovering:
                raise fault
        return exit(value, *exc)

    monkeypatch.setattr(http_gateway, "validate_response_body", validate_body)
    monkeypatch.setattr(resources._MemoryReservation, "__enter__", acquire)
    monkeypatch.setattr(resources._MemoryReservation, "__exit__", refund)
    monkeypatch.setattr(tokenizer_admission._PromptMetadataSnapshot, "tokenizer", construct)
    monkeypatch.setattr(tokenizer_admission._PromptTokenizer, "encode", render)
    with pytest.raises(resources.NetworkError, match="closed") as caught:
        facade.responses.create(input="hello", max_output_tokens=4)
    assert len(observed) == 1 and not f.state["topologies"] and f.lease._requests == 0
    owner, reservation = observed[0]
    if failed_refund and reservation is not None:
        assert f.lease._request_owner is owner and owner._metadata_owner is reservation
        assert reservation._token in f.capacity._owners and not owner._closed
        assert exits == [reservation]  # No retry in create's except/finally.
        assert any("cleanup retained" in note for note in caught.value.__notes__)
    recovering = True
    f.lease.close()
    assert f.lease._cleanup_complete and not f.capacity._owners
    assert owner._closed


@pytest.mark.parametrize("failed", [False, True])
def test_prepared_close_callbacks_outside_lock_and_waiter_never_retries(monkeypatch, failed):
    import threading
    from pllm.runtime.responses import ResponsesError

    f = _request_metadata_repair_fixture(monkeypatch)
    owner = f.core._prepare_request_prompt(body(), attempt=f.attempt)
    reservation = owner._metadata_owner
    entered, release, waiter_started = [threading.Event() for _ in range(3)]
    original_close = f.tokenizer.close
    fault = RuntimeError("prepared cleanup failure")
    errors, calls = [], []
    recovering = False

    def close():
        calls.append(True)
        assert owner._lock.acquire(blocking=False)
        owner._lock.release()
        with pytest.raises(ResponsesError, match="already running"):
            owner.close()
        entered.set()
        assert release.wait(timeout=3)
        if failed and not recovering:
            raise fault
        original_close()

    def run(waiter=False):
        try:
            owner.close()
        except BaseException as error:
            errors.append(error)

    monkeypatch.setattr(f.tokenizer, "close", close)
    workers = [threading.Thread(target=run), threading.Thread(target=run, args=(True,))]
    started = []
    try:
        _start_custody_worker(workers[0], started)
        assert entered.wait(timeout=3)
        actual_event = owner._close_event

        wait = actual_event.wait

        def observed_wait(timeout=None):
            waiter_started.set()
            assert wait(3 if timeout is None else timeout)
            return True

        monkeypatch.setattr(actual_event, "wait", observed_wait)
        _start_custody_worker(workers[1], started)
        assert waiter_started.wait(timeout=3)
        assert reservation._token in f.capacity._owners and not owner._closed
    finally:
        _assert_custody_workers_joined(started, releases=(release.set,), timeout=6)
    if failed:
        assert errors == [fault, fault] and calls == [True]
        assert owner._tokenizer is f.tokenizer and owner._payload is not None
        assert owner._metadata_owner is reservation and reservation._token in f.capacity._owners
        recovering = True
        owner.close()
    else:
        assert not errors and calls == [True]
    assert owner._closed and not f.capacity._owners


def test_gkm_gate_competing_closer_and_producer_reentry_no_wait_cycle(tmp_path, monkeypatch):
    import threading
    f = _gkm_custody_gate_fixture(tmp_path, monkeypatch)
    entered, closer_entered, reentered, release = (threading.Event() for _ in range(4))
    errors = []

    def hook(name):
        if name == "client-close":
            closer_entered.set()

    f.state["hook"] = hook

    def produce(attempt):
        entered.set()
        assert closer_entered.wait(timeout=3)
        f.lease.close()  # External closer owns retiring and waits for this frame.
        assert f.lease._operation == "retiring" and not attempt._completion.is_set()
        assert f.lease._request_owner is attempt._prepared_prompt
        assert f.lease._topology.rows and f.capacity._owners
        reentered.set()
        assert release.wait(timeout=3)
        yield SimpleNamespace(type="response.output_text.delta", response=None)

    monkeypatch.setattr(f.core, "_produce", produce)
    stream = f.lease._construct_client().responses.create(input="hello", max_output_tokens=4, stream=True)
    attempt = f.core._active

    def call(function):
        try:
            function()
        except BaseException as error:
            errors.append(error)

    producer = threading.Thread(target=call, args=(lambda: next(stream),))
    closer = threading.Thread(target=call, args=(f.lease.close,))
    started = []
    try:
        _start_custody_worker(producer, started)
        assert entered.wait(timeout=3)
        _start_custody_worker(closer, started)
        assert reentered.wait(timeout=3)
        assert closer.is_alive() and not f.lease._cleanup_complete
    finally:
        _assert_custody_workers_joined(started, releases=(release.set, closer_entered.set))
    assert not producer.is_alive() and not closer.is_alive() and not errors
    assert attempt._completion.is_set() and f.lease._cleanup_complete and not f.capacity._owners
    stream.close()


def test_gkm_gate_terminal_cleanup_failure_retains_original_attempt(tmp_path, monkeypatch):
    from pllm.deployment import resources
    f = _gkm_custody_gate_fixture(tmp_path, monkeypatch)

    def produce(attempt):
        yield SimpleNamespace(type="response.completed", response={"id": "gate-fixture"})

    monkeypatch.setattr(f.core, "_produce", produce)
    stream = f.lease._construct_client().responses.create(input="hello", max_output_tokens=4, stream=True)
    attempt = f.core._active
    owner, client, topology = attempt._prepared_prompt, f.lease._client, f.lease._topology
    reservation = owner._metadata_owner
    original_exit = resources._MemoryReservation.__exit__
    exits = []
    primary = KeyboardInterrupt("terminal metadata refund")

    def refund(current, *exc):
        if current is reservation:
            exits.append(current)
            if len(exits) == 1:
                raise primary
        return original_exit(current, *exc)

    monkeypatch.setattr(resources._MemoryReservation, "__exit__", refund)
    _, errors = _gkm_joined_call(lambda: next(stream))
    assert errors == [primary] and not attempt._completion.is_set() and not attempt._retired
    assert attempt.running_thread is None and f.core._active is attempt
    assert f.lease._request_owner is attempt._prepared_prompt is owner
    assert f.lease._client is client and f.lease._topology is topology
    assert reservation._token in f.capacity._owners and not f.lease._cleanup_complete
    _, errors = _gkm_joined_call(f.lease.close)
    assert not errors and attempt._completion.is_set() and attempt._retired
    assert owner._closed and not f.capacity._owners and f.lease._cleanup_complete
    assert exits == [reservation, reservation]
    stream.close()


@pytest.mark.parametrize("failed_refund", [False, True])
def test_gkm_gate_real_async_owned_thread_cancel_waits_for_actual_attempt(tmp_path, monkeypatch, failed_refund):
    import asyncio
    import threading
    from pllm.deployment import resources
    from pllm.runtime.client import AsyncResponsesResource
    from pllm.runtime.delegated_source import AdmittedSource
    f = _gkm_custody_gate_fixture(tmp_path, monkeypatch)
    entered, release, retry, refund_allowed = (threading.Event() for _ in range(4))
    producer_unwound, failed_retirement = threading.Event(), threading.Event()
    original_source = f.state["owned"]
    assert type(original_source) is AdmittedSource
    captured, exits = [], []
    original_exit = resources._MemoryReservation.__exit__

    def produce(attempt):
        owner = attempt._prepared_prompt
        captured.append((attempt, owner, owner._metadata_owner))
        assert f.lease._source_owner is f.core._source is original_source
        f.lease.close()
        assert f.lease._request_owner is owner and not attempt._completion.is_set()
        assert f.lease._source_owner is f.core._source is original_source
        entered.set()
        assert release.wait(timeout=3)
        try:
            attempt.check_cancelled()
        finally:
            producer_unwound.set()
        yield SimpleNamespace(type="response.output_text.delta", response=None)

    def refund(reservation, *exc):
        if captured and reservation is captured[0][2]:
            attempt, owner, _ = captured[0]
            # The producer generator has unwound, but retirement still borrows
            # the ORIGINAL source until this actual ledger refund succeeds.
            assert producer_unwound.is_set()
            assert attempt.running_thread == threading.get_ident()
            assert not attempt._completion.is_set() and not attempt._retired
            assert attempt._prepared_prompt is f.lease._request_owner is owner
            assert f.lease._drain_attempt is attempt and f.core._drain_attempt is attempt
            assert f.lease._source_owner is f.core._source is original_source
            assert f.core._source_checked and reservation._token in f.capacity._owners
            exits.append(reservation)
            if failed_refund and len(exits) == 1:
                failed_retirement.set()
                raise RuntimeError("first async refund")
            if failed_refund:
                retry.set()
                assert refund_allowed.wait(timeout=3)
        return original_exit(reservation, *exc)

    monkeypatch.setattr(f.core, "_produce", produce)
    monkeypatch.setattr(resources._MemoryReservation, "__exit__", refund)

    async def exercise():
        stream = advancing = None
        try:
            resource = AsyncResponsesResource(f.lease._construct_client().responses)
            stream = await asyncio.wait_for(resource.create(input="hello", max_output_tokens=4, stream=True), 4)
            assert f.lease._source_owner is f.core._source is original_source
            advancing = asyncio.create_task(stream.__anext__())
            assert await asyncio.wait_for(asyncio.to_thread(entered.wait, 3), 4)
            attempt, owner, reservation = captured[0]
            assert not producer_unwound.is_set() and attempt.running_thread is not None
            assert f.lease._source_owner is f.core._source is original_source
            advancing.cancel()
            assert await asyncio.wait_for(asyncio.to_thread(attempt.cancellation.event.wait, 3), 4)
            assert not advancing.done() and not attempt._completion.is_set()
            assert f.lease._drain_attempt is attempt and f.lease._request_owner is owner
            assert reservation._token in f.capacity._owners and f.lease._topology.rows
            assert not producer_unwound.is_set() and attempt.running_thread is not None
            assert f.lease._source_owner is f.core._source is original_source
            assert "release" not in f.state["calls"]
            release.set()
            if failed_refund:
                assert await asyncio.wait_for(asyncio.to_thread(retry.wait, 3), 4)
                assert not attempt._completion.is_set() and attempt._prepared_prompt is owner
                assert reservation._token in f.capacity._owners and not f.lease._cleanup_complete
                assert producer_unwound.is_set() and failed_retirement.is_set()
                assert not attempt._retired and f.lease._drain_attempt is f.core._drain_attempt is attempt
                assert f.lease._source_owner is f.core._source is original_source
                assert f.core._source_checked and "release" not in f.state["calls"]
                refund_allowed.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(advancing, 4)
            assert attempt._completion.is_set() and owner._closed
            assert producer_unwound.is_set() and attempt.running_thread is None and attempt._retired
            assert reservation._token not in f.capacity._owners
            # AdmittedSource is immutable metadata, not a closable producer.
            # Client custody retires after refund/drain; lease custody persists
            # until the explicit dependent topology retirement below.
            assert f.core._source is None and not f.core._source_checked
            assert f.lease._source_owner is original_source and f.lease._topology.rows
            assert "release" not in f.state["calls"]
            await asyncio.wait_for(stream.aclose(), 4)
        finally:
            import sys
            primary, secondary = sys.exception(), []
            release_ok = False
            try:
                release_ok = _assert_custody_workers_joined((),
                    releases=(release.set, refund_allowed.set), primary=primary)
            except BaseException as error:
                secondary.append(error)
            awaited = advancing is None and stream is not None
            if advancing is not None:
                try:
                    if not advancing.done():
                        advancing.cancel()
                except BaseException as error:
                    secondary.append(error)
                # Attempt the owned-thread drain even if cancel/release failed.
                try:
                    await asyncio.wait_for(advancing, 4)
                    awaited = True
                except asyncio.CancelledError:
                    awaited = True
                except BaseException as error:
                    secondary.append(error)
            eligible = False
            try:
                # Timeout/task.done is NOT executor completion. Keep actual GKM
                # custody if the original producer retirement was not observed.
                eligible = (release_ok and not secondary and awaited and
                    (advancing is None or (captured and captured[0][0]._completion.is_set()
                        and captured[0][0].running_thread is None)))
            except BaseException as error:
                secondary.append(error)
            if eligible:
                try:
                    await asyncio.wait_for(asyncio.to_thread(f.lease.close), 4)
                except BaseException as error:
                    secondary.append(error)
            elif not secondary and primary is None:
                secondary.append(AssertionError("async producer custody unresolved; no lease cleanup"))
            if secondary:
                selected = primary if primary is not None else secondary[0]
                for error in secondary:
                    try:
                        selected.add_note(f"async custody teardown: {type(error).__name__}")
                    except BaseException:
                        pass
                if primary is None:
                    raise selected

    try:
        _, errors = _gkm_joined_call(lambda: asyncio.run(exercise()), timeout=30,
            releases=(release.set, refund_allowed.set))
        assert not errors
    finally:
        _assert_custody_workers_joined((), releases=(release.set, refund_allowed.set))
    assert f.lease._cleanup_complete and not f.capacity._owners
    assert f.lease._source_owner is None and f.core._source is None
    assert original_source is f.state["owned"]  # Test reference is not production custody.
    assert f.state["calls"].count("release") == 1
    assert exits == [captured[0][2]] * (2 if failed_refund else 1)


@pytest.mark.parametrize("kind", ["fss", "puma"])
def test_non_gkm_protected_factory_keeps_legacy_retirement_branch(tmp_path, monkeypatch, kind):
    lease, core, state = _real_activation_closure_fixture(tmp_path, monkeypatch, kind)
    assert not lease._uses_gkm_custody() and lease._expected_protected_type is type(core)
    value = lease._construct_client().responses.create(input="hello", max_output_tokens=4)
    assert value.simulated and lease._operation == "idle"
    lease.close()
    assert lease._cleanup_complete and state["topologies"][0].cleanup_complete
    core.close()


@pytest.mark.parametrize("busy", [False, True])
def test_ordinary_runtime_client_legacy_retirement(tmp_path, monkeypatch, busy):
    from pllm import Deployment, ExecutionBudget, Model, model_loader, modeling
    from pllm.configuration import Experiment
    from pllm.deployment import execution
    from pllm.profiles import ClientOnlyCpu, resolve_runtime_composition
    from pllm.runtime import servers
    from pllm.runtime.client import OpenAI, ResponsesResource, RuntimeClient, _protected_client_type
    from pllm.runtime.transformer_engine import MaskedTransformerEngine

    # Ordinary eager branch, not a protected pending factory or constructor tag.
    # Only replay, lowering, source I/O and the local engine are doubled. Real
    # composition dispatch, RuntimeClient construction/close and lease stay real.
    experiment = Experiment("ordinary-legacy-retirement",
        ClientOnlyCpu(Model.path(str(tmp_path), model_id="ordinary-legacy")),
        Deployment.local(root="local://ordinary-legacy"),
        ExecutionBudget(requests=1, max_input_tokens=16, max_new_tokens=4))
    config = {"model_type": "qwen3", "vocab_size": 8}
    (tmp_path / "config.json").write_text(json.dumps(config))
    result, snapshot = _pending_result_stub(experiment, "1" * 64)
    network = SimpleNamespace(backend="local", snapshot=snapshot)
    calls, clients = [], []
    manifest = SimpleNamespace(chat_template=None)
    source = SimpleNamespace(source_lock_digest="1" * 64, manifest=manifest)

    def replay(_cls, document):
        assert execution.canonical(document) == result.canonical_bytes()
        calls.append("replay")
        return result

    def validate(value, **kwargs):
        assert value is result and kwargs == {"snapshot": snapshot, "evaluated_at_ms": 0}
        calls.append("validate")
        return {}

    def lower(value, **kwargs):
        assert value == config and kwargs == {"batch": 1, "max_input_tokens": 16, "max_new_tokens": 4}
        calls.append("lower")
        return SimpleNamespace(digest="3" * 64)

    def resolve(model):
        assert model is experiment.pipeline.model
        calls.append("source")
        return source

    def descriptor(path, value, cfg):
        assert path == tmp_path and value is manifest and cfg == config
        calls.append("tokenizer-metadata")
        return {"kind": "alphabet", "alphabet": "abc"}

    def forbidden_gate(*args, **kwargs):
        pytest.fail("ordinary RuntimeClient entered the GKM operation gate")

    real_close = RuntimeClient.close

    def close_core(core):
        assert type(core) is RuntimeClient
        calls.append("client-close")
        real_close(core)
        assert core._closing
        calls.append("client-closed")

    engine = SimpleNamespace(kernel=SimpleNamespace(threads=1))

    class Topology:
        cleanup_complete = False

        def start(self):
            calls.append("topology-start")

        def client(self):
            calls.append("client")
            client = OpenAI.__new__(OpenAI)
            client._core = RuntimeClient(base_url="", api_key="", experiment=experiment,
                local_engine=engine)
            client._owned_topology = None
            client.responses = ResponsesResource(client._core)
            clients.append(client)
            return client

        def close(self):
            core = clients[0]._core
            if core._foreground_active:
                assert calls[-1] == "client-close" and not core._closing
            else:
                assert calls[-1] == "client-closed" and core._closing
            assert lease._client is clients[0]  # HEAD eager finally retains this root.
            calls.append("topology-close")
            self.cleanup_complete = True

    topology = Topology()

    def build(declaration):
        assert declaration is experiment
        calls.append("topology-build")
        return topology

    monkeypatch.setattr(execution, "NetworkSpec", type(network))  # Typed planning double, not routing.
    monkeypatch.setattr(execution.PlanningResult, "from_spec", classmethod(replay))
    monkeypatch.setattr(execution.PlanningResult, "validate", validate)
    monkeypatch.setattr(modeling, "lower_model", lower)
    monkeypatch.setattr(model_loader, "resolve_model", resolve)
    monkeypatch.setattr(MaskedTransformerEngine, "_load_tokenizer_descriptor", staticmethod(descriptor))
    monkeypatch.setattr(servers, "build_roles", build)
    monkeypatch.setattr(RuntimeClient, "close", close_core)
    monkeypatch.setattr(execution.ExecutionLease, "_begin_gkm_operation", forbidden_gate)
    assert _protected_client_type(resolve_runtime_composition(experiment.pipeline)) is None
    assert _protected_client_type(experiment.resolve()) is None

    lease = execution.open_execution(result, network=network, clock=lambda: 0)
    assert calls == ["replay", "validate", "lower", "source", "tokenizer-metadata",
                     "topology-build", "topology-start", "validate"]
    assert not lease._deferred and lease._state == "active" and lease._topology is topology
    assert lease._expected_protected_type is None and not lease._uses_gkm_custody()
    try:
        client = lease._construct_client()
        core = client._core
        assert type(core) is RuntimeClient and core._local_engine is engine
        assert not core._closing and not topology.cleanup_complete
        assert lease._client is client and core._native_worker_limit == 1
        if busy:
            from pllm.runtime.client import _ClientBorrowedError
            core._begin_foreground()
            try:
                with pytest.raises(_ClientBorrowedError):
                    lease.close()
                assert calls[-2:] == ["client-close", "topology-close"]
                assert lease._client is client and not lease._cleanup_complete
                assert topology.cleanup_complete and not core._closing
            finally:
                core._end_foreground()
        lease.close()
        assert calls[-3:] == ["client-close", "client-closed", "topology-close"]
        assert lease.closed and lease._cleanup_complete and topology.cleanup_complete
        assert lease._client is None and lease._topology is None and core._closing
        closed_calls = list(calls)
        lease.close()
        lease.close()
        assert calls == closed_calls and lease._operation == "idle"
    finally:
        lease.close()


def _fss_custody_stream(tmp_path, monkeypatch):
    """Actual inherited execution/retirement; only source and transport doubled."""
    from pllm.runtime.delegated_client import _Attempt
    from pllm.runtime.fss_client import FssClient

    f = _gkm_custody_gate_fixture(tmp_path, monkeypatch, kind="fss")
    assert type(f.core) is FssClient and not f.lease._uses_gkm_custody()
    monkeypatch.setattr(f.core, "_produce", lambda attempt: (
        SimpleNamespace(type="response.completed", response={"id": "fss-custody"}),))
    monkeypatch.setattr(FssClient, "_lease_drain_snapshot",
        lambda *a, **k: pytest.fail("legacy FSS consulted exact-GKM completion"))
    stream = f.lease._construct_client().responses.create(input="hello", max_output_tokens=4, stream=True)
    attempt = f.core._active
    assert type(attempt) is _Attempt
    return f, stream, attempt, attempt._prepared_prompt


@pytest.mark.parametrize("failure", [None, "tokenizer", "refund"])
def test_fss_tokenizer_retirement_foreign_closer_reentry_exact_cycle(tmp_path, monkeypatch, failure):
    import threading
    from pllm.deployment import resources

    f, stream, attempt, owner = _fss_custody_stream(tmp_path, monkeypatch)
    reservation = owner._metadata_owner
    token, manager, amount = reservation._token, reservation._manager, reservation._amount
    topology, client, source_owner = f.lease._topology, f.lease._client, f.lease._source_owner
    tokenizer_entered, closer_waiting, reentered, release = [threading.Event() for _ in range(4)]
    actual_lock = attempt.lock
    errors = {"producer": [], "closer": []}
    calls, refunds = [], []
    fault = RuntimeError("original failed retirement")
    recovering = False
    closing_thread = None
    close_tokenizer = type(owner._tokenizer).close
    exit_reservation = resources._MemoryReservation.__exit__

    class ContendedAttemptLock:
        def acquire(self):
            if actual_lock.acquire(blocking=False):
                return True
            assert threading.current_thread() is closing_thread
            closer_waiting.set()
            assert actual_lock.acquire(timeout=2), "contended attempt-lock escape timeout"
            return True

        def release(self):
            actual_lock.release()

    attempt.lock = ContendedAttemptLock()

    def tokenizer_close(value):
        calls.append(value)
        if len(calls) == 1:
            tokenizer_entered.set()
            assert closer_waiting.wait(timeout=3)
            assert f.lease._operation == "retiring"
            assert f.lease._operation_thread == closing_thread.ident
            f.lease.close()  # REAL producer, foreign retirement owner, same mutex.
            assert f.lease._request_owner is attempt._prepared_prompt is owner
            assert f.lease._client is client and f.lease._source_owner is source_owner
            assert f.lease._topology is topology and topology.rows
            assert not f.lease._cleanup_complete and token in f.capacity._owners
            assert reservation._manager is manager and reservation._amount == amount
            reentered.set()
            assert release.wait(timeout=3)
        if failure == "tokenizer" and not recovering:
            raise fault
        return close_tokenizer(value)

    def refund(value, *exc):
        if value is reservation:
            refunds.append(value)
            if failure == "refund" and not recovering:
                raise fault
        return exit_reservation(value, *exc)

    def run(label, function):
        try:
            function()
        except BaseException as error:
            errors[label].append(error)

    monkeypatch.setattr(type(owner._tokenizer), "close", tokenizer_close)
    monkeypatch.setattr(resources._MemoryReservation, "__exit__", refund)
    producer = threading.Thread(target=run, args=("producer", lambda: next(stream.iterator)))
    closing_thread = threading.Thread(target=run, args=("closer", f.lease.close))
    started = []
    try:
        _start_custody_worker(producer, started)
        assert tokenizer_entered.wait(timeout=3)
        _start_custody_worker(closing_thread, started)
        assert reentered.wait(timeout=3)
        assert producer.is_alive() and closing_thread.is_alive()
        assert not topology.cleanup_complete and token in f.capacity._owners
    finally:
        _assert_custody_workers_joined(started,
            releases=(closer_waiting.set, release.set), timeout=6)
    if failure is not None:
        assert errors == {"producer": [fault], "closer": [fault]}
        assert attempt._prepared_prompt is owner and f.lease._request_owner is owner
        assert f.lease._client is client and f.lease._source_owner is source_owner
        assert owner._metadata_owner is reservation and token in f.capacity._owners
        # Preserve HEAD's terminal failure cleanup; neither waiter nor ticket
        # adds another attempt or erases the SAME unresolved custody.
        assert len(calls) == (2 if failure == "tokenizer" else 1)
        assert len(refunds) == (2 if failure == "refund" else 0)
        assert not f.lease._cleanup_complete
        recovering = True
        f.lease.close()  # Explicit retry of the SAME manager/token only.
    else:
        assert errors == {"producer": [], "closer": []}
    assert f.lease._cleanup_complete and not f.capacity._owners
    assert owner._closed and reservation._manager is manager and reservation._closed
    assert len(calls) == (3 if failure == "tokenizer" else 1)
    assert len(refunds) == (3 if failure == "refund" else 1)
    assert f.state["calls"].count("release") == 1
    stream.close()


@pytest.mark.parametrize("phase", ["produce", "tokenizer", "attempt-close", "core-close"])
def test_fss_producer_only_close_defers_to_outer_physical_extent(tmp_path, monkeypatch, phase):
    from pllm.runtime import delegated_client

    f, stream, attempt, owner = _fss_custody_stream(tmp_path, monkeypatch)
    reservation, topology = owner._metadata_owner, f.lease._topology
    tokenizer_close = type(owner._tokenizer).close
    clear_owned = f.core._clear_owned_state
    observations = []

    def observe():
        f.lease.close()
        assert delegated_client._has_delegated_extent(f.core)
        assert f.lease._request_owner is owner and f.lease._topology is topology
        assert not topology.cleanup_complete and not f.lease._cleanup_complete
        observations.append(True)

    def close_tokenizer(value):
        if phase in {"tokenizer", "attempt-close"}:
            observe()
            assert reservation._token in f.capacity._owners
        return tokenizer_close(value)

    def clear():
        if phase == "core-close" and not observations:
            assert f.core._active is None  # Actual original frame still exists.
            observe()
        return clear_owned()

    def produce(original):
        assert original is attempt
        observe()
        return (SimpleNamespace(type="response.completed", response={"id": "fss-custody"}),)

    monkeypatch.setattr(type(owner._tokenizer), "close", close_tokenizer)
    monkeypatch.setattr(f.core, "_clear_owned_state", clear)
    if phase == "produce":
        monkeypatch.setattr(f.core, "_produce", produce)
        from pllm.runtime.client import ProtocolError
        with pytest.raises(ProtocolError, match="cancelled"):
            next(stream.iterator)
        # An exceptional boundary does not authorize an automatic cleanup retry.
        assert not f.lease._cleanup_complete
        f.lease.close()
    elif phase == "attempt-close":
        attempt.close()
    elif phase == "core-close":
        f.core.close()
    else:
        next(stream.iterator)
    assert observations == [True]
    assert not delegated_client._has_delegated_extent(f.core)
    assert f.lease._cleanup_complete and not f.capacity._owners
    assert f.state["calls"].count("release") == 1
    stream.close()


@pytest.mark.parametrize("before_ticket", [False, True])
@pytest.mark.parametrize("other_core", [False, True])
def test_fss_nested_core_boundary_sticky_failure_never_refreshes_ticket(
        tmp_path, monkeypatch, before_ticket, other_core):
    from pllm.runtime.delegated_client import DelegatedClient, _has_delegated_extent

    f, stream, attempt, owner = _fss_custody_stream(tmp_path, monkeypatch)
    actual_clear = f.core._clear_owned_state
    fault = RuntimeError("caught nested cleanup sentinel")
    calls = []
    second = DelegatedClient.__new__(DelegatedClient)
    # A second REAL core.close extent between same-core extents; no fake TLS.
    import threading
    second._lock = threading.RLock()
    second._active = second._drain_attempt = None
    second._count_preparation = second._failed_request_cleanup = None
    second._compact_session = second._compact_context = None
    second._clear_owned_state = lambda: nested()

    def nested():
        calls.append("nested")
        if not before_ticket:
            f.lease.close()
        raise fault

    def clear():
        calls.append("outer")
        if calls.count("outer") == 1:
            try:
                if other_core:
                    second.close()
                else:
                    f.core.close()
            except RuntimeError as error:
                assert error is fault
            # Failure may precede the first ticket: poison belongs to extent,
            # not the ticket's creation-time failure snapshot.
            f.lease.close()
            assert _has_delegated_extent(f.core)
            assert not f.lease._cleanup_complete and f.lease._source_owner is not None
        elif calls.count("outer") == 2 and not other_core:
            nested()
        return actual_clear()

    monkeypatch.setattr(f.core, "_clear_owned_state", clear)
    f.core.close()
    assert not _has_delegated_extent(f.core) and not _has_delegated_extent(second)
    assert not f.lease._cleanup_complete and f.lease._source_owner is f.state["owned"]
    assert f.lease._request_owner is owner  # Original custody, no replacement.
    f.lease.close()
    assert f.lease._cleanup_complete and not f.capacity._owners
    stream.close()


def test_fss_ticket_consumed_by_intervening_failed_and_successful_closer(tmp_path, monkeypatch):
    import threading

    f, stream, attempt, owner = _fss_custody_stream(tmp_path, monkeypatch)
    entered, release, first_done, retry_done = [threading.Event() for _ in range(4)]
    fault = RuntimeError("external topology failure")
    events, errors = [], []
    producer = None
    failing = True

    def drain():
        if threading.current_thread() is producer:
            f.lease.close()
            events.append(("ticket", f.lease._close_ticket))
            entered.set()
            assert release.wait(timeout=3)
        cleanup.closed = True

    def phase(name):
        if name == "release" and failing:
            raise fault

    def direct_core_close():
        try:
            f.core.close()
        except BaseException as error:
            errors.append(error)

    # Real core.close's drain phase is outside core/attempt locks. Its actual
    # outer TLS extent remains live while explicit closers fail and then succeed.
    # Double only the low-level cleanup callback, not producer identity, the
    # inherited core-close extent, lease gate, or any completion assertion.
    cleanup = SimpleNamespace(drain=drain, closed=False)
    f.core._compact_cleanup = cleanup
    f.state["hook"] = phase
    producer = threading.Thread(target=direct_core_close)
    started = []
    try:
        _start_custody_worker(producer, started)
        assert entered.wait(timeout=3)
        with pytest.raises(RuntimeError) as caught:
            f.lease.close()
        assert caught.value is fault
        assert f.lease._source_owner is f.state["owned"]
        first_done.set()
        failing = False
        f.lease.close()
        retry_done.set()
        assert f.lease._cleanup_complete
        assert f.lease._close_ticket is None
    finally:
        _assert_custody_workers_joined(started, releases=(release.set,), timeout=6)
    assert first_done.is_set() and retry_done.is_set() and not errors
    assert f.lease._cleanup_complete  # Old ticket did NOT resurrect at IDLE.
    assert f.state["calls"].count("release") == 2  # First failure + explicit retry only.
    failing = False
    f.state["hook"] = None
    f.lease.close()
    assert f.lease._cleanup_complete and not f.capacity._owners
    stream.close()


@pytest.mark.parametrize("failed", [False, True])
def test_memory_reservation_backend_callback_close_is_not_recursive(monkeypatch, failed):
    from pllm.deployment import resources

    capacity = resources._ClientCapacity()
    monkeypatch.setattr(capacity, "observe", lambda: SimpleNamespace(headroom_bytes=8 << 30))
    hold = capacity.hold_bytes
    slot, calls = [], []
    fault = RuntimeError("original backend exit fault")
    failing = failed

    def manager(amount, *, _token):
        backend = hold(amount, _token=_token)

        class Manager:
            def __enter__(self):
                return backend.__enter__()

            def __exit__(self, *exc):
                reservation = slot[0]
                calls.append(reservation)
                assert reservation._finish_lock.acquire(blocking=False)
                reservation._finish_lock.release()
                assert capacity.lock.acquire(blocking=False)
                capacity.lock.release()
                with pytest.raises(resources.NetworkError, match="RETIREMENT_BUSY"):
                    reservation.close()
                if failing:
                    raise fault
                return backend.__exit__(*exc)

        return Manager()

    monkeypatch.setattr(capacity, "hold_bytes", manager)
    reservation = resources._MemoryReservation(capacity, 64 << 10)
    slot.append(reservation)
    original_manager, token, amount = reservation._manager, reservation._token, reservation._amount
    reservation.__enter__()
    try:
        if failed:
            with pytest.raises(RuntimeError) as caught:
                reservation.close()
            assert caught.value is fault and capacity._owners[token] == amount
            assert reservation._manager is original_manager and not reservation._closed
        failing = False
        reservation.close()
        reservation.close()
        assert reservation._closed and token not in capacity._owners
        assert reservation._manager is original_manager and reservation._token is token
        assert len(calls) == (2 if failed else 1)
    finally:
        failing = False
        reservation.close()


def test_fss_normal_stop_services_only_after_actual_retirement(tmp_path, monkeypatch):
    f, stream, attempt, owner = _fss_custody_stream(tmp_path, monkeypatch)
    close_tokenizer = type(owner._tokenizer).close

    def close(value):
        f.lease.close()
        assert not f.lease._cleanup_complete and owner._metadata_owner._token in f.capacity._owners
        return close_tokenizer(value)

    monkeypatch.setattr(type(owner._tokenizer), "close", close)
    monkeypatch.setattr(f.core, "_produce", lambda original: ())
    with pytest.raises(StopIteration):
        next(stream.iterator)
    assert attempt.closed and owner._closed
    assert f.lease._cleanup_complete and not f.capacity._owners
    assert f.state["calls"].count("release") == 1
    stream.close()


def test_fss_core_close_outer_cleanup_failure_cannot_service_inner_ticket(tmp_path, monkeypatch):
    f, stream, attempt, owner = _fss_custody_stream(tmp_path, monkeypatch)
    original_source, topology = f.lease._source_owner, f.lease._topology
    actual_clear = f.core._clear_owned_state
    actual_close = type(owner._tokenizer).close
    fault = RuntimeError("outer core cleanup after attempt mutex release")
    calls = []

    def tokenizer_close(value):
        f.lease.close()
        assert f.lease._source_owner is original_source and not topology.cleanup_complete
        return actual_close(value)

    def clear():
        calls.append(True)
        assert f.lease._source_owner is original_source and not topology.cleanup_complete
        if len(calls) == 2:
            assert attempt.closed and owner._closed
            assert f.core._active is None
            assert attempt.lock.acquire(blocking=False)
            attempt.lock.release()
            raise fault
        return actual_clear()

    monkeypatch.setattr(type(owner._tokenizer), "close", tokenizer_close)
    monkeypatch.setattr(f.core, "_clear_owned_state", clear)
    with pytest.raises(RuntimeError) as caught:
        f.core.close()
    assert caught.value is fault and len(calls) == 2
    assert f.lease._source_owner is original_source and f.lease._topology is topology
    assert not topology.cleanup_complete and not f.lease._cleanup_complete
    f.lease.close()
    assert f.lease._cleanup_complete and not f.capacity._owners
    stream.close()


def test_fss_ticket_crosses_other_core_but_only_outer_same_core_services(tmp_path, monkeypatch):
    import threading
    from pllm.runtime.delegated_client import DelegatedClient, _has_delegated_extent

    f, stream, attempt, owner = _fss_custody_stream(tmp_path, monkeypatch)
    actual_clear = f.core._clear_owned_state
    second = DelegatedClient.__new__(DelegatedClient)
    second._lock = threading.RLock()
    second._active = second._drain_attempt = None
    second._count_preparation = second._failed_request_cleanup = None
    second._compact_session = second._compact_context = None
    calls = []

    def second_clear():
        f.core.close()  # A -> B -> A, all real inherited close boundaries.
        assert not f.lease._cleanup_complete

    def clear():
        calls.append(True)
        if len(calls) == 1:
            second.close()
            assert _has_delegated_extent(f.core) and not _has_delegated_extent(second)
            assert not f.lease._cleanup_complete
        elif len(calls) == 2:
            f.lease.close()
            assert _has_delegated_extent(second) and not f.lease._cleanup_complete
        return actual_clear()

    second._clear_owned_state = second_clear
    monkeypatch.setattr(f.core, "_clear_owned_state", clear)
    f.core.close()
    assert not _has_delegated_extent(f.core) and not _has_delegated_extent(second)
    assert f.lease._cleanup_complete and not f.capacity._owners
    assert f.state["calls"].count("release") == 1
    stream.close()


def test_legacy_preparation_deferred_service_keeps_selected_primary(tmp_path, monkeypatch):
    from pllm.deployment import resources
    from pllm.runtime import tokenizer_admission

    f = _gkm_custody_gate_fixture(tmp_path, monkeypatch, kind="fss")
    reservation = f.lease._pending_metadata_owner
    original_exit = resources._MemoryReservation.__exit__
    fault = RuntimeError("new deferred-service cleanup failure")
    calls = []
    failing = True

    construct = tokenizer_admission._PromptMetadataSnapshot.tokenizer

    def tokenizer(snapshot):
        value = construct(snapshot)
        f.lease.close()
        return value

    def exit(value, *exc):
        if value is reservation:
            calls.append(value)
            if failing:
                raise fault
        return original_exit(value, *exc)

    monkeypatch.setattr(tokenizer_admission._PromptMetadataSnapshot, "tokenizer", tokenizer)
    monkeypatch.setattr(resources._MemoryReservation, "__exit__", exit)
    try:
        with pytest.raises(resources.NetworkError, match="closed") as caught:
            f.lease._construct_client().responses.create(input="hello", max_output_tokens=4)
        assert any("deferred lease cleanup retained owners: RuntimeError" in note
                   for note in getattr(caught.value, "__notes__", ()))
        assert calls == [reservation] and not f.lease._cleanup_complete
        assert reservation._token in f.capacity._owners
        assert f.lease._pending_metadata_owner is reservation
    finally:
        failing = False
        f.lease.close()
    assert calls == [reservation, reservation] and not f.capacity._owners


@pytest.mark.parametrize("gate", ["reservation", "prepared", "lease", "stream"])
@pytest.mark.parametrize("recovery_fails", [False, True])
def test_custody_failed_generation_survives_overtaking_explicit_recovery(
        tmp_path, monkeypatch, gate, recovery_fails):
    """C2 keeps C1's actual event/outcome while C3 explicitly recovers.

    All waits observe real blocking edges; no event timeout authorizes cleanup.
    This is reference/exception custody, NOT a control-memory funding proof.
    """
    import threading
    from pllm.deployment import execution, resources

    entered, release, waiting, signaled, resume = [threading.Event() for _ in range(5)]
    first, third = RuntimeError("C1 original outcome"), RuntimeError("C3 recovery outcome")
    calls, errors, started, generated = [], {"C1": [], "C2": []}, [], []
    stream = owner = attempt = lease = None

    def controlled_cleanup(actual):
        calls.append(True)
        if len(calls) == 1:
            entered.set()
            assert release.wait(3), "first cleanup not released"
            raise first
        if len(calls) == 2 and recovery_fails:
            raise third
        return actual()

    if gate == "reservation":
        _bounded_custody_edges(monkeypatch)
        capacity = resources._ClientCapacity()
        monkeypatch.setattr(capacity, "observe", lambda: SimpleNamespace(headroom_bytes=1 << 40))
        hold = capacity.hold_bytes

        class Backend:
            def __init__(self, actual):
                self.actual = actual

            def __enter__(self):
                return self.actual.__enter__()

            def __exit__(self, *exc):
                return controlled_cleanup(lambda: self.actual.__exit__(*exc))

        monkeypatch.setattr(capacity, "hold_bytes",
            lambda amount, **kwargs: Backend(hold(amount, **kwargs)))
        reservation = resources._MemoryReservation(capacity, 1 << 20)
        reservation.__enter__()
        close = reservation.close
    elif gate == "prepared":
        f = _request_metadata_repair_fixture(monkeypatch)
        owner = f.core._prepare_request_prompt(body(), attempt=f.attempt)
        attempt, capacity = f.attempt, f.capacity
        reservation = owner._metadata_owner
        actual = f.tokenizer.close
        monkeypatch.setattr(f.tokenizer, "close", lambda: controlled_cleanup(actual))
        close = owner.close
    else:
        f, stream, attempt, owner = _fss_custody_stream(tmp_path, monkeypatch)
        lease, capacity = f.lease, f.capacity
        reservation = owner._metadata_owner
        actual = type(owner._tokenizer).close
        monkeypatch.setattr(type(owner._tokenizer), "close",
            lambda value: controlled_cleanup(lambda: actual(value)))
        if gate == "lease":
            close = lease.close
        else:
            # Actual local stream retirement closure; ResponseStream.close itself
            # consumes its callback once and is not the concurrent gate under test.
            close = stream._close
            edges = execution.threading

            class ObserveEvents:
                def Event(self):
                    event = edges.Event()
                    generated.append(event)
                    return event

                def __getattr__(self, name):
                    return getattr(edges, name)

            monkeypatch.setattr(execution, "threading", ObserveEvents())

    token, manager, amount = reservation._token, reservation._manager, reservation._amount

    # ResponseStream consumes its underlying close callback once. Recovery of
    # that failed actual attempt is the existing explicit LEASE close contract,
    # not a new promise that repeated wrapper.close drains a failed producer.
    recover = lease.close if gate == "stream" else close

    def run(label):
        try:
            close()
        except BaseException as error:
            errors[label].append(error)

    workers = [threading.Thread(target=run, args=(label,)) for label in ("C1", "C2")]
    try:
        _start_custody_worker(workers[0], started)
        assert entered.wait(3)
        assert capacity._owners[token] == amount
        if gate == "reservation":
            generation = reservation._exit_event
        elif gate == "prepared":
            generation = owner._close_event
        elif gate == "lease":
            generation = lease._operation_event
        else:
            assert len(generated) == 1
            generation = generated[0]
        wait = generation.wait

        def delayed_wait(timeout=None):
            waiting.set()
            assert wait(3 if timeout is None else timeout)
            signaled.set()  # Original generation really completed, not C3.
            assert resume.wait(3), "captured waiter not released"
            return True

        monkeypatch.setattr(generation, "wait", delayed_wait)
        _start_custody_worker(workers[1], started)
        assert waiting.wait(3) and calls == [True]
        release.set()
        assert signaled.wait(3)
        assert _assert_custody_workers_joined(started[:1])
        assert not workers[0].is_alive() and errors["C1"] == [first]
        assert generation._error is first and capacity._owners[token] == amount
        if owner is not None:
            assert attempt._prepared_prompt is owner and owner._metadata_owner is reservation
        if lease is not None:
            assert lease._request_owner is owner and lease._request_busy
        if recovery_fails:
            with pytest.raises(RuntimeError) as caught:
                recover()  # C3 is a genuinely subsequent explicit invocation.
            assert caught.value is third and capacity._owners[token] == amount
            if owner is not None:
                assert owner._metadata_owner is reservation and not owner._closed
        else:
            recover()
            assert token not in capacity._owners
        assert calls == [True, True] and generation._error is first
        assert reservation._manager is manager and reservation._amount == amount
        resume.set()
        assert _assert_custody_workers_joined(started[1:])
        assert not workers[1].is_alive() and errors["C2"] == [first]
        assert calls == [True, True]  # Failed waiter did NOT acquire a retry.
        if recovery_fails:
            recover()
        assert not capacity._owners and reservation._manager is manager
    finally:
        import sys
        primary = sys.exception()
        eligible = _assert_custody_workers_joined(started,
            releases=(release.set, resume.set), primary=primary)
        # Expected injected faults are finite. Teardown preserves any primary;
        # it does not turn a failed assertion into a successful retirement claim.
        if eligible:
            try:
                close()
                if lease is not None:
                    lease.close()
                elif attempt is not None:
                    attempt.close()
                if stream is not None:
                    stream.close()
            except BaseException as cleanup:
                if primary is None:
                    raise
                primary.add_note(f"test teardown failed: {type(cleanup).__name__}")


@pytest.mark.parametrize("ticket_before", [False, True])
def test_fss_postpop_service_failure_poisons_all_actual_core_ancestors(
        tmp_path, monkeypatch, ticket_before):
    from pllm.runtime.delegated_client import _has_delegated_extent

    outer_dir, inner_dir = tmp_path / "outer", tmp_path / "inner"
    outer_dir.mkdir()
    inner_dir.mkdir()
    outer = inner = None
    outer_stream = inner_stream = None
    recovering = False
    fault = RuntimeError("inner post-pop deferred topology failure")
    caught, outer_calls, inner_calls = [], [], []
    try:
        outer, outer_stream, _, _ = _fss_custody_stream(outer_dir, monkeypatch)
        inner, inner_stream, _, _ = _fss_custody_stream(inner_dir, monkeypatch)
        original_source, topology = outer.lease._source_owner, outer.lease._topology
        serial = outer.lease._operation_serial
        outer_clear, inner_clear = outer.core._clear_owned_state, inner.core._clear_owned_state
        close_topology = type(inner.lease._topology).close
        inner_topology = inner.lease._topology

        def close(value):
            if value is inner_topology and not recovering:
                assert _has_delegated_extent(outer.core)
                assert not _has_delegated_extent(inner.core)  # Service is post-pop.
                raise fault
            return close_topology(value)

        def clear_inner():
            inner_calls.append(True)
            if len(inner_calls) == 1:
                inner.lease.close()
            return inner_clear()

        def clear_outer():
            outer_calls.append(True)
            if len(outer_calls) == 1:
                if ticket_before:
                    outer.lease.close()
                try:
                    inner.core.close()
                except RuntimeError as error:
                    assert error is fault
                    caught.append(error)
                if not ticket_before:
                    outer.lease.close()  # Cannot refresh the poisoned ancestor.
            return outer_clear()

        monkeypatch.setattr(type(inner_topology), "close", close)
        monkeypatch.setattr(inner.core, "_clear_owned_state", clear_inner)
        monkeypatch.setattr(outer.core, "_clear_owned_state", clear_outer)
        outer.core.close()
        assert caught == [fault]
        assert not _has_delegated_extent(outer.core) and not _has_delegated_extent(inner.core)
        assert outer.lease._operation_serial == serial  # No automatic service.
        assert outer.lease._source_owner is original_source and outer.lease._topology is topology
        assert not outer.lease._cleanup_complete and not topology.cleanup_complete
        recovering = True
        outer.lease.close()
        inner.lease.close()
        assert outer.lease._cleanup_complete and inner.lease._cleanup_complete
        assert not outer.capacity._owners and not inner.capacity._owners
    finally:
        recovering = True
        import sys
        primary = sys.exception()
        for value in (outer, inner):
            if value is not None:
                try:
                    value.lease.close()
                except BaseException as cleanup:
                    if primary is None:
                        raise
                    primary.add_note(f"test teardown failed: {type(cleanup).__name__}")
        for stream in (outer_stream, inner_stream):
            if stream is not None:
                try:
                    stream.close()
                except BaseException as cleanup:
                    if primary is None:
                        raise
                    primary.add_note(f"test teardown failed: {type(cleanup).__name__}")


def test_lease_rejected_preparation_does_not_poison_unadmitted_claim(tmp_path, monkeypatch):
    from pllm.deployment.network import NetworkError

    f, stream, attempt, owner = _fss_custody_stream(tmp_path, monkeypatch)
    reservation = owner._metadata_owner
    failures, ledger = f.lease._failure_serial, dict(f.capacity._owners)
    try:
        with pytest.raises(NetworkError, match="already owns request preparation/cleanup"):
            f.lease._admit_request(body())
        assert f.lease._failure_serial == failures
        assert f.lease._request_owner is attempt._prepared_prompt is owner
        assert owner._metadata_owner is reservation and f.capacity._owners == ledger
        assert f.lease._operation == "idle" and not owner._closed
    finally:
        import sys
        primary = sys.exception()
        try:
            stream.close()
            f.lease.close()
        except BaseException as cleanup:
            if primary is None:
                raise
            primary.add_note(f"test teardown failed: {type(cleanup).__name__}")


def test_custody_teardown_double_attempts_every_edge_and_preserves_primary():
    """Thread-shaped doubles witness orchestration ONLY, not physical drain."""
    calls, workers, retirement = [], [], []
    primary, release_error = ValueError("selected primary"), KeyboardInterrupt("release")
    join_error, observation_error = RuntimeError("join"), RuntimeError("observation")

    class Double:
        def __init__(self, name):
            self.name = name

        def start(self):
            calls.append((self.name, "start"))

        def join(self, timeout):
            assert timeout == 3
            calls.append((self.name, "join"))
            if self.name == "first":
                raise join_error

        @property
        def ident(self):
            calls.append((self.name, "ident"))
            if self.name == "first":
                raise observation_error
            return 17

        def is_alive(self):
            calls.append((self.name, "alive"))
            return False

    def bad_release():
        calls.append("bad-release")
        raise release_error

    eligible = False
    with pytest.raises(ValueError) as caught:
        try:
            _start_custody_worker(Double("first"), workers)
            _start_custody_worker(Double("second"), workers)
            raise primary
        finally:
            eligible = _assert_custody_workers_joined(workers,
                releases=(bad_release, lambda: calls.append("last-release")))
            if eligible:
                retirement.extend(["close", "refund"])
    assert caught.value is primary and not eligible and not retirement
    assert calls == [("first", "start"), ("second", "start"), "bad-release", "last-release",
        ("first", "join"), ("first", "ident"), ("first", "alive"),
        ("second", "join"), ("second", "ident"), ("second", "alive")]
    assert workers[0]["join_error"] is join_error
    assert workers[0]["observation_error"] is observation_error
    assert primary.__notes__  # Secondary diagnostics did not select a new primary.
    calls.clear()
    with pytest.raises(KeyboardInterrupt) as caught:
        _assert_custody_workers_joined(workers,
            releases=(bad_release, lambda: calls.append("last-release")))
    assert caught.value is release_error  # Surfaced only AFTER every attempt.
    assert calls == ["bad-release", "last-release",
        ("first", "join"), ("first", "ident"), ("first", "alive"),
        ("second", "join"), ("second", "ident"), ("second", "alive")]


@pytest.mark.parametrize("state", ["live", "join-error", "ident-error", "alive-error", "completed"])
def test_custody_teardown_double_eligibility_controls_recovery(state):
    """Controlled observations test the gate; no double proves worker drain."""
    calls, workers, retirement = [], [], []
    primary, fault = ValueError("active primary"), RuntimeError("observation/join failed")

    class Double:
        def start(self):
            calls.append("start")

        def join(self, timeout):
            assert timeout == 3
            calls.append("join")
            if state == "join-error":
                raise fault

        @property
        def ident(self):
            calls.append("ident")
            if state == "ident-error":
                raise fault
            return 17

        def is_alive(self):
            calls.append("alive")
            if state == "alive-error":
                raise fault
            return state == "live"

    eligible = False
    try:
        _start_custody_worker(Double(), workers)
    finally:
        eligible = _assert_custody_workers_joined(workers, primary=primary)
        if eligible:
            retirement.extend(["close", "refund"])
    assert eligible is (state == "completed")
    assert retirement == (["close", "refund"] if state == "completed" else [])
    assert calls == ["start", "join", "ident", "alive"]
    if state != "completed":
        assert primary.__notes__
        state = "completed"
        # A later apparently clean observation cannot erase an earlier failed
        # join/observation or live-worker refusal in this teardown registration.
        assert not _assert_custody_workers_joined(workers, primary=primary)
        assert calls[-3:] == ["join", "ident", "alive"]
        assert not retirement


def test_custody_teardown_double_partial_start_is_registered_joined_and_unresolved(monkeypatch):
    """A raised start + ident None/alive False is NOT stopped proof."""
    import threading
    calls, workers, retirement = [], [], []
    fault = RuntimeError("partial start")

    class Partial:
        def start(self):
            calls.append("start")
            assert workers and workers[0]["worker"] is self  # Before attempting launch.
            raise fault

        def join(self, timeout):
            assert timeout == 3
            calls.append("join")
            raise RuntimeError("cannot join thread before it is started")

        @property
        def ident(self):
            calls.append("ident")
            return None

        def is_alive(self):
            calls.append("alive")
            return False

    partial, eligible = Partial(), False
    with pytest.raises(RuntimeError) as caught:
        try:
            _start_custody_worker(partial, workers)
        finally:
            eligible = _assert_custody_workers_joined(workers)
            if eligible:
                retirement.extend(["close", "refund"])
    assert caught.value is fault and not eligible and not retirement
    assert calls == ["start", "join", "ident", "alive"]
    assert workers[0]["start_error"] is fault and not workers[0]["returned"]

    # Exercise _gkm_joined_call's no-normal-return rule with the same narrow
    # orchestration double. It is NOT a producer/completion stub in a real test.
    calls.clear()
    workers.clear()

    class CallPartial(Partial):
        def start(self):
            calls.append("start")
            raise fault

    monkeypatch.setattr(threading, "Thread", lambda **kwargs: CallPartial())
    with pytest.raises(RuntimeError) as caught:
        _gkm_joined_call(lambda: retirement.append("must-not-run"),
            releases=(lambda: calls.append("release-one"), lambda: calls.append("release-two")))
    assert caught.value is fault and not retirement
    assert calls == ["start", "release-one", "release-two", "join", "ident", "alive"]


@pytest.mark.parametrize("window", ["handoff", "chain", "acquired-before-status"])
def test_gkm_original_attempt_publication_tracks_all_physical_invocations(tmp_path, monkeypatch, window):
    """Real RLock handoffs; post-release/pre-publication pauses never imply drain."""
    import sys
    import threading
    from pllm.runtime.delegated_client import _Attempt

    f = _gkm_custody_gate_fixture(tmp_path, monkeypatch)

    def produce(attempt):
        # The pre-status gap MUST already be retired: otherwise an identity-only
        # finalizer would not try to signal, and the observable race is untested.
        yield SimpleNamespace(type=("response.completed" if window == "acquired-before-status"
            else "response.output_text.delta"), response={"id": "publication"})

    monkeypatch.setattr(f.core, "_produce", produce)
    facade = f.lease._construct_client()
    stream = facade.responses.create(input="hello", max_output_tokens=4, stream=True)
    client = f.lease._client
    assert f.lease._facade is facade and client is not facade and client._core is f.core
    attempt, source, topology = f.core._active, f.lease._source_owner, f.lease._topology
    assert attempt.owner is f.core
    owner, metadata = attempt._prepared_prompt, f.lease._pending_metadata_owner
    reservation = owner._metadata_owner
    retained = {token: amount for token, amount in f.capacity._owners.items()
                if token is not reservation._token}
    assert metadata is not None and retained and source is f.state["owned"]
    actual_lock, retire = attempt.lock, _Attempt._retire
    a_unlocked, a_finalize, b_acquired, b_publish, b_retired, b_release = [threading.Event() for _ in range(6)]
    b_unlocked, b_finalize, c_waiting, c_retired, c_release = [threading.Event() for _ in range(5)]
    c_unlocked, c_finalize = threading.Event(), threading.Event()
    a = b = c = None
    workers, errors = [], {"A": [], "B": [], "C": []}

    class ForwardingLock:
        def acquire(self):
            current = threading.current_thread()
            if not actual_lock.acquire(blocking=False):
                assert current is c
                c_waiting.set()
                assert actual_lock.acquire(timeout=3), "actual mutex did not acquire"
            if current is b and window == "acquired-before-status":
                b_acquired.set()
                try:
                    assert b_publish.wait(3), "pre-status observation did not release"
                except BaseException:
                    actual_lock.release()  # A failed wrapper entry must not strand its real mutex.
                    raise
            return True

        def release(self):
            actual_lock.release()  # REAL release precedes each post-unlock latch.
            current = threading.current_thread()
            if current is a:
                a_unlocked.set()
                assert a_finalize.wait(3), "A finalizer observation did not release"
            elif current is b and window == "chain":
                b_unlocked.set()
                assert b_finalize.wait(3), "B finalizer observation did not release"
            elif current is c:
                c_unlocked.set()
                assert c_finalize.wait(3), "C finalizer observation did not release"

    def retired(value, *, delivered):
        retire(value, delivered=delivered)  # Complete REAL retirement first.
        if value is attempt:
            current = threading.current_thread()
            if current is b:
                b_retired.set()
                assert b_release.wait(3), "B still-held mutex observation did not release"
            elif current is c:
                c_retired.set()
                assert c_release.wait(3), "C still-held mutex observation did not release"

    def roots():
        assert f.lease._client is client and f.lease._source_owner is source
        assert f.lease._topology is topology and not topology.cleanup_complete
        assert f.lease._request_owner is owner and not f.lease._cleanup_complete
        assert f.lease._pending_metadata_owner is metadata
        assert all(f.capacity._owners.get(token) == amount for token, amount in retained.items())
        assert f.core._drain_attempt is attempt and not attempt._completion.is_set()

    def assert_physically_held():
        acquired = actual_lock.acquire(blocking=False)
        if acquired:
            actual_lock.release()  # Even an unexpected successful observation must unwind.
        assert not acquired

    def run(label, function):
        try:
            function()
        except BaseException as error:
            errors[label].append(error)

    def close_original():
        # Real lease.close captures the original incomplete attempt; client.close
        # falls back from cleared _active to that _drain_attempt and closes it.
        # Once the nested invocation really finalizes, cleanup/refund is legitimate.
        f.lease.close()
        # Observe C's own successful return BEFORE joined-worker fallback cleanup
        # can repair anything. This is nested-attempt completion, NOT C thread exit.
        assert attempt._completion.is_set() and attempt._outstanding_calls == 0
        assert attempt.running_thread is None
        assert f.lease._cleanup_complete and not f.capacity._owners
        assert f.lease._client is None and f.lease._request_owner is None
        assert f.lease._topology is None and f.lease._pending_metadata_owner is None
        assert f.lease._source_owner is None
        assert topology.cleanup_complete and not topology.rows
        assert f.state["calls"].count("release") == 1

    attempt.lock = ForwardingLock()
    monkeypatch.setattr(_Attempt, "_retire", retired)
    a = threading.Thread(target=run, args=("A", lambda: next(attempt)))
    b = threading.Thread(target=run, args=("B", attempt.close))
    c = threading.Thread(target=run, args=("C", close_original))
    releases = (a_finalize.set, b_publish.set, b_release.set, b_finalize.set,
                c_release.set, c_finalize.set)
    eligible = False
    try:
        _start_custody_worker(a, workers)
        assert a_unlocked.wait(3) and not attempt._completion.is_set()
        _start_custody_worker(b, workers)
        if window == "acquired-before-status":
            assert b_acquired.wait(3)
            a_finalize.set()
            assert _assert_custody_workers_joined(workers[:1])
            assert attempt.running_thread is None and attempt._outstanding_calls == 1
            assert not attempt._completion.is_set()
            assert attempt._retired and owner._closed and attempt._prepared_prompt is None
            assert reservation._token not in f.capacity._owners
            assert_physically_held()  # B REALLY holds it, unpublished.
            roots()
            b_publish.set()
        assert b_retired.wait(3)
        assert attempt._retired and attempt._prepared_prompt is None and owner._closed
        assert reservation._token not in f.capacity._owners  # Successful metadata retirement IS refunded.
        assert attempt.running_thread == b.ident and not attempt._completion.is_set()
        if window != "acquired-before-status":
            a_finalize.set()
            assert _assert_custody_workers_joined(workers[:1])
        assert attempt.running_thread == b.ident and not attempt._completion.is_set()
        roots()
        _start_custody_worker(c, workers)
        assert c_waiting.wait(3)
        assert_physically_held()
        roots()
        b_release.set()
        assert c_retired.wait(3)
        assert attempt.running_thread == c.ident and not attempt._completion.is_set()
        assert owner._closed and attempt._prepared_prompt is None
        assert reservation._token not in f.capacity._owners
        assert_physically_held()
        roots()
        if window == "chain":
            assert b_unlocked.wait(3)
            b_finalize.set()
        # B must be fully joined while C is still physically held in EVERY case.
        assert _assert_custody_workers_joined(workers[1:2])
        assert attempt.running_thread == c.ident and not attempt._completion.is_set()
        assert attempt._outstanding_calls == 1
        assert_physically_held()
        roots()
        c_release.set()
        assert c_unlocked.wait(3)
        assert attempt.running_thread == c.ident and attempt._outstanding_calls == 1
        assert not attempt._completion.is_set()
        roots()
        acquired = actual_lock.acquire(blocking=False)
        if acquired:
            actual_lock.release()
        assert acquired  # Real unlock is observed; C's publication is STILL outstanding.
        c_finalize.set()
    finally:
        primary = sys.exception()
        eligible = _assert_custody_workers_joined(workers, primary=primary,
            releases=releases)
        if eligible:
            try:
                # C's actual production close may already have retired/refunded.
                # EXTRA cleanup is allowed only after every actual worker joined.
                f.lease.close()
                stream.close()
            except BaseException as cleanup:
                if primary is None:
                    raise
                primary.add_note(f"publication teardown: {type(cleanup).__name__}")
    assert eligible and errors == {"A": [], "B": [], "C": []}
    assert attempt._outstanding_calls == 0 and attempt.running_thread is None
    assert attempt._completion.is_set() and f.lease._cleanup_complete and not f.capacity._owners
    assert topology.cleanup_complete and not topology.rows
    assert f.lease._client is None and f.lease._request_owner is None and f.lease._topology is None
    assert f.lease._pending_metadata_owner is None and f.lease._source_owner is None
    assert f.state["calls"].count("release") == 1


@pytest.mark.parametrize("recursive", ["next", "close"])
def test_gkm_invocation_recursive_parent_and_terminal_admission(tmp_path, monkeypatch, recursive):
    """Keep actual recursive-next behavior; recursive close is cancellation-only."""
    f = _gkm_custody_gate_fixture(tmp_path, monkeypatch)
    calls, snapshots = [], []

    def produce(attempt):
        parent = attempt._invocation_owner
        if recursive == "close":
            attempt.close()
            assert attempt.cancelled.is_set() and not attempt._retired
        else:
            # The REAL executing generator rejects recursion as before; its
            # existing error/retirement path runs, not a replacement policy.
            with pytest.raises(ValueError, match="generator already executing"):
                next(attempt)
            assert attempt._retired
        assert attempt._invocation_owner is parent and attempt.running_thread == threading.get_ident()
        assert attempt._outstanding_calls == 1 and not attempt._completion.is_set()
        snapshots.append(parent)
        yield SimpleNamespace(type="response.output_text.delta", response={"id": "recursive"})

    import threading
    monkeypatch.setattr(f.core, "_produce", produce)
    stream = f.lease._construct_client().responses.create(input="hello", max_output_tokens=4, stream=True)
    attempt, actual = f.core._active, f.core._active.lock

    class ForwardingLock:
        def acquire(self):
            calls.append("acquire")
            return actual.acquire()

        def release(self):
            actual.release()

    attempt.lock = ForwardingLock()
    try:
        next(attempt)
        assert snapshots and attempt._invocation_owner is None and attempt.running_thread is None
        assert attempt._outstanding_calls == 0
        if recursive == "close":
            assert not attempt._completion.is_set()
            attempt.close()
        assert attempt._completion.is_set()
        acquisitions = len(calls)
        with pytest.raises(StopIteration):
            next(attempt)
        attempt.close()
        assert len(calls) == acquisitions and attempt._outstanding_calls == 0
    finally:
        import sys
        primary = sys.exception()
        with attempt._state_lock:
            eligible = attempt._outstanding_calls == 0 and attempt.running_thread is None
        if eligible:
            for cleanup in (stream.close, f.lease.close):
                try:
                    cleanup()
                except BaseException as error:
                    if primary is None:
                        raise
                    primary.add_note(f"recursive publication teardown: {type(error).__name__}")
                    break  # Do not make another cleanup retry after a failure.
        elif primary is None:
            raise AssertionError("recursive invocation unresolved; no retirement")
        else:
            primary.add_note("recursive invocation unresolved; no retirement")
