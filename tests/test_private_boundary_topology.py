from __future__ import annotations

import json
import hashlib
from types import SimpleNamespace

import pytest

from pllm import Experiment
from pllm.runtime.servers import LocalTopology, TopologyError, build_roles, role_client


@pytest.fixture
def experiment(monkeypatch):
    value = Experiment.from_spec({
        "schema": "pllm.experiment.v2", "name": "private-boundary",
        "pipeline": {
            "model": {"source": "org/model", "revision": "a" * 40, "local_files_only": True},
            "components": {
                "inference": {"component": "pllm/inference", "params": {}},
                "kernels": {"component": "pllm/cpu", "params": {"threads": 1}},
                "linear": {"component": "pllm/masked-linear", "params": {}},
                "preparation": {"component": "pllm/model-aware-corrections", "params": {}},
                "boundary": {"component": "pllm/private-lookup-full-head/v1", "params": {}},
                "topology": {"component": "pllm/prepared-body-two-online-boundary/v1", "params": {}},
            },
        },
        "deployment": {"kind": "local", "root": "local://private-boundary"},
        "budget": {"requests": 1, "max_input_tokens": 2, "max_new_tokens": 1},
    })
    # Launcher tests do not claim the old installed native wheel admits the new
    # composition. Native structural admission is tested by its owner after build.
    monkeypatch.setattr(Experiment, "resolve", lambda self: SimpleNamespace(model="org/model"))
    return value


def test_generated_boundary_children_credentials_and_gateway(experiment, monkeypatch):
    monkeypatch.setenv("PLLM_API_KEY", "inherited-provider-secret")
    monkeypatch.setenv("PLLM_PUSH_API_KEY", "inherited-push-secret")
    monkeypatch.setenv("PLLM_BOUNDARY_API_KEYS_JSON", "inherited-boundary-secret")
    monkeypatch.setenv("CUSTOM_INFERENCE_API_KEY", "parentwide-inference-secret")
    topology = build_roles(experiment)
    roles = ("inference", "preparation", "boundary_a", "boundary_b")
    assert topology._role_ids == roles
    ports = {role: 9101 + index for index, role in enumerate(roles)}
    commands = topology._commands(ports)
    assert set(commands) == set(roles)
    for role in ("inference", "preparation"):
        assert "--private-boundary" in commands[role]
        assert commands[role][commands[role].index("--revision") + 1] == "a" * 40
    for role in ("boundary_a", "boundary_b"):
        assert commands[role][1:3] == ["-m", "pllm.runtime.private_boundary_role"]
        assert commands[role][commands[role].index("--role") + 1] == role
    topology._role_credentials = {role: role + "-distinct-secret" for role in roles}
    topology._role_urls = {role: f"http://127.0.0.1:{ports[role]}" for role in roles}
    topology._inference_url, topology._preparation_url = (topology._role_urls[r] for r in roles[:2])
    topology._inference_key, topology._preparation_key = (topology._role_credentials[r] for r in roles[:2])
    topology._push_key = "private-push-secret"
    for role in roles:
        env = topology._environment(role)
        assert "inherited" not in repr(env)
        assert not any(value in commands[role] for value in topology._role_credentials.values())
        if role.startswith("boundary"):
            assert env["PLLM_BOUNDARY_API_KEY"] == topology._role_credentials[role]
            peer = "boundary_b" if role == "boundary_a" else "boundary_a"
            assert env["PLLM_BOUNDARY_PEER_CREDENTIAL_DIGEST"] == hashlib.sha256(
                topology._role_credentials[peer].encode()).hexdigest()
            assert topology._role_credentials[peer] not in repr(env)
            assert "parentwide-inference-secret" not in repr(env)
            assert "PLLM_BOUNDARY_API_KEYS_JSON" not in env
            assert "PLLM_API_KEY" not in env and "PLLM_PUSH_API_KEY" not in env
        else:
            assert "PLLM_BOUNDARY_API_KEYS_JSON" not in env
    monkeypatch.setattr(LocalTopology, "is_healthy", lambda self: True)
    monkeypatch.setattr("pllm.runtime.client.OpenAI", lambda **kwargs: kwargs)
    monkeypatch.setattr("pllm.runtime.sidecar.create_sidecar_app", lambda **kwargs: kwargs)
    for kwargs in (topology.client(), topology.gateway_app(local_api_key="gateway-secret")):
        assert kwargs["boundary_urls"] == tuple(topology._role_urls[r] for r in roles[2:])
        assert kwargs["boundary_api_keys"] == tuple(topology._role_credentials[r] for r in roles[2:])
        assert "worker_url_a" not in kwargs and "worker_url_b" not in kwargs
    for name in ("boundary_urls", "boundary_api_keys"):
        with pytest.raises(TopologyError, match="cannot be overridden"):
            topology.client(**{name: ("x", "y")})
        with pytest.raises(TopologyError, match="cannot be overridden"):
            topology.gateway_app(local_api_key="local", **{name: ("x", "y")})


@pytest.mark.parametrize("failed_role", ["boundary_a", "boundary_b"])
def test_boundary_child_failure_rolls_back_every_started_group(experiment, monkeypatch, failed_role):
    spawned, stopped = [], []
    ports = iter(range(9101, 9105))
    monkeypatch.setattr(LocalTopology, "_free_port", staticmethod(lambda excluded: next(ports)))

    def spawn(self, role, command):
        process = SimpleNamespace(pid=100 + len(spawned), poll=lambda: None)
        self._processes[role] = process
        spawned.append(role)
        return process

    def wait(self, role, expected, url):
        assert expected == ("trusted-preparation" if role == "preparation" else role)
        if role == failed_role:
            raise TopologyError("boundary startup failed")

    monkeypatch.setattr(LocalTopology, "_spawn", spawn)
    monkeypatch.setattr(LocalTopology, "_wait", wait)
    monkeypatch.setattr(LocalTopology, "_stop_all", staticmethod(lambda processes: stopped.extend(processes)))
    topology = build_roles(experiment)
    with pytest.raises(TopologyError, match="boundary startup failed"):
        topology.start()
    assert topology.closed
    assert [p.pid for p in stopped] == [topology._processes[r].pid for r in reversed(spawned)]
    assert topology._role_credentials == {}
    assert topology._inference_key == topology._preparation_key == topology._push_key == ""


def test_private_boundary_external_execution_rejected(experiment):
    with pytest.raises(TopologyError, match="UNSUPPORTED_NETWORK_GRAPH"):
        build_roles(experiment, docker=True)
    with pytest.raises(TopologyError, match="bound Experiment"):
        build_roles(experiment.pipeline)
    with pytest.raises(TopologyError, match="UNSUPPORTED_NETWORK_GRAPH"):
        role_client(experiment, {"boundary_a": ("http://remote", "secret")})


def test_standalone_boundary_requires_distinct_credentials_before_source_load(monkeypatch):
    from pllm.runtime.private_boundary_loader import create_private_boundary_role_app
    from pllm.runtime.private_boundary_protocol import BoundaryError

    monkeypatch.setattr("pllm.runtime.private_boundary_loader.load_private_boundary",
                        lambda *_args, **_kwargs: pytest.fail("loaded weights before admission bridge"))
    key = "own-boundary-secret"
    with pytest.raises(BoundaryError, match="distinct"):
        create_private_boundary_role_app(object(), "boundary_a", key,
                                         hashlib.sha256(key.encode()).hexdigest())


def test_standalone_boundary_nonloopback_fails_before_source_load():
    from pllm.runtime.private_boundary_role import main

    with pytest.raises(SystemExit, match="2"):
        main(["--role", "boundary_a", "--host", "0.0.0.0", "--port", "9000"])


def test_sidecar_factory_forwards_boundary_pair(monkeypatch):
    from pllm.runtime.sidecar import create_sidecar_app

    captured = {}

    class Client:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def close(self):
            pass

    monkeypatch.setattr("pllm.runtime.client.OpenAI", Client)
    create_sidecar_app(remote_base_url="http://inference", remote_api_key="inference-key",
                       boundary_urls=("http://a", "http://b"),
                       boundary_api_keys=("boundary-a-key", "boundary-b-key"))
    assert captured["boundary_urls"] == ("http://a", "http://b")
    assert captured["boundary_api_keys"] == ("boundary-a-key", "boundary-b-key")
    assert "worker_url_a" not in captured


def test_head_only_loader_never_loads_decoder_or_transposes_table(experiment, monkeypatch, tmp_path):
    from pllm.runtime import private_boundary_loader as loader
    from pllm.runtime.models import ModelManifest

    (tmp_path / "config.json").write_text(json.dumps({"tie_word_embeddings": True}))
    manifest = ModelManifest("org/model", "DenseDecoder", "huggingface", str(tmp_path),
                             5, 3, 6, 1, 1, 1, 3, 32, tied_embeddings=True)
    source = SimpleNamespace(path=tmp_path, source_lock_digest="a" * 64,
                             checkpoint_digest="b" * 64, manifest=manifest)
    monkeypatch.setattr(loader, "resolve_model", lambda model, cache_dir: source)
    operations = [
        {"id": "tokens", "operator": "token_lookup", "attributes": {"weight": "table.weight"},
         "output_shape": [1, 2, 3]},
        {"id": "body", "operator": "rms_norm", "output_shape": [1, 2, 3]},
        {"id": "head", "operator": "output_head", "attributes": {"weight": "table.weight"},
         "inputs": ["body"], "output_shape": [1, 2, 5]},
    ]
    plan = SimpleNamespace(to_dict=lambda: {phase: {"operations": operations}
                                           for phase in ("prefill", "decode")},
                           runtime_schedule=lambda pipeline: SimpleNamespace(
                               to_dict=lambda: {"private_boundary": {}}))
    monkeypatch.setattr(loader, "lower_model", lambda *args, **kwargs: plan)
    store = SimpleNamespace(clear_cache=lambda: None)
    monkeypatch.setattr(loader, "SafeTensorStore", lambda path: store)
    monkeypatch.setattr(loader, "preflight_semantic_checkpoint", lambda p, s: None)
    imported = []
    head = SimpleNamespace(bias=None, compiled_weight=SimpleNamespace(_matrix=object()))

    class Importer:
        def __init__(self, **kwargs):
            pass

        def load(self, *args):
            pytest.fail("whole decoder loaded in boundary worker")

        def _load_stage(self, actual_store, spec, actual_manifest):
            assert actual_store is store
            assert spec.id == "lm_head" and not spec.transpose_weight
            assert spec.weight_keys == ("table.weight",)
            assert actual_manifest.metadata["stage_origin"] == "semantic_schedule_v1"
            imported.append(spec)
            return head

        def _load_tokenizer_descriptor(self, *args):
            return {"type": "test-public-tokenizer"}

    monkeypatch.setattr("pllm.runtime.transformer_engine.MaskedTransformerEngine", Importer)
    loaded = loader.load_private_boundary(experiment)
    assert loaded.head is head and loaded.plan is plan
    assert len(imported) == 1


@pytest.fixture
def imported_boundary(experiment, monkeypatch):
    import numpy as np
    from pllm import _native
    from pllm.runtime.models import StageSpec
    from pllm.runtime.private_boundary_loader import LoadedPrivateBoundary

    values = np.array([[1, -2], [3, 4], [-5, 6]], dtype=np.int8)
    scales = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    native_handle = object()
    head = SimpleNamespace(spec=StageSpec("lm_head", "lm_head", 2, 3, weight_bits=8,
                                        activation_bits=8, weight_keys=("table.weight",)),
                           compiled_weight=SimpleNamespace(_matrix=native_handle,
                                                           weight_view=lambda: values),
                           weight_scales=scales,
                           weight_digest=hashlib.sha256(values.tobytes()).hexdigest(),
                           signed_output_bound=11 * 127)
    document = {"config_digest": "d" * 64}
    for phase, rows in (("prefill", 2), ("decode", 1)):
        document[phase] = {
            "query_sequence": rows, "maximum_key_sequence": 2,
            "operations": [
                {"id": f"{phase}.lookup", "operator": "token_lookup",
                 "attributes": {"weight": "table.weight"}, "output_shape": [1, rows, 2]},
                {"id": f"{phase}.head", "operator": "output_head",
                 "attributes": {"weight": "table.weight"}, "output_shape": [1, 3]},
            ],
        }
    contract = {
        "schema_version": "pllm.private_boundary.v1", "model_plan_digest": "c" * 64,
        "model_config_digest": "d" * 64, "composition_digest": experiment.pipeline.digest(),
        "weight_id": "table.weight", "vocabulary": 3, "hidden_width": 2,
        "owners": ["boundary_a", "boundary_b"], "lookup_protocol": "packed128_xor_dpf_v2",
        "head_protocol": "two_party_additive_full_head_v1", "weight_bits": 8,
        "activation_bits": 8, "head_ring_bits": 16, "head_signed_output_bound": 2 * 127 * 127,
        "embedding_numeric": "tied_per_token_row_f32_v1",
        "head_numeric": "ordinary_symmetric_per_row_integer_then_f32_v1",
        "lookup_batch_bound": 64, "lookup_query_bound": 2, "matrix_bytes": 6,
        "scale_bytes": 12, "lookup_scratch_bytes": 192, "client_weight_bytes": 0,
        **{phase: {"lookup_operation": f"{phase}.lookup", "head_operation": f"{phase}.head",
                   "lookup_rows": 2 if phase == "prefill" else 1, "head_rows": 1}
           for phase in ("prefill", "decode")},
    }
    schedule_doc = {"private_boundary": contract, "model_plan_digest": "c" * 64,
                    "model_config_digest": "d" * 64}
    plan = SimpleNamespace(digest="c" * 64, to_dict=lambda: document)
    schedule = SimpleNamespace(digest="e" * 64, composition_digest=experiment.pipeline.digest(),
                               to_dict=lambda: schedule_doc)

    class LookupRoot:
        @staticmethod
        def from_matrix(matrix, original_scales, party):
            assert matrix is native_handle
            assert original_scales == scales.astype("<f4", copy=False).tobytes()
            assert party == 0
            return LookupRoot()

        def descriptor(self):
            return bytes.fromhex("f" * 64), 3, 6

    # Lightweight fixture exercises the bridge, not new-wheel native admission.
    monkeypatch.setattr(_native, "PrivatePageServer", LookupRoot)
    return LoadedPrivateBoundary(SimpleNamespace(source_lock_digest="a" * 64,
                                                  checkpoint_digest="b" * 64),
                                 plan, schedule, head, {}, {})


def test_default_trusted_contract_admits_tighter_regular_requests(experiment, imported_boundary):
    from pllm.runtime.private_boundary_loader import build_private_boundary_contract
    from pllm.runtime.private_boundary_protocol import BoundaryAdmission, BoundaryError
    from dataclasses import replace

    descriptor, admit = build_private_boundary_contract(imported_boundary, experiment)
    assert descriptor.source_digest == "a" * 64
    assert descriptor.checkpoint_digest == "b" * 64
    assert descriptor.model_config_digest == "d" * 64
    for input_rows in (1, 2):
        request = BoundaryAdmission(descriptor.digest, "c" * 64, input_rows, 1)
        assert admit(descriptor, request) == request
    for request in (BoundaryAdmission(descriptor.digest, "0" * 64, 1, 1),
                    BoundaryAdmission(descriptor.digest, "c" * 64, 3, 1),
                    BoundaryAdmission(descriptor.digest, "c" * 64, 2, 2)):
        with pytest.raises(BoundaryError):
            admit(descriptor, request)
    with pytest.raises(BoundaryError):
        admit(replace(descriptor, source_digest="0" * 64),
              BoundaryAdmission(descriptor.digest, "c" * 64, 1, 1))


@pytest.mark.parametrize("field", [
    "source_digest", "checkpoint_digest", "model_config_digest", "composition_digest",
    "schedule_digest", "head_weight_digest", "head_scales_digest", "lookup_digest",
])
def test_independently_imported_contract_rejects_factory_forgery(
    experiment, imported_boundary, monkeypatch, field,
):
    from dataclasses import replace
    from pllm.runtime import private_boundary_loader as loader
    from pllm.runtime.private_boundary_protocol import BoundaryError

    descriptor, admit = loader.build_private_boundary_contract(imported_boundary, experiment)
    monkeypatch.setattr(loader, "load_private_boundary", lambda *args, **kwargs: imported_boundary)
    with pytest.raises(BoundaryError, match="differs from imported source"):
        loader.create_private_boundary_role_app(
            experiment, "boundary_a", "own-boundary-secret", "9" * 64,
            contract_factory=lambda *args: (replace(descriptor, **{field: "0" * 64}), admit),
        )


@pytest.mark.parametrize("tamper", ["absent", "plan", "weight", "numeric", "diagnostic"])
def test_bridge_rejects_absent_or_forged_native_contract(experiment, imported_boundary, tamper):
    from pllm.runtime.private_boundary_loader import build_private_boundary_contract
    from pllm.runtime.private_boundary_protocol import BoundaryError

    schedule = imported_boundary.schedule.to_dict()
    if tamper == "absent":
        del schedule["private_boundary"]
    elif tamper == "plan":
        schedule["private_boundary"]["model_plan_digest"] = "0" * 64
    elif tamper == "weight":
        imported_boundary.head.weight_digest = "0" * 64
    elif tamper == "numeric":
        schedule["private_boundary"]["head_ring_bits"] = 32
    elif tamper == "diagnostic":
        imported_boundary.plan.to_dict()["prefill"]["operations"][-1]["output_shape"] = [1, 2, 3]
    with pytest.raises(BoundaryError):
        build_private_boundary_contract(imported_boundary, experiment)


def test_default_standalone_factory_passes_only_own_key_and_peer_hash(
    experiment, imported_boundary, monkeypatch,
):
    from pllm.runtime import private_boundary_loader as loader

    captured = {}
    monkeypatch.setattr(loader, "load_private_boundary", lambda *args, **kwargs: imported_boundary)
    monkeypatch.setattr("pllm.runtime.private_boundary_worker.create_private_boundary_worker_app",
                        lambda **kwargs: (captured.update(kwargs) or SimpleNamespace(state=SimpleNamespace())))
    app = loader.create_private_boundary_role_app(experiment, "boundary_b", "own-boundary-secret", "9" * 64)
    assert app.state.private_boundary_source is imported_boundary
    assert captured["api_key"] == "own-boundary-secret"
    assert captured["peer_credential_digest"] == "9" * 64
    assert "credentials" not in captured


def test_standalone_entrypoint_uses_own_environment_credentials(experiment, monkeypatch):
    from pllm.runtime import private_boundary_role as role

    monkeypatch.setenv("PLLM_ROLE_EXPERIMENT_JSON", experiment.canonical_bytes().decode())
    monkeypatch.setenv("PLLM_BOUNDARY_API_KEY", "own-boundary-secret")
    monkeypatch.setenv("PLLM_BOUNDARY_PEER_CREDENTIAL_DIGEST", "9" * 64)
    captured = {}
    app = object()

    def factory(bound_experiment, role_id, api_key, peer_digest, **kwargs):
        assert bound_experiment.canonical_bytes() == experiment.canonical_bytes()
        assert role_id == "boundary_b"
        assert api_key == "own-boundary-secret"
        assert peer_digest == "9" * 64
        assert "contract_factory" not in kwargs  # Real native bridge is the default.
        return app

    monkeypatch.setattr(role, "create_private_boundary_role_app", factory)
    monkeypatch.setattr("uvicorn.run", lambda actual_app, **kwargs: captured.update(app=actual_app, **kwargs))
    role.main(["--role", "boundary_b", "--port", "9000"])
    assert captured["app"] is app
    assert captured["host"] == "127.0.0.1" and captured["port"] == 9000


@pytest.mark.parametrize("family", ["qwen2", "qwen3", "llama"])
@pytest.mark.parametrize("max_new", [1, 4])
@pytest.mark.parametrize("compare_client", [False, True], ids=["native-loader", "thin-client"])
def test_generated_native_loader_matches_independent_thin_client(
    tmp_path, monkeypatch, family, max_new, compare_client,
):
    import asyncio
    from dataclasses import replace

    from pllm import Model, _native, lower_model
    from pllm.configuration import Deployment, ExecutionBudget
    from pllm.model_loader import resolve_model
    from pllm.profiles import MaskedLinearCpu
    from pllm.roles import PrivateLookupAndFullHead, PreparedBoundaryRoles
    from pllm.runtime.model_binding import compile_runtime_model
    from pllm.runtime.private_boundary_loader import load_private_boundary, build_private_boundary_contract
    from pllm.runtime.private_boundary_protocol import BoundaryAdmission, BoundaryError
    from pllm.runtime.tiny_llama import create_tiny_llama_checkpoint
    from pllm.runtime.transformer_client import ClientBundle
    from pllm.runtime.transformer_engine import MaskedTransformerEngine

    if not hasattr(_native.PrivatePageServer, "from_matrix"):
        pytest.skip("generated integration requires shared native private table API")
    # One generated decoder layer, 258x8 head, no external checkpoint/network.
    path = create_tiny_llama_checkpoint(
        tmp_path / "generated", model_type=family, hidden_size=8, intermediate_size=16,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=4,
        with_qkv_bias=family == "qwen2", qk_norm=family == "qwen3",
    )
    model = Model.path(str(path), model_id="generated-private-boundary")
    pipeline = MaskedLinearCpu(model, boundary=PrivateLookupAndFullHead(), topology=PreparedBoundaryRoles())
    experiment = Experiment("generated-private-boundary", pipeline,
                            Deployment.local(root=str(tmp_path / "deployment")),
                            ExecutionBudget(requests=1, max_input_tokens=8, max_new_tokens=max_new))
    imported = []
    original_import = MaskedTransformerEngine._load_stage

    def head_only(self, store, spec, manifest, **kwargs):
        imported.append(spec.id)
        assert spec.id == "lm_head" and not spec.transpose_weight
        return original_import(self, store, spec, manifest, **kwargs)

    def forbidden_decoder_load(*args, **kwargs):
        pytest.fail("boundary worker imported full decoder")

    with monkeypatch.context() as custody:
        custody.setattr(MaskedTransformerEngine, "_load_stage", head_only)
        custody.setattr(MaskedTransformerEngine, "load", forbidden_decoder_load)
        loaded = load_private_boundary(experiment)
    assert imported == ["lm_head"]
    descriptor, admit = build_private_boundary_contract(loaded, experiment)
    actual_plan = loaded.plan.to_dict()
    assert actual_plan["prefill"]["maximum_key_sequence"] == 8
    assert actual_plan["decode"]["maximum_key_sequence"] == 8 + max_new - 1
    assert loaded.schedule.to_dict()["private_boundary"]["lookup_query_bound"] == 8 + max_new - 1
    for input_rows in (1, 3, 8):
        for head_calls in sorted({1, max_new}):
            request = BoundaryAdmission(descriptor.digest, loaded.plan.digest,
                                        input_rows + head_calls - 1, head_calls)
            assert admit(descriptor, request) == request
    with pytest.raises(BoundaryError):
        admit(descriptor, BoundaryAdmission(descriptor.digest, loaded.plan.digest, 8 + max_new, max_new))
    if not compare_client:
        # Real default worker factory, including immutable shared-matrix lookup.
        from pllm.runtime.private_boundary_loader import create_private_boundary_role_app

        app = create_private_boundary_role_app(experiment, "boundary_a", "generated-own-key-a",
                                               hashlib.sha256(b"generated-own-key-b").hexdigest())
        actual, actual_admit = build_private_boundary_contract(app.state.private_boundary_source, experiment)
        assert actual == descriptor
        assert actual_admit(actual, request) == request
        return

    # Independent source resolution, publisher, head compilation, and client binding.
    source = resolve_model(model)
    publisher = MaskedTransformerEngine(threads=1, weight_bits=8, activation_bits=8, private_boundary=True)
    asyncio.run(publisher.load(source.manifest))
    try:
        import msgpack

        raw_bundle = publisher.client_bundle(model.model_id)
        public = msgpack.unpackb(raw_bundle, raw=False)
        assert descriptor.lookup_digest == public["privacy"]["private_boundary"]["lookup"]["digest"]
        assert descriptor.head_weight_digest == public["stages"]["lm_head"]["weight_digest"]
        assert descriptor.head_scales_digest == hashlib.sha256(
            public["stages"]["lm_head"]["weight_scales"]).hexdigest()
        assert descriptor.source_digest == source.source_lock_digest
        assert descriptor.checkpoint_digest == source.checkpoint_digest
        bundle = ClientBundle.unpack(raw_bundle)
        client_plan = lower_model(json.loads((path / "config.json").read_text()), batch=1,
                                  max_input_tokens=8, max_new_tokens=max_new)
        compiled = compile_runtime_model(client_plan, bundle, composition=pipeline)
        assert descriptor == compiled.private_boundary_descriptor()
        assert loaded.plan.digest == compiled.model_plan_digest
        for input_rows in (1, 3, 8):
            for decode_steps in sorted({0, max_new - 1}):
                request = compiled.private_boundary_admission(input_rows, decode_steps)
                assert request.lookup_rows == input_rows + decode_steps
                assert request.head_calls == 1 + decode_steps
                assert admit(descriptor, request) == request
        request = compiled.private_boundary_admission(8, max_new - 1)
        # Real native descriptor rejects independent source/config/head forgeries.
        for field in ("source_digest", "checkpoint_digest", "model_config_digest", "head_weight_digest"):
            with pytest.raises(BoundaryError):
                admit(replace(descriptor, **{field: "0" * 64}), request)
        for invalid in (
            BoundaryAdmission(descriptor.digest, "0" * 64, request.lookup_rows, request.head_calls),
            BoundaryAdmission(descriptor.digest, loaded.plan.digest, request.lookup_rows + 1, request.head_calls),
            BoundaryAdmission(descriptor.digest, loaded.plan.digest, request.lookup_rows + 1, max_new + 1),
        ):
            with pytest.raises(BoundaryError):
                admit(descriptor, invalid)
    finally:
        asyncio.run(publisher.unload(model.model_id))


def test_bridge_rejects_non_native_prefill_key_geometry(experiment, imported_boundary):
    from pllm.runtime.private_boundary_loader import build_private_boundary_contract
    from pllm.runtime.private_boundary_protocol import BoundaryError

    imported_boundary.plan.to_dict()["prefill"]["maximum_key_sequence"] = 3
    with pytest.raises(BoundaryError, match="plan or composition differs"):
        build_private_boundary_contract(imported_boundary, experiment)
