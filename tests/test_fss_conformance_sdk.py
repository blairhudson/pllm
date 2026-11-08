"""Frozen Qwen2 bytes/layout and installed-native final-only SDK/benchmark regressions.

Cheap checks against the installed wheel:
    uv run --no-sync python -m pytest -o pythonpath= tests/test_fss_conformance_sdk.py -m 'not slow'
After rebuilding the installed wheel, with no competing build/model run:
    uv run --no-sync python -m pytest -o pythonpath= tests/test_fss_conformance_sdk.py -m slow -v

The generated control tests arithmetic/transport, not trained-model quality.
"""

import dataclasses
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import threading
import time

import httpx
import psutil
import pytest
from safetensors import safe_open
from tokenizers import Tokenizer

from examples.papers.fusefss_conformance import FIXTURE_ID, INPUTS, experiment
from pllm import OpenAI, _native
from pllm.metrics import (BenchmarkCase, BenchmarkRun, BenchmarkSuite, BenchmarkTarget,
                          Sampling, Scenario, benchmark_suite)
from pllm.runtime import fss_source, fss_transport
from pllm.runtime.fss_protocol import HOLDERS, ROLES
from pllm.runtime.responses import _render_prompt as shared_render_prompt
from pllm.runtime.responses import normalize_input
from pllm.runtime.transformer_client import TokenizersJSONTokenizer


@pytest.fixture
def control(tmp_path):
    case, = [c for c in json.loads(INPUTS.read_bytes())["cases"] if c["id"] == FIXTURE_ID]
    exp = experiment(tmp_path)
    return case, exp, Path(exp.pipeline.model.source)


def tensor_file(path):
    data = path.read_bytes()
    size = int.from_bytes(data[:8], "little")
    return json.loads(data[8:8 + size]), data[8 + size:]


def test_original_ieee_bytes_and_complete_qwen2_source_layout(control):
    case, _, root = control
    header, payload = tensor_file(root / "model.safetensors")
    # Independent expected mapping: every public tensor must consume one original
    # source tensor, with no cast, transpose, fused QKV or exchanged gate/up roles.
    expected = {"model.embed_tokens.weight": "embedding", "model.norm.weight": "final.gamma"}
    suffixes = {
        "input_layernorm.weight": "norm1.gamma",
        "post_attention_layernorm.weight": "norm2.gamma",
        "self_attn.q_proj.weight": "q.weight", "self_attn.q_proj.bias": "q.bias",
        "self_attn.k_proj.weight": "k.weight", "self_attn.k_proj.bias": "k.bias",
        "self_attn.v_proj.weight": "v.weight", "self_attn.v_proj.bias": "v.bias",
        "self_attn.o_proj.weight": "o.weight", "mlp.gate_proj.weight": "gate.weight",
        "mlp.up_proj.weight": "up.weight", "mlp.down_proj.weight": "down.weight",
    }
    for layer in range(2):
        expected.update({f"model.layers.{layer}.{name}": f"layer.{layer}.{source}"
                         for name, source in suffixes.items()})
    assert set(header) == set(expected)
    assert set(expected.values()) == set(case["tensors"]) - {"rope.cos", "rope.sin"}
    assert {row["dtype"] for row in header.values()} == {"F32", "F16", "BF16"}
    end = 0
    with safe_open(root / "model.safetensors", framework="np") as stored:
        assert set(stored.keys()) == set(expected)
        for name in sorted(expected):
            original, item = case["tensors"][expected[name]], header[name]
            begin, end = item["data_offsets"]
            assert item["dtype"] == original["dtype"].upper()
            assert item["shape"] == original["shape"] == stored.get_slice(name).get_shape()
            assert payload[begin:end] == b"".join(bytes.fromhex(w)[::-1] for w in original["words"])
    assert end == len(payload)
    assert sum(math.prod(row["shape"]) for row in header.values()) == 2944
    for layer in range(2):
        base = f"model.layers.{layer}."
        for role, shape in (("q", [16, 16]), ("k", [8, 16]), ("v", [8, 16]), ("o", [16, 16])):
            assert header[base + f"self_attn.{role}_proj.weight"]["shape"] == shape
        for role, shape in (("gate", [12, 16]), ("up", [12, 16]), ("down", [16, 12])):
            assert header[base + f"mlp.{role}_proj.weight"]["shape"] == shape


def test_deterministic_public_experiment_and_ordinary_prompt(control, tmp_path):
    case, exp, root = control
    again = experiment(tmp_path / "again")
    files = {"config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"}
    assert {p.name for p in root.iterdir()} == files
    assert all((root / name).read_bytes() == (Path(again.pipeline.model.source) / name).read_bytes()
               for name in files)
    assert exp.budget.requests == 1
    assert exp.budget.max_input_tokens == len(case["prefix"]) == 2
    assert exp.budget.max_new_tokens == case["generation_count"] == 3
    assert exp.pipeline.components["linear"].params["numeric_profile"] == case["profile"]
    config = json.loads((root / "config.json").read_bytes())
    assert config["model_type"] == "qwen2"
    assert config["tie_word_embeddings"] is case["tied_head"] is True
    assert config["eos_token_id"] == case["eos_id"] == 0
    assert config["bos_token_id"] == 1
    assert [config[k] for k in ("hidden_size", "intermediate_size", "num_hidden_layers",
                               "num_attention_heads", "num_key_value_heads", "head_dim",
                               "max_position_embeddings", "vocab_size")] == [16, 12, 2, 4, 2, 4, 6, 7]
    options = json.loads((root / "tokenizer_config.json").read_bytes())
    document = json.loads((root / "tokenizer.json").read_bytes())
    vocab = {"zero": 0, "<unk>": 1, "user:": 2, "three": 3,
             "hello": 4, "five": 5, "six": 6}
    assert document["model"] == {"type": "WordLevel", "vocab": vocab, "unk_token": "<unk>"}
    assert document["pre_tokenizer"] == {"type": "WhitespaceSplit"}
    assert document["added_tokens"] == []
    assert all(document[key] is None for key in ("normalizer", "post_processor", "decoder"))
    inner = Tokenizer.from_file(str(root / "tokenizer.json"))
    assert inner.get_vocab() == vocab
    assert inner.post_processor is None
    assert options["add_bos_token"] is False
    tokenizer = TokenizersJSONTokenizer(inner, config["bos_token_id"], config["eos_token_id"])
    source = fss_source.resolve_client_source(exp)
    assert source.tokenizer_descriptor["chat_template"] == options["chat_template"]
    rendered = shared_render_prompt(normalize_input("hello"), source.tokenizer_descriptor)
    assert rendered == "user: hello"
    assert tokenizer.encode(rendered, add_bos=options["add_bos_token"]) == case["prefix"] == [2, 4]


def test_ordinary_rotary_generation_matches_frozen_f32_words(control):
    case, _, root = control
    config = json.loads((root / "config.json").read_bytes())
    shape = [config["max_position_embeddings"], config["head_dim"]]
    frames = list(fss_source._rotary_frames(shape, config["rope_theta"]))
    for index, name in ((1, "rope.cos"), (2, "rope.sin")):
        original = case["tensors"][name]
        assert original["shape"] == shape and original["dtype"] == "f32"
        assert b"".join(f[index] for f in frames) == b"".join(
            bytes.fromhex(word)[::-1] for word in original["words"])
    assert struct.pack("<f", 1 / math.sqrt(config["head_dim"])) == bytes.fromhex(
        case["attention_scale"]["words"][0])[::-1]


@pytest.mark.rust
def test_source_layout_matches_native_metadata_without_tensor_import(control):
    case, exp, root = control
    assert callable(getattr(_native, "fss_source_contract", None)), "rebuild the installed wheel"
    source = fss_source.resolve_client_source(exp)
    header, _ = tensor_file(root / "model.safetensors")
    contract = source.contract
    assert contract["source_shapes"] == {name: row["shape"] for name, row in header.items()}
    assert contract["source_layouts"] == {
        name: "features" if len(row["shape"]) == 1 else
        "rows_features" if name == "model.embed_tokens.weight" else "output_input"
        for name, row in header.items()
    }
    assert contract["input_rows"] == 2 and contract["output_tokens"] == 3
    assert contract["maximum_sequence"] == 4
    assert len(contract["rotary_shapes"]) == 4
    assert all(shape == [4, 4] for shape in contract["rotary_shapes"].values())
    assert contract["numeric_profile"] == case["profile"]
    assert contract["admission"]["numeric"] == "not_performed"
    assert contract["admission"]["program_resources"] is None


def assert_retired(core):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        counts = []
        for role in ROLES:
            url, key = core.role_connections[role]
            response = httpx.get(url.removesuffix("/v1") + "/v1/fss/metrics",
                headers={"authorization": f"Bearer {key}"}, timeout=10, trust_env=False)
            if not response.is_success:
                code = response.headers.get("x-pllm-error-code")
                if code not in {"CLIENT_MEMORY_OBSERVATION_UNAVAILABLE", "CLIENT_MEMORY_PRESSURE",
                                "CLIENT_MEMORY_CAPACITY_CHANGED", "FSS_METRICS_FAILED"}:
                    code = "UNAVAILABLE"
                raise AssertionError(
                    f"Qwen2 FSS retirement metrics failed: role={role} "
                    f"status={response.status_code} code={code}") from None
            counts.append(response.json()["active_sessions"])
        if counts == [0, 0, 0]:
            assert core._active is None
            return
        time.sleep(0.02)
    raise AssertionError("Qwen2 FSS role owners did not retire")


@contextmanager
def _observe_fss_benchmark_children():
    """Observe fresh owned processes only; exit is not graceful provider drain."""
    parent = psutil.Process()
    existing = set()
    for child in parent.children(recursive=True):
        try:
            existing.add((child.pid, child.create_time()))
        except psutil.NoSuchProcess:
            pass
    workers, roles, errors = {}, {}, []
    stop = threading.Event()
    deadline = time.monotonic() + 600

    def module_args(command, module):
        for index in range(len(command) - 1):
            if command[index:index + 2] == ["-m", module]:
                return command[index + 2:]
        return None

    def observe():
        try:
            while not stop.is_set():
                if time.monotonic() >= deadline:
                    errors.append("observer_deadline")
                    return
                for child in parent.children(recursive=True):
                    try:
                        identity = (child.pid, child.create_time())
                        if identity in existing:
                            continue
                        command = child.cmdline()
                        if module_args(command, "pllm._benchmark_worker") is not None:
                            if "-I" not in command:
                                errors.append("benchmark_worker_not_isolated")
                                return
                            workers[identity] = child
                    except psutil.NoSuchProcess:
                        continue
                for worker in tuple(workers.values()):
                    try:
                        children = worker.children(recursive=True)
                    except psutil.NoSuchProcess:
                        continue
                    for child in children:
                        try:
                            args = module_args(child.cmdline(), "pllm.runtime.fss_worker")
                            if args is None:
                                continue
                            if len(args) < 2 or args[0] != "--role" or args[1] not in ROLES:
                                errors.append("unexpected_fss_role_command")
                                return
                            identity = (child.pid, child.create_time())
                            roles[identity] = (args[1], child)
                        except psutil.NoSuchProcess:
                            continue
                stop.wait(0.02)
        except Exception as error:
            errors.append(type(error).__name__)

    observer = threading.Thread(target=observe, daemon=True)
    observer.start()
    try:
        yield
    finally:
        stop.set()
        observer.join(timeout=2)
        assert not observer.is_alive(), "FSS benchmark observer did not stop"
        assert not errors, errors
        assert len(workers) == 1, "expected one fresh benchmark worker"
        assert sorted(role for role, _ in roles.values()) == sorted(ROLES), "missing or duplicate FSS roles"
        # Ordinary runner already terminates survivors. This checks observed exit,
        # not graceful shutdown, security, custody or a parent qualification proof.
        _, alive = psutil.wait_procs(
            [*workers.values(), *(child for _, child in roles.values())], timeout=10)
        assert not alive, "observed benchmark worker or FSS role survived runner cleanup"


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.rust
def test_qwen2_ordinary_response_target_oracle_and_report_reload(control, tmp_path):
    case, exp, root = control
    assert root.is_absolute()
    source = fss_source.resolve_client_source(exp)
    oracle = json.loads(INPUTS.with_name("semantic-expected.json").read_bytes())
    expected, = [row for row in oracle["results"] if row["fixture_id"] == case["id"]]
    assert expected["profile"] == case["profile"]
    assert case["prefix"] == [2, 4]
    final = expected["final"]
    assert final["generated_ids"] == [4, 4, 4]
    assert final["token_ids"] == case["prefix"] + final["generated_ids"]
    assert final["selection_count"] == 3
    # Fixed independent vocabulary, never candidate Tokenizer.decode as oracle.
    reverse_vocab = {0: "zero", 1: "<unk>", 2: "user:", 3: "three",
                     4: "hello", 5: "five", 6: "six"}
    expected_text = " ".join(reverse_vocab[token] for token in final["generated_ids"])
    assert expected_text == "hello hello hello"
    expected_digest = hashlib.sha256(json.dumps(expected_text, sort_keys=True).encode()).hexdigest()
    aggregate_digest = hashlib.sha256(json.dumps(
        [expected_digest], sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    target = BenchmarkTarget.response(exp, scenario=Scenario.cold(max_output_tokens=3),
        backend="native", expected_input_tokens=2, expected_output_tokens=3)
    benchmark_case = BenchmarkCase("fss-qwen2-response", target)
    # Functional admission envelope, not an RSS bound or memory qualification.
    suite = BenchmarkSuite([benchmark_case], Sampling(processes=1, repetitions=1,
        warmups=0, timeout_seconds=300, memory_bytes=2 << 30))
    output = tmp_path / "benchmark.json"
    with _observe_fss_benchmark_children():
        result = benchmark_suite(suite, inputs={benchmark_case.name: {"prompt": "hello"}},
                                 output=output)
    value, = [row.to_dict() for row in result.results]
    assert value["status"] == "completed", value
    execution = value["execution"]
    assert execution["case"] == benchmark_case.to_spec()
    assert execution["isolation"]["fresh_process"] is True
    report = execution["details"]["report"]
    assert report["checks"]["passed"] is True, report["checks"]
    for key in ("expected_input_tokens", "expected_output_tokens", "fixed_count_final_ids_delivery"):
        assert report["checks"][key] is True
    request, = report["runs"]
    assert request["status"] == "completed"
    assert request["tokens"]["authoritative"] is True
    assert request["tokens"]["input_tokens"] == 2
    assert request["tokens"]["output_tokens"] == 3
    assert request["privacy"]["plaintext_prompt_bytes_sent"] == 0
    assert request["privacy"]["plaintext_token_ids_sent"] == 0
    assert request["measurement"]["output_delivery"] == "buffered_final_ids"
    contract = report["runtime_contract"]
    assert contract["runtime_kind"] == "fss"
    assert contract["output_delivery"] == "buffered_final_ids"
    assert contract["generation_policy"] == "fixed_length_greedy"
    assert contract["decode_tps_supported"] is False
    assert report["summary"]["median_tokens_per_second"] is None
    metrics = {entry["id"]: entry for entry in value["metrics"]}
    assert metrics["response.decode_tps"]["value"] is None
    assert metrics["response.decode_tps"]["unavailable_reason"] == "final_only_private_token_disclosure"
    for role in ROLES:
        assert metrics[f"{role}.cold_cpu"]["value"] > 0
        assert metrics[f"{role}.peak_rss"]["value"] > 0
    assert report["configuration"]["source_lock_digest"] == source.source_lock_digest
    assert execution["comparison"]["source_digest"] == source.source_lock_digest
    assert request["generation"]["output_text_digest"] == expected_digest
    assert execution["comparison"]["output_digest"] == aggregate_digest
    loaded = BenchmarkRun.load(output)
    assert loaded.digest == result.digest
    assert loaded.to_dict() == result.to_dict()
    assert loaded.to_dict()["complete"] is True
    assert loaded.to_dict()["results"] == [value]
    assert loaded.to_dict()["results"][0]["execution"]["case"] == benchmark_case.to_spec()


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.rust
@pytest.mark.parametrize(("capacity", "material_delivery", "online_schedule"), [
    ("capsule", "single", "serial"), ("decoder", "single", "serial"), ("decoder", "batch-v1", "serial"),
    ("decoder", "batch-v1", "wavefront-v1"),
    ("decoder", "batch-v1", "tensor-v1"),
])
def test_qwen2_three_child_sdk_final_oracle_and_large_tape(
        control, monkeypatch, capacity, material_delivery, online_schedule):
    from pllm.runtime.benchmark_memory import execution_memory_admission

    case, exp, root = control
    if capacity == "decoder":
        exp = experiment(root.parent, capacity=capacity)
    if material_delivery != "single":
        exp = exp.with_params(pipeline__linear__material_delivery=material_delivery)
    if online_schedule != "serial":
        exp = exp.with_params(pipeline__linear__online_schedule=online_schedule)
    required = ("FssSourceBuilder", "FssClientRequest", "FssRoleRequest", "fss_transport_admission")
    assert all(callable(getattr(_native, name, None)) for name in required), "rebuild the installed wheel"
    oracle = json.loads(INPUTS.with_name("semantic-expected.json").read_bytes())
    expected, = [row for row in oracle["results"] if row["fixture_id"] == case["id"]]
    final = expected["final"]
    assert expected["profile"] == case["profile"]
    assert final["token_ids"] == case["prefix"] + final["generated_ids"]
    assert final["selection_count"] == len(final["generated_ids"]) == 3
    # Real host memory/disk pricing; a denial or native failure is a failed test,
    # never a skip, substituted callback, smaller geometry or expanded cap.
    price = execution_memory_admission(exp, backend="native")
    assert price["admitted"], price["candidates"]["native"]
    estimate = price["estimate"]
    assert set(estimate["provider_peak_bytes"]) == set(ROLES)
    assert all(value > 0 for value in estimate["provider_peak_bytes"].values())
    if capacity == "capsule":
        assert estimate["components"]["fss_tape_disk_bytes"] > 0
    else:
        assert price["admission_scope"] == "source_qualification_only"
        assert price["full_topology_admitted"] is False
    assert estimate["fss_admission"]["source_pins_verified"] is True

    captured, calls, records = [], [], {}
    post, receipt = fss_transport._Peers.post, fss_transport._receipt

    def observe_post(self, index, action, *args, **kwargs):
        calls.append((index, action, kwargs.get("phase")))  # Public control metadata only.
        return post(self, index, action, *args, **kwargs)

    def observe_receipt(raw, role, **options):
        result = receipt(raw, role, **options)
        if role in HOLDERS:
            storage = result[2]
            records[role] = (storage["records_written"], storage["records_read"])
        return result

    monkeypatch.setattr(fss_transport._Peers, "post", observe_post)
    monkeypatch.setattr(fss_transport, "_receipt", observe_receipt)
    with OpenAI(experiment=exp, timeout=300) as client:
        topology = client._owned_topology
        if capacity == "decoder":
            admitted = topology.memory_admission
            assert admitted["full_topology_admitted"] is True
            assert admitted["estimate"]["components"]["fss_tape_disk_bytes"] > 0
            assert admitted["estimate"]["fss_admission"]["numeric_admission"] == "native_imported_program"
            assert topology.source_qualification["owner_retired"] is True
            assert topology.source_qualification["native_program_priced"] is True
        statuses = topology.statuses
        assert {s.role for s in statuses} == set(ROLES)
        assert len({s.pid for s in statuses}) == 3
        assert all(s.running and s.pid != os.getpid() for s in statuses)
        client._core._result_hook = captured.append
        response = client.responses.create(input="hello", max_output_tokens=3, store=False)
        result, = captured
        assert tuple(f.name for f in dataclasses.fields(result)) == ("selected",)
        assert result.selected == tuple(final["generated_ids"])
        assert response.usage.input_tokens == 2 and response.usage.output_tokens == 3
        assert response.status == "incomplete"
        assert not client._core.records
        assert set(records) == set(HOLDERS)
        assert all(written == read and read > 4096 for written, read in records.values())
        assert_retired(client._core)
        online = [(index, action) for index, action, phase in calls if phase == "online"]
        assert online == [(0, "input"), (1, "input"), (0, "start"), (1, "start"),
                          (0, "finish"), (1, "finish")]
        assert calls.index((2, "finish", "setup")) < calls.index((0, "input", "online"))
        audit = client._core.audit.to_dict()
        assert audit["plaintext_prompt_bytes_sent"] == audit["plaintext_token_ids_sent"] == 0
        for role in HOLDERS:
            assert audit[f"peer_link.dealer.{role}.online_bytes"] == 0
            assert audit[f"peer_link.{role}.dealer.online_bytes"] == 0
            assert audit[f"fss_storage.{role}.read_bytes"] > 0
        assert audit["peer_link.worker_a.worker_b.online_bytes"] > 0
        assert audit["peer_link.worker_b.worker_a.online_bytes"] > 0
    assert all(not status.running for status in topology.statuses)
