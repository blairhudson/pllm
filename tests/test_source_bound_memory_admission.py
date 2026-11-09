"""Generated-source metadata contracts only; never execute a method or launch roles.

Parity oracle is the existing unbound estimator plus independently recorded
source report fields, not the production envelope-binding helper. Independent
arithmetic checks cover native payload/backing additions and whole-role margin;
Python HTTP copy adequacy remains parity-only, not an RSS or security proof.
"""
from contextlib import contextmanager
from dataclasses import replace
import json
import socket
import subprocess
import threading

import pytest

from pllm import Model, _native
from pllm.configuration import Deployment, ExecutionBudget, Experiment
from pllm.profiles import OpenWeightVerifiableCpu, PumaThreePartyCpu
from pllm.protocols import OpenWeightVerifiableOutsourcing, PumaThreePartyDecoder
from pllm.runtime import benchmark_memory as memory
from pllm.runtime import delegated_decoder, delegated_source, replicated_source
from pllm.runtime import delegated_transport, replicated_transport, tokenizer_admission
from pllm.runtime.safetensors_store import SafeTensorStore

from test_delegated_source import write_source_fixture


GKM_CASES = [f"gkm-{capacity}-{sharing}{batch}"
             for capacity in ("capsule", "decoder")
             for sharing in ("full", "seeded")
             for batch in ("", "-batch")]
CASES = GKM_CASES + ["puma-private", "puma-public", "puma-streams", "puma-checked-streams"]
HOST = memory.HostMemory(64 * memory.GiB, 48 * memory.GiB, 0, 256 * memory.GiB)


def forbidden(*args, **kwargs):
    raise AssertionError("metadata-only test crossed host/launch/value boundary")


@pytest.fixture(autouse=True)
def isolate_boundaries(monkeypatch, request):
    assert tokenizer_admission.DECODER_METADATA_MEMORY_BYTES == 384 << 20
    reservations = []
    active = []

    @contextmanager
    def metadata_reservation():
        token = object()
        active.append(token)
        reservations.append(("enter", token))
        try:
            yield None
        finally:
            assert active.pop() is token
            reservations.append(("exit", token))

    # Exercise exceptional exit without acquiring a real resource reservation.
    with pytest.raises(ValueError, match="reservation isolation"):
        with metadata_reservation():
            raise ValueError("reservation isolation")
    assert not active and reservations[0][1] is reservations[1][1]
    reservations.clear()
    monkeypatch.setattr(tokenizer_admission, "reserve_decoder_metadata", metadata_reservation)
    monkeypatch.setattr(memory, "host_memory", lambda: HOST)
    monkeypatch.setattr(memory, "_docker_capacity", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(threading.Thread, "start", forbidden)
    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr("pllm.runtime.hf_hub.resolve_huggingface_source", forbidden)
    for name in ("get", "get_slice", "iter_slices", "get_linear"):
        monkeypatch.setattr(SafeTensorStore, name, forbidden)
    monkeypatch.setattr(delegated_source._AdmittedTensorStore, "get", forbidden)
    monkeypatch.setattr(delegated_source.AdmittedSource, "load_values", forbidden)
    monkeypatch.setattr(delegated_source.AdmittedSource, "load_transport", forbidden)
    monkeypatch.setattr(replicated_source.ReplicatedSource, "publisher", forbidden)
    monkeypatch.setattr(replicated_source.ReplicatedSource, "populate_public_worker", forbidden)
    monkeypatch.setattr(delegated_decoder, "execute_gkm_reference", forbidden)
    monkeypatch.setattr(delegated_decoder, "_execute_verified_window", forbidden)
    monkeypatch.setattr(delegated_transport, "_execute_verified_window", forbidden)
    assert replicated_transport._Peers is delegated_transport._Peers
    monkeypatch.setattr(delegated_transport._Peers, "__init__", forbidden)
    for transport in (delegated_transport, replicated_transport):
        monkeypatch.setattr(transport, "execute_request", forbidden)
        monkeypatch.setattr(transport, "reserve_incremental_memory", forbidden)
    monkeypatch.setattr(replicated_source, "reserve_incremental_memory", forbidden)
    monkeypatch.setattr(replicated_source, "reserve_incremental_disk", forbidden)
    monkeypatch.setattr(memory.MemoryWatchdog, "__init__", forbidden)
    monkeypatch.setattr(memory, "process_memory", forbidden)
    for name in ("_GkmDecoderSession", "_GkmTransportImport", "_GkmTransportClient", "_GkmTransportWorker"):
        # Replace whole class: static entry points must not escape via forwarding.
        monkeypatch.setattr(_native, name, forbidden)
    monkeypatch.setattr(_native, "ReplicatedPublisher", forbidden)
    monkeypatch.setattr(_native, "ReplicatedWorker", forbidden)
    try:
        yield
    finally:
        assert not active
        entries = [token for action, token in reservations if action == "enter"]
        exits = [token for action, token in reservations if action == "exit"]
        assert entries == exits
        case = request.node.callspec.params.get("case", "")
        # PUMA and capsule source admission must not acquire decoder workspace.
        assert bool(entries) == case.startswith("gkm-decoder-")


def _assert_policy(experiment, source, case):
    runtime = json.loads(source.runtime_contract_bytes)
    assert source.experiment_bytes == experiment.canonical_bytes()
    assert runtime["composition"] == experiment.pipeline.digest()
    if case.startswith("gkm-"):
        _, capacity, sharing, *batch = case.split("-")
        assert source.capacity == capacity
        assert runtime["decoder"] == source.contract
        if capacity == "decoder":
            assert runtime["decoder"]["decoder_resources"]["profile"] == "pllm.gkm.decoder-capacity.v1"
        else:
            assert "decoder_resources" not in runtime["decoder"]
        if sharing == "seeded":
            assert runtime["sharing_profile"] == "gkm.seeded.aes128.duplicate-le64.two-purpose.window.v1"
        else:
            assert "sharing_profile" not in runtime
        if batch:
            assert runtime["batch_execution"]["execution"] == "source_stage_batch_v1"
            assert runtime["batch_execution"]["descriptors"]
        else:
            assert "batch_execution" not in runtime
    else:
        if case == "puma-private":
            assert "weights" not in runtime
        else:
            assert runtime["weights"] == "public"
            assert runtime["public_weight_snapshot_bytes_per_worker"] == 2 * runtime["source_float32_bytes"]
            assert runtime["public_weight_read_buffer_bytes"] == 65536
        if "streams" in case:
            checked = "checked" in case
            assert runtime["execution"] == {
                "range_policy": "checked" if checked else "static",
                "max_abs_activation": 128 if checked else 256,
                "lookup": "tiled" if checked else "per_token",
                "selection": "tiled" if checked else "expanded",
                "peer_transport": "streams",
            }
            assert runtime["peer_stream_workspace_bytes_per_worker"] == 8 << 20
        else:
            assert "execution" not in runtime
            assert "peer_stream_workspace_bytes_per_worker" not in runtime


def _assert_charges(report, runtime, case, margin):
    """Independent native/backing arithmetic; HTTP copy prices parity-only."""
    mib = 1 << 20
    process, transport = 256 * mib, 64 * mib

    def rounded(value):
        return (value * (100 + margin) + 99) // 100

    estimate = report["estimate"]
    assert type(estimate["client_peak_bytes"]) is int
    assert type(estimate["provider_peak_bytes"]) is dict
    assert all(type(value) is int for value in estimate["provider_peak_bytes"].values())
    if case.startswith("puma-"):
        payload = runtime["execution_payload_estimate_bytes"]
        workspace = runtime.get("peer_stream_workspace_bytes_per_worker", 0)
        client = rounded(process + payload)
        workers = {role: rounded(process + payload + workspace)
                   for role in ("worker_a", "worker_b", "worker_c")}
        components = estimate["components"]
        assert components["per_engine_compiled_disk_bytes"] == runtime.get("public_weight_snapshot_bytes_per_worker", 0)
        assert components["public_weight_read_buffer_bytes"] == runtime.get("public_weight_read_buffer_bytes", 0)
        if workspace:
            assert components["peer_stream_workspace_bytes_per_worker"] == workspace
        else:
            assert "peer_stream_workspace_bytes_per_worker" not in components
    else:
        decoder = runtime["decoder"]
        expanded = case.startswith("gkm-decoder-")
        source = 0 if expanded else 3 * decoder["source_float32_bytes"]
        native_client = decoder["decoder_resources"]["client_payload_bytes"] if expanded else 64 * mib
        native_worker = decoder["decoder_resources"]["worker_payload_bytes"] if expanded else 64 * mib
        client_base = process + native_client + transport + source
        worker_base = process + native_worker + transport + source
        allocation = estimate["verification_allocation"]
        assert estimate["components"]["source_import_and_retention_bytes_per_role"] == source
        assert allocation["contract_execution_payload_estimate_bytes"] == decoder["execution_payload_estimate_bytes"]
        assert allocation["native_allowance_per_role_bytes"] == native_client
        assert allocation["native_allowance_by_role_bytes"] == {
            "client": native_client, "worker_a": native_worker, "worker_b": native_worker}
        if case.endswith("-batch"):
            batch = runtime["batch_execution"]
            assert type(batch["descriptors"]) is list and batch["descriptors"]
            for descriptor in batch["descriptors"]:
                assert type(descriptor["rows"]) is int and descriptor["rows"] > 0
                assert type(descriptor["columns"]) is int and descriptor["columns"] >= 0
                assert descriptor["kind"] in {"linear", "lookup"}
            backing = max(4 * d["rows"] * d["columns"] for d in batch["descriptors"])
            encoded = max((8 * d["rows"] * d["columns"] for d in batch["descriptors"]
                           if d["kind"] == "linear"), default=0)
            resources = batch["resources"]
            for key in ("extra_native_owner_bytes", "extra_wire_bytes"):
                assert type(resources[key]) is int and resources[key] >= 0
            delta = resources["extra_native_owner_bytes"] + resources["extra_wire_bytes"]
            scalar_worker = (decoder["decoder_resources"]["worker_payload_bytes"] if expanded
                             else decoder["execution_payload_estimate_bytes"])
            assert type(batch["client_payload_bytes"]) is int
            assert type(batch["worker_payload_bytes"]) is int
            assert type(batch["compiler_encoded_input_peak_bytes"]) is int
            assert batch["compiler_encoded_input_peak_bytes"] == encoded
            assert batch["client_payload_bytes"] == decoder["execution_payload_estimate_bytes"] + delta + encoded
            assert batch["worker_payload_bytes"] == scalar_worker + delta
            assert allocation["batch_runtime_backing_delta_bytes"] == backing
            python_client = allocation["batch_python_client_peak_bytes"]
            python_workers = allocation["batch_python_worker_peak_bytes"]
            assert type(python_client) is int and python_client >= 0
            assert type(python_workers) is dict and set(python_workers) == {"worker_a", "worker_b"}
            assert all(type(value) is int and value >= 0 for value in python_workers.values())
            client = rounded(client_base + delta + encoded + python_client + backing)
            workers = {role: rounded(worker_base + delta + python_workers[role])
                       for role in ("worker_a", "worker_b")}
        else:
            client = rounded(client_base)
            workers = {role: rounded(worker_base) for role in ("worker_a", "worker_b")}
    assert estimate["client_peak_bytes"] == client
    assert estimate["provider_peak_bytes"] == workers
    assert estimate["native_total_peak_bytes"] == client + sum(workers.values())


def make_source(path, case):
    """Reuse tiny local fixture; only decoder/SDK metadata constructors run."""
    if case.startswith("gkm-"):
        _, capacity, sharing, *batch = case.split("-")
        write_source_fixture(path / "source", capacity=capacity)
        linear = OpenWeightVerifiableOutsourcing(capacity=capacity, sharing=sharing,
            execution="source_stage_batch_v1" if batch else None)
        pipeline = OpenWeightVerifiableCpu(Model.path(str(path / "source")), linear=linear)
        module = delegated_source
    else:
        # Same original Safetensors/tokenizer fixture, PUMA's small geometry.
        write_source_fixture(path / "source", tied=False, vocabulary=5,
            config_overrides={"hidden_size": 2, "intermediate_size": 2,
                "num_hidden_layers": 1, "num_attention_heads": 1,
                "num_key_value_heads": 1, "head_dim": 2,
                "attention_bias": True, "max_position_embeddings": 64})
        choices = {"weights": "private" if case == "puma-private" else "public"}
        if "streams" in case:
            choices["peer_transport"] = "streams"
        if "checked" in case:
            choices.update(range_policy="checked", max_abs_activation=128,
                           lookup="tiled", selection="tiled")
        pipeline = PumaThreePartyCpu(Model.path(str(path / "source")),
                                     linear=PumaThreePartyDecoder(**choices))
        module = replicated_source
    experiment = Experiment("source-memory", pipeline,
        Deployment.local(root=str(path / "roles")),
        ExecutionBudget(requests=2, max_input_tokens=8, max_new_tokens=3))
    source = module.admit_source(experiment)
    _assert_policy(experiment, source, case)
    return experiment, source, module


def checkpoint_bytes(source):
    return sum(row["size"] for row in source.source_lock["files"]
               if row["path"].endswith(".safetensors"))


def legacy_estimate(experiment, source, margin=25):
    return memory.estimate_memory(source.config, experiment.pipeline,
        max_input_tokens=experiment.budget.max_input_tokens,
        max_output_tokens=experiment.budget.max_new_tokens,
        checkpoint_bytes=checkpoint_bytes(source), memory_safety_margin_percent=margin)


def with_source_report(estimate, source, case):
    if case.startswith("gkm-"):
        decoder = source.contract
        estimate["gkm_admission"] = {
            "checkpoint_digest": source.checkpoint_digest,
            "source_lock_digest": source.source_lock_digest,
            "tokenizer_digest": source.tokenizer_digest,
            "decoder_contract_digest": source.contract_digest,
            "runtime_contract_digest": source.runtime_contract_digest,
            "numeric_profile": decoder["numeric_profile"],
            "local_numeric_profile": decoder["local_numeric_profile"],
            "max_input_tokens": decoder["input_rows"],
            "max_output_tokens": decoder["decode_steps"] + 1,
            "execution_payload_estimate_bytes": source.execution_payload_estimate_bytes,
        }
    return estimate


def forbid_placeholder_regeneration(monkeypatch, experiment):
    # Do not patch functions legitimately called by owner request rebinds.
    # Reject only estimator's synthetic Experiment or placeholder decoder path.
    canonical = experiment.canonical_bytes()
    original_gkm = _native.gkm_runtime_contract
    original_puma = _native.replicated_source_contract

    def gkm(encoded, config, rows, outputs, lock, tokenizer):
        assert encoded == canonical and lock not in {"0" * 64, "1" * 64}
        assert tokenizer not in {"0" * 64, "2" * 64}
        return original_gkm(encoded, config, rows, outputs, lock, tokenizer)

    def puma(encoded, config, rows, outputs, lock, tokenizer):
        assert encoded == canonical and lock != "0" * 64 and tokenizer != "0" * 64
        return original_puma(encoded, config, rows, outputs, lock, tokenizer)

    monkeypatch.setattr(_native, "gkm_runtime_contract", gkm)
    monkeypatch.setattr(_native, "replicated_source_contract", puma)
    original_decoder = delegated_decoder.decoder_contract

    def decoder(*args, **kwargs):
        # benchmark_memory's local import is a different seam from the source
        # module's already imported binder. Authentic source rebind stays real.
        forbidden()

    assert delegated_source.decoder_contract is original_decoder
    monkeypatch.setattr(delegated_decoder, "decoder_contract", decoder)


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("margin", [0, 25])
def test_bound_and_smaller_request_match_every_legacy_report_field(tmp_path, monkeypatch, case, margin):
    experiment, source, module = make_source(tmp_path, case)
    original_bytes = source.runtime_contract_bytes
    # Independent full-envelope native result, with actual pins and Experiment.
    assert source.runtime_contract(experiment, 8, 3) == original_bytes
    expected = memory.admit_memory(with_source_report(legacy_estimate(experiment, source, margin),
        source, case), HOST, backend="native")
    monkeypatch.setattr(module, "admit_source", lambda *a, **kw: source)
    forbid_placeholder_regeneration(monkeypatch, experiment)
    full = memory.benchmark_memory(experiment, backend="native", memory_safety_margin_percent=margin)
    small = memory.benchmark_memory(experiment, max_input_tokens=2, max_output_tokens=1,
                                   backend="native", memory_safety_margin_percent=margin)
    assert full == small == expected
    assert full["estimate"]["max_input_tokens"] == 8
    assert full["estimate"]["max_output_tokens"] == 3
    runtime = json.loads(original_bytes)
    _assert_charges(full, runtime, case, margin)
    _assert_charges(small, runtime, case, margin)
    if case.startswith("gkm-"):
        full["estimate"]["verification_allocation"]["native_transport_resources"].clear()
        full["estimate"]["gkm_admission"].clear()
    else:
        full["estimate"]["components"].clear()
    assert source.runtime_contract_bytes is original_bytes
    assert source.runtime_contract(experiment, 8, 3) == original_bytes
    assert memory.benchmark_memory(experiment, backend="native",
                                  memory_safety_margin_percent=margin) == expected


CORRUPTIONS = ["foreign-experiment", "missing-experiment", "source-pin", "tokenizer-pin",
               "missing-runtime", "mutable-runtime", "substituted-runtime", "decoder",
               "saved-geometry", "wrong-owner"]


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("change", CORRUPTIONS)
def test_bad_saved_owner_rejected_before_common_admission(tmp_path, monkeypatch, case, change):
    experiment, source, module = make_source(tmp_path, case)
    if change == "foreign-experiment":
        source = replace(source, experiment_bytes=replace(experiment, name="foreign").canonical_bytes())
    elif change == "missing-experiment":
        source = replace(source, experiment_bytes=None)
    elif change == "source-pin":
        source = replace(source, source_lock_digest="a" * 64)
    elif change == "tokenizer-pin":
        source = replace(source, tokenizer_digest="b" * 64)
    elif change == "missing-runtime":
        source = replace(source, runtime_contract_bytes=None)
    elif change == "mutable-runtime":
        source = replace(source, runtime_contract_bytes=bytearray(source.runtime_contract_bytes))
    elif change == "substituted-runtime":
        # Valid native bytes for smaller geometry must not fund saved envelope.
        source = replace(source, runtime_contract_bytes=source.runtime_contract(experiment, 2, 1))
    elif change in {"decoder", "saved-geometry"}:
        decoder = source.contract
        if change == "decoder":
            decoder["source_float32_bytes"] += 4
        else:
            decoder["input_rows"] = 2
        source = replace(source, contract_bytes=json.dumps(decoder).encode())
    else:
        # An arbitrary document is not a trusted internal admission owner.
        source = {"runtime_contract_bytes": source.runtime_contract_bytes}
    monkeypatch.setattr(module, "admit_source", lambda *a, **kw: source)
    monkeypatch.setattr(memory, "admit_memory", forbidden)
    with pytest.raises(memory.BenchmarkMemoryError, match="source-bound memory envelope") as error:
        memory.benchmark_memory(experiment, backend="native")
    assert isinstance(error.value.__cause__, (ValueError, TypeError, KeyError, OverflowError))


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("rows,outputs", [(True, 1), (1, False), (0, 1), (1, 0),
                                          (-1, 1), (32769, 1), (1, 32769), (1.5, 1)])
def test_invalid_request_geometry_rejected_after_source_before_admission(tmp_path, monkeypatch,
                                                                      case, rows, outputs):
    experiment, source, module = make_source(tmp_path, case)
    seen = []

    def admitted(*args, **kwargs):
        seen.append(True)
        return source

    monkeypatch.setattr(module, "admit_source", admitted)
    monkeypatch.setattr(memory, "admit_memory", forbidden)
    with pytest.raises(memory.BenchmarkMemoryError, match="source-bound memory envelope") as error:
        memory.benchmark_memory(experiment, max_input_tokens=rows, max_output_tokens=outputs,
                                backend="native")
    assert seen == [True]
    assert isinstance(error.value.__cause__, ValueError)


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("rows,outputs", [(9, 1), (1, 4)])
def test_over_envelope_keeps_existing_error_and_source_precedence(tmp_path, monkeypatch,
                                                                case, rows, outputs):
    experiment, source, module = make_source(tmp_path, case)
    monkeypatch.setattr(module, "admit_source", lambda *a, **kw: source)
    monkeypatch.setattr(memory, "admit_memory", forbidden)
    with pytest.raises(memory.BenchmarkMemoryError, match="GKM measurement exceeds its saved workload envelope"):
        memory.benchmark_memory(experiment, max_input_tokens=rows, max_output_tokens=outputs)
    failure = ValueError("source admission first")

    def rejected(*args, **kwargs):
        raise failure

    monkeypatch.setattr(module, "admit_source", rejected)
    with pytest.raises(ValueError) as error:
        memory.benchmark_memory(experiment, max_input_tokens=rows, max_output_tokens=outputs)
    assert error.value is failure


@pytest.mark.parametrize("case", CASES)
def test_smaller_request_uses_existing_request_binder_but_prices_saved_bytes(tmp_path, monkeypatch, case):
    experiment, source, module = make_source(tmp_path, case)
    calls = []
    binder = type(source).runtime_contract

    def recording(self, actual, rows, outputs):
        calls.append((actual, rows, outputs))
        return binder(self, actual, rows, outputs)

    monkeypatch.setattr(type(source), "runtime_contract", recording)
    monkeypatch.setattr(module, "admit_source", lambda *a, **kw: source)
    report = memory.benchmark_memory(experiment, max_input_tokens=2, max_output_tokens=1)
    assert calls == [(experiment, 8, 3), (experiment, 2, 1)]
    assert report["estimate"]["max_input_tokens"] == 8
    assert report["estimate"]["max_output_tokens"] == 3


@pytest.mark.parametrize("case", CASES)
def test_cancellation_not_translated_and_metadata_owner_has_no_close(tmp_path, monkeypatch, case):
    experiment, source, module = make_source(tmp_path, case)
    assert not hasattr(source, "close")
    monkeypatch.setattr(module, "admit_source", lambda *a, **kw: source)
    failure = KeyboardInterrupt("cancel metadata bind")

    def cancel(*args, **kwargs):
        raise failure

    monkeypatch.setattr(type(source), "runtime_contract", cancel)
    monkeypatch.setattr(memory, "admit_memory", forbidden)
    with pytest.raises(KeyboardInterrupt) as error:
        memory.benchmark_memory(experiment)
    assert error.value is failure


@pytest.mark.parametrize("case", CASES)
def test_native_byte_identity_not_json_reserialization(tmp_path, monkeypatch, case):
    experiment, source, module = make_source(tmp_path, case)
    # Equal JSON documents, unequal native bytes: never normalize identity.
    changed = source.runtime_contract_bytes + b"\n"
    assert json.loads(changed) == json.loads(source.runtime_contract_bytes)
    source = replace(source, runtime_contract_bytes=changed)
    monkeypatch.setattr(module, "admit_source", lambda *a, **kw: source)
    monkeypatch.setattr(memory, "admit_memory", forbidden)
    with pytest.raises(memory.BenchmarkMemoryError, match="source-bound memory envelope") as error:
        memory.benchmark_memory(experiment)
    assert isinstance(error.value.__cause__, ValueError)


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("cache", [{"cache_bytes": 1}, {"cache_bound_tokens": 1}])
def test_bound_path_preserves_existing_cache_errors(tmp_path, monkeypatch, case, cache):
    experiment, source, module = make_source(tmp_path, case)
    monkeypatch.setattr(module, "admit_source", lambda *a, **kw: source)
    monkeypatch.setattr(memory, "admit_memory", forbidden)
    message = ("GKM requires a bounded full-KV response without cache reuse"
               if case.startswith("gkm-") else
               "replicated response does not admit client cache reuse")
    with pytest.raises(memory.BenchmarkMemoryError, match=message):
        memory.benchmark_memory(experiment, **cache)


@pytest.mark.parametrize("case", CASES[-4:])
@pytest.mark.parametrize("field", ["numeric_profile", "execution", "peer_stream_workspace_bytes_per_worker"])
def test_puma_validates_actual_experiment_policy_before_consuming_price(tmp_path, monkeypatch, case, field):
    experiment, source, _ = make_source(tmp_path, case)
    contract = json.loads(source.runtime_contract_bytes)
    contract[field] = ({"range_policy": "unchecked"} if field == "execution" else
                       "foreign-profile" if field == "numeric_profile" else
                       contract.get(field, 0) + 1)
    # If validation is moved after arithmetic, this deliberately cannot price.
    contract["execution_payload_estimate_bytes"] = "must validate before arithmetic"
    seen = []
    validate = replicated_source.validate_execution

    def recorded(actual, value):
        assert actual is experiment
        seen.append(True)
        return validate(actual, value)

    monkeypatch.setattr(replicated_source, "validate_execution", recorded)
    with pytest.raises(memory.BenchmarkMemoryError, match="replicated source exceeds native admission") as error:
        memory._replicated_memory(source.config, experiment.pipeline, 8, 3, 0, None,
            checkpoint_bytes(source), runtime_contract=json.dumps(contract).encode(), experiment=experiment)
    assert seen == [True]
    assert isinstance(error.value.__cause__, ValueError)


@pytest.mark.parametrize("as_pipeline", [False, True])
def test_legacy_model_pipeline_owner_without_sdk_bytes_keeps_unbound_path(tmp_path, monkeypatch,
                                                                       as_pipeline):
    write_source_fixture(tmp_path / "source")
    model = Model.path(str(tmp_path / "source"))
    pipeline = OpenWeightVerifiableCpu(model)
    source = delegated_source.admit_source(model, input_rows=3, decode_steps=2)
    assert source.experiment_bytes is source.runtime_contract_bytes is None
    expected = memory.estimate_memory(source.config, pipeline, max_input_tokens=3,
        max_output_tokens=3, checkpoint_bytes=checkpoint_bytes(source))
    decoder = source.contract
    expected["gkm_admission"] = {
        "checkpoint_digest": source.checkpoint_digest, "source_lock_digest": source.source_lock_digest,
        "tokenizer_digest": source.tokenizer_digest, "decoder_contract_digest": source.contract_digest,
        "runtime_contract_digest": None, "numeric_profile": decoder["numeric_profile"],
        "local_numeric_profile": decoder["local_numeric_profile"], "max_input_tokens": 3,
        "max_output_tokens": 3, "execution_payload_estimate_bytes": source.execution_payload_estimate_bytes,
    }
    monkeypatch.setattr(memory, "_bound_envelope_contract", forbidden)
    seen = []
    admit = delegated_source.admit_source

    def record(*args, **kwargs):
        result = admit(*args, **kwargs)
        seen.append(result)
        return result

    monkeypatch.setattr(delegated_source, "admit_source", record)
    if not as_pipeline:
        # Existing Model call selects default profile internally; isolate only
        # that profile choice so this exercises real Model coercion/source flow.
        monkeypatch.setattr("pllm.profiles.MaskedLinearCpu", lambda actual: pipeline)
    report = memory.benchmark_memory(pipeline if as_pipeline else model,
        max_input_tokens=3, max_output_tokens=3, backend="native")
    assert len(seen) == 1
    assert seen[0].experiment_bytes is seen[0].runtime_contract_bytes is None
    assert report == memory.admit_memory(expected, HOST, backend="native")
