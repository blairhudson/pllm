"""Explicit public-weight adaptation: private feedback, storage and ownership."""

import json
import math
import os
from pathlib import Path
import threading

import numpy as np
import psutil
import pytest

from pllm import Deployment, ExecutionBudget, Model, OpenAI, _native
from pllm.configuration import Experiment
from pllm.protocols import PumaThreePartyDecoder
from pllm.profiles import PumaThreePartyCpu, resolve_runtime_composition
from pllm.runtime import delegated_source as source_io
from pllm.runtime.replicated_source import admit_source

from test_replicated_sdk import active_sessions, experiment_fixture, native_execution


def test_public_mode_is_explicit_serializable_and_has_separate_numeric_identity(tmp_path):
    private = experiment_fixture(tmp_path)
    public = private.with_params(pipeline__components__linear__weights="public")
    assert not private.pipeline.components["linear"].params
    assert public.pipeline.components["linear"].params["weights"] == "public"
    assert private.configuration_digest() != public.configuration_digest()
    assert resolve_runtime_composition(private.pipeline).numeric_profile != resolve_runtime_composition(public.pipeline).numeric_profile
    assert Experiment.from_spec(public.to_spec()).configuration_digest() == public.configuration_digest()
    for invalid in ("secret", True, None):
        with pytest.raises((TypeError, ValueError)):
            PumaThreePartyDecoder(weights=invalid)


@pytest.mark.parametrize("family", ["qwen2", "qwen3", "llama"])
def test_public_source_keeps_native_feedback_private(tmp_path, family):
    experiment = experiment_fixture(tmp_path, family=family, weights="public", active=True)
    source = admit_source(experiment)
    assert source.contract["weights"] == "public"
    # Independent sign-margin oracle from the active-body fixture, with the
    # same full-prefix recomputation and fresh party-local PRSS each request.
    assert native_execution(source, experiment, (2, 2, 3, 3), 2) == (4, 4)
    assert native_execution(source, experiment, (2, 2, 4, 3), 2) == (1, 1)


def test_public_weights_are_local_files_and_only_tokens_get_secret_share_frames(tmp_path):
    experiment = experiment_fixture(tmp_path, weights="public")
    source = admit_source(experiment)
    publisher, manifest = source.publisher(experiment, (3, 4), 2, lambda: False)
    worker = None
    before = psutil.Process().num_fds()
    try:
        value = json.loads(manifest)
        assert value["weights"] == "public"
        key, offset, frames = publisher.next_chunk()
        assert (key, offset) == ("input.tokens", 0)
        assert [len(frame) for frame in frames] == [32, 32, 32]
        assert publisher.next_chunk() is None
        worker = _native.ReplicatedWorker(manifest, 0)
        source.populate_public_worker(worker, lambda: False)
        assert psutil.Process().num_fds() > before
    finally:
        if worker is not None:
            worker.close()
        publisher.close()
    assert psutil.Process().num_fds() == before


def test_public_file_import_rejects_value_substitution_and_burns_owner(tmp_path, monkeypatch):
    experiment = experiment_fixture(tmp_path, weights="public")
    source = admit_source(experiment)
    publisher, manifest = source.publisher(experiment, (3, 4), 2, lambda: False)
    worker = _native.ReplicatedWorker(manifest, 0)
    before = psutil.Process().num_fds()
    original = source_io._source_float_frame
    changed = []

    def corrupt(*args):
        data = original(*args)
        if not changed:
            array = np.frombuffer(data, dtype="<f4").copy()
            array[0] += 0.125
            changed.append(True)
            return array.tobytes()
        return data

    monkeypatch.setattr(source_io, "_source_float_frame", corrupt)
    try:
        with pytest.raises((ValueError, source_io.DelegatedSourceError)):
            source.populate_public_worker(worker, lambda: False)
        assert changed
        with pytest.raises(ValueError, match="retired"):
            worker.call("offer", b"", 0)
    finally:
        worker.close()
        publisher.close()
    assert psutil.Process().num_fds() == before


def test_public_matrix_bounds_match_independent_source_rows_and_bind_manifest(tmp_path):
    experiment = experiment_fixture(tmp_path, weights="public", active=True)
    source = admit_source(experiment)
    publisher, manifest = source.publisher(experiment, (3, 4), 2, lambda: False)
    try:
        value = json.loads(manifest)
        encoded = {}

        def collect(key, offset, frame):
            # Independent source float32 multiplication and truncation to Q18.
            words = (np.frombuffer(frame, dtype="<f4") * np.float32(1 << 18)).astype(np.int64)
            target = encoded.setdefault(key, [])
            assert len(target) == offset
            target.extend(map(int, words))

        source._append_source(sorted(source.source_shapes.items()), collect, lambda: False)
        for key, descriptor in value["tensors"].items():
            shape = descriptor["shape"]
            if len(shape) == 2:
                rows = np.asarray(encoded[key], dtype=np.int64).reshape(shape)
                assert descriptor["row_l1"] == max(sum(abs(int(v)) for v in row) for row in rows)
                assert descriptor["row_squared_norm"] == max(sum(int(v) ** 2 for v in row) for row in rows)
                assert descriptor["row_l1_squared_sum"] == sum(sum(abs(int(v)) for v in row) ** 2 for row in rows)
            else:
                assert "row_l1" not in descriptor
                assert "row_squared_norm" not in descriptor
                assert "row_l1_squared_sum" not in descriptor
        # Native program commitments reject a changed certificate before a
        # worker can accept checkpoint frames or enter setup.
        value["tensors"]["model.embed_tokens.weight"]["row_l1"] -= 1
        with pytest.raises(ValueError, match="program differs|row bound"):
            _native.ReplicatedWorker(source_io._canonical(value), 0)
    finally:
        publisher.close()


def test_public_sdk_three_workers_complete_two_private_branches(tmp_path):
    experiment = experiment_fixture(tmp_path, weights="public", active=True)
    captured = []
    with OpenAI(experiment=experiment, timeout=300) as client:
        client._core._result_hook = captured.append
        for prompt, expected in [("hello hello", (4, 4)), ("world hello", (1, 1))]:
            response = client.responses.create(input=prompt, max_output_tokens=2)
            assert response.usage.output_tokens == 2
            assert captured[-1].selected == expected
            assert active_sessions(client._core) == [0, 0, 0]
        audit = client._core.audit.to_dict()
        assert audit["plaintext_prompt_bytes_sent"] == audit["plaintext_token_ids_sent"] == 0


@pytest.mark.parametrize("family", ["qwen2", "qwen3", "llama"])
def test_public_guarded_softmax_accepts_wide_scores_and_keeps_private_feedback(tmp_path, family):
    from safetensors.numpy import load_file, save_file

    experiment = experiment_fixture(tmp_path, family=family, weights="public")
    path = Path(experiment.pipeline.model.source) / "model.safetensors"
    values = load_file(path)
    for key, value in values.items():
        if key.endswith(("q_proj.weight", "k_proj.weight", "q_norm.weight", "k_norm.weight")):
            value.fill(64)
    save_file(values, path, metadata={"format": "pt"})
    source = admit_source(experiment)
    # Nonzero Q/K force wide score bounds and clipped exponentials. V/O/MLP
    # remain zero, so the independent signed-head feedback oracle is unchanged.
    assert native_execution(source, experiment, (3, 4), 2) == (2, 1)
    if family == "qwen2":
        captured = []
        with OpenAI(experiment=experiment, timeout=300) as client:
            client._core._result_hook = captured.append
            response = client.responses.create(input="hello", max_output_tokens=2)
            assert response.usage.output_tokens == 2
            assert captured[-1].selected == (2, 1)
            assert active_sessions(client._core) == [0, 0, 0]
            audit = client._core.audit.to_dict()
            assert audit["plaintext_prompt_bytes_sent"] == audit["plaintext_token_ids_sent"] == 0


@pytest.mark.parametrize("family", ["qwen2", "qwen3", "llama"])
@pytest.mark.parametrize("concentrated", [False, True])
def test_public_softmax_mass_admits_attention_residual_and_private_feedback(tmp_path, family, concentrated):
    from safetensors.numpy import load_file, save_file

    experiment = experiment_fixture(tmp_path, family=family, weights="public")
    path = Path(experiment.pipeline.model.source) / "model.safetensors"
    values = load_file(path)
    for key, value in values.items():
        if concentrated and key.endswith("o_proj.weight"):
            value[:] = np.diag([28, 0]).astype(np.float32)
        elif key.endswith(("v_proj.weight", "o_proj.weight")):
            value[:] = 8 * np.eye(2, dtype=np.float32)
    save_file(values, path, metadata={"format": "pt"})
    source = admit_source(experiment)
    # Q=K=0 gives uniform attention. V=O=8I amplifies the signed prefix
    # mean; the independent head oracle has a wide margin under every rounding.
    # Uncorrelated softmax boxes reject the following RMSNorm on the old wheel.
    # Concentrated output additionally needs the certified complete-row energy:
    # repeating its largest coordinate's bound across both features rejects.
    assert native_execution(source, experiment, (3, 4), 2) == (2, 2)
    assert native_execution(source, experiment, (2, 2), 2) == (1, 1)
    if family == "qwen2":
        captured = []
        with OpenAI(experiment=experiment, timeout=300) as client:
            client._core._result_hook = captured.append
            # Rendered user: contributes two positive unknown embeddings.
            response = client.responses.create(input="hello", max_output_tokens=2)
            assert response.usage.output_tokens == 2
            assert captured[-1].selected == (1, 2)
            assert active_sessions(client._core) == [0, 0, 0]
            audit = client._core.audit.to_dict()
            assert audit["plaintext_prompt_bytes_sent"] == audit["plaintext_token_ids_sent"] == 0


@pytest.mark.parametrize("family", ["qwen2", "qwen3", "llama"])
def test_public_guarded_silu_keeps_identity_tails_and_private_feedback(tmp_path, family):
    from safetensors.numpy import load_file, save_file

    experiment = experiment_fixture(tmp_path, family=family, weights="public")
    path = Path(experiment.pipeline.model.source) / "model.safetensors"
    values = load_file(path)
    for key, value in values.items():
        if key.endswith("gate_proj.weight"):
            value[:] = 64 * np.eye(2, dtype=np.float32)
        elif key.endswith(("up_proj.weight", "down_proj.weight")):
            value[:] = np.eye(2, dtype=np.float32) / 128
    save_file(values, path, metadata={"format": "pt"})
    source = admit_source(experiment)
    # Zero attention, signed embeddings, diagonal positive MLP. Negative gates
    # select exact zero; positive gates retain identity, reinforcing the input
    # sign. The independent signed-head oracle therefore alternates feedback.
    # Eager evaluation on this source rejects the unused sixth-degree tails.
    assert native_execution(source, experiment, (3, 4), 2) == (2, 1)
    if family == "qwen2":
        captured = []
        with OpenAI(experiment=experiment, timeout=300) as client:
            client._core._result_hook = captured.append
            response = client.responses.create(input="hello", max_output_tokens=2)
            assert response.usage.output_tokens == 2
            assert captured[-1].selected == (2, 1)
            assert active_sessions(client._core) == [0, 0, 0]
            audit = client._core.audit.to_dict()
            assert audit["plaintext_prompt_bytes_sent"] == audit["plaintext_token_ids_sent"] == 0


def test_public_import_cancellation_closes_partial_files(tmp_path):
    experiment = experiment_fixture(tmp_path, weights="public")
    source = admit_source(experiment)
    publisher, manifest = source.publisher(experiment, (3, 4), 2, lambda: False)
    worker = _native.ReplicatedWorker(manifest, 0)
    before = psutil.Process().num_fds()
    calls = 0
    opened = False

    def cancel():
        nonlocal calls, opened
        calls += 1
        opened |= psutil.Process().num_fds() > before
        return calls >= 3

    try:
        with pytest.raises(ValueError, match="cancelled"):
            source.populate_public_worker(worker, cancel)
        assert opened
        assert psutil.Process().num_fds() == before
        with pytest.raises(ValueError, match="retired"):
            worker.call("offer", b"", 0)
    finally:
        worker.close()
        publisher.close()


def test_public_wide_mlp_runs_compact_loops_through_installed_binding(tmp_path):
    experiment = experiment_fixture(tmp_path, weights="public", intermediate_size=4864)
    source = admit_source(experiment)
    # Zero body projections preserve signed token embeddings. All 4,864 lanes
    # and complete-sum truncations still execute.
    assert native_execution(source, experiment, (3,), 2) == (2, 1)


def test_pinned_public_qwen_prices_shape_work_before_checkpoint_import():
    config = (Path(__file__).parent / "fixtures/qwen25_0_5b_config.json").read_bytes()
    public = Experiment(
        "puma-metadata-gate",
        PumaThreePartyCpu(Model.hf("Qwen/Qwen2.5-0.5B-Instruct",
            revision="7ae557604adf67be50417f59c2c2f167def9a775",
            local_files_only=True), weights="public"),
        Deployment.local(root="local://puma-metadata-gate"),
        ExecutionBudget(requests=1, max_input_tokens=4, max_new_tokens=2),
    )
    # Placeholder source commitments: this test admits metadata, not checkpoint values.
    source_lock = checkpoint = "00" * 32
    contract = json.loads(_native.replicated_source_contract(public.canonical_bytes(),
        config, 4, 2, source_lock, checkpoint))
    assert contract["weights"] == "public"
    assert contract["vocabulary"] == 151936
    assert contract["selection_prefix_rows"] == [4, 5]
    assert contract["public_weight_snapshot_bytes_per_worker"] == 2 * contract["source_float32_bytes"]
    larger = public.with_params(budget__max_input_tokens=16)
    with pytest.raises(ValueError, match="public lookup operation count"):
        _native.replicated_source_contract(larger.canonical_bytes(),
            config, 16, 2, source_lock, checkpoint)
    private = public.with_params(pipeline__linear__weights="private")
    with pytest.raises(ValueError, match="source expansion budget"):
        _native.replicated_source_contract(private.canonical_bytes(),
            config, 4, 2, source_lock, checkpoint)


@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get("PLLM_RUN_REAL_QWEN25") != "1",
    reason="opt in to the pinned cached real Qwen2.5 checkpoint",
)
def test_pinned_public_qwen_checked_trie_source_publication():
    """Publish the installed SDK source only; never start roles or decode a prompt."""
    import pllm
    from pllm.runtime.benchmark_memory import (
        BenchmarkMemoryError,
        MemoryWatchdog,
        benchmark_memory,
        require_admission,
    )
    from pllm.runtime.replicated_protocol import execution_binding

    # pyproject's default pythonpath otherwise shadows the rebuilt wheel. Do not
    # silently exercise mixed Python/native versions or skip an outdated wheel.
    checkout_python = Path(__file__).resolve().parents[1] / "python"
    for module in (pllm, _native, source_io):
        assert not Path(module.__file__).resolve().is_relative_to(checkout_python), (
            "Run with the rebuilt installed wheel, PYTHONPATH unset and -o pythonpath="
        )

    experiment = Experiment(
        "puma-pinned-public-source-publication",
        PumaThreePartyCpu(
            Model.hf(
                "Qwen/Qwen2.5-0.5B-Instruct",
                revision="7ae557604adf67be50417f59c2c2f167def9a775",
                local_files_only=True,
            ),
            weights="public",
            range_policy="checked",
            max_abs_activation=256,
            lookup="trie",
            selection="tiled",
            peer_transport="streams",
        ),
        Deployment.local(root="local://puma-pinned-public-source-publication"),
        ExecutionBudget(requests=1, max_input_tokens=4, max_new_tokens=2),
    )
    # Keep the ordinary whole-topology estimate, allocation margin, pressure
    # floor and swap guard, even though this test allocates no workers.
    report = benchmark_memory(experiment, backend="native")
    require_admission(report)
    cancelled = threading.Event()
    monitor = MemoryWatchdog(report, lambda _reason: cancelled.set())
    publisher = None

    def check_memory():
        if monitor.error is not None:
            raise BenchmarkMemoryError(monitor.error)
        monitor.check()

    try:
        monitor.start()
        source = admit_source(experiment)
        check_memory()
        assert source.source_lock_digest == "383f5ed6947ba56ceed6083a9503d6302f09559e1e57ee614409900b36084c29"
        assert source.checkpoint_digest == "f69322057253c4fb853c10e73da5d84111262e0fb1edba33933b9e5b8caf59fd"
        assert source.contract["weights"] == "public"
        assert source.contract["selection_prefix_rows"] == [4, 5]
        assert source.contract["execution"] == execution_binding(experiment)
        try:
            # Fixed public synthetic IDs; no tokenizer or private prompt decode.
            publisher, manifest = source.publisher(
                experiment, (0, 0, 0, 0), 2, cancelled.is_set,
            )
        except ValueError as error:
            check_memory()
            # Preserve the native public context/stage/ledger verbatim, without
            # persisting exceptions or guessing which numeric contract failed.
            pytest.fail(f"Pinned public source publication rejected: {error}", pytrace=False)
        check_memory()
        value = json.loads(manifest)
        assert value["weights"] == "public"
        assert value["execution"] == execution_binding(experiment)
        assert value.get("norms")
    finally:
        if publisher is not None:
            publisher.close()
        monitor.close()


def large_public_source_fixture(path, dtype="F32", change=None):
    """Native-admitted geometry crossing both old private-header limits.

    Sparse zero payloads avoid allocating full tensors. Three table sentinels
    exercise conversion at a frame boundary and the final partial frame.
    """
    experiment = experiment_fixture(path, weights="public")
    root = Path(experiment.pipeline.model.source)
    config = json.loads((root / "config.json").read_text())
    config.update(hidden_size=128, head_dim=128, intermediate_size=128, vocab_size=8193)
    (root / "config.json").write_text(json.dumps(config))
    tokenizer = json.loads((root / "tokenizer.json").read_text())
    tokenizer["model"]["vocab"].update({f"word{i}": i for i in range(5, 8193)})
    (root / "tokenizer.json").write_text(json.dumps(tokenizer))
    contract = json.loads(_native.replicated_source_contract(
        experiment.canonical_bytes(), source_io._encode_config(config), 8, 2, "0" * 64, "0" * 64,
    ))
    shapes = dict(contract["source_shapes"])
    if change == "shape":
        shapes["model.norm.weight"] = [64, 2]  # Same bytes; wrong declared layout.
    elif change == "extra":
        shapes["undeclared.weight"] = [1]
    elif change == "missing":
        shapes.pop("model.norm.weight")
    itemsize = source_io._DTYPES[dtype].itemsize
    header, offset = {}, 0
    for name, shape in sorted(shapes.items()):
        size = math.prod(shape) * itemsize
        header[name] = {"shape": shape, "dtype": dtype, "data_offsets": [offset, offset + size]}
        offset += size
    table = "model.embed_tokens.weight"
    if change == "dtype":
        header[table]["dtype"] = "I32"
    elif change == "range":
        header[table]["data_offsets"][0] += itemsize
    elif change == "dimension":
        header[table]["shape"] = [8194, 128]
    encoded = source_io._canonical(header)
    encoded += b" " * (-len(encoded) % 8)
    with (root / "model.safetensors").open("wb") as stream:
        stream.write(len(encoded).to_bytes(8, "little") + encoded)
        stream.truncate(8 + len(encoded) + offset)
        begin = header[table]["data_offsets"][0]
        for index, value in ((0, -0.0), (4095, -1.5), (8193 * 128 - 1, 1.25)):
            stream.seek(8 + len(encoded) + begin + index * itemsize)
            stream.write(np.array([value], dtype=source_io._DTYPES[dtype]).tobytes())
    return experiment, contract, 8 + len(encoded)


@pytest.mark.parametrize("dtype", ["F32", "F16", "BF16"])
def test_public_source_uses_admitted_header_geometry_and_bounded_frames(tmp_path, monkeypatch, dtype):
    from test_delegated_source import forbid_values

    experiment, contract, data_offset = large_public_source_fixture(tmp_path, dtype)
    forbid_values(monkeypatch)
    original = source_io._read_range

    def headers_only(path, identity, offset, size, **kwargs):
        if path.name.endswith(".safetensors"):
            assert offset + size <= data_offset
        return original(path, identity, offset, size, **kwargs)

    monkeypatch.setattr(source_io, "_read_range", headers_only)
    source = admit_source(experiment)
    assert source.source_shapes == {k: tuple(v) for k, v in contract["source_shapes"].items()}
    monkeypatch.setattr(source_io, "_read_range", original)
    received, frames, tails = {}, [], []

    def append(key, offset, frame):
        assert offset == received.get(key, 0)
        assert 0 < len(frame) <= 16384 and len(frame) % 4 == 0
        words = len(frame) // 4
        expected = np.zeros(words, dtype="<u4")
        if key == "model.embed_tokens.weight":
            for index, bits in ((0, 0x80000000), (4095, 0xBFC00000), (8193 * 128 - 1, 0x3FA00000)):
                if offset <= index < offset + words:
                    expected[index - offset] = bits
        np.testing.assert_array_equal(np.frombuffer(frame, dtype="<u4"), expected)
        received[key] = offset + words
        frames.append(len(frame))
        if words < 4096:
            tails.append((key, words))

    source._append_source(sorted(source.source_shapes.items()), append, lambda: False)
    assert received == {k: math.prod(v) for k, v in source.source_shapes.items()}
    assert sum(frames) == source.source_float32_bytes
    assert max(frames) == 16384 and ("model.embed_tokens.weight", 128) in tails


@pytest.mark.parametrize("change", ["shape", "dimension", "dtype", "range", "extra", "missing"])
def test_large_public_headers_reject_substitution_before_hashing_or_values(tmp_path, monkeypatch, change):
    from test_delegated_source import forbid_values, forbidden

    experiment, _, _ = large_public_source_fixture(tmp_path, change=change)
    forbid_values(monkeypatch)
    monkeypatch.setattr(source_io, "_locked", forbidden)
    monkeypatch.setattr(source_io, "_source_float_frame", forbidden)
    with pytest.raises(ValueError, match="shape|dtype|range|cardinality"):
        admit_source(experiment)
