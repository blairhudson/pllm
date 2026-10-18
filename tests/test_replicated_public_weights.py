"""Explicit public-weight adaptation: private feedback, storage and ownership."""

import json

import numpy as np
import psutil
import pytest

from pllm import OpenAI, _native
from pllm.configuration import Experiment
from pllm.protocols import PumaThreePartyDecoder
from pllm.profiles import resolve_runtime_composition
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


def test_pinned_public_qwen_rejects_known_work_before_checkpoint_import():
    from pathlib import Path

    replay = json.loads((Path(__file__).resolve().parents[1] /
        "paper/artifacts/puma-sdk/local-v4/qwen-metadata-replay.json").read_text())
    case = replay["modes"]["public"]
    with pytest.raises(ValueError, match="public lookup work charge"):
        _native.replicated_source_contract(case["experiment"].encode(),
            replay["normalized_config"].encode(), replay["rows"], replay["outputs"],
            replay["source_lock"], replay["checkpoint"])
