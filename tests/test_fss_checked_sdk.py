"""Checked-range acceptance across actual SDK role processes.

The source has an independent two-dimensional integer oracle. Its first selected
token reaches an invalid embedding only on private feedback. This is generated
functionality and rejection coverage, not trained-model quality.
"""

import json
import struct

import pytest

from pllm import ExecutionBudget, Experiment, Model, OpenAI, _native
from pllm.profiles import FuseFssSourceCpu
from pllm.runtime import fss_transport
from test_fss_sdk import assert_retired


CHECKED = "fusefss.dense-rope-checked-ranges-ring50-q12.v1"
READY = all(callable(getattr(_native, name, None)) for name in (
    "FssSourceBuilder", "FssClientRequest", "FssRoleRequest", "fss_source_contract",
))
pytestmark = pytest.mark.skipif(not READY, reason="requires installed native FSS bridge")


def checked_experiment(path):
    from examples.papers.fusefss_smoke import dense_experiment
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel

    base = dense_experiment(path, requests=4)
    root = path / "source"
    config = json.loads((root / "config.json").read_bytes())
    config["tie_word_embeddings"] = False
    (root / "config.json").write_text(json.dumps(config))
    Tokenizer(WordLevel({"user: safe": 0, "user: hello": 1, "<unk>": 2},
                        unk_token="<unk>")).save(str(root / "tokenizer.json"))
    data = (root / "model.safetensors").read_bytes()
    length, = struct.unpack_from("<Q", data)
    original = json.loads(data[8:8 + length])
    tensors = {}
    for name, spec in original.items():
        count = (spec["data_offsets"][1] - spec["data_offsets"][0]) // 4
        norm = "layernorm.weight" in name or name == "model.norm.weight"
        tensors[name] = (spec["shape"], [4096 if norm else 0] * count)
    # B=floor_sqrt((2^49-2)/2)=2^24-1. All values are exact F32 encodings.
    tensors["model.embed_tokens.weight"] = ([3, 2], [0, 0, -256, 1024, 1 << 24, 1 << 24])
    tensors["lm_head.weight"] = ([3, 2], [0, 0, 0, 0, -4096, 4096])
    header, payload = {}, bytearray()
    for name, (shape, codes) in sorted(tensors.items()):
        raw = struct.pack("<" + "f" * len(codes), *(value / 4096 for value in codes))
        header[name] = {"dtype": "F32", "shape": shape,
                        "data_offsets": [len(payload), len(payload) + len(raw)]}
        payload.extend(raw)
    encoded = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    (root / "model.safetensors").write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)
    return Experiment("generated-fss-checked-sdk",
        FuseFssSourceCpu(Model.path(str(root)), numeric_profile=CHECKED, capacity="decoder"),
        base.deployment, ExecutionBudget(requests=4, max_input_tokens=2, max_new_tokens=2))


@pytest.mark.slow
@pytest.mark.parametrize(("material_delivery", "online_schedule"), [
    ("single", "serial"), ("batch-v1", "serial"), ("batch-v1", "wavefront-v1"),
    ("batch-v1", "tensor-v1"),
])
def test_checked_feedback_rejects_all_tokens_and_fresh_request_recovers(
        tmp_path, monkeypatch, material_delivery, online_schedule):
    captured, receipts = [], []
    original = fss_transport._receipt

    def observe(raw, role, **options):
        result = original(raw, role, **options)
        if role in ("worker_a", "worker_b"):
            storage = result[2]
            receipts.append((role, storage["records_written"], storage["records_read"]))
        return result

    monkeypatch.setattr(fss_transport, "_receipt", observe)
    experiment = checked_experiment(tmp_path)
    if material_delivery != "single":
        experiment = experiment.with_params(pipeline__linear__material_delivery=material_delivery)
    if online_schedule != "serial":
        experiment = experiment.with_params(pipeline__linear__online_schedule=online_schedule)
    with OpenAI(experiment=experiment, timeout=300) as client:
        topology = client._owned_topology
        client._core._result_hook = captured.append
        # Complete body projections are zero, leaving the original embedding.
        # Its normalized first coordinate is negative, second positive, so the
        # [-1,+1] head row uniquely wins. No native helper supplies this oracle.
        response = client.responses.create(input="hello", max_output_tokens=1, store=False)
        assert response.usage.input_tokens == 1
        assert response.usage.output_tokens == 1
        assert captured[-1].selected == (2,)
        assert_retired(client._core)
        captured.clear()
        receipts.clear()

        # Feedback introduces embedding 2 only in the second selection. The
        # first, valid, nonzero selection must also be suppressed at final reject.
        with pytest.raises(_native.NumericDomainRejected):
            with client.responses.create(input="hello", max_output_tokens=2,
                                         stream=True, store=False) as stream:
                for _ in stream:
                    pytest.fail("rejected request released an application event")
        assert not captured and not client._core.records
        assert_retired(client._core)
        rejected_counts = sorted(receipts)
        assert len(rejected_counts) == 2
        assert all(written == read and read > 0 for _, written, read in rejected_counts)
        receipts.clear()

        # Token zero has zero embedding. Complete zero head ties select zero;
        # EOS metadata must not truncate D9's fixed two selections.
        response = client.responses.create(input="safe", max_output_tokens=2, store=False)
        assert captured[-1].selected == (0, 0)
        assert response.usage.input_tokens == 1
        assert response.usage.output_tokens == 2
        assert sorted(receipts) == rejected_counts  # Same whole public schedule consumed.
        assert_retired(client._core)
    assert all(not status.running for status in topology.statuses)
