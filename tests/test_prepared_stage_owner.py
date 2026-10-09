"""One-use native ownership, independent boundary arithmetic and frame admission."""
import hashlib
import msgpack
import numpy as np
import pytest

from pllm import _native


def fixture(bits=32, precision=8, rows=17, chunk=4):
    width, outputs = 13, 9
    widths = bytes([1, 7, 8, 11, 15, 16, min(23, bits), min(24, bits), bits])
    rng = np.random.default_rng(819)
    values = rng.normal(size=(rows, width)).astype("<f4")
    values[0] = 0
    scale = np.where(np.abs(values).max(axis=1) > 0,
        np.abs(values).max(axis=1) / ((1 << (precision - 1)) - 1), 1).astype("<f4")
    codes = np.rint(values / scale[:, None]).astype(np.int64)
    weights = rng.uniform(.01, 1, outputs).astype("<f4")
    bias = rng.normal(size=outputs).astype("<f4")
    seed = bytes(range(32))
    def masks():
        return _native.PreparedMaskRows(seed, b"input", b"output", rows, width, outputs, bits)
    r, s = masks().take(0, rows)
    r = np.frombuffer(r, "<u4").reshape(rows, width)
    s = np.frombuffer(s, "<u4").reshape(rows, outputs)
    signed = np.empty((rows, outputs), np.int64)
    for col, w in enumerate(widths):
        signed[:, col] = rng.integers(-(1 << (w - 1)), 1 << (w - 1), rows)
    expected = signed.astype("<f4") * scale[:, None] * weights + bias
    tickets = [f"{i:032x}" for i in range((rows + chunk - 1) // chunk)]
    layout = _native.PreparedStageLayout("stage", width, widths, bits, bits, precision,
                                         rows, weights.tobytes(), bias.tobytes())
    replies, packed_inputs = [], []
    for i, start in enumerate(range(0, rows, chunk)):
        end = min(rows, start + chunk)
        packed = 0
        for j, value in enumerate(((codes[start:end] - r[start:end]) & ((1 << bits) - 1)).flat):
            packed |= int(value) << (bits * j)
        packed_inputs.append(packed.to_bytes(((end - start) * width * bits + 7) // 8, "little"))
        offset, packed = 0, 0
        for value, mask, w in zip(signed[start:end].flat, s[start:end].flat, list(widths) * (end - start)):
            packed |= ((int(value) - int(mask)) & ((1 << w) - 1)) << offset
            offset += w
        replies.append(b"PLLMPR01" + msgpack.packb([tickets[i], "stage", end - start, bits,
            hashlib.sha256(widths).digest(), i + 1, packed.to_bytes((offset + 7) // 8, "little")], use_bin_type=True))
    return layout, masks(), values, tickets, replies, packed_inputs, expected


@pytest.mark.parametrize("bits", [16, 24, 32])
@pytest.mark.parametrize("precision", [4, 8])
@pytest.mark.parametrize("rows,chunk", [(1, 1), (17, 4)])
def test_native_stage_exact_oracle_and_owned_readonly_result(bits, precision, rows, chunk):
    layout, masks, values, tickets, replies, expected_inputs, expected = fixture(bits, precision, rows, chunk)
    query, requests = layout.prepare(masks, 0, rows, chunk, tickets, values.tobytes())
    assert requests == expected_inputs
    for reply in replies:
        query.receive(reply)
    output, ns = query.finish()
    actual = np.frombuffer(output, "<f4").reshape(expected.shape)
    assert actual.tobytes() == expected.tobytes()
    assert ns == sum(range(1, len(replies) + 1))
    assert query.retained_bytes == 0 and not actual.flags.writeable
    del output, query, layout, masks
    assert actual.tobytes() == expected.tobytes()
    with pytest.raises(ValueError):
        actual.setflags(write=True)


@pytest.mark.parametrize("failure", ["type", "truncated", "trailing", "namespace", "ticket", "stage",
    "rows", "bool", "bits", "digest", "duration", "padding", "replay", "reordered", "incomplete", "cancel"])
def test_native_stage_burns_all_state_on_invalid_frames(failure):
    layout, masks, values, tickets, replies, _, _ = fixture()
    query, _ = layout.prepare(masks, 0, 17, 4, tickets, values.tobytes())
    frame = replies[0]
    fields = msgpack.unpackb(frame[8:])
    if failure in {"ticket", "stage", "rows", "bool", "bits", "digest", "duration", "padding"}:
        index, value = {"ticket": (0, "f" * 32), "stage": (1, "other"), "rows": (2, 1),
            "bool": (2, True), "bits": (3, 24), "digest": (4, b"x" * 32),
            "duration": (5, 1 << 63), "padding": (6, fields[6] + b"\0")}[failure]
        fields[index] = value
        frame = b"PLLMPR01" + msgpack.packb(fields, use_bin_type=True)
    elif failure == "type":
        frame = None
    elif failure == "truncated":
        frame = frame[:-1]
    elif failure == "trailing":
        frame += b"\0"
    elif failure == "namespace":
        frame = b"PLLMOR01" + frame[8:]
    elif failure == "replay":
        query.receive(frame)
    elif failure == "reordered":
        frame = replies[1]
    elif failure == "cancel":
        query.cancel()
    with pytest.raises((ValueError, TypeError)):
        query.finish() if failure == "incomplete" else query.receive(frame)
    assert query.retained_bytes == 0
    with pytest.raises(ValueError):
        query.receive(replies[0])
    with pytest.raises(ValueError):
        masks.take(0, 1)


def test_native_stage_truncations_and_geometry_admit_before_claim():
    layout, _, values, tickets, replies, _, _ = fixture(rows=1, chunk=1)
    for length in range(len(replies[0])):
        masks = _native.PreparedMaskRows(bytes(range(32)), b"input", b"output", 1, 13, 9, 32)
        query, _ = layout.prepare(masks, 0, 1, 1, tickets, values.tobytes())
        with pytest.raises(ValueError):
            query.receive(replies[0][:length])
        assert query.retained_bytes == 0
    masks = _native.PreparedMaskRows(bytes(range(32)), b"input", b"output", 1, 13, 9, 16)
    with pytest.raises(ValueError):
        layout.prepare(masks, 0, 1, 1, tickets, values.tobytes())
    with pytest.raises(ValueError):
        masks.take(0, 1)
    with pytest.raises(ValueError):
        _native.PreparedStageLayout("stage", 1, b"\x20", 32, 32, 8, 4097, b"\0" * 4, None)


@pytest.mark.parametrize("failure", ["malformed", "cancel", "transport"])
def test_native_transport_failure_burns_claimed_and_unused_lease_rows(failure):
    from types import SimpleNamespace
    from test_prepared_mask_stream import request
    from pllm.runtime.prepared_stage import NativePreparedLinear
    from pllm.runtime.transformer_client import PreparedInventory, PreparedStageRows, StageClientStats

    req = request(rows=5, width=2, output=1)
    stage = PreparedStageRows(req)
    inventory = PreparedInventory("inventory", 5, {req.stage_id: stage})
    lease = inventory.reserve(5)
    layout = _native.PreparedStageLayout(req.stage_id, 2, b"\x10", 24, 16, 8, 5,
                                         np.ones(1, "<f4").tobytes(), None)

    def exchange(*_):
        if failure == "malformed":
            return [b"bad"]
        raise KeyboardInterrupt() if failure == "cancel" else OSError("injected")

    remote = SimpleNamespace(stats=StageClientStats(), inventory=lease, inference=exchange,
        request_encoding="stage_packed", prefill_chunk_rows=4,
        stages={req.stage_id: SimpleNamespace(in_features=2, out_features=1,
            input_equalization=None, output_residue_bits=b"\x10")})
    with pytest.raises((ValueError, KeyboardInterrupt, OSError)):
        NativePreparedLinear._execute(remote, layout, req.stage_id, np.ones((2, 2), "<f4"))
    assert lease._closed and inventory.status()["stage_rows_burned"] == 3
    with pytest.raises(ValueError):
        stage.take_masks(2, 1)


@pytest.mark.integration
@pytest.mark.parametrize("family", ["qwen2", "qwen3", "llama"])
def test_compiler_bound_stage_preserves_prefill_decode_logits_and_kv(tmp_path, monkeypatch, family):
    from pllm import Model
    from test_prepared_row_residues import experiment
    from pllm.runtime.model_binding import CompiledRuntimeModel
    from pllm.runtime.prepared_stage import NativePreparedLinear
    from pllm.runtime.semantic_executor import SemanticDecoderRuntime
    from pllm.runtime.servers import build_roles
    from pllm.runtime.tiny_llama import create_tiny_llama_checkpoint
    from pllm.runtime.transformer_client import PreparedRemoteLinear

    source = Model.path(str(create_tiny_llama_checkpoint(tmp_path / "model", model_type=family,
        seed=81, num_hidden_layers=2, with_qkv_bias=family == "qwen2", qk_norm=family == "qwen3")), model_id="residue-test")
    original_runtime = CompiledRuntimeModel.runtime
    original_prefill, original_decode = SemanticDecoderRuntime.prepare_ids, SemanticDecoderRuntime.decode_step
    outputs, observed, selected, timing_bytes = [], [], [], []

    def save(runtime, logits):
        observed.append((logits.tobytes(), [(c.key[:c.length].tobytes(), c.value[:c.length].tobytes()) for c in runtime.caches]))

    def prefill(runtime, *args, **kwargs):
        result = original_prefill(runtime, *args, **kwargs)
        save(runtime, result[1])
        return result

    def decode(runtime, *args, **kwargs):
        result = original_decode(runtime, *args, **kwargs)
        save(runtime, result[0])
        return result

    def bind(compiled, remote, **kwargs):
        assert type(remote) is PreparedRemoteLinear
        exchange = remote.inference

        def measured_exchange(stage, requests):
            replies = exchange(stage, requests)
            for reply in replies:
                assert reply.startswith(b"PLLMPR01")
                fields = msgpack.unpackb(reply[8:], raw=False)
                # Account exactly for the variable-width server duration; all
                # remaining application bytes must still have equal lengths.
                timing_bytes.append(len(msgpack.packb(fields[5])))
            return replies

        remote.inference = measured_exchange
        if not native:
            return original_runtime(compiled, remote, **kwargs)
        stage = next(s.stage_id for s in compiled.stage_bindings if s.role == "mlp_gate_up")
        selected.append(stage)
        return original_runtime(compiled, NativePreparedLinear(compiled, remote, [stage]), **kwargs)

    monkeypatch.setattr(SemanticDecoderRuntime, "prepare_ids", prefill)
    monkeypatch.setattr(SemanticDecoderRuntime, "decode_step", decode)
    for native in (False, True):
        exp = experiment(source, tmp_path / str(native), "row_residues").with_params(
            pipeline__linear__request_encoding="stage_packed", pipeline__linear__prefill_chunk_rows=4)
        with monkeypatch.context() as patch:
            patch.setattr(CompiledRuntimeModel, "runtime", bind)
            with build_roles(exp, engine_threads=1) as topology, topology.client(bundle_cache_dir=tmp_path / f"cache-{native}") as client:
                result = client.responses.create(input="abcde " * 8, temperature=0, max_output_tokens=3)
                outputs.append((result.output_text, observed.copy(), client.privacy_audit.masked_online_upload_bytes,
                                client.privacy_audit.masked_online_download_bytes - sum(timing_bytes)))
                observed.clear()
                timing_bytes.clear()
    assert selected and outputs[0] == outputs[1]
