import copy
from dataclasses import replace
import hashlib
import pickle
import platform
import secrets

import numpy as np
import pytest

from pllm.native import MaskedGEMM
from pllm.preparation import CorrectionBatch, PreparationRequest
from pllm.runtime.preparation_protocol import prepared_mask_rows, seeded_ring_profile
from pllm.runtime.stage_protocol import unpack_residues


def request(weights, rows, *, bound=10000):
    profile = seeded_ring_profile(bound)
    return PreparationRequest(secrets.token_hex(16), secrets.token_hex(16), "test-model", "test-body",
        "test-stage", hashlib.sha256(weights.tobytes()).hexdigest(), rows, weights.shape[1], weights.shape[0],
        8, 8, bound, profile.ring, profile.modulus, profile.wire_bits, secrets.token_bytes(32))


@pytest.mark.parametrize("bound", [12000, 100000, 10000000])
def test_batch_cross_inventory_windows_match_independent_oracle_and_burn(bound):
    weights = np.arange(-13, 14, dtype=np.int8).reshape(3, 9)
    matrix = MaskedGEMM(threads=1).compile(weights)
    requests = [request(weights, rows, bound=bound) for rows in (2, 5, 3)]
    with CorrectionBatch(matrix, window_rows=4) as batch:
        tickets = [batch.add(req) for req in requests]
        assert all(t._request.seed == b"" for t in tickets)
        result = batch.execute()
        assert result["rows"] == 10 and result["windows"] == 3
        assert result["kernel_calls"]["native"] == 3
        for req, ticket in zip(requests, tickets, strict=True):
            cursor = prepared_mask_rows(req)
            try:
                r, s = cursor.take(0, req.rows)
                r = np.frombuffer(r, "<u4").reshape(req.rows, 9)
                s = np.frombuffer(s, "<u4").reshape(req.rows, 3)
                expected = (r.astype(np.int64) @ weights.astype(np.int64).T - s.astype(np.int64)) % req.modulus
                first = unpack_residues(ticket.take_rows(1), (1, 3), req.wire_bits)
                tail = unpack_residues(ticket.take_rows(req.rows - 1), (req.rows - 1, 3), req.wire_bits)
                np.testing.assert_array_equal(np.concatenate((first, tail)), expected)
                assert ticket.state == "consumed" and ticket._file.closed
            finally:
                cursor.cancel()
    assert batch._matrix is None and batch._key == b""
    with pytest.raises(ValueError, match="unavailable"):
        tickets[0].take_rows(1)


@pytest.mark.parametrize("failure", ["replay", "context", "weights", "bound", "capacity", "cancel", "frame", "copy", "read"])
def test_batch_rejection_burns_every_owned_cursor_and_file(monkeypatch, failure):
    weights = np.ones((3, 9), np.int8)
    matrix = MaskedGEMM(threads=1).compile(weights)
    batch = CorrectionBatch(matrix, window_rows=2, max_tickets=2)
    req = request(weights, 3)
    first = batch.add(req)
    if failure == "copy":
        for owner in (first, batch):
            for operation in (copy.copy, copy.deepcopy, pickle.dumps):
                with pytest.raises(TypeError):
                    operation(owner)
        batch.close()
    elif failure == "cancel":
        first.close()
        with pytest.raises(ValueError):
            batch.execute()
    elif failure == "frame":
        accept = batch._accept
        monkeypatch.setattr(batch, "_accept", lambda ticket, offset, rows, payload, tag:
                            accept(ticket, offset + 1, rows, payload, tag))
        with pytest.raises(ValueError, match="attribution"):
            batch.execute()
    elif failure == "read":
        with pytest.raises(ValueError, match="unavailable"):
            first.take_rows(1)
    else:
        other = request(weights, 3)
        if failure == "replay":
            other = req
        elif failure == "context":
            other = replace(other, stage_id="other-stage")
        elif failure == "weights":
            other = replace(other, weight_digest="a" * 64)
        elif failure == "bound":
            other = replace(other, signed_output_bound=1)
        else:
            batch.add(other)
            other = request(weights, 3)
        with pytest.raises(ValueError):
            batch.add(other)
    assert first.state == "burned" and first._file.closed
    assert batch._key == b"" and batch._closed


def test_batch_admission_precedes_mask_allocation(monkeypatch):
    import pllm.runtime.correction_batch as implementation
    weights = np.ones((3, 9), np.int8)
    matrix = MaskedGEMM(threads=1).compile(weights)
    monkeypatch.setattr(implementation, "prepared_mask_rows", lambda _: pytest.fail("allocated masks"))
    with pytest.raises(ValueError, match="workspace"):
        CorrectionBatch(matrix, window_rows=128, workspace_bytes=1)
    with CorrectionBatch(matrix) as batch:
        with pytest.raises(ValueError):
            batch.add(replace(request(weights, 3), rows=True))


@pytest.mark.parametrize("kernel", ["digit8", "metal", "adaptive"])
def test_explicit_accelerated_batches_preserve_ring32(kernel):
    from pllm.native import capabilities
    if kernel == "digit8" and not capabilities().get("digit8_cpu"):
        pytest.skip("native digit8 unavailable")
    if kernel in {"metal", "adaptive"}:
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            pytest.skip("Metal requires Apple Silicon")
        pytest.importorskip("mlx.core")
    weights = np.resize(np.array([-128, 127, 0, 1], np.int8), (3, 19))
    matrix = MaskedGEMM(threads=1).compile(weights)
    requests = [request(weights, rows, bound=10000000) for rows in (3, 2)]
    with CorrectionBatch(matrix, kernel=kernel, window_rows=4, min_gpu_rows=4) as batch:
        tickets = [batch.add(req) for req in requests]
        result = batch.execute()
        if kernel == "adaptive":
            assert result["kernel_calls"] == {"native": 1, "metal": 1, "digit8": 0}
        for req, ticket in zip(requests, tickets, strict=True):
            cursor = prepared_mask_rows(req)
            try:
                r, s = cursor.take(0, req.rows)
                r = np.frombuffer(r, "<u4").reshape(req.rows, 19)
                s = np.frombuffer(s, "<u4").reshape(req.rows, 3)
                expected = (r.astype(np.int64) @ weights.astype(np.int64).T - s) % req.modulus
                actual = unpack_residues(ticket.take_rows(req.rows), (req.rows, 3), req.wire_bits)
                np.testing.assert_array_equal(actual, expected)
            finally:
                cursor.cancel()
