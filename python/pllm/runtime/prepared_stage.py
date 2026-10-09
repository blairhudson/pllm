"""Experimental compiler-bound native client ownership for selected linear stages."""
from __future__ import annotations

import numpy as np

from pllm import _native
from .transformer_client import (PreparedInventoryLease, PreparedRemoteLinear,
    TransformerClientError, equalize_activation, record_protocol_bytes,
    record_protocol_operation, start_protocol_span)


def stage_layout(stage, maximum_rows, request_encoding):
    profile = stage.seeded_profile
    widths = stage.output_residue_bits
    if (profile is None or not widths or request_encoding not in {"compact", "stage_packed"}
            or profile.modulus != 1 << profile.wire_bits):
        raise TransformerClientError("native stage requires admitted packed/residue boundaries")
    return _native.PreparedStageLayout(stage.id, stage.in_features, widths, profile.wire_bits,
        max(widths) if request_encoding == "stage_packed" else profile.wire_bits,
        stage.activation_bits, maximum_rows, stage.weight_scales.astype("<f4", copy=False).tobytes(),
        None if stage.bias is None else stage.bias.astype("<f4", copy=False).tobytes())


class NativePreparedLinear:
    """A checked stage subset; transport/session authentication remains with the SDK.

    Construction validates the compiled binding before copying immutable numeric
    metadata. The prototype is internal and is not a Pipeline capability.
    """

    def __init__(self, compiled, remote, stage_ids):
        compiled.validate()
        if (type(remote) is not PreparedRemoteLinear or remote.verification_component != "none"
                or remote.model_id != compiled._bundle.model_id
                or remote.body_fingerprint != compiled._bundle.privacy.get("body_fingerprint")
                or remote.stages is not compiled._bundle.stages
                or type(remote.inventory) is not PreparedInventoryLease or remote.inventory._closed):
            raise TransformerClientError("native stage owner differs from compiled prepared binding")
        selected = set(stage_ids)
        bindings = {s.stage_id: s for s in compiled.stage_bindings if s.client_weight_layout is None}
        if not selected or not selected <= bindings.keys():
            raise TransformerClientError("native stage owner requires compiled remote stages")
        self._layouts = {name: stage_layout(remote.stages[name],
            1 if bindings[name].role == "lm_head" else min(compiled._plan.prefill["query_sequence"],
                4_000_000 // max(remote.stages[name].in_features, remote.stages[name].out_features)),
            remote.request_encoding) for name in selected}
        self._binding_digest = compiled.digest
        self._remote = remote
        self.stats = remote.stats

    def __call__(self, stage_id, activation):
        if stage_id not in self._layouts:
            return self._remote(stage_id, activation)
        try:
            return self._execute(self._remote, self._layouts[stage_id], stage_id, activation)
        except BaseException:
            self._remote.inventory.close()
            raise

    @staticmethod
    def _execute(remote, layout, stage_id, activation):
        stats = remote.stats
        stage = remote.stages[stage_id]
        value = np.asarray(activation, dtype=np.float32)
        if not value.ndim or value.shape[-1] != stage.in_features:
            raise TransformerClientError("native prepared input differs from compiled stage")
        shape = value.shape[:-1] + (stage.out_features,)
        if stage.input_equalization is not None:
            value = equalize_activation(value, stage.input_equalization)
        input_bits = max(stage.output_residue_bits) if remote.request_encoding == "stage_packed" else None
        query, requests = remote.inventory._prepare_native(stage_id, value.reshape(-1, stage.in_features),
            layout, remote.prefill_chunk_rows, input_bits)
        del value
        upload = sum(map(len, requests))
        stats.inference_upload_bytes += upload
        stats.upload_bytes += upload
        record_protocol_operation(stage_id)
        record_protocol_bytes("client", "inference", upload, stage_id)
        span = start_protocol_span(stage_id, 0, upload)
        try:
            replies = remote.inference(stage_id, requests)
            download = sum(map(len, replies))
            stats.inference_download_bytes += download
            stats.download_bytes += download
            record_protocol_bytes("inference", "client", download, stage_id)
            if len(replies) != len(requests):
                raise TransformerClientError("inference provider returned the wrong chunk count")
            for reply in replies:
                query.receive(reply)
            output, server_ns = query.finish()
            span.set_attribute("pllm.inference_client.bytes", download)
        except BaseException as exc:
            remote.inventory.close()
            span.record_exception(exc)
            raise
        finally:
            query.cancel()
            span.end()
        stats.calls += 1
        stats.rows += int(np.prod(shape[:-1]))
        stats.inference_server_ns += server_ns
        stats.server_ns += server_ns
        return np.frombuffer(output, dtype="<f4").reshape(shape)
