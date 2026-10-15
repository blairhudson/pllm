"""Certificate-domain codec coverage independent of source weight scanning."""
from copy import deepcopy
import json

import numpy as np
import pytest

from scripts import audit_compact_private_head as audit
from scripts import probe_compact_private_head as probe


@pytest.fixture
def report():
    weights = np.array([[0, 0], [-128, 127], [-1, 2], [127, 127]], dtype=np.int8)
    scales = np.array([1, 0.5, 0.25, 0.125], dtype=np.float32)
    pin = {key: probe.digest(key.encode()) for key in (
        "source_lock_sha256", "checkpoint_sha256", "model_config_sha256", "descriptor_sha256")}
    pin.update(weight_sha256=probe.digest(weights.tobytes()), scales_sha256=probe.digest(scales.tobytes()),
               vocabulary=4, hidden=2, qmax=127, weight_dtype="i8", activation_quantizer=probe.QUANTIZER)
    from pllm import _native
    cert = probe.scan(lambda a, b: weights[a:b], lambda a, b: scales[a:b], pin,
                      chunk_rows=4, native_widths=_native.offset_row_bits)
    return {
        "schema": "pllm.compact-private-head-scan.v1", "pin": pin,
        "certificate": json.loads(cert["metadata"]),
        "certificate_sha256": probe.digest(cert["metadata"]),
        "widths_hex": cert["widths"].hex(), "bounds_le_u64_hex": cert["bounds"].hex(),
        "verification": {"full_passes": 2, "every_row_scalar_verified": True,
                         "installed_native_every_row_verified": True},
        "costs": probe.costs(cert, 2),
    }


def test_actual_native_reconstruction_covers_every_row_and_endpoint(report):
    result = audit.audit(report)
    assert result["checked_integer_outputs"] == 20
    assert len(result["output_sha256"]) == 5
    assert result["payload_bytes_both"] < result["baseline_payload_bytes_both"]


@pytest.mark.parametrize("field", ["reply_payload_bytes_both", "saved_reply_bytes_both_per_head_estimate",
                                  "net_saved_bytes_at_head_calls_estimate"])
def test_rejects_unreconciled_costs(report, field):
    forged = deepcopy(report)
    forged["costs"][field] += 1
    with pytest.raises(ValueError, match="ledger"):
        audit.audit(forged)


def test_rejects_altered_layout_even_with_new_array_hash(report):
    widths = bytearray.fromhex(report["widths_hex"])
    widths[1] -= 1
    report["widths_hex"] = widths.hex()
    report["certificate"]["width_sha256"] = audit.sha(widths)
    report["certificate_sha256"] = audit.sha(audit.canonical(report["certificate"]))
    with pytest.raises(ValueError, match="signed-center"):
        audit.audit(report)


def test_rejects_substituted_native_result(report):
    from pllm import _native

    class Broken:
        offset_pack_rows = staticmethod(_native.offset_pack_rows)

        @staticmethod
        def offset_reconstruct_rows(*_):
            return bytes(4 * 8)

    with pytest.raises(ValueError, match="native reconstruction"):
        audit.audit(report, native=Broken)
