"""Public head evidence reconciles actual frames without retaining private shares."""
import hashlib
from dataclasses import replace

import httpx
import numpy as np
import pytest

from pllm.runtime.private_boundary_head_codec import HEADER, HeadOutputCodec
from pllm.runtime.private_boundary_protocol import ROLES, pack_frame, unpack_frame

from test_private_boundary_row_certificate import fixture_certificate
from test_private_boundary_transport import Harness


@pytest.mark.parametrize("encoding", ["u32", "uniform_residues", "row_residues"])
@pytest.mark.parametrize("dispatch", ["serial", "parallel"])
def test_actual_head_envelopes_subframes_and_public_source_reconcile(tmp_path, encoding, dispatch):
    harness = Harness()
    pin = None
    if encoding == "row_residues":
        pin = fixture_certificate(tmp_path / "head.cert", harness.d, harness.weights)
    harness.d = replace(harness.d, head_dispatch=dispatch, head_output_encoding=encoding,
                        head_certificate_digest="" if pin is None else pin["digest"])
    harness.admission = replace(harness.admission, descriptor_digest=harness.d.digest)
    codec = HeadOutputCodec(harness.d, certificate=pin, weights=harness.weights if pin else None)
    actual = {role: [] for role in ROLES}

    def reply(request):
        response = harness.handle(request)
        if not request.url.path.endswith("/execute"):
            return response
        row = unpack_frame(response.content, response=True)
        assert row["operation"] == "head"
        frame = codec.encode(row["frames"][0], admission=harness.admission, role=row["role"],
            paired_session=row["session"], epoch=row["epoch"], sequence=row["sequence"])
        body = pack_frame(role=row["role"], paired_session=row["session"], epoch=row["epoch"],
            sequence=row["sequence"], operation="head", frames=[frame], response=True)
        actual[row["role"]].append((len(body), len(frame)))
        return httpx.Response(200, content=body)

    harness.close()
    harness.clients = [httpx.Client(base_url=f"http://127.0.0.1:{8001 + index}",
        transport=httpx.MockTransport(reply)) for index in (0, 1)]
    transport = harness.transport(head_certificate=pin).open(harness.admission)
    try:
        for _ in range(3):
            transport.linear("lm_head", np.array([[.25, -.5, .125]], np.float32))
        evidence = transport.head_output_accounting
        assert evidence["descriptor"] == harness.d.to_dict()
        assert evidence["descriptor_digest"] == harness.d.digest
        layout = evidence["layout"]
        expected_widths = (bytes([32]) * 4 if encoding == "u32" else
            bytes([(2 * 3 * 128 * 127).bit_length()]) * 4 if encoding == "uniform_residues" else
            bytes(max(1, (127 * sum(abs(int(x)) for x in row)).bit_length() + 1)
                  for row in harness.weights))
        assert layout["widths_digest"] == hashlib.sha256(expected_widths).hexdigest()
        assert layout["width_counts"] == {str(w): expected_widths.count(w) for w in set(expected_widths)}
        assert layout["total_bits"] == sum(expected_widths)
        assert layout["payload_bytes_per_role"] == (sum(expected_widths) + 7) // 8
        header = 0 if encoding == "u32" else HEADER.size
        assert layout["subframe_header_bytes"] == header
        for role, frames in actual.items():
            counts = evidence["roles"][role]
            assert counts == {"replies": 3, "outer_body_bytes": sum(row[0] for row in frames),
                "subframe_bytes": sum(row[1] for row in frames), "header_bytes": 3 * header,
                "payload_bytes": 3 * layout["payload_bytes_per_role"]}
        assert sum(row["outer_body_bytes"] for row in evidence["roles"].values()) == (
            transport.operation_counters["head"]["download_bytes"])
        if pin:
            assert layout["certificate"]["digest"] == pin["digest"]
            assert layout["certificate"]["size_bytes"] == pin["size_bytes"]
            assert layout["certificate"]["metadata"]["widths_digest"] == layout["widths_digest"]
            assert layout["certificate"]["metadata"]["source"]["head_weight_digest"] == harness.d.head_weight_digest
        else:
            assert layout["certificate"] is None
        # Consumers cannot alter the original evidence or execution owner.
        evidence["roles"][ROLES[0]]["payload_bytes"] = -1
        layout["width_counts"].clear()
        assert transport.head_output_accounting["layout"]["width_counts"]
        transport.complete()
        saved = transport.head_output_accounting
        transport.close()
        assert transport.head_output_accounting == saved
    finally:
        transport.close()
        harness.close()
