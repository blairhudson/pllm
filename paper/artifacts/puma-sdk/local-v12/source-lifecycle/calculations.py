"""Read-only dimension/framing arithmetic; no pllm imports or model/file access.

Run: python3 -B target/puma-source-delivery-next/calculations.py
Not an execution benchmark, admission override, or proposed wire format.
"""
import json

LAYERS, HIDDEN, INTERMEDIATE, VOCABULARY = 24, 896, 4864, 151936
MATRICES = VOCABULARY * HIDDEN + LAYERS * (
    2 * HIDDEN**2 + 2 * HIDDEN * 128 + 3 * HIDDEN * INTERMEDIATE
)
GAMMA = (2 * LAYERS + 1) * HIDDEN
BIAS = LAYERS * (HIDDEN + 2 * 128)
N = MATRICES + GAMMA + BIAS
assert N == 494032768
assert MATRICES % 4096 == 0
W = 8 * (N - GAMMA)
MIN_MANIFEST = 49 * (2 * HIDDEN + 1)  # Gamma arrays alone; other JSON omitted.


def delivery(n):
    return {"bytes": n, "seconds_at_100_mbps": n * 8 / 100000000,
            "seconds_at_40_mbps": n * 8 / 40000000}


report = {
    "source_values": N,
    "matrix_values": MATRICES,
    "gamma_values": GAMMA,
    "bias_values": BIAS,
    "source_tensors": 290,
    "worker_private_files": 241,
    "append_calls_per_complete_import": MATRICES // 4096 + 49 + 72,
    "append_calls_client_plus_three_workers": 4 * (MATRICES // 4096 + 49 + 72),
    "hypothetical_distribution_payload_floors_not_current_request_traffic": {
        "half_precision_checkpoint_selected_tensors": delivery(2 * N),
        "fp32_checkpoint_selected_tensors": delivery(4 * N),
        "all_q18_equivalent_one_copy": delivery(8 * N),
        "all_q18_equivalent_three_copies_shared_uplink": delivery(24 * N),
        "worker_private_q18_files_one_copy": delivery(W),
        "worker_private_q18_files_three_copies_shared_uplink": delivery(3 * W),
    },
    "current_local_import": {
        "float32_bridge_bytes_per_import": 4 * N,
        "float32_bridge_bytes_all_four_imports": 16 * N,
        "q18_bytes_generated_and_hashed_per_import": 8 * N,
        "worker_private_file_bytes": W,
        "three_workers_private_file_bytes": 3 * W,
        "worker_disk_headroom_admission_bytes": 8 * N + (64 << 20),
        "half_precision_min_logical_io_per_worker_import": 6 * N + W,
        "half_precision_min_logical_io_client_and_three_imports": 24 * N + 3 * W,
        "half_precision_min_selected_reads_all_four_imports": 24 * N,
    },
    "actual_request_publication": {
        "bulk_q18_weight_upload_bytes": 0,
        "manifest_bytes_B": "unknown without actual certified coefficients; B >= 87857",
        "admit_body_bytes_per_worker": "B + 161 (bin32 manifest, 64-byte request/epoch strings)",
        "three_admit_upload_plus_ack_bytes": "3*B + 516",
        "other_setup_and_http_tls_tcp_bytes": "additional; not included",
        "private_input": [
            {"rows": r, "raw_share_bytes_per_worker": 16 * r,
             "msgpack_input_body_bytes_per_worker": 16 * r + 32 + (2 if r < 16 else 3),
             "three_upload_plus_ack_bytes": 3 * (16 * r + 32 + (2 if r < 16 else 3) + 11),
             "three_admit_and_input_upload_plus_ack_bytes":
                 f"3*B + {516 + 3 * (16 * r + 32 + (2 if r < 16 else 3) + 11)}",
             "dimension_only_known_component_framed_floor": delivery(
                 3 * MIN_MANIFEST + 516
                 + 3 * (16 * r + 32 + (2 if r < 16 else 3) + 11))}
            for r in (4, 16, 64)
        ],
    },
    "time_formula_for_any_framed_body_B_bytes": {
        "seconds_at_100_mbps": "B / 12500000",
        "seconds_at_40_mbps": "B / 5000000",
    },
}
print(json.dumps(report, indent=2))
