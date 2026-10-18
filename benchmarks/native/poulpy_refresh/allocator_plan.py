"""Fixed-parameter mapped-allocation plan. Budgets are not completion proofs."""

from pathlib import Path
import json
import runpy

HERE = Path(__file__).resolve().parent
S = runpy.run_path(str(HERE / "staged_admission.py"))
PAGE = 16_384
HEADER = 64
TICKET = 2 * PAGE
CAP = 8 << 30
GUARD = 64 << 20
RUNTIME = 64 << 20
PAGE_TABLES = 16 << 20
CHARGE_CEILING = CAP - GUARD - RUNTIME - PAGE_TABLES
ISIZE_MAX = (1 << 63) - 1
USIZE_MAX = (1 << 64) - 1


def allocation_geometry(size, alignment, *, page=PAGE, ticket=TICKET):
    """Independent integer model, including unrepresentable Layout rejection."""
    if any(type(v) is not int for v in (size, alignment, page, ticket)):
        raise ValueError("integer byte geometry required")
    if (size <= 0 or alignment <= 0 or alignment & (alignment - 1)
            or page < HEADER or page & (page - 1) or ticket < 0):
        raise ValueError("invalid or zero-size geometry")
    if size + alignment - 1 > ISIZE_MAX:
        raise ValueError("invalid Rust Layout")
    alignment = max(alignment, 16)
    offset = alignment if alignment > page else ((HEADER + alignment - 1) // alignment) * alignment
    mapped = ((size + offset + page - 1) // page) * page
    if mapped > ISIZE_MAX or mapped + ticket > USIZE_MAX:
        raise ValueError("mapping/charge overflow")
    return {"mapped_bytes": mapped, "charge_bytes": mapped + ticket, "maximum_offset": offset}


def buffer_charge(size, count=1, alignment=64):
    return count * allocation_geometry(size, alignment)["charge_bytes"]


def plan():
    # Source-sized persistent payloads; each real allocation is rounded separately.
    # Container/noise/plan metadata is NOT silently included in these leaf buffers.
    common = {
        "five_wide_ct_and_entry": buffer_charge(S["z"](2, 28), 5) + buffer_charge(S["z"](2, 1)),
        "two_ntt_families": 4 * sum(buffer_charge(64 * (1 << i)) for i in range(17)),
        "two_encoding_twiddle_families": 4 * sum(buffer_charge(((16 * (1 << i) + 63) // 64) * 64) for i in range(16)),
        "both_prepared_dft_matrices": 2 * sum(buffer_charge(n * 32, count) for n, count in ((65536, 16), (4096, 31), (256, 31), (16, 15))),
        "both_raw_and_prepared_secret_pairs": 2 * (buffer_charge(S["z"](1, 1)) + buffer_charge(S["d"](1, 1))),
    }
    key_pool = buffer_charge(S["HIGH_PREPARED"], 7)
    babies = buffer_charge(S["d"](2, 27), 4)
    matrix_scratch = buffer_charge(S["matrix_query_certificate"]()["matrix_query_bytes"])
    raw = buffer_charge(S["HIGH_RAW"])
    # Preparation scratch and private per-column Vec coexist during key loading.
    preparation = 2 * buffer_charge(2 << 20)
    known_evaluation = sum(common.values()) + key_pool + babies + matrix_scratch
    known_loading = sum(common.values()) + key_pool + babies + raw + preparation
    # Uniform DFT envelope even if all seven owners remain allocated in factor 1:
    # use its larger 28-limb baby cache instead of the factor-2 27-limb cache.
    uniform_dft = sum(common.values()) + key_pool + buffer_charge(S["d"](2, 28), 4) + matrix_scratch
    # Explicit ceilings to be PROVED sufficient, not fabricated source bounds.
    other_heap_budget = 96 << 20
    stream_io_budget = 1 << 20
    return {
        "schema": "pllm.poulpy-mmap-allocation-plan.v1",
        "admitted": False,
        "full_owner_completion_bound_bytes": None,
        "allocator_charge_ceiling_bytes": CHARGE_CEILING,
        "fixed_external_operational_reserves_bytes": GUARD + RUNTIME + PAGE_TABLES,
        "per_mapping_kernel_bookkeeping_ticket_bytes": TICKET,
        "bookkeeping_ticket_is_proven_kernel_bound": False,
        "common_source_payload_mapping_charges": common,
        "factor_2_loading_known_charge_bytes": known_loading,
        "factor_2_evaluation_known_charge_bytes": known_evaluation,
        "factor_2_known_mapping_count": 6 + 68 + 64 + 186 + 4 + 7 + 4 + 1,
        "factor_2_evaluation_residual_charge_bytes": CHARGE_CEILING - known_evaluation,
        "uniform_dft_known_charge_bytes": uniform_dft,
        "uniform_dft_residual_charge_bytes": CHARGE_CEILING - uniform_dft,
        "other_heap_operational_charge_budget_bytes": other_heap_budget,
        "stream_io_operational_charge_budget_bytes": stream_io_budget,
        "conditional_factor_2_total_bytes": known_evaluation + other_heap_budget + stream_io_budget + GUARD + RUNTIME + PAGE_TABLES,
        "conditional_factor_2_margin_bytes": CAP - known_evaluation - other_heap_budget - stream_io_budget - GUARD - RUNTIME - PAGE_TABLES,
        "conditional_uniform_dft_total_bytes": uniform_dft + other_heap_budget + stream_io_budget + GUARD + RUNTIME + PAGE_TABLES,
        "conditional_uniform_dft_margin_bytes": CAP - uniform_dft - other_heap_budget - stream_io_budget - GUARD - RUNTIME - PAGE_TABLES,
        "unproved_sufficiency": [
            "96-MiB charged allowance for all remaining contexts, plan/noise/container capacities, client codec and oracle owners",
            "1-MiB charged stream buffers and attributable kernel I/O outside Rust allocator",
            "64-MiB non-Rust runtime and 16-MiB page-table allowances; 32-KiB per-mapping kernel ticket",
            "solver and context setup peak, all non-DFT phase scratch/copies, and full typed adapter liveness",
            "pinned-nightly native allocator tests and effective dependency closure",
        ],
    }


if __name__ == "__main__":
    print(json.dumps(plan(), indent=2))
    raise SystemExit(78)
