"""Fixed-parameter mapped-allocation plan. Budgets are not completion proofs."""

from pathlib import Path
from dataclasses import dataclass
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
MAX_MAPPINGS = 4096


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
    if type(count) is not int or count <= 0:
        raise ValueError("positive allocation count required")
    return count * allocation_geometry(size, alignment)["charge_bytes"]


@dataclass
class _Mapping:
    capacity: int
    alignment: int
    mapped: int
    charge: int
    retired: bool = False
    unmap_failed: bool = False


class MappingLedger:
    """Serial integer simulation of allocator events, NOT an OS release proof.

    Capacity means original allocation Layout.size(), never Vec length or an
    AlignedBuf's truncated logical size. Retire ends the modelled owner only;
    release credit requires a separately supplied successful-unmap outcome.
    No payload memory or real mapping is allocated here.
    """

    def __init__(self, ceiling=CHARGE_CEILING, max_mappings=MAX_MAPPINGS):
        if any(type(v) is not int or v < 0 for v in (ceiling, max_mappings)):
            raise ValueError("nonnegative integer limits required")
        self.ceiling = ceiling
        self.max_mappings = max_mappings
        self.mappings = {}
        self.charged = 0
        self.peak_charged = 0
        self.poisoned = False

    def allocate(self, name, capacity, alignment=64):
        if name in self.mappings:
            raise ValueError("duplicate mapping")
        g = allocation_geometry(capacity, alignment)
        if (self.poisoned or len(self.mappings) >= self.max_mappings
                or self.charged + g["charge_bytes"] > self.ceiling):
            raise ValueError("allocation rejected")
        self.mappings[name] = _Mapping(capacity, alignment, g["mapped_bytes"], g["charge_bytes"])
        self.charged += g["charge_bytes"]
        self.peak_charged = max(self.peak_charged, self.charged)

    def retire(self, name):
        mapping = self.mappings[name]
        if mapping.retired:
            raise ValueError("owner already retired")
        mapping.retired = True

    def unmap_result(self, name, *, succeeded):
        if type(succeeded) is not bool:
            raise ValueError("explicit unmap outcome required")
        mapping = self.mappings[name]
        if not mapping.retired:
            raise ValueError("owner still live")
        if mapping.unmap_failed:
            raise ValueError("no production retry for failed unmap")
        if not succeeded:
            mapping.unmap_failed = True
            self.poisoned = True
            return
        self.charged -= mapping.charge
        del self.mappings[name]

    def realloc(self, old, new, capacity, *, old_unmap_succeeded):
        mapping = self.mappings[old]
        if type(old_unmap_succeeded) is not bool:
            raise ValueError("explicit unmap outcome required")
        if mapping.retired or self.poisoned:
            raise ValueError("allocation rejected")
        allocation_geometry(capacity, mapping.alignment)
        if capacity == mapping.capacity:
            return old
        # Reserve BOTH capacities, even for shrink. Rejection preserves old.
        self.allocate(new, capacity, mapping.alignment)
        self.retire(old)
        self.unmap_result(old, succeeded=old_unmap_succeeded)
        return new


def dft_phase_owners(baby_limbs=28, *, loading=False):
    """Explicit leaf Layouts for candidate DFT phase, not complete liveness.

    One tuple describes each allocation, (name, capacity bytes, alignment).
    Context/solver/container capacities remain unknown and are never zero-filled.
    """
    if type(baby_limbs) is not int or baby_limbs not in (27, 28):
        raise ValueError("only reviewed DFT baby widths allowed")
    owners = []

    def add(prefix, capacity, count=1, alignment=64):
        owners.extend((f"{prefix}/{i}", capacity, alignment) for i in range(count))

    add("wide_ct", S["z"](2, 28), 5)
    add("entry", S["z"](2, 1))
    for i in range(17):
        add(f"ntt/{i}", 64 * (1 << i), 4)
    for i in range(16):
        add(f"twiddle/{i}", ((16 * (1 << i) + 63) // 64) * 64, 4)
    for n, count in ((65536, 16), (4096, 31), (256, 31), (16, 15)):
        add(f"diagonal/{n}", n * 32, 2 * count)
    add("raw_secret", S["z"](1, 1), 2)
    add("prepared_secret", S["d"](1, 1), 2)
    add("high_key", S["HIGH_PREPARED"], 7)
    add("baby", S["d"](2, baby_limbs), 4)
    if loading:
        add("raw_import", S["HIGH_RAW"])
        add("prepare_scratch", 2 << 20)
        add("vmp_private", 2 << 20, alignment=4)
    else:
        add("matrix_scratch", S["matrix_query_certificate"]()["matrix_query_bytes"])
    return owners


def phase_certificate(owners):
    ledger = MappingLedger()
    for name, capacity, alignment in owners:
        ledger.allocate(name, capacity, alignment)
    return {
        "leaf_owners": [{"name": name, "capacity_bytes": capacity, "alignment": alignment}
                        for name, capacity, alignment in owners],
        "known_charge_bytes": ledger.charged,
        "known_mapping_count": len(ledger.mappings),
        "complete_upper_bound_bytes": None,
    }


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
        "poulpy_commit": S["COMMIT"],
        "cpu_execution_threads": 1,
        "maximum_refreshes": 2,
        "admitted": False,
        "execution_performed": False,
        "budget_sufficiency_proven": False,
        "conditional_total_is_complete_owner_bound": False,
        "solver_dependency_audit": S["solver_dependency_audit"](),
        "full_owner_completion_bound_bytes": None,
        "complete_setup_upper_bound_bytes": None,
        "attributable_kernel_io_upper_bound_bytes": None,
        "host_reserve_bytes": 4 << 30,
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
        "candidate_dft_phases": {
            f"{width}_limb_{phase}": phase_certificate(dft_phase_owners(width, loading=loading))
            for width in (27, 28)
            for phase, loading in (("loading", True), ("evaluation", False))
        },
        "unclosed_phases": ["context_and_solver_setup", "key_generation_export",
                            "affine_modup", "split_evalmod", "s2c_reentry"],
        "transition_requirements": [
            "end all typed borrows before refill or retirement",
            "successful unmap of raw import, preparation scratch and private vector before evaluation scratch",
            "successful unmap of prior baby cache before allocating next cache",
            "successful unmap of DFT keys/scratch before typed tensor allocation; no cross-role cast",
            "retain original Layout capacity through truncation; charge both mappings during every realloc",
        ],
        "other_heap_operational_charge_budget_bytes": other_heap_budget,
        "stream_io_operational_charge_budget_bytes": stream_io_budget,
        "conditional_factor_2_total_bytes": known_evaluation + other_heap_budget + stream_io_budget + GUARD + RUNTIME + PAGE_TABLES,
        "conditional_factor_2_margin_bytes": CAP - known_evaluation - other_heap_budget - stream_io_budget - GUARD - RUNTIME - PAGE_TABLES,
        "conditional_uniform_dft_total_bytes": uniform_dft + other_heap_budget + stream_io_budget + GUARD + RUNTIME + PAGE_TABLES,
        "conditional_uniform_dft_margin_bytes": CAP - uniform_dft - other_heap_budget - stream_io_budget - GUARD - RUNTIME - PAGE_TABLES,
        "unproved_sufficiency": [
            "96-MiB charged allowance for all remaining contexts, plan/noise/container capacities, client codec and oracle owners",
            "1-MiB charged Rust stream buffers; attributable kernel I/O has no proved separate bound",
            "64-MiB non-Rust runtime and 16-MiB page-table allowances; 32-KiB per-mapping kernel ticket",
            "solver and context setup peak, all non-DFT phase scratch/copies, and full typed adapter liveness",
            "pinned-nightly native allocator tests and effective dependency closure; cfg-if source identity alone is not closure",
        ],
    }


if __name__ == "__main__":
    print(json.dumps(plan(), indent=2))
    raise SystemExit(78)
