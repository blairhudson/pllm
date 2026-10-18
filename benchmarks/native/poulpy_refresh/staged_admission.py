"""Integer-only lifetime certificates for the fixed Poulpy qualification.

No backend import, allocation proportional to HE sizes, subprocess, or execution
admission. The ledger distinguishes live payload from the conservative charge
when reclamation is unproved. Neither quantity is a process-memory measurement.
See paper/methods/poulpy-lifetime-admission.md for the source derivation.
"""

from contextlib import contextmanager
from dataclasses import dataclass
import argparse
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
COMMIT = "ca3532570ff62b96599a74f7adb85de7726b0446"
N = 65_536
CAP = 8 << 30
GUARD = 64 << 20


def z(cols, limbs):
    return N * cols * limbs * 8


def d(cols, limbs):
    return N * cols * limbs * 32


def g(cols, limbs):
    return N * cols * limbs * 16


HIGH_RAW = 7 * z(2, 33)
HIGH_PREPARED = 7 * d(2, 33)
LOW_PREPARED = d(2, 3)
VMP_PRIVATE = 8 * N * 4


def matrix_query_certificate():
    """Exact fixed-shape query, proved by dominance over bounded other branches.

    S07:313-377, S22:75-117,257-291. All operands are the public query's
    28-limb envelopes, with actual one-limb diagonals. Gadget output is <=33
    limbs (S26:1521-1541); using 33 gives monotone upper bounds for siblings.
    These are scratch reservations only, not bounds on private heap owners.
    """
    a, pt, key = 28, 1, 33
    prod = a + pt
    normal = 3 * N * 8
    big_normal = 3 * N * 16
    idft = d(1, 1)
    prepare = 8 * N * 8

    def gadget(size):
        digits = min((size + 3) // 4, 7)
        return d(1, digits) + (16 + 8 * digits) * 8

    internal = z(2, a) + max(d(1, a) + gadget(a), normal)
    consume = g(1, key) + big_normal
    big_work = g(2, key) + max(idft, big_normal)
    keyswitch = z(2, a) + max(d(2, key) + max(consume, big_work, internal), normal)
    automorphism = max(
        keyswitch,
        d(2, key) + z(2, a) + max(normal, internal, big_work),
    )
    babies = z(2, a) + max(
        d(1, a) + z(2, a) + max(d(2, key) + max(gadget(a), consume), prepare),
        z(2, a) + max(automorphism, prepare), prepare, normal,
    )
    convolution = 16 * (a + 2 * (4 - 1) + 2 * pt) * 4 + 8 * 16 * prod * 8
    multiply_plain = z(2, a) + max(
        d(2, a) + d(1, pt) + max(
            prepare,
            g(1, prod) + z(1, a) + d(1, prod) + max(convolution, big_normal),
        ), normal,
    )
    inner = 8 * 16 * prod * 8
    lazy_rotation = (
        d(1, prod) + g(1, prod) + z(1, prod) + idft + d(2, key) + gadget(prod)
    )
    lazy = d(2, prod) + d(2, key) + inner + d(2, key) + lazy_rotation + g(2, key)
    other_branches = {
        "automorphism_upper": automorphism,
        "multiply_plain_upper": multiply_plain,
        "prepare_right": prepare,
        "baby_prepare_upper": babies,
        "lazy_from_small_upper": d(2, key) + max(internal, idft, g(1, 1)),
        "fallback_upper": d(2, prod) + g(1, prod) + max(inner, automorphism, normal),
        "copy_add_sub_imag_upper": z(1, a) + normal,
    }
    if max(other_branches.values()) >= lazy:
        raise ValueError("dominance proof failed")
    return {
        "other_branch_bounds_bytes": other_branches,
        "prepared_core_query_bytes": lazy,
        "assign_copy_bytes": z(2, a),
        "streamed_rhs_slot_bytes": d(1, pt),
        "streamed_rhs_prepare_bytes": prepare,
        "matrix_query_bytes": z(2, a) + lazy + d(1, pt) + prepare,
    }


@dataclass
class _Slot:
    role: str
    size: int
    live: bool = True
    borrows: int = 0


class RetentionLedger:
    """Model explicit payload ownership, never infer page release from drop.

    This models a no-reclamation charge, not a claim that an OS retains pages.
    In-place refill of an existing same-role owner has no new payload. There is
    deliberately no release-credit or arbitrary cross-role conversion method.
    """

    def __init__(self):
        self._slots = {}
        self.live_bytes = 0
        self.charged_bytes = 0
        self.peak_live_bytes = 0

    def allocate(self, name, role, size):
        if name in self._slots or not isinstance(role, str) or not role:
            raise ValueError("duplicate owner or missing role")
        if type(size) is not int or size <= 0:
            raise ValueError("payload must be a positive integer")
        self._slots[name] = _Slot(role, size)
        self.live_bytes += size
        self.charged_bytes += size
        self.peak_live_bytes = max(self.peak_live_bytes, self.live_bytes)

    def _live(self, name):
        slot = self._slots[name]
        if not slot.live:
            raise ValueError("owner was retired")
        return slot

    @contextmanager
    def borrow(self, name):
        slot = self._live(name)
        slot.borrows += 1
        try:
            yield
        finally:
            slot.borrows -= 1

    def refill(self, name, role, size):
        slot = self._live(name)
        if slot.borrows:
            raise ValueError("outstanding key views")
        if role != slot.role or type(size) is not int or size != slot.size:
            raise ValueError("no public same-shape, same-role refill proof")

    def retire(self, name):
        slot = self._live(name)
        if slot.borrows:
            raise ValueError("outstanding key views")
        slot.live = False
        self.live_bytes -= slot.size
        # Rust drop is not a proved physical-memory credit.


def verify_sources(source_root):
    """Hash existing text only. Never fetch dependencies or execute source."""
    manifest = json.loads((HERE / "staged_admission.sources.json").read_text())
    entries = list(manifest["supplementary_sources"])
    for inherited in manifest["inherited_manifests"]:
        path = ROOT / inherited["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != inherited["sha256"]:
            raise ValueError(f"manifest drift: {path}")
        entries.extend(json.loads(path.read_text())["sources"])
    for entry in entries:
        digest = hashlib.sha256()
        with (Path(source_root) / entry["path"]).open("rb") as stream:
            while chunk := stream.read(64 << 10):
                digest.update(chunk)
        if digest.hexdigest() != entry["sha256"]:
            raise ValueError(f"source drift: {entry['path']}")
    return len(entries)


def report():
    query = matrix_query_certificate()
    matrices = 2 * sum(n * count * 32 for n, count in ((N, 16), (4096, 31), (256, 31), (16, 15)))
    common = 7 * HIGH_PREPARED + 4 * d(2, 27) + matrices + 5 * z(2, 28) + z(2, 1)
    common += 2 * 2 * 64 * ((1 << 17) - 1) + GUARD
    subset = common + query["matrix_query_bytes"]
    remaining_after_raw = CAP - subset - HIGH_RAW
    return {
        "schema": "pllm.poulpy-staged-lifetime-admission.v1",
        "commit": COMMIT,
        "scope": "source calculations and payload lifetime model; no measurements",
        "admitted": False,
        "execution_performed": False,
        "complete_owner_upper_bound_bytes": None,
        "complete_setup_upper_bound_bytes": None,
        "measured_peak_bytes": None,
        "owner_ceiling_bytes": CAP,
        "guard_charge_bytes": GUARD,
        "matrix_query": query,
        "matrix_policy_subset_bytes": subset,
        "matrix_policy_residual_upper_bytes": CAP - subset,
        "matrix_policy_with_raw_subset_bytes": subset + HIGH_RAW,
        "residual_after_raw_upper_bytes": remaining_after_raw,
        "retained_first_baby_cache_bytes": 4 * d(2, 28),
        "live_first_baby_cache_policy_excess_bytes": subset + 4 * d(2, 28) - CAP,
        "live_separate_tensor_policy_excess_bytes": subset + HIGH_PREPARED - CAP,
        "distinct_role_persistent_keys_plus_guard_bytes": 9 * HIGH_PREPARED + LOW_PREPARED + GUARD,
        "vmp_private_live_bytes": VMP_PRIVATE,
        "vmp_private_allocations_per_high_key": 7 * 2 * 33,
        "vmp_private_requested_bytes_per_high_key": 7 * 2 * 33 * VMP_PRIVATE,
        "maximum_retained_vmp_blocks_after_raw_before_other_owners": remaining_after_raw // VMP_PRIVATE,
        "blockers": [
            "no bounded allocator reuse/reclamation for private VMP, baby caches, tensor and scratch",
            "no public automorphism-to-tensor prepared-owner conversion",
            "coefficient solver dependency allocation contract absent; probe dependency closure unlocked",
            "remaining contexts, containers, client, runtime and attributable I/O upper bounds absent",
            "independent parameter receipts and numerical re-entry qualification absent",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, help="optionally verify an existing pinned source tree")
    args = parser.parse_args()
    result = report()
    if args.source_dir is not None:
        result["verified_source_hash_count"] = verify_sources(args.source_dir)
    print(json.dumps(result, indent=2))
    return 78  # Admission remains blocked even when every source hash matches.


if __name__ == "__main__":
    raise SystemExit(main())
