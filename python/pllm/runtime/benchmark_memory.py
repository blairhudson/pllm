"""Whole-topology benchmark memory admission, before loading tensor values.

These are conservative allocation estimates, not measured peak RSS or a proof
against OOM. Runtime pressure monitoring and Docker hard limits supplement them.
Swap never increases the admission budget. Backend selection never changes a
Pipeline, privacy topology, kernel, or an enforced-WAN measurement into an estimate.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import platform
import re
import shutil
import subprocess
import threading
from typing import Any, Callable

MiB = 1 << 20
GiB = 1 << 30
_PROCESS_BYTES = 256 * MiB
_HELPER_BYTES = 512 * MiB


class BenchmarkMemoryError(RuntimeError):
    def __init__(self, message: str, report: dict[str, Any] | None = None):
        super().__init__(message)
        self.report = report


def host_reserve_bytes(total: int) -> int:
    return max(GiB, total // 8)


@dataclass(frozen=True)
class HostMemory:
    total: int
    available: int
    swap_used: int
    disk_free: int
    availability_source: str = "psutil_available"
    pressure: str | None = None
    free: int | None = None
    cached: int | None = None
    purgeable: int | None = None
    reusable: int | None = None

    def __post_init__(self):
        if (any(type(value) is not int or value < 0 for value in
                (self.total, self.available, self.swap_used, self.disk_free))
                or not 0 <= self.available <= self.total or self.total == 0):
            raise BenchmarkMemoryError("host memory telemetry is unavailable or invalid")
        if self.availability_source == "macos_mach_dispatch_v1":
            if (self.pressure not in {"normal", "warning", "critical"}
                    or any(type(n) is not int or n < 0 for n in (self.free, self.cached, self.purgeable))
                    or self.free + self.cached + self.purgeable != self.available
                    or self.reusable is not None):
                raise BenchmarkMemoryError("macOS memory telemetry is unavailable or invalid")
        elif self.availability_source == "macos_mach_dispatch_v2":
            if (self.pressure not in {"normal", "warning", "critical"}
                    or any(type(n) is not int or n < 0 for n in
                           (self.free, self.cached, self.purgeable, self.reusable))
                    or self.free + self.cached + max(self.purgeable, self.reusable) != self.available):
                raise BenchmarkMemoryError("macOS memory telemetry is unavailable or invalid")
        elif self.availability_source != "psutil_available" or self.pressure is not None:
            raise BenchmarkMemoryError("unsupported host memory telemetry")

    @property
    def reserve(self) -> int:
        # Reclaimable physical pages and live kernel pressure replace the fixed
        # fraction-of-machine reserve. Allocation estimates already add a
        # workload-sized safety margin; do not subtract another total-RAM tax.
        if self.availability_source in ("macos_mach_dispatch_v1", "macos_mach_dispatch_v2"):
            return 0
        return host_reserve_bytes(self.total)


def host_memory() -> HostMemory:
    import psutil

    try:
        if platform.system() == "Darwin":
            from pllm import _native

            sample = json.loads(_native.macos_host_memory())
            return HostMemory(**sample, availability_source="macos_mach_dispatch_v2",
                              swap_used=int(psutil.swap_memory().used),
                              disk_free=int(shutil.disk_usage(Path.home()).free))
        memory = psutil.virtual_memory()
        return HostMemory(int(memory.total), int(memory.available),
                          int(psutil.swap_memory().used), int(shutil.disk_usage(Path.home()).free))
    except (OSError, ValueError, AttributeError, ImportError, TypeError, psutil.Error) as exc:
        raise BenchmarkMemoryError("cannot sample host memory for benchmark admission") from exc


def process_memory() -> dict[str, int | None]:
    """OS process samples. A lifetime high-water mark is not a per-run peak."""
    import psutil
    try:
        rss = int(psutil.Process().memory_info().rss)
    except (OSError, psutil.Error):
        rss = None
    peak = None
    try:
        import resource
        peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if platform.system() != "Darwin":
            peak *= 1024
    except (ImportError, OSError, AttributeError):
        pass
    return {"rss_bytes": rss, "lifetime_peak_rss_bytes": peak}


def _docker_capacity() -> tuple[int, int]:
    """Read existing capacity; never reconfigure Docker Desktop or other containers."""
    try:
        result = subprocess.run(["docker", "info", "--format", "{{.MemTotal}}"],
                                capture_output=True, text=True, check=True, timeout=10)
        total = int(result.stdout.strip())
        result = subprocess.run(["docker", "stats", "--no-stream", "--format", "{{.MemUsage}}"],
                                capture_output=True, text=True, check=True, timeout=10)
        # MemPerc is relative to each container's own limit, NOT VM capacity.
        # Parse measured bytes with explicit units and round display uncertainty up.
        used = sum(_docker_usage_bytes(line.split("/", 1)[0].strip())
                   for line in result.stdout.splitlines())
        if total <= 0:
            raise ValueError("invalid Docker memory telemetry")
        return total, used
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        raise BenchmarkMemoryError("Docker memory capacity is unavailable") from exc


def _docker_usage_bytes(text: str) -> int:
    match = re.fullmatch(r"(\d+(?:\.\d+)?)(B|kB|KB|MB|GB|TB|KiB|MiB|GiB|TiB)", text)
    if match is None:
        raise ValueError("unknown Docker memory unit")
    number, unit = match.groups()
    units = {"B": 1, "kB": 1000, "KB": 1000, "MB": 1000**2, "GB": 1000**3, "TB": 1000**4,
             "KiB": 1024, "MiB": MiB, "GiB": GiB, "TiB": 1024 * GiB}
    places = len(number.partition(".")[2])
    return math.ceil((float(number) + 10 ** -places) * units[unit])


def _memory_margin(percent: int) -> int:
    if type(percent) is not int or not 0 <= percent <= 100:
        raise ValueError("memory safety margin must be an integer from 0 to 100 percent")
    return percent


def _memory_budget(value: int | None) -> int | None:
    if value is not None and (type(value) is not int or value <= 0):
        raise ValueError("memory budget must be a positive byte count")
    return value


def _slack(value: int, percent: int = 25) -> int:
    return (value * (100 + _memory_margin(percent)) + 99) // 100


def _freivalds_memory(native, stages, *, rows, failure_bits, stage_window):
    """Price native u32 projections, their owners, and bounded Python/FFI copies.

    The policy query does not register a session or allocate material. Its failure
    target already includes any cache-lineage uplift from composition resolution.
    Uniform rows upper-bound demand allocation's attempt and storage budgets.
    """
    if not stages:
        raise BenchmarkMemoryError("verified memory admission requires remote stages")
    attempts = rows * len(stages)
    try:
        policy = native.FreivaldsPolicy(
            failure_bits, attempts, 1, 1, 1, 1, 1, 1, bytes(32), False)
        checks = policy.checks
        if type(checks) is not int or not 1 <= checks <= 8:
            raise ValueError("unknown native verification layout")
    except (AttributeError, TypeError, ValueError, OverflowError) as exc:
        raise BenchmarkMemoryError("native Freivalds allocation policy is unavailable") from exc
    projections = 4 * rows * checks * sum(s.in_features for s in stages)
    largest_projection = 4 * rows * checks * max(s.in_features for s in stages)
    # Row burns, material identities, bounded bindings and per-stage handle overhead.
    ledger = len(stages) * (8192 + 32 * rows)
    # A live inventory and one idle-refill spare can coexist. Challenges are only
    # expanded for a claimed stage, not for every retained inventory row at once.
    retained = 2 * (projections + ledger)
    # Receive bytearray, HTTP/MessagePack frames, parsed payload, binding copy and
    # native import overlap. Eight full frame copies conservatively bound these.
    import_work = stage_window * 8 * (4096 + largest_projection)
    claim_work = max(rows * (4 * checks * (s.in_features + s.out_features)
                            + 8 * s.in_features + 40 * s.out_features) for s in stages)
    # Preparation copies one stage's i8 snapshot through bytes and Vec<i8>, holds
    # one projection batch and frame copies, and expands one row of challenges.
    preparation = stage_window * max(
        2 * s.in_features * s.out_features
        + 8 * (4096 + 4 * rows * checks * s.in_features)
        + 4 * checks * s.out_features + 8192 + 32 * rows for s in stages)
    return {"scope": "native_freivalds_u32_owners_v1", "checks": checks,
            "target_failure_bits": failure_bits, "max_attempts": attempts,
            "one_inventory_projection_bytes": projections,
            "client_retained_bytes": retained, "client_import_work_bytes": import_work,
            "client_claim_work_bytes": claim_work,
            "client_peak_bytes": retained + max(import_work, claim_work),
            "preparation_peak_bytes": preparation, "inference_peak_bytes": 0}


def _validate_workload(max_input_tokens, max_output_tokens, inventory_rows, cache_bytes, cache_bound_tokens):
    for value in (max_input_tokens, max_output_tokens, inventory_rows):
        if type(value) is not int or not 1 <= value <= 32768:
            raise BenchmarkMemoryError("memory admission requires bounded positive workload counts")
    if type(cache_bytes) is not int or cache_bytes < 0:
        raise BenchmarkMemoryError("invalid client cache memory bound")
    if cache_bound_tokens is not None and (
        type(cache_bound_tokens) is not int or not 1 <= cache_bound_tokens <= 32768
    ):
        raise BenchmarkMemoryError("invalid client cache input bound")


def private_boundary_memory(vocabulary, hidden, *, lookup_rows, head_calls,
                            concurrent_requests=1, shared_matrix=True,
                            head_dispatch="serial", head_output_encoding="u32", head_certificate=None):
    """Allocation owners for packed128 lookup plus full additive wrap32 head.

    A context shares immutable table storage, never its replay ledger. Scratch
    is per active lookup (not per issued row); framing/import/export copies are
    deliberately additional owners. This is not a measured RSS result.
    """
    for value, limit in ((vocabulary, 1048576), (hidden, 1048576),
                         (lookup_rows, 4096), (head_calls, 4096),
                         (concurrent_requests, 8)):
        if type(value) is not int or not 1 <= value <= limit:
            raise BenchmarkMemoryError("private boundary allocation geometry exceeds bound")
    if type(shared_matrix) is not bool:
        raise BenchmarkMemoryError("private boundary sharing evidence must be explicit")
    if head_dispatch not in ("serial", "parallel") or head_output_encoding not in (
        "u32", "uniform_residues", "row_residues"
    ):
        raise BenchmarkMemoryError("private boundary head execution choice is unsupported")
    table, scales = vocabulary * hidden, 4 * vocabulary
    # Protocol accepts up to 64 lookup rows per frame; expansion is serial.
    lookup_frame = 8192 + min(64, lookup_rows) * (hidden + 4 + 256)
    head_frame = 8192 + 4 * (hidden + vocabulary)
    if max(lookup_frame, head_frame) > 16 * MiB:
        raise BenchmarkMemoryError("private boundary allocation exceeds frame bound")
    ledger = 8192 + 64 * (lookup_rows + head_calls)
    root_ledger = 4096 * 256  # Native context set plus Python worker replay set.
    scratch = vocabulary * 64
    worker_work = concurrent_requests * (ledger + scratch + 8 * max(lookup_frame, head_frame))
    # Keep the original full-width frame allowance. Parallel HTTP owners add two
    # conservative 8 MiB thread stacks and simultaneous framing/response copies.
    parallel_work = concurrent_requests * (16 * MiB + 4 * head_frame) if head_dispatch == "parallel" else 0
    head_bits = (2 * hidden * 128 * 127).bit_length() if head_output_encoding != "u32" else 32
    if head_bits > 32:
        raise BenchmarkMemoryError("private boundary head exceeds exact wrapping ring")
    head_payload = (vocabulary * head_bits + 7) // 8
    # Width arrays, packed bodies and native conversion overlap are additional
    # owners; reduced traffic never licenses a smaller original allocation bound.
    client_codec_work = concurrent_requests * (10 * vocabulary + 4 * head_payload) if head_output_encoding == "uniform_residues" else 0
    worker_codec_work = concurrent_requests * (9 * vocabulary + 2 * head_payload) if head_output_encoding == "uniform_residues" else 0
    certificate_work = {}
    if head_output_encoding == "row_residues":
        from .private_boundary_certificate import validate_pin
        try:
            artifact_bytes = validate_pin(head_certificate)["size_bytes"]
        except ValueError as exc:
            raise BenchmarkMemoryError(str(exc)) from exc
        # Full-width original frame owners remain. Certificate read/hash/JSON,
        # bounded zlib expansion, bytearray conversion, native Vec + Python i64
        # output overlap and every retained u8 layout are additional owners.
        client_codec_work = concurrent_requests * (24 * vocabulary + 4 * head_payload
                                                   + 4 * artifact_bytes + 128 * 1024)
        worker_codec_work = concurrent_requests * (16 * vocabulary + 2 * head_payload
                                                   + 4 * artifact_bytes + 2 * MiB)
        certificate_work = {"head_layout_bytes_per_owner": vocabulary,
            "certificate_artifact_bytes_per_recipient": artifact_bytes,
            "certificate_three_recipient_delivery_bytes": 3 * artifact_bytes,
            "certificate_publication_work_bytes": 2 * MiB + 8 * vocabulary + 4 * artifact_bytes,
            "certificate_publication_source_import_separate": True}
    elif head_certificate is not None:
        raise BenchmarkMemoryError("unexpected head certificate")
    worker_work += worker_codec_work
    native_table = table + scales + (0 if shared_matrix else table + scales)
    # The native publisher borrows the existing head Arc, hashes one row at a
    # time, and exports only its descriptor. Python/native scale buffers and the
    # row hash buffer remain additional owners. Unsupported older bindings keep
    # their conservative full-table serialization allowance.
    publisher_table_export = 0 if shared_matrix else 6 * table
    publisher_descriptor_work = 4 * scales + hidden
    # Matrix import and bytes/Vec overlap; source publisher remains separately priced.
    chunk_elements = min(vocabulary, 64, max(1, (1 << 20) // hidden)) * hidden
    import_work = max(3 * table + 4 * scales, 28 * chunk_elements,
                      4 * min(table, max(1 << 20, hidden)))
    # Source stage keeps its NumPy scales while native lookup owns scales Vec.
    # Head-only loader retains config/tokenizer/plan descriptors, not decoder W.
    retained_metadata = 96 * MiB + scales
    worker_peak = (_PROCESS_BYTES + native_table + retained_metadata + root_ledger
                   + max(import_work, worker_work))
    # Both returned u32 shares, reconstruction, float logits and client scales.
    client_work = scales + concurrent_requests * (
        ledger + 8 * max(lookup_frame, head_frame) + 24 * vocabulary + 16 * hidden)
    client_work += parallel_work + client_codec_work
    return {"scope": "private_boundary_allocation_owners_v1", "table_bytes": table,
            "scales_bytes": scales, "shared_immutable_matrix": shared_matrix,
            "worker_native_table_bytes": native_table, "worker_import_work_bytes": import_work,
            "publisher_table_export_bytes": publisher_table_export,
            "publisher_descriptor_work_bytes": publisher_descriptor_work,
            "worker_retained_metadata_and_source_scales_bytes": retained_metadata,
            "worker_root_context_ledger_bytes": root_ledger,
            "worker_session_ledger_bytes": concurrent_requests * ledger,
            "worker_lookup_scratch_bytes": concurrent_requests * scratch,
            "head_dispatch": head_dispatch, "head_output_encoding": head_output_encoding,
            "head_reply_bits": head_bits, "head_payload_bytes_per_worker": head_payload,
            "client_parallel_head_work_bytes": parallel_work,
            "client_head_codec_work_bytes": client_codec_work,
            "worker_head_codec_work_bytes": worker_codec_work,
            "worker_peak_bytes": worker_peak, "client_work_bytes": client_work,
            "client_tied_weight_bytes": 0, "lookup_rows_bound": lookup_rows,
            "head_calls_bound": head_calls, "concurrent_request_bound": concurrent_requests,
            **certificate_work}


def estimate_memory(config, pipeline, *, max_input_tokens, max_output_tokens,
                     inventory_rows=64, cache_bytes=0, cache_bound_tokens=None, checkpoint_bytes=0, store=None,
                     memory_safety_margin_percent=25):
    """Model-neutral graph geometry; no arrays or checkpoint tensor reads."""
    from pllm.modeling import lower_model
    from pllm.profiles import resolve_runtime_composition

    _memory_margin(memory_safety_margin_percent)
    options = resolve_runtime_composition(pipeline)
    if options is None:
        raise BenchmarkMemoryError("memory estimation requires an implemented composition")
    if options.client_runtime == "compiled_fss_v1":
        return _fss_memory(config, pipeline, max_input_tokens, max_output_tokens,
            cache_bytes, cache_bound_tokens, checkpoint_bytes, memory_safety_margin_percent)
    if options.client_runtime == "compiled_replicated_v1":
        return _replicated_memory(config, pipeline, max_input_tokens, max_output_tokens,
            cache_bytes, cache_bound_tokens, checkpoint_bytes, memory_safety_margin_percent)
    if options.client_runtime == "compiled_gkm_v1":
        return _delegated_memory(config, pipeline, max_input_tokens, max_output_tokens,
                                 cache_bytes, cache_bound_tokens, checkpoint_bytes, memory_safety_margin_percent)
    _validate_workload(max_input_tokens, max_output_tokens, inventory_rows, cache_bytes, cache_bound_tokens)
    compiled_input_bound = max(max_input_tokens, cache_bound_tokens or 0,
                               options.prefix_cache_bound_tokens or 0)
    plan = lower_model(config, batch=1, max_input_tokens=compiled_input_bound,
                       max_new_tokens=max_output_tokens)
    return estimate_plan_memory(plan, pipeline, max_input_tokens=max_input_tokens,
        max_output_tokens=max_output_tokens, inventory_rows=inventory_rows, cache_bytes=cache_bytes,
        cache_bound_tokens=cache_bound_tokens, checkpoint_bytes=checkpoint_bytes, store=store,
        memory_safety_margin_percent=memory_safety_margin_percent)


def _fss_memory(config, pipeline, rows, outputs, cache_bytes, cache_bound, checkpoint_bytes,
                memory_safety_margin_percent=25, *, contract=None, source_pins=None,
                transport_admission=None):
    """Bounded metadata/owner estimate; never admits numeric/program resources."""
    from .fss_protocol import ROLE_MEMORY, FRAME_LIMIT, ADMISSION_LIMIT, CONTROL_LIMIT, ROLES
    from .fss_tape import MAX_PACKET_BYTES, capacity_limits
    from .fss_transport import client_memory_bytes, price_transport, transport_capacity, MAX_GUARD_DISK_BYTES
    from . import delegated_source as source_io
    from .fss_source import _FRAME, _material_delivery
    from .tokenizer_admission import DECODER_METADATA_MEMORY_BYTES
    from pllm.configuration import Deployment, ExecutionBudget, Experiment
    _validate_workload(rows, outputs, 1, cache_bytes, cache_bound)
    if cache_bytes or cache_bound is not None or outputs > 4 or rows + outputs - 1 > 16:
        raise BenchmarkMemoryError("FuseFSS requires bounded fixed-count complete-prefix recomputation")
    try:
        capacity = pipeline.components["linear"].params.get("capacity", "capsule")
        delivery = _material_delivery(pipeline.components["linear"].params)
        expanded = capacity == "decoder"
        total_packet_limit, tape_limit = capacity_limits(capacity)
        raw = json.dumps(config, separators=(",", ":"), allow_nan=False).encode()
        if len(raw) > 16384:
            raise ValueError("config exceeds native metadata bound")
        if contract is None:
            from .fss_source import _contract, _geometry
            experiment = Experiment("fss-memory-estimate", pipeline, Deployment.local(root="."),
                budget=ExecutionBudget(requests=1, max_input_tokens=rows, max_new_tokens=outputs))
            _geometry(raw, rows, outputs, pipeline.components["linear"].params["numeric_profile"],
                      **({"capacity": capacity} if expanded else {}))
            # Native requires nonzero pins. These intentionally synthetic pins
            # are never represented as source verification or serving admission.
            _, contract = _contract(experiment, experiment.canonical_bytes(), raw, rows, outputs,
                                    "1" * 64, "2" * 64)
        if (contract["schema_version"] != "pllm.fss.sdk-contract.v1"
                or contract["runtime_kind"] != "fss" or contract["ring_bits"] != 50
                or contract["fractional_bits"] != 12 or contract["input_rows"] != rows
                or contract["output_tokens"] != outputs or transport_capacity(contract) != capacity
                or _material_delivery(contract) != delivery):
            raise ValueError("native metadata identity differs")
        source = contract["source_float32_bytes"]
        minimum = contract["admission"]["source_minimum_owner_bytes"]
        if (type(source) is not int or not 0 < source <= (source_io._DECODER_SOURCE_BYTES if expanded else 8 * MiB)
                or type(minimum) is not int or not 0 < minimum <= 64 * MiB
                or contract["admission"]["numeric"] != "not_performed"
                or contract["admission"]["program_resources"] is not None):
            raise ValueError("native metadata admission differs")
        source_disk, source_resident = 0, 64 * MiB
        if expanded:
            source_disk = contract["source_disk_bytes"]
            source_resident = contract["source_resident_bound_bytes"]
            q12 = contract["source_q12_value_bytes"]
            if (type(q12) is not int or q12 != 2 * source
                    or type(source_disk) is not int or not 0 <= source_disk <= 2 * source_io._DECODER_SOURCE_BYTES
                    or type(source_resident) is not int or not 0 < source_resident <= 8 * MiB):
                raise ValueError("native source storage admission differs")
        guard_disk = MAX_GUARD_DISK_BYTES if expanded else 0
        disk_scope = "capacity ceilings before native import"
        python_batch, native_batch = 0, None
        batch_scope = "conservative transport envelope before native import"
        if delivery == "batch-v1":
            from .fss_protocol import MATERIAL_BATCH_PYTHON_BYTES
            python_batch = MATERIAL_BATCH_PYTHON_BYTES
        if transport_admission is not None:
            resources = price_transport(transport_admission, json.dumps(contract).encode())
            tape_limit, guard_disk = resources.tape.max_file_bytes, resources.guard_disk_bytes
            total_packet_limit = resources.tape.total_packet_bytes
            disk_scope = "native transport admission after import"
            if delivery == "batch-v1":
                # Shared transport admission validates the native bound/mode.
                # Native packet/queue leases are already inside the fixed native
                # kernel owner; Python callback/HTTP copies are a separate owner.
                python_batch = resources.python_batch_reservation_bytes
                native_batch = resources.native_batch_reservation_bytes
                batch_scope = "shared transport price from imported native program"
    except (ValueError, TypeError, KeyError, AttributeError, OverflowError, ImportError) as error:
        raise BenchmarkMemoryError("FuseFSS source exceeds metadata-only native admission") from error
    metadata = 8 * (len(raw) + len(json.dumps(contract).encode()))
    client_parts = {"native_request_workspace": 8 * MiB, "metadata_parse_copies": metadata,
                    "hash_and_source_inspection_scratch": 4 * MiB,
                    # Match the shared decoder's bounded 64MiB tokenizer/outer
                    # workspace convention, derived from its 16MiB byte limit.
                    "tokenizer_parse_copies": 4 * source_io._DECODER_TOKENIZER_BYTES if expanded else 32 * MiB,
                    "http_control_overlap": 6 * ADMISSION_LIMIT + 8 * CONTROL_LIMIT}
    role_parts = {"native_kernel_owner": 64 * MiB, "native_source_program_owner": source_resident,
                  "metadata_parse_copies": metadata, "http_native_aead_tape_overlap": 6 * FRAME_LIMIT,
                  "source_import_scratch": 7 * _FRAME + 4 * MiB,
                  "thread_stack_allowance": 8 * MiB}
    if expanded:
        role_parts["tokenizer_parse_copies"] = 4 * source_io._DECODER_TOKENIZER_BYTES
    if delivery == "batch-v1":
        role_parts["material_batch_python_overlap"] = python_batch
    client_work, role_work = sum(client_parts.values()), sum(role_parts.values())
    client_bound = client_memory_bytes(capacity)
    # The authenticated worker reserves its existing role envelope plus the
    # explicit batch Python overlap. Native's combined <=64MiB proof remains
    # independent; this matches the outer Python lease rather than raising it.
    role_bound = ROLE_MEMORY + python_batch
    if client_work > client_bound or role_work > role_bound:
        raise BenchmarkMemoryError("FuseFSS conservative incremental owners exceed client/role memory caps")
    # Source metadata preflight precedes role import/execution. Price its shared
    # transient JSON/reverse-index reservation as a separate peak, not another
    # live copy of the checkpoint or an increase to the native primitive bound.
    metadata_preflight = DECODER_METADATA_MEMORY_BYTES if expanded else 0
    client_work, role_work = max(client_work, metadata_preflight), max(role_work, metadata_preflight)
    margin = memory_safety_margin_percent
    providers = {role: _slack(_PROCESS_BYTES + role_work, margin) for role in ROLES}
    # SDK, supervisor, dashboard and relay can overlap, plus three role processes.
    client = _slack(3 * _PROCESS_BYTES + client_work + 32 * MiB, margin)
    return {"scope": "conservative allocation estimate; not measured peak RSS",
        "runtime_kind": "fss", "model_plan_digest": None,
        "model_config_digest": contract["plan"]["config_digest"], "pipeline_digest": pipeline.digest(),
        "max_input_tokens": rows, "max_output_tokens": outputs, "compiled_input_bound": rows,
        "inventory_rows": 0, "kernel": "cpu", "concurrent_response_bound": 1,
        "allocation_slack_percent": margin, "client_peak_bytes": client,
        "provider_peak_bytes": providers, "docker_provider_peak_bytes": {},
        "native_total_peak_bytes": client + sum(providers.values()), "limitations": [],
        "preparation_storage": "dealer-generated encrypted holder disk tapes",
        "preparation_kernel": "native_trusted_dealer", "preparation_roles": ["dealer"],
        "weight_storage": "public original IEEE to exact nearest-even ring50/Q12",
        "bundle_storage": "no source checkpoint delivery measured", "mask_storage": "one-use offline tape",
        "tensor_storage": "private complete-prefix recomputation; no persistent KV",
        "fss_admission": {"source_pins": source_pins, "source_pins_verified": source_pins is not None,
            "synthetic_nonzero_pins_used": source_pins is None, "metadata_only": True,
            "numeric_admission": "not_performed", "complete_program_resources": None,
            "transport_admission": transport_admission, "disk_price_scope": disk_scope,
            **({"material_delivery": delivery, "material_batch_memory_scope": batch_scope,
                "material_batch_python_reservation_bytes": python_batch,
                "material_batch_native_reservation_bytes": native_batch,
                "material_batch_native_in_kernel_owner": True} if delivery == "batch-v1" else {}),
            "native_admission": contract["admission"], "serving_qualified": False},
        "incremental_owner_bounds": {"client": max(client_bound, metadata_preflight),
            "per_role": max(role_bound, metadata_preflight), "client_execution": client_bound,
            "per_role_execution": role_bound, "source_metadata_preflight": metadata_preflight},
        "allocation_owners": {"client": client_parts, "each_role": role_parts,
            "source_metadata_preflight_bytes_per_process": metadata_preflight,
            "client_sdk_supervisor_dashboard_processes": 3, "relay_workspace_bytes": 32 * MiB},
        "components": {"checkpoint_bytes": checkpoint_bytes, "source_float32_bytes": source,
            "checkpoint_byte_count_scope": "caller_supplied_or_float32_geometry_upper_bound_not_delivery",
            # The shared whole-topology gate multiplies per-engine storage by
            # all three roles, and adds its holder-transport bucket exactly once.
            "per_engine_compiled_disk_bytes": source_disk, "raw_client_bundle_bytes": 0,
            "fss_source_disk_bytes": 3 * source_disk,
            "fss_tape_disk_bytes": 2 * (tape_limit + guard_disk),
            "fss_tape_disk_scope": "two holder tapes plus two separate native replay guards",
            "fss_encrypted_tape_disk_bytes": 2 * tape_limit, "fss_guard_disk_bytes": 2 * guard_disk,
            "fss_packet_bound_bytes": MAX_PACKET_BYTES,
            "fss_cumulative_packets_per_holder_bytes": total_packet_limit,
            "fss_tape_file_per_holder_bytes": tape_limit}}


def _replicated_memory(config, pipeline, rows, outputs, cache_bytes, cache_bound, checkpoint_bytes,
                       memory_safety_margin_percent=25):
    from pllm import _native
    from pllm.configuration import Deployment, ExecutionBudget, Experiment

    if cache_bytes or cache_bound is not None:
        raise BenchmarkMemoryError("replicated response does not admit client cache reuse")
    try:
        experiment = Experiment("replicated-admission", pipeline, Deployment.local(root="."),
            budget=ExecutionBudget(requests=1, max_input_tokens=rows, max_new_tokens=outputs))
        contract = json.loads(_native.replicated_source_contract(experiment.canonical_bytes(),
            json.dumps(config).encode(), rows, outputs, "0" * 64, "0" * 64))
        from .replicated_source import validate_execution
        validate_execution(experiment, contract)
    except (ValueError, TypeError, OverflowError) as error:
        raise BenchmarkMemoryError("replicated source exceeds native admission") from error
    role = _slack(_PROCESS_BYTES + contract["execution_payload_estimate_bytes"], memory_safety_margin_percent)
    from .replicated_protocol import execution_policy, STREAM_MEMORY
    selected = execution_policy(experiment)["peer_transport"] == "streams"
    stream_memory = STREAM_MEMORY if selected else 0
    worker = _slack(_PROCESS_BYTES + contract["execution_payload_estimate_bytes"] + stream_memory,
                    memory_safety_margin_percent)
    public_weights = contract.get("weights") == "public"
    return {
        "scope": "conservative allocation estimate; not measured peak RSS",
        "model_plan_digest": None, "model_config_digest": contract["plan"]["config_digest"],
        "pipeline_digest": pipeline.digest(), "max_input_tokens": rows, "max_output_tokens": outputs,
        "compiled_input_bound": rows, "inventory_rows": 0, "kernel": "cpu", "runtime_kind": "replicated",
        "concurrent_response_bound": 1, "allocation_slack_percent": memory_safety_margin_percent,
        "client_peak_bytes": role, "provider_peak_bytes": {name: worker for name in ("worker_a", "worker_b", "worker_c")},
        "docker_provider_peak_bytes": {}, "native_total_peak_bytes": role + 3 * worker,
        "preparation_storage": "none", "preparation_kernel": "none",
        "weight_storage": "verified file-backed public Q18 weights" if public_weights else "party-local Q18 shares",
        "bundle_storage": "trusted client source publication", "mask_storage": "party-local pairwise PRSS",
        "tensor_storage": "party-local recomputed prefix, KV and Q18 activations",
        "components": {"source_float32_bytes": contract["source_float32_bytes"],
            "checkpoint_bytes": checkpoint_bytes,
            "per_engine_compiled_disk_bytes": contract.get("public_weight_snapshot_bytes_per_worker", 0),
            "public_weight_read_buffer_bytes": contract.get("public_weight_read_buffer_bytes", 0),
            **({"peer_stream_workspace_bytes_per_worker": stream_memory} if stream_memory else {}),
            "raw_client_bundle_bytes": 0},
        "limitations": [],
    }


def _delegated_memory(config, pipeline, rows, outputs, cache_bytes, cache_bound, checkpoint_bytes,
                      memory_safety_margin_percent=25):
    """Price the bounded source/verified-window owners, never an int8 schedule."""
    from .delegated_decoder import decoder_contract
    from .delegated_protocol import BATCH_EXECUTION, batch_memory_bytes

    capacity = pipeline.components["linear"].params.get("capacity", "capsule")
    expanded = capacity == "decoder"
    if (type(outputs) is not int or not 1 <= outputs <= (65 if expanded else 9)
            or cache_bytes or cache_bound is not None):
        raise BenchmarkMemoryError("GKM requires a bounded full-KV response without cache reuse")
    try:
        raw_contract = (decoder_contract(config, input_rows=rows, decode_steps=outputs - 1,
                                        capacity=capacity, streamed=True) if expanded else
                        decoder_contract(config, input_rows=rows, decode_steps=outputs - 1,
                                         streamed=True))
        contract = json.loads(raw_contract)
    except (ValueError, TypeError, OverflowError) as exc:
        raise BenchmarkMemoryError("GKM source or workload exceeds native numerical admission") from exc
    source = contract["source_float32_bytes"]
    batch_prices = {}
    execution = pipeline.components["linear"].params.get("execution")
    if execution is not None:
        from pllm import _native
        from pllm.configuration import Deployment, ExecutionBudget, Experiment
        from .delegated_decoder import _encode_config

        if execution != BATCH_EXECUTION:
            raise BenchmarkMemoryError("GKM unsupported execution selector")
        try:
            experiment = Experiment("gkm-memory-estimate", pipeline, Deployment.local(root="."),
                                    ExecutionBudget(requests=1, max_input_tokens=rows, max_new_tokens=outputs))
            # Geometry-only quote: nonzero placeholders satisfy native pin syntax.
            # Neither pin authorizes source loading, role admission or execution.
            runtime = json.loads(_native.gkm_runtime_contract(experiment.canonical_bytes(),
                _encode_config(config), rows, outputs, "1" * 64, "2" * 64))
            if runtime.get("decoder") != contract or "batch_execution" not in runtime:
                raise ValueError("native batch memory contract differs")
            batch_prices = batch_memory_bytes(runtime, capacity)
            if not batch_prices:
                raise ValueError("native batch memory metadata is missing")
        except (ValueError, TypeError, OverflowError) as exc:
            raise BenchmarkMemoryError("GKM batch HTTP/copy admission is unavailable or exceeds capacity") from exc
    # Capsule preserves its complete 64 MiB allowance and legacy source-copy
    # estimate in every role. Decoder uses separate native client/worker bounds
    # for streamed sources, publication, windows, tensors and metadata. Both add
    # outer process/transport allowance and the ordinary allocation margin.
    # These are conservative owner allowances, not measured RSS or tier results.
    native = 64 * MiB
    transport = 64 * MiB
    source_owners = 3 * source
    if expanded:
        native = contract["decoder_resources"]["client_payload_bytes"]
        source_owners = 0  # Streamed native ownership is already included above.
    role = _slack(_PROCESS_BYTES + native + transport + source_owners, memory_safety_margin_percent)
    worker = (_slack(_PROCESS_BYTES + contract["decoder_resources"]["worker_payload_bytes"] + transport,
                     memory_safety_margin_percent)
              if expanded else role)
    providers = {name: worker for name in ("worker_a", "worker_b")}
    if batch_prices:
        # Keep historical process/source/transport terms and margin unchanged.
        # Replace only the payload term with the shared native+Python+backing price.
        base_client = contract["execution_payload_estimate_bytes"]
        base_worker = contract["decoder_resources"]["worker_payload_bytes"] if expanded else base_client
        role = _slack(_PROCESS_BYTES + native + transport + source_owners
                      + batch_prices["client_payload_bytes"] - base_client, memory_safety_margin_percent)
        providers = {name: _slack(_PROCESS_BYTES + (base_worker if expanded else native)
                     + transport + source_owners + batch_prices["worker_payload_bytes"][name] - base_worker,
                     memory_safety_margin_percent) for name in providers}
    return {
        "scope": "conservative allocation estimate; not measured peak RSS",
        "model_plan_digest": None, "model_config_digest": contract["plan"]["config_digest"],
        "pipeline_digest": pipeline.digest(), "max_input_tokens": rows,
        "max_output_tokens": outputs, "compiled_input_bound": rows,
        "inventory_rows": 0, "kernel": "cpu", "runtime_kind": "gkm",
        "concurrent_response_bound": 1, "allocation_slack_percent": memory_safety_margin_percent,
        "client_peak_bytes": role, "provider_peak_bytes": providers,
        "docker_provider_peak_bytes": {}, "native_total_peak_bytes": role + sum(providers.values()),
        "preparation_storage": "none", "preparation_kernel": "none",
        "weight_storage": ("streamed native i16 weights and original float32 lookup" if expanded
                           else "source float32 plus exact i16 native snapshots"),
        "bundle_storage": "trusted client-owned publication; bounded source loading",
        "mask_storage": (
            "one pending complementary share and fresh client/worker AES window streams"
            if pipeline.components["linear"].params.get("sharing") == "seeded"
            else "one pending explicit-share pair per verified window"
        ),
        "tensor_storage": "complete client-local float32 logits and full KV",
        "verification_allocation": {
            "native_allowance_per_role_bytes": native,
            "native_allowance_by_role_bytes": {
                "client": native,
                **{name: (contract["decoder_resources"]["worker_payload_bytes"]
                          if expanded else native) for name in providers},
            },
            "contract_execution_payload_estimate_bytes": contract["execution_payload_estimate_bytes"],
            "publication_window_payload_allowance_bytes": contract["capsule_payload_allowance_bytes"],
            "native_transport_resources": deepcopy(contract["transport_resources"]),
            "transport_and_tokenizer_allowance_per_role_bytes": transport,
            **({key: value for key, value in batch_prices.items() if key.startswith("batch_")}
               if batch_prices else {}),
        },
        "components": {"source_float32_bytes": source,
                       "source_import_and_retention_bytes_per_role": source_owners,
                       "checkpoint_bytes": checkpoint_bytes,
                       "per_engine_compiled_disk_bytes": 0, "raw_client_bundle_bytes": 0},
        "limitations": [],
    }


def estimate_plan_memory(plan, pipeline, *, max_input_tokens, max_output_tokens,
                         inventory_rows=64, cache_bytes=0, cache_bound_tokens=None,
                         checkpoint_bytes=0, store=None, memory_safety_margin_percent=25):
    """Shared native allocation-owner estimate from an admitted semantic plan."""
    from pllm.profiles import resolve_runtime_composition
    from pllm.roles.topology import graph_for_runtime
    from .semantic_stages import client_owns_linear, provider_owns_linear, scheduled_stage_specs
    from ._native_support import extension

    _memory_margin(memory_safety_margin_percent)
    # Reference executors and stale wheels have different allocations. Check a
    # one-byte snapshot capability before inspecting model geometry or values.
    native = extension()
    snapshot_storage = False
    if native is not None:
        try:
            with memoryview(native.Matrix(b"\x00", 1, 1)) as view:
                snapshot_storage = view.readonly
        except (AttributeError, TypeError, BufferError):
            pass
    if not snapshot_storage or not hasattr(native, "PreparedMaskRows"):
        raise BenchmarkMemoryError("memory admission requires the installed native snapshot storage and mask-stream backend")

    _validate_workload(max_input_tokens, max_output_tokens, inventory_rows, cache_bytes, cache_bound_tokens)
    options = resolve_runtime_composition(pipeline)
    if options is None or options.privacy_mode not in {"public", "offset_public", "client_only"}:
        raise BenchmarkMemoryError("memory estimation requires an implemented public compiled topology")
    delivery = pipeline.components.get("delivery")
    paged = delivery is not None and delivery.params.get("storage") == "paged"
    if paged and (not hasattr(getattr(native, "PagedMatrix", None), "from_public_file")
                  or not hasattr(native, "PublicFileWriter")):
        raise BenchmarkMemoryError("paged memory admission requires native shared authenticated snapshots")
    compiled_input_bound = max(max_input_tokens, cache_bound_tokens or 0,
                               options.prefix_cache_bound_tokens or 0)
    if (plan.prefill["query_sequence"] != compiled_input_bound
            or plan.decode["maximum_key_sequence"] != compiled_input_bound + max_output_tokens - 1):
        raise BenchmarkMemoryError("allocation plan and compiled workload bounds differ")
    stages = scheduled_stage_specs(plan, pipeline)
    from .semantic_tensors import required_client_tensors, preflight_semantic_checkpoint
    local_tensors = 4 * sum(math.prod(shape) for shape in required_client_tensors(plan).values())
    if store is not None:
        preflight_semantic_checkpoint(plan, store)
        for stage in stages:
            if any(store.tensor_dtype(key).upper() not in {"F16", "BF16", "F32", "F64"}
                   for key in stage.weight_keys):
                raise BenchmarkMemoryError("packed checkpoint loading has no admitted allocation estimate")
    roles = [role.id for role in graph_for_runtime(options).roles if role.id != "client"]
    ownership: dict[str, Any] = dict(client_prefix_layers=options.client_prefix_layers,
                     client_linear_roles=options.client_linear_roles,
                     client_linear_layer_roles=options.client_linear_layer_roles)
    remote = [s for s in stages if roles and provider_owns_linear(
        s, remote_output_head=options.remote_output_head, **ownership)]
    client_body = [s for s in stages if s.role not in {"token_lookup", "lm_head"}
                   and (not roles or client_owns_linear(s, **ownership))]
    weights = sum(s.in_features * s.out_features for s in stages)
    largest = max(s.in_features * s.out_features for s in stages)
    local_weights = sum(s.in_features * s.out_features for s in client_body)
    boundary = sum(s.in_features * s.out_features for s in stages
                   if s.role in {"token_lookup", "lm_head"}
                   and not (s.role == "lm_head" and options.remote_output_head))
    token = next(s for s in stages if s.role == "token_lookup")
    head = next(s for s in stages if s.role == "lm_head")
    shared_main_table = bool(token.weight_keys and head.weight_keys
                              and token.weight_keys[0] == head.weight_keys[0])
    if shared_main_table:
        boundary -= next(s.in_features * s.out_features for s in stages if s.role == "lm_head")
    private_boundary = bool(getattr(options, "private_boundary", False))
    if private_boundary:
        if not shared_main_table:
            raise BenchmarkMemoryError("private boundary admission requires canonical tied weights")
        boundary = 0
    # Descriptors include scales for every stage, not only client-owned weights.
    metadata = 16 * sum(s.out_features + s.in_features for s in stages) + 32 * MiB + local_tensors
    bundle = boundary + local_weights + metadata
    client_stages = [*client_body, *([] if private_boundary else [token])]
    if not options.remote_output_head and not private_boundary:
        client_stages.append(head)
    # Raw paged import shares one executor and never creates a whole-matrix
    # Python array. Metadata is still owned by the decoded document and bundle;
    # include two copies, page indices/row bounds and bounded I/O/kernel tiles.
    paged_indices = sum(256 + 128 * math.ceil(s.in_features * s.out_features / MiB)
                        + 8 * max(s.in_features, s.out_features) for s in client_stages) if paged else 0
    bundle_work = 2 * metadata + paged_indices + 16 * MiB if paged else 6 * bundle
    weight_snapshot_disk = boundary + local_weights if paged else 0
    paged_disk = weight_snapshot_disk  # delivery and kernels retain the same immutable file
    if paged:
        from .bundle_storage import MAX_TOKENIZER_BYTES
        paged_disk += 2 * MAX_TOKENIZER_BYTES  # transient source and private tokenizer snapshot
    cpu_weights = weights  # NumPy stage views retain the same immutable Rust owner
    from .transformer_engine import WEIGHT_CHUNK_ELEMENTS
    # Whole-stage i8 staging and its immutable binding input are still charged.
    # Float conversion/quantization and metadata validation are chunk-bounded.
    chunk_elements = max(min(s.out_features, 64, max(1, WEIGHT_CHUNK_ELEMENTS // s.in_features))
                         * s.in_features for s in stages)
    quantization = 28 * chunk_elements
    validation = 4 * min(largest, max(WEIGHT_CHUNK_ELEMENTS, max(s.in_features for s in stages)))
    loading = max(quantization, validation, largest * 2)
    # A supplied Docker image can predate the single-snapshot loader. Keep its
    # legacy upper bound and hard limits; host-native savings cannot price it down.
    legacy_quantization = max(28 * (s.in_features * s.out_features
                             if s.in_features * s.out_features < 50_000_000
                             else min(64, s.out_features) * s.in_features) for s in stages)
    legacy_loading = max(legacy_quantization, 4 * largest, largest * 2)
    rows = max(inventory_rows, max_input_tokens + max_output_tokens - 1)
    legacy_masks = 16 * rows * sum(s.in_features + s.out_features for s in remote) if options.requires_preparation else 0
    # Only one stage's claimed mask rows are returned at a time. Price native
    # output buffers, Python conversion and the per-stage cursor/burn ledger.
    mask_cursors = (2048 + rows) * len(remote) if options.requires_preparation else 0
    masks = (16 * rows * max(
        (s.in_features + s.out_features for s in remote), default=0) + mask_cursors
        if options.requires_preparation else 0)
    corrections = 8 * rows * sum(s.out_features for s in remote) if options.requires_preparation else 0
    work = max((4096 + rows * (16 * s.in_features + 24 * s.out_features) for s in remote), default=0)
    window = pipeline.components.get("inventory")
    work *= window.params.get("stage_window", 1) if window is not None else 1
    # Epochs limit fresh inventory, never full-cap semantic/client state. Keep
    # the full inventory upper bound too until role lifetimes are independently
    # measured. In-flight responses may each own an inventory and active claim.
    concurrent_requests = options.inflight_max_requests if options.inflight else 1
    boundary_allocation = None
    if private_boundary:
        server = getattr(native, "PrivatePageServer", None)
        shared = server is not None and all(hasattr(server, name) for name in ("from_matrix", "bind_context"))
        boundary_allocation = private_boundary_memory(head.out_features, head.in_features,
            lookup_rows=compiled_input_bound + max_output_tokens - 1,
            head_calls=max_output_tokens, concurrent_requests=concurrent_requests,
            shared_matrix=shared,
            head_dispatch=pipeline.components["boundary"].params.get("head_dispatch", "serial"),
            head_output_encoding=pipeline.components["boundary"].params.get("head_output_encoding", "u32"),
            head_certificate=pipeline.components["boundary"].params.get("head_certificate"))
    masks *= concurrent_requests
    mask_cursors *= concurrent_requests
    legacy_masks *= concurrent_requests
    corrections *= concurrent_requests
    work *= concurrent_requests
    from .decoder_memory import decoder_memory
    workspace = decoder_memory(plan, plan.runtime_schedule(pipeline),
        initial_prefill_only=not (cache_bytes or options.prefix_cache_bytes))
    tensors = workspace["working_bytes"] + workspace["state_bytes"]
    tensors *= concurrent_requests
    # Shared pending owner is NOT the completed cache or any branch's live KV.
    # Charge owner plus two bounded snapshot/export copies during fork/import.
    inflight_owner = options.inflight_max_bytes if options.inflight else 0
    inflight_copy_work = 2 * inflight_owner
    # Caller-owned decode cache is additional to ordinary KV, prefix cache,
    # snapshots and response owners. Its bound covers transient/metadata work;
    # no projected hit licenses smaller full-cap decoder or correction bounds.
    decode_owner = options.decode_max_bytes if options.decode else 0
    # Serial dispatch owns one client chunk; overlap also owns one prefetched
    # chunk. A worker may own one chunk in each of
    # eight admitted sessions. Charge the native allocation ceiling, packet
    # copies and bounded HTTP buffers, rather than retaining a response's keys.
    nonlinear_client = nonlinear_worker = 0
    if options.nonlinear_layout is not None:
        from .masked_wire import MAX_FRAME

        nonlinear_client = (2 if options.nonlinear_dispatch == "overlap" else 1) * (64 * MiB + 6 * MAX_FRAME)
        nonlinear_worker = 8 * (64 * MiB + 3 * MAX_FRAME)
    cache_bytes = max(cache_bytes, options.prefix_cache_bytes)
    verification = legacy_verification = 0
    verification_allocation = None
    if options.verification_target_failure_bits:
        # Keep the historical provider upper bound for potentially older images.
        legacy_verification = 8 * rows * sum(s.in_features for s in remote) * (
            options.verification_target_failure_bits + (12 if cache_bytes else 0))
        verification_allocation = _freivalds_memory(native, remote, rows=rows,
            failure_bits=options.verification_target_failure_bits,
            stage_window=window.params.get("stage_window", 1) if window else 1)
        verification = verification_allocation["client_peak_bytes"]
    kernel = pipeline.components.get("kernels")
    metal = kernel is not None and kernel.component == "pllm/apple-metal-int8/v1"
    gpu_provider = sum(s.in_features * s.out_features for s in remote) if metal else 0
    gpu_client = local_weights if metal else 0
    # Client-only binds all stages in its local engine, including boundary matrices.
    if metal and not roles:
        gpu_client = weights
    limitations = []
    if private_boundary and not boundary_allocation["shared_immutable_matrix"]:
        limitations.append("private boundary requires native from_matrix and bind_context shared immutable storage")
    paged_preparation = options.preparation_storage == "paged"
    digit_workspace = 0
    if options.preparation_kernel == "digit8":
        # 64 MiB native tiles, binding/page/output overlap, and a conservative
        # 64 MiB allowance for the platform BLAS implementation per active stage.
        digit_workspace = (224 * MiB * concurrent_requests
                           * (window.params.get("stage_window", 1) if window else 1))
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            limitations.append("digit8 requires Apple Silicon CPU Accelerate")
    preparation_disk = sum(s.in_features * s.out_features for s in remote) if paged_preparation else 0
    if paged_preparation and any(s.in_features * s.out_features > 2 * GiB
                                 or max(s.in_features, s.out_features) > 1048576 for s in remote):
        limitations.append("paged preparation matrix exceeds native snapshot geometry bounds")
    if paged and any(s.in_features * s.out_features > 2 * GiB
                     or max(s.in_features, s.out_features) > 1048576 for s in client_stages):
        limitations.append("paged client matrix exceeds native snapshot geometry bounds")
    if metal:
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            limitations.append("Metal requires an Apple Silicon native host")
        if max(gpu_provider, gpu_client) > 2 * GiB:
            limitations.append("Metal stage snapshots exceed the existing 2 GiB per-role bound")
        if max((s.in_features * s.out_features for s in (*remote, *client_body)), default=0) > 512 * MiB:
            limitations.append("Metal stage exceeds its 512 MiB working-set bound")
    engine = cpu_weights + metadata + _PROCESS_BYTES
    role_peaks, docker_role_peaks = {}, {}
    role_weights, role_loading = {}, {}
    from .encoded_artifacts import WORK_BYTES as ARTIFACT_CACHE_WORK_BYTES
    artifact_cache_roles = {role: options.provider_artifact_cache_bytes for role in roles
                            if role not in {"preparation", "boundary_a", "boundary_b"}
                            and options.provider_artifact_cache_bytes}
    for role in roles:
        if role in {"boundary_a", "boundary_b"} and boundary_allocation is not None:
            role_weights[role] = boundary_allocation["worker_native_table_bytes"]
            role_loading[role] = boundary_allocation["worker_import_work_bytes"]
            role_peaks[role] = _slack(boundary_allocation["worker_peak_bytes"], memory_safety_margin_percent)
            # Unknown/stale image imports may retain full original source engine.
            docker_role_peaks[role] = _slack(boundary_allocation["worker_peak_bytes"] + engine + legacy_loading,
                                           memory_safety_margin_percent)
            continue
        # Preparation never exports a client bundle; inference and either offset
        # worker may. Include retained corrections while preparing new material.
        # Native delivery streams immutable weight views and retains only copied
        # metadata plus bounded compression/batch frames. Existing images may
        # still pack/unpack full bundles and cache copied artifact objects.
        delivery = 2 * metadata + 8 * MiB if role != "preparation" else 0
        if private_boundary and role == "inference":
            delivery += (boundary_allocation["publisher_table_export_bytes"]
                         + boundary_allocation["publisher_descriptor_work_bytes"])
        cache_work = ARTIFACT_CACHE_WORK_BYTES if role in artifact_cache_roles else 0
        legacy_delivery = 5 * bundle if role != "preparation" else 0
        retained = {s.id for s in remote} if role == "preparation" else {
            s.id for s in stages if not (
                shared_main_table and token.out_features == head.in_features and s.id == token.id)}
        role_weights[role] = sum(s.in_features * s.out_features for s in stages if s.id in retained)
        # Omitted weights are still imported and validated. Charge their full
        # i8 staging array alongside bounded float/metadata work, even though
        # no native snapshot will be allocated or retained for them.
        role_loading[role] = max(
            max(quantization, validation, 2 * s.in_features * s.out_features) if s.id in retained
            else s.in_features * s.out_features + max(quantization, validation) for s in stages)
        role_engine = role_weights[role] + metadata + _PROCESS_BYTES + nonlinear_worker + cache_work
        if role == "preparation" and paged_preparation:
            page_indices = sum(256 + 128 * math.ceil(s.in_features * s.out_features / MiB)
                               + 8 * max(s.in_features, s.out_features) for s in remote)
            active_pages = (16 * MiB * concurrent_requests
                            * (window.params.get("stage_window", 1) if window else 1))
            role_weights[role] = 0
            role_engine = metadata + page_indices + active_pages + _PROCESS_BYTES
        role_verification = (verification_allocation["preparation_peak_bytes"]
                              if verification_allocation and role == "preparation" else 0)
        if role == "preparation":
            role_engine += digit_workspace
        role_peaks[role] = _slack(role_engine + gpu_provider + max(
            role_loading[role], delivery + corrections, work + corrections + role_verification),
            memory_safety_margin_percent)
        docker_role_peaks[role] = _slack(engine + weights + gpu_provider + nonlinear_worker + cache_work + max(
            legacy_loading, legacy_delivery + corrections, work + corrections + legacy_verification),
            memory_safety_margin_percent)
    tokenizer = pipeline.components.get("tokenizer")
    capsule = pipeline.components.get("public_prefix")
    # Public artifacts are pre-positioned. Price bounded index query caches,
    # authenticated private DB snapshot disk, and capsule parse/copy overlap.
    public_artifact_work = (32 * MiB if tokenizer else 0) + (4 * capsule.params["size_bytes"] if capsule else 0)
    client = _PROCESS_BYTES + bundle_work + masks + verification + tensors + cache_bytes + gpu_client + public_artifact_work + nonlinear_client
    client += inflight_owner + inflight_copy_work
    client += decode_owner
    if boundary_allocation is not None:
        client += boundary_allocation["client_work_bytes"]
    if not roles:
        client += engine + loading
    client_peak = _slack(client, memory_safety_margin_percent)
    return {
        "scope": "conservative allocation estimate; not measured peak RSS",
        "model_plan_digest": plan.digest, "pipeline_digest": pipeline.digest(),
        "max_input_tokens": max_input_tokens, "max_output_tokens": max_output_tokens,
        "compiled_input_bound": compiled_input_bound,
        "inventory_rows": rows, "kernel": "metal" if metal else "cpu",
        "preparation_decode_epoch_tokens": options.preparation_decode_epoch_tokens,
        **({"preparation_decode_epoch_schedule": options.preparation_decode_epoch_schedule}
           if options.preparation_decode_epoch_schedule is not None else {}),
        **({"decode": True, "decode_max_bytes": options.decode_max_bytes} if options.decode else {}),
        "concurrent_response_bound": concurrent_requests,
        "allocation_slack_percent": memory_safety_margin_percent, "client_peak_bytes": client_peak,
        "provider_peak_bytes": role_peaks,
        "docker_provider_peak_bytes": docker_role_peaks,
        "weight_storage": "native_snapshot_v1; Docker retains legacy allocation upper bounds",
        "preparation_storage": options.preparation_storage,
        "preparation_kernel": options.preparation_kernel,
        "bundle_storage": ("streamed_paged_v2; shared native weight snapshots; raw cache remains a separate disk owner" if paged else
                           "immutable_segments_v1; client import retains conservative copy bounds"),
        "mask_storage": "one_active_stage_v1; one benchmark response at a time",
        "verification_allocation": verification_allocation,
        **({"private_boundary_allocation": boundary_allocation} if private_boundary else {}),
        "tensor_storage": workspace["mode"] + "; completed benchmark response histories are retired",
        "native_total_peak_bytes": client_peak + sum(role_peaks.values()),
        "components": {"per_engine_i8_and_native_bytes": cpu_weights,
            "per_role_i8_and_native_bytes": role_weights,
            "per_role_loading_temporary_bytes": role_loading,
            "legacy_per_engine_i8_and_native_bytes": 2 * weights,
            "float_quantization_work_bytes": quantization,
            "legacy_largest_loading_temporary_bytes": legacy_loading,
            "largest_loading_temporary_bytes": loading, "raw_client_bundle_bytes": bundle,
            "client_bundle_work_bytes": bundle_work, "client_paged_metadata_bytes": paged_indices,
            "client_additional_paged_disk_bytes": paged_disk,
            "client_weight_snapshot_disk_bytes": weight_snapshot_disk,
            "client_public_artifact_work_bytes": public_artifact_work,
            "client_tokenizer_snapshot_disk_bytes": 64 * MiB if tokenizer else 0,
            "preparation_additional_paged_disk_bytes": preparation_disk + (largest if paged_preparation else 0),
             "client_mask_bytes": masks, "provider_correction_bytes": corrections,
             "client_nonlinear_work_bytes": nonlinear_client,
             "per_provider_nonlinear_work_bytes": nonlinear_worker,
             "provider_artifact_cache_disk_bytes": artifact_cache_roles,
             "preparation_digit_workspace_bytes": digit_workspace,
             "provider_artifact_cache_work_bytes": {role: ARTIFACT_CACHE_WORK_BYTES for role in artifact_cache_roles},
            "legacy_client_mask_bytes": legacy_masks, "client_mask_cursor_bytes": mask_cursors,
            "verifier_bytes": verification, "legacy_verifier_bytes": legacy_verification,
            "client_tensor_work_bytes": tensors,
             "client_live_tensor_bytes": concurrent_requests * workspace["working_bytes"],
             "client_rotary_coefficient_bytes": concurrent_requests * workspace["rotary_coefficient_bytes"],
             "client_state_work_bytes": concurrent_requests * workspace["state_bytes"],
             "legacy_client_tensor_work_bytes": concurrent_requests * workspace["legacy_working_bytes"],
             "client_cache_bytes": cache_bytes, "per_provider_metal_bytes": gpu_provider,
             "client_inflight_owner_bytes": inflight_owner,
              "client_inflight_copy_work_bytes": inflight_copy_work,
              **({"client_decode_owner_bytes": decode_owner} if options.decode else {}),
            "client_metal_bytes": gpu_client, "checkpoint_bytes": checkpoint_bytes,
            "per_engine_compiled_disk_bytes": sum(s.in_features * s.out_features
                for s in stages if s.in_features * s.out_features >= 50_000_000)},
        "limitations": limitations,
    }


def admit_memory(estimate, host: HostMemory, *, backend="native", docker_capacity=None,
                 enforced_wan=False, memory_budget_bytes=None):
    if backend not in {"native", "docker", "auto"}:
        raise ValueError("benchmark backend must be native, docker or auto")
    memory_budget_bytes = _memory_budget(memory_budget_bytes)
    budget = max(0, host.available - host.reserve)
    if memory_budget_bytes is not None:
        budget = min(budget, memory_budget_bytes)  # explicit budget can only tighten admission
    margin_percent = _memory_margin(estimate["allocation_slack_percent"])

    def before_margin(peak):
        # Each role was rounded separately. Invert that rounding per owner;
        # never treat a whole Docker VM's fixed capacity as a margined estimate.
        return peak * 100 // (100 + margin_percent)

    candidates = {}
    for kind in ("native", "docker"):
        reasons = list(estimate["limitations"])
        if host.pressure in {"warning", "critical"}:
            reasons.append(f"macOS memory pressure is {host.pressure}")
        required = estimate["native_total_peak_bytes"]
        expected = (before_margin(estimate["client_peak_bytes"])
                    + sum(before_margin(n) for n in estimate["provider_peak_bytes"].values()))
        domain = None
        if kind == "native":
            if enforced_wan:
                if backend == "native":
                    required += _slack(32 * MiB, margin_percent)  # relay buffers/control thread
                    expected += 32 * MiB
                else:
                    reasons.append("enforced WAN auto fallback requires explicit native TCP pacing selection")
        else:
            if estimate.get("runtime_kind") in {"gkm", "replicated", "fss"}:
                reasons.append("selected runtime requires native role owners; Docker is not admitted")
            if estimate.get("preparation_kernel") == "digit8":
                reasons.append("digit8 requires native Apple Silicon CPU execution")
            if estimate["kernel"] == "metal":
                reasons.append("Metal requires native provider processes")
            if docker_capacity is None:
                reasons.append("Docker memory capacity is unavailable")
            else:
                total, used = docker_capacity
                reserve = max(GiB, total // 8)
                domain = {"total_bytes": total, "other_container_bytes": used, "reserve_bytes": reserve,
                          "required_bytes": sum(estimate["docker_provider_peak_bytes"].values())
                                            + _slack(_HELPER_BYTES, margin_percent)}
                if domain["required_bytes"] > total - used - reserve:
                    reasons.append("provider peaks and helpers exceed existing Docker memory headroom")
                # On Desktop the *whole* existing VM can compete with the host
                # client. Never infer safety from a VM's currently idle RSS.
                if platform.system() != "Linux":
                    required = total + estimate["client_peak_bytes"]
                    expected = total + before_margin(estimate["client_peak_bytes"])
                else:
                    required = estimate["client_peak_bytes"] + domain["required_bytes"]
                    expected = (before_margin(estimate["client_peak_bytes"]) + _HELPER_BYTES
                                + sum(before_margin(n) for n in estimate["docker_provider_peak_bytes"].values()))
        if required > budget:
            reasons.append(f"estimated host peak {required / GiB:.2f} GiB exceeds safe headroom {budget / GiB:.2f} GiB")
        # Budget cold compiled-cache files and raw/reconstructed client bundles;
        # cached objects are not assumed resident without verified cache evidence.
        disk_required = (estimate["components"]["per_engine_compiled_disk_bytes"]
                         * max(1, len(estimate["provider_peak_bytes"]))
                         + 2 * estimate["components"]["raw_client_bundle_bytes"] + 2 * GiB
                         + estimate["components"].get("client_additional_paged_disk_bytes", 0))
        disk_required += estimate["components"].get("preparation_additional_paged_disk_bytes", 0)
        disk_required += estimate["components"].get("client_tokenizer_snapshot_disk_bytes", 0)
        disk_required += sum(estimate["components"].get("provider_artifact_cache_disk_bytes", {}).values())
        disk_required += estimate["components"].get("fss_tape_disk_bytes", 0)
        if host.disk_free < disk_required:
            reasons.append("insufficient disk headroom for compiled caches, bundles and 2 GiB reserve")
        candidates[kind] = {"admitted": not reasons, "reasons": reasons,
                            "required_host_bytes": required, "docker": domain,
                            "expected_peak_bytes": expected, "safety_margin_bytes": required - expected,
                            "required_disk_bytes": disk_required}
    preference = ("native",) if backend == "native" else ("docker",) if backend == "docker" else (
        ("native", "docker") if estimate["kernel"] == "metal" else ("docker", "native"))
    selected = next((name for name in preference if candidates[name]["admitted"]), None)
    runtime_floor = (candidates[selected or preference[0]]["safety_margin_bytes"]
                     if host.availability_source in ("macos_mach_dispatch_v1", "macos_mach_dispatch_v2")
                     else max(GiB, host.reserve // 2))
    return {"schema": "pllm.benchmark_memory.v1", "admitted": selected is not None,
            "requested_backend": backend, "selected_backend": selected,
            "runtime_floor_bytes": runtime_floor,
            "host": asdict(host) | {"reserve_bytes": host.reserve, "admission_budget_bytes": budget},
            "estimate": estimate, "candidates": candidates, "enforced_wan": enforced_wan}


def benchmark_memory(model, *, max_input_tokens=None, max_output_tokens=None, inventory_rows=None,
                     cache_bytes=0, cache_bound_tokens=None, backend="native", enforced_wan=False,
                     memory_budget_bytes=None, memory_safety_margin_percent=25, fss_imported=None):
    """Inspect without importing tensors or starting roles.

    An optional live LoadedFssSource supplies native program pricing. The caller
    owns its prior import reservation, setup measurements and lifetime; a JSON
    price cannot replace it. Omitting it preserves metadata-only preflight.
    """
    from pllm.configuration import Experiment, Model, Pipeline
    from pllm.profiles import MaskedLinearCpu, resolve_runtime_composition

    _memory_margin(memory_safety_margin_percent)
    initial = host_memory()
    if initial.pressure in {"warning", "critical"}:
        raise BenchmarkMemoryError(f"macOS memory pressure is {initial.pressure} before model inspection")
    if initial.available <= initial.reserve:
        raise BenchmarkMemoryError("host has no safe memory headroom before model inspection")
    if isinstance(model, Experiment):
        model.resolve()
        pipeline = model.pipeline
        if max_input_tokens is None:
            max_input_tokens = model.budget.max_input_tokens
        if max_output_tokens is None:
            max_output_tokens = model.budget.max_new_tokens
    elif isinstance(model, Pipeline):
        pipeline = model
    else:
        pipeline = MaskedLinearCpu(model if isinstance(model, Model) else Model(model))
    max_input_tokens = 256 if max_input_tokens is None else max_input_tokens
    max_output_tokens = 24 if max_output_tokens is None else max_output_tokens
    options = resolve_runtime_composition(pipeline)
    if options is None:
        raise BenchmarkMemoryError("no allocation model for the selected runtime composition")
    if fss_imported is not None and options.client_runtime != "compiled_fss_v1":
        raise BenchmarkMemoryError("imported FuseFSS pricing requires its complete Experiment")
    if inventory_rows is None:
        inventory_rows = options.prepared_inventory_rows
    source = pipeline.model
    checkpoint_bytes = 0
    store = None
    if options.client_runtime == "compiled_fss_v1":
        if not isinstance(model, Experiment):
            raise BenchmarkMemoryError("FuseFSS source preflight requires complete Experiment")
        if max_input_tokens > model.budget.max_input_tokens or max_output_tokens > model.budget.max_new_tokens:
            raise BenchmarkMemoryError("FuseFSS measurement exceeds saved workload envelope")
        from .fss_source import LoadedFssSource, resolve_client_source
        pricing = program_resources = None
        try:
            if fss_imported is None:
                admitted = resolve_client_source(model)
            else:
                if type(fss_imported) is not LoadedFssSource or fss_imported._closed:
                    raise ValueError("FuseFSS pricing requires a live imported source owner")
                admitted = fss_imported.source
                # Reuse the frozen source's Experiment, file-identity and full
                # saved-envelope checks. A smaller prompt's price is not enough.
                raw = admitted.runtime_contract(model, model.budget.max_input_tokens,
                                                 model.budget.max_new_tokens)
                if raw != admitted.contract_bytes or raw != fss_imported.contract_bytes:
                    raise ValueError("FuseFSS imported pricing differs from saved source envelope")
                from .benchmark_delegated import inspect_fss_imported
                pricing = inspect_fss_imported(fss_imported)
                program_resources = json.loads(fss_imported.native_program.metadata())
                if (program_resources.get("schema") != "pllm.fss.program_resources.v1"
                        or any(program_resources.get(key) != pricing.get(field) for key, field in (
                            ("program_digest", "program_digest"), ("source_digest", "source_digest"),
                            ("original_input_digest", "input_digest"), ("numeric_profile", "numeric_profile"),
                            ("initial_tokens", "initial_tokens"), ("selections", "selections"),
                            ("vocabulary", "vocabulary")))):
                    raise ValueError("FuseFSS native program resource identity differs")
        except ValueError as error:
            scope = ("source metadata preflight rejected before model loading" if fss_imported is None
                     else "imported source pricing rejected")
            raise BenchmarkMemoryError(f"FuseFSS {scope}") from error
        contract = admitted.contract
        estimate = _fss_memory(json.loads(admitted.config_bytes), pipeline,
            contract["input_rows"], contract["output_tokens"], cache_bytes, cache_bound_tokens,
            contract["source_float32_bytes"], memory_safety_margin_percent,
            contract=contract, source_pins=admitted.snapshot_pins, transport_admission=pricing)
        if program_resources is not None:
            estimate["fss_admission"].update(metadata_only=False,
                numeric_admission="native_imported_program", complete_program_resources=program_resources,
                pricing_scope="saved_full_envelope; not_request_admission",
                qualification_owner_lifetime="caller_owned; retire_before_full_topology_source_imports")
        return admit_memory(estimate, host_memory(), backend=backend, enforced_wan=enforced_wan,
                            memory_budget_bytes=memory_budget_bytes)
    if options.client_runtime in {"compiled_gkm_v1", "compiled_replicated_v1"}:
        if options.client_runtime == "compiled_replicated_v1":
            from .replicated_source import admit_source
            if not isinstance(model, Experiment):
                raise BenchmarkMemoryError("replicated memory admission requires its complete Experiment")
        else:
            from .delegated_source import admit_source

        if isinstance(model, Experiment):
            admitted = admit_source(model)
            if (max_input_tokens > model.budget.max_input_tokens
                    or max_output_tokens > model.budget.max_new_tokens):
                raise BenchmarkMemoryError("GKM measurement exceeds its saved workload envelope")
        else:
            admitted = admit_source(source, input_rows=max_input_tokens,
                                    decode_steps=max_output_tokens - 1)
        config = admitted.config
        checkpoint_bytes = sum(row["size"] for row in admitted.source_lock["files"]
                               if row["path"].endswith(".safetensors"))
        # Runtime owners are bound to the saved full envelope, even for a
        # smaller measurement. Price the contract those owners will allocate.
        contract = admitted.contract
        max_input_tokens = contract["input_rows"]
        max_output_tokens = contract["decode_steps"] + 1
    elif source.kind == "tiny":
        from pllm.sources import _tiny_model_config
        config = _tiny_model_config()
    else:
        from pllm.model_loader import resolve_model
        resolved = resolve_model(source)
        if resolved.path is None:
            raise BenchmarkMemoryError("memory admission requires a checkpoint configuration")
        path = Path(resolved.path)
        config_path = path / "config.json"
        if config_path.stat().st_size > 2 * MiB:
            raise BenchmarkMemoryError("checkpoint configuration exceeds inspection bound")
        config = json.loads(config_path.read_text())
        from .safetensors_store import SafeTensorStore
        store = SafeTensorStore(path)
        # Header validation below checks required shapes and dtypes. Counting
        # source bytes does not read tensor values into model arrays.
        checkpoint_bytes = sum(p.stat().st_size for p in path.glob("*.safetensors"))
    estimate = estimate_memory(config, pipeline, max_input_tokens=max_input_tokens,
                                max_output_tokens=max_output_tokens, inventory_rows=inventory_rows,
                                cache_bytes=cache_bytes, cache_bound_tokens=cache_bound_tokens,
                                checkpoint_bytes=checkpoint_bytes, store=store,
                                memory_safety_margin_percent=memory_safety_margin_percent)
    if options.client_runtime == "compiled_gkm_v1":
        # Keep the saved full-envelope admission explicit even when a benchmark
        # requests fewer rows/outputs. These pins come from bounded source reads.
        estimate["gkm_admission"] = {
            "checkpoint_digest": admitted.checkpoint_digest,
            "source_lock_digest": admitted.source_lock_digest,
            "tokenizer_digest": admitted.tokenizer_digest,
            "decoder_contract_digest": admitted.contract_digest,
            "runtime_contract_digest": (admitted.runtime_contract_digest
                                        if admitted.runtime_contract_bytes is not None else None),
            "numeric_profile": contract["numeric_profile"],
            "local_numeric_profile": contract["local_numeric_profile"],
            "max_input_tokens": contract["input_rows"],
            "max_output_tokens": contract["decode_steps"] + 1,
            "execution_payload_estimate_bytes": admitted.execution_payload_estimate_bytes,
        }
    docker = None
    if backend in {"docker", "auto"}:
        try:
            docker = _docker_capacity()
        except BenchmarkMemoryError:
            pass
    return admit_memory(estimate, host_memory(), backend=backend, docker_capacity=docker,
                        enforced_wan=enforced_wan, memory_budget_bytes=memory_budget_bytes)


def _fss_source_qualification_estimate(estimate):
    """Single coordinator source owner on disk; unchanged whole-topology memory.

    Only the decoder execution path uses this phase. Transport is not yet
    priced, no role may start from this estimate, and the final native program
    estimate must replace it before allocating the complete graph.
    """
    if (estimate.get("runtime_kind") != "fss"
            or not estimate["incremental_owner_bounds"]["source_metadata_preflight"]
            or not estimate["fss_admission"]["metadata_only"]):
        raise BenchmarkMemoryError("source qualification requires decoder metadata admission")
    value = deepcopy(estimate)
    original = estimate["components"]
    source_disk = original["per_engine_compiled_disk_bytes"]
    value["components"] = {
        "checkpoint_bytes": original["checkpoint_bytes"],
        "source_float32_bytes": original["source_float32_bytes"],
        "per_engine_compiled_disk_bytes": 0, "raw_client_bundle_bytes": 0,
        "client_additional_paged_disk_bytes": source_disk,
        "source_qualification_disk_bytes": source_disk,
    }
    value["fss_admission"].update(disk_price_scope="single_source_qualification_only; graph_transport_unpriced",
        full_graph_disk_admitted=False, qualification_required_before_children=True)
    return value


def execution_memory_admission(model, *, max_input_tokens=None, max_output_tokens=None, inventory_rows=None,
                               cache_bytes=0, cache_bound_tokens=None, backend="native", enforced_wan=False,
                               memory_budget_bytes=None, memory_safety_margin_percent=25):
    """Admission for the next execution phase; never imports tensor values.

    Decoder FuseFSS first admits one source-qualification owner on disk while
    retaining the full graph's conservative memory estimate and watchdog floor.
    LocalTopology must qualify and admit the complete graph before any child.
    Other runtimes and the capsule retain ordinary benchmark_memory behavior.
    """
    from pllm.configuration import Experiment
    from pllm.metrics import benchmark_memory as inspect_memory
    from .benchmark_delegated import is_fss

    report = inspect_memory(model, max_input_tokens=max_input_tokens, max_output_tokens=max_output_tokens,
        inventory_rows=inventory_rows, cache_bytes=cache_bytes, cache_bound_tokens=cache_bound_tokens,
        backend=backend, enforced_wan=enforced_wan, memory_budget_bytes=memory_budget_bytes,
        memory_safety_margin_percent=memory_safety_margin_percent)
    if (not isinstance(model, Experiment)
            or model.pipeline.components.get("linear") is None
            or model.pipeline.components["linear"].params.get("capacity", "capsule") != "decoder"
            or not is_fss(model)):
        return report
    estimate = _fss_source_qualification_estimate(report["estimate"])
    report = admit_memory(estimate, host_memory(), backend=backend, enforced_wan=enforced_wan,
                          memory_budget_bytes=memory_budget_bytes)
    report.update(admission_scope="source_qualification_only", full_topology_admitted=False)
    return report


def require_admission(report):
    if not report["admitted"]:
        backends = (report["requested_backend"],) if report["requested_backend"] != "auto" else ("docker", "native")
        detail = "; ".join(f"{name}: {', '.join(report['candidates'][name]['reasons'])}" for name in backends)
        raise BenchmarkMemoryError("benchmark memory preflight rejected: " + detail, report)


class MemoryWatchdog:
    """Abort owned work when host pressure changes after admission."""
    def __init__(self, report, abort: Callable[[str], None], *, sample=None):
        self.report, self.abort, self.sample = report, abort, sample or host_memory
        self.error = None
        self.minimum_available_bytes = report["host"]["available"]
        self._minimum_swap = report["host"]["swap_used"]
        self.maximum_swap_growth_bytes = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="pllm-memory-guard", daemon=True)

    def check(self):
        current = self.sample()
        initial = self.report["host"]
        self.minimum_available_bytes = min(self.minimum_available_bytes, current.available)
        self._minimum_swap = min(self._minimum_swap, current.swap_used)
        self.maximum_swap_growth_bytes = max(self.maximum_swap_growth_bytes, current.swap_used - self._minimum_swap)
        if current.pressure in {"warning", "critical"}:
            raise BenchmarkMemoryError(f"benchmark stopped: macOS memory pressure is {current.pressure}")
        # macOS keeps the workload-sized margin; other platforms retain the
        # existing runtime reserve. Swap never supplies capacity.
        floor = self.report.get("runtime_floor_bytes", max(GiB, initial["reserve_bytes"] // 2))
        if current.available < floor:
            raise BenchmarkMemoryError("benchmark stopped: host memory fell below runtime reserve")
        if current.swap_used > self._minimum_swap + 256 * MiB:
            raise BenchmarkMemoryError("benchmark stopped: host swap grew by more than 256 MiB")

    def _run(self):
        while not self._stop.wait(0.25):
            try:
                self.check()
            except Exception as exc:
                self.error = str(exc)
                self.abort(self.error)
                return

    def start(self):
        self.check()
        self._thread.start()

    def close(self):
        self._stop.set()
        if self._thread.ident is not None and threading.current_thread() is not self._thread:
            self._thread.join(timeout=2)
