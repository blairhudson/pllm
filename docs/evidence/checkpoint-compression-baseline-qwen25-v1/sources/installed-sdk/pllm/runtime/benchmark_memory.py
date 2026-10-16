"""Whole-topology benchmark memory admission, before loading tensor values.

These are conservative allocation estimates, not measured peak RSS or a proof
against OOM. Runtime pressure monitoring and Docker hard limits supplement them.
Swap never increases the admission budget. Backend selection never changes a
Pipeline, privacy topology, kernel, or an enforced-WAN measurement into an estimate.
"""
from __future__ import annotations

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

    def __post_init__(self):
        if (any(type(value) is not int or value < 0 for value in asdict(self).values())
                or not 0 < self.available <= self.total):
            raise BenchmarkMemoryError("host memory telemetry is unavailable or invalid")

    @property
    def reserve(self) -> int:
        return host_reserve_bytes(self.total)


def host_memory() -> HostMemory:
    import psutil

    try:
        memory = psutil.virtual_memory()
        return HostMemory(int(memory.total), int(memory.available),
                          int(psutil.swap_memory().used), int(shutil.disk_usage(Path.home()).free))
    except (OSError, ValueError, psutil.Error) as exc:
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


def _slack(value: int) -> int:
    return (value * 5 + 3) // 4


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
                    inventory_rows=64, cache_bytes=0, cache_bound_tokens=None, checkpoint_bytes=0, store=None):
    """Model-neutral graph geometry; no arrays or checkpoint tensor reads."""
    from pllm.modeling import lower_model
    from pllm.profiles import resolve_runtime_composition

    _validate_workload(max_input_tokens, max_output_tokens, inventory_rows, cache_bytes, cache_bound_tokens)
    options = resolve_runtime_composition(pipeline)
    if options is None:
        raise BenchmarkMemoryError("memory estimation requires an implemented composition")
    compiled_input_bound = max(max_input_tokens, cache_bound_tokens or 0,
                               options.prefix_cache_bound_tokens or 0)
    plan = lower_model(config, batch=1, max_input_tokens=compiled_input_bound,
                       max_new_tokens=max_output_tokens)
    return estimate_plan_memory(plan, pipeline, max_input_tokens=max_input_tokens,
        max_output_tokens=max_output_tokens, inventory_rows=inventory_rows, cache_bytes=cache_bytes,
        cache_bound_tokens=cache_bound_tokens, checkpoint_bytes=checkpoint_bytes, store=store)


def estimate_plan_memory(plan, pipeline, *, max_input_tokens, max_output_tokens,
                         inventory_rows=64, cache_bytes=0, cache_bound_tokens=None,
                         checkpoint_bytes=0, store=None):
    """Shared native allocation-owner estimate from an admitted semantic plan."""
    from pllm.profiles import resolve_runtime_composition
    from pllm.roles.topology import graph_for_runtime
    from .semantic_stages import client_owns_linear, provider_owns_linear, scheduled_stage_specs
    from ._native_support import extension

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
            role_peaks[role] = _slack(boundary_allocation["worker_peak_bytes"])
            # Unknown/stale image imports may retain full original source engine.
            docker_role_peaks[role] = _slack(boundary_allocation["worker_peak_bytes"] + engine + legacy_loading)
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
            role_loading[role], delivery + corrections, work + corrections + role_verification))
        docker_role_peaks[role] = _slack(engine + weights + gpu_provider + nonlinear_worker + cache_work + max(
            legacy_loading, legacy_delivery + corrections, work + corrections + legacy_verification))
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
    client_peak = _slack(client)
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
        "allocation_slack_percent": 25, "client_peak_bytes": client_peak,
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
    if memory_budget_bytes is not None and (type(memory_budget_bytes) is not int or memory_budget_bytes <= 0):
        raise ValueError("memory budget must be a positive byte count")
    budget = max(0, host.available - host.reserve)
    if memory_budget_bytes is not None:
        budget = min(budget, memory_budget_bytes)  # explicit budget can only tighten admission
    candidates = {}
    for kind in ("native", "docker"):
        reasons = list(estimate["limitations"])
        required = estimate["native_total_peak_bytes"]
        domain = None
        if kind == "native":
            if enforced_wan:
                if backend == "native":
                    required += 32 * MiB  # bounded TCP relay buffers and control thread
                else:
                    reasons.append("enforced WAN auto fallback requires explicit native TCP pacing selection")
        else:
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
                           "required_bytes": sum(estimate["docker_provider_peak_bytes"].values()) + _HELPER_BYTES}
                if domain["required_bytes"] > total - used - reserve:
                    reasons.append("provider peaks and helpers exceed existing Docker memory headroom")
                # On Desktop the *whole* existing VM can compete with the host
                # client. Never infer safety from a VM's currently idle RSS.
                required = (total + estimate["client_peak_bytes"] if platform.system() != "Linux"
                            else estimate["client_peak_bytes"]
                                 + sum(estimate["docker_provider_peak_bytes"].values()) + _HELPER_BYTES)
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
        if host.disk_free < disk_required:
            reasons.append("insufficient disk headroom for compiled caches, bundles and 2 GiB reserve")
        candidates[kind] = {"admitted": not reasons, "reasons": reasons,
                            "required_host_bytes": required, "docker": domain,
                            "required_disk_bytes": disk_required}
    preference = ("native",) if backend == "native" else ("docker",) if backend == "docker" else (
        ("native", "docker") if estimate["kernel"] == "metal" else ("docker", "native"))
    selected = next((name for name in preference if candidates[name]["admitted"]), None)
    return {"schema": "pllm.benchmark_memory.v1", "admitted": selected is not None,
            "requested_backend": backend, "selected_backend": selected,
            "host": asdict(host) | {"reserve_bytes": host.reserve, "admission_budget_bytes": budget},
            "estimate": estimate, "candidates": candidates, "enforced_wan": enforced_wan}


def benchmark_memory(model, *, max_input_tokens=None, max_output_tokens=None, inventory_rows=None,
                     cache_bytes=0, cache_bound_tokens=None, backend="native", enforced_wan=False,
                     memory_budget_bytes=None):
    """Inspect an Experiment/Model without loading tensor values or starting roles."""
    from pllm.configuration import Experiment, Model, Pipeline
    from pllm.profiles import MaskedLinearCpu, resolve_runtime_composition

    initial = host_memory()
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
    if inventory_rows is None:
        inventory_rows = options.prepared_inventory_rows
    source = pipeline.model
    checkpoint_bytes = 0
    store = None
    if source.kind == "tiny":
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
                                checkpoint_bytes=checkpoint_bytes, store=store)
    docker = None
    if backend in {"docker", "auto"}:
        try:
            docker = _docker_capacity()
        except BenchmarkMemoryError:
            pass
    return admit_memory(estimate, host_memory(), backend=backend, docker_capacity=docker,
                        enforced_wan=enforced_wan, memory_budget_bytes=memory_budget_bytes)


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
        # Keep half the preflight OS/user reserve untouched at runtime. Abort on
        # fresh swap growth too; pre-existing swapped-out pages are not capacity.
        floor = max(GiB, initial["reserve_bytes"] // 2)
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
