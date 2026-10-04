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
        return max(2 * GiB, self.total // 4)


def host_memory() -> HostMemory:
    import psutil

    try:
        memory = psutil.virtual_memory()
        return HostMemory(int(memory.total), int(memory.available),
                          int(psutil.swap_memory().used), int(shutil.disk_usage(Path.home()).free))
    except (OSError, ValueError, psutil.Error) as exc:
        raise BenchmarkMemoryError("cannot sample host memory for benchmark admission") from exc


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


def estimate_memory(config, pipeline, *, max_input_tokens, max_output_tokens,
                    inventory_rows=64, cache_bytes=0, cache_bound_tokens=None, checkpoint_bytes=0, store=None):
    """Model-neutral graph geometry; no arrays or checkpoint tensor reads."""
    from pllm.modeling import lower_model
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
    if not snapshot_storage:
        raise BenchmarkMemoryError("memory admission requires the installed native snapshot storage backend")

    for value in (max_input_tokens, max_output_tokens, inventory_rows):
        if type(value) is not int or not 1 <= value <= 32768:
            raise BenchmarkMemoryError("memory admission requires bounded positive workload counts")
    if type(cache_bytes) is not int or cache_bytes < 0:
        raise BenchmarkMemoryError("invalid client cache memory bound")
    if cache_bound_tokens is not None and (
        type(cache_bound_tokens) is not int or not 1 <= cache_bound_tokens <= 32768
    ):
        raise BenchmarkMemoryError("invalid client cache input bound")
    options = resolve_runtime_composition(pipeline)
    if options is None or options.privacy_mode not in {"public", "offset_public", "client_only"}:
        raise BenchmarkMemoryError("memory estimation requires an implemented public compiled topology")
    compiled_input_bound = max(max_input_tokens, cache_bound_tokens or 0,
                               options.prefix_cache_bound_tokens or 0)
    plan = lower_model(config, batch=1, max_input_tokens=compiled_input_bound,
                       max_new_tokens=max_output_tokens)
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
                     client_linear_roles=options.client_linear_roles)
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
    raw_config = config.get("text_config", config)
    if raw_config.get("tie_word_embeddings"):
        boundary -= next(s.in_features * s.out_features for s in stages if s.role == "lm_head")
    # Descriptors include scales for every stage, not only client-owned weights.
    metadata = 16 * sum(s.out_features + s.in_features for s in stages) + 32 * MiB + local_tensors
    bundle = boundary + local_weights + metadata
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
    masks = 16 * rows * sum(s.in_features + s.out_features for s in remote) if options.requires_preparation else 0
    corrections = 8 * rows * sum(s.out_features for s in remote) if options.requires_preparation else 0
    work = max((4096 + rows * (16 * s.in_features + 24 * s.out_features) for s in remote), default=0)
    window = pipeline.components.get("inventory")
    work *= window.params.get("stage_window", 1) if window is not None else 1
    # Summing graph tensors intentionally prices more than a live DAG traversal.
    # Includes full-KV, sliding and recurrent state-bearing graph outputs.
    tensors = 8 * sum(math.prod(op["output_shape"]) for graph in (plan.prefill, plan.decode)
                      for op in graph["operations"])
    cache_bytes = max(cache_bytes, options.prefix_cache_bytes)
    verification = 0
    if options.verification_target_failure_bits:
        # Backend has at most 80 failure bits. Charge one projection round per
        # requested bit plus the cache union-budget uplift (deliberate overestimate).
        verification = 8 * rows * sum(s.in_features for s in remote) * (
            options.verification_target_failure_bits + (12 if cache_bytes else 0))
    kernel = pipeline.components.get("kernels")
    metal = kernel is not None and kernel.component == "pllm/apple-metal-int8/v1"
    gpu_provider = sum(s.in_features * s.out_features for s in remote) if metal else 0
    gpu_client = local_weights if metal else 0
    # Client-only binds all stages in its local engine, including boundary matrices.
    if metal and not roles:
        gpu_client = weights
    limitations = []
    if metal:
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            limitations.append("Metal requires an Apple Silicon native host")
        if max(gpu_provider, gpu_client) > 2 * GiB:
            limitations.append("Metal stage snapshots exceed the existing 2 GiB per-role bound")
        if max((s.in_features * s.out_features for s in (*remote, *client_body)), default=0) > 512 * MiB:
            limitations.append("Metal stage exceeds its 512 MiB working-set bound")
    engine = cpu_weights + metadata + _PROCESS_BYTES
    role_peaks, docker_role_peaks = {}, {}
    for role in roles:
        # Preparation never exports a client bundle; inference and either offset
        # worker may. Include retained corrections while preparing new material.
        # Native delivery streams immutable weight views and retains only copied
        # metadata plus bounded compression/batch frames. Existing images may
        # still pack/unpack full bundles and cache copied artifact objects.
        delivery = 2 * metadata + 8 * MiB if role != "preparation" else 0
        legacy_delivery = 5 * bundle if role != "preparation" else 0
        role_peaks[role] = _slack(engine + gpu_provider + max(
            loading, delivery + corrections, work + corrections + verification))
        docker_role_peaks[role] = _slack(engine + weights + gpu_provider + max(
            legacy_loading, legacy_delivery + corrections, work + corrections + verification))
    client = _PROCESS_BYTES + 6 * bundle + masks + verification + tensors + cache_bytes + gpu_client
    if not roles:
        client += engine + loading
    client_peak = _slack(client)
    return {
        "scope": "conservative allocation estimate; not measured peak RSS",
        "model_plan_digest": plan.digest, "pipeline_digest": pipeline.digest(),
        "max_input_tokens": max_input_tokens, "max_output_tokens": max_output_tokens,
        "compiled_input_bound": compiled_input_bound,
        "inventory_rows": rows, "kernel": "metal" if metal else "cpu",
        "allocation_slack_percent": 25, "client_peak_bytes": client_peak,
        "provider_peak_bytes": role_peaks,
        "docker_provider_peak_bytes": docker_role_peaks,
        "weight_storage": "native_snapshot_v1; Docker retains legacy allocation upper bounds",
        "bundle_storage": "immutable_segments_v1; client import retains conservative copy bounds",
        "native_total_peak_bytes": client_peak + sum(role_peaks.values()),
        "components": {"per_engine_i8_and_native_bytes": cpu_weights,
            "legacy_per_engine_i8_and_native_bytes": 2 * weights,
            "float_quantization_work_bytes": quantization,
            "legacy_largest_loading_temporary_bytes": legacy_loading,
            "largest_loading_temporary_bytes": loading, "raw_client_bundle_bytes": bundle,
            "client_mask_bytes": masks, "provider_correction_bytes": corrections,
            "verifier_bytes": verification, "client_tensor_work_bytes": tensors,
            "client_cache_bytes": cache_bytes, "per_provider_metal_bytes": gpu_provider,
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
                reasons.append("enforced WAN requires Docker; native supports explicit --wan-estimate")
        else:
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
                         + 2 * estimate["components"]["raw_client_bundle_bytes"] + 2 * GiB)
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
