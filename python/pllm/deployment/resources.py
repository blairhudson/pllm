"""Client-local resource observations and process-wide concurrent ownership."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
import platform
import re
import threading
import time

from pllm.deployment.network import NetworkError, PublicRecord, integer


@dataclass(frozen=True, slots=True)
class ClientResources(PublicRecord):
    """A local observation, never proof of an RSS ceiling or part of role admission."""

    SCHEMA = "pllm.client_resources.v1"
    observed_at_ms: int
    available_memory_bytes: int
    process_memory_bytes: int
    reserve_bytes: int
    reserved_memory_bytes: int = 0
    container_headroom_bytes: int | None = None

    def __post_init__(self):
        for name in ("observed_at_ms", "available_memory_bytes", "process_memory_bytes",
                     "reserve_bytes", "reserved_memory_bytes"):
            integer(getattr(self, name), name)
        if self.container_headroom_bytes is not None:
            integer(self.container_headroom_bytes, "container_headroom_bytes")

    @classmethod
    def from_spec(cls, value):
        return cls(**cls._fields({"container_headroom_bytes": None, **value}))

    def to_spec(self):
        value = super(ClientResources, self).to_spec()
        if self.container_headroom_bytes is None:
            value.pop("container_headroom_bytes")
        return value

    @classmethod
    def observe(cls):
        """Observe this process and its live client reservations without allocating a model."""
        return _capacity.observe()

    @property
    def headroom_bytes(self):
        headroom = max(0, self.available_memory_bytes - self.reserve_bytes)
        if self.container_headroom_bytes is not None:
            headroom = min(headroom, self.container_headroom_bytes)
        return max(0, headroom - self.reserved_memory_bytes)


def _bounded_text(path, size):
    with path.open("r", encoding="utf-8") as stream:
        value = stream.read(size + 1)
    if len(value) > size:
        raise ValueError("resource observation exceeds bound")
    return value


def _linux_memory_headroom(proc_root=Path("/proc")):
    """Remaining hard cgroup memory across visible ancestors, excluding swap/cache credit."""
    memberships = {}
    for line in _bounded_text(proc_root / "self/cgroup", 65536).splitlines():
        _hierarchy, controllers, path = line.split(":", 2)
        if not controllers or "memory" in controllers.split(","):
            memberships["cgroup2" if not controllers else "cgroup"] = PurePosixPath(path)
    if not memberships:
        return None  # Kernel has no memory controller attached to this process.
    headroom, found = None, set()
    for line in _bounded_text(proc_root / "self/mountinfo", 131072).splitlines():
        left, right = line.split(" - ", 1)
        columns, mounted = left.split(), right.split()
        kind = mounted[0]
        if kind not in memberships or (kind == "cgroup" and "memory" not in mounted[2].split(",")):
            continue
        unescape = lambda text: re.sub(r"\\([0-7]{3})", lambda match: chr(int(match[1], 8)), text)
        root, mount = PurePosixPath(unescape(columns[3])), Path(unescape(columns[4]))
        member = memberships[kind]
        if not root.is_absolute() or not mount.is_absolute() or not member.is_absolute():
            raise ValueError("invalid cgroup memory path")
        if any(".." in path.parts for path in (root, mount, member)):
            raise ValueError("invalid cgroup memory path")
        if member.is_relative_to(root):
            relative = member.relative_to(root)
        elif member == PurePosixPath("/"):
            relative = PurePosixPath(".")  # Namespace-relative membership at the mount root.
        else:
            continue
        found.add(kind)
        current = mount / relative
        limit_name, usage_name = (("memory.max", "memory.current") if kind == "cgroup2"
                                  else ("memory.limit_in_bytes", "memory.usage_in_bytes"))
        for _ in range(256):
            try:
                limit_text = _bounded_text(current / limit_name, 64).strip()
            except FileNotFoundError:
                # The v2 hierarchy root has no memory.max. A delegated child with
                # an unreadable controller cannot be treated as unlimited.
                if kind != "cgroup2" or current != mount or (current / usage_name).exists():
                    raise
            else:
                if limit_text != "max":
                    limit = int(limit_text)
                    usage = int(_bounded_text(current / usage_name, 64).strip())
                    if limit < 0 or usage < 0:
                        raise ValueError("invalid cgroup memory observation")
                    remaining = max(0, limit - usage)
                    headroom = remaining if headroom is None else min(headroom, remaining)
            if current == mount:
                break
            current = current.parent
        else:
            raise ValueError("cgroup memory ancestry exceeds bound")
    if set(memberships) != found:
        raise ValueError("cgroup memory controller is not visible")
    return headroom


class _ClientCapacity:
    def __init__(self):
        self.lock = threading.RLock()
        self._owners = {}

    def observe(self):
        import psutil

        # Keep the benchmark's existing host/application reserve. Available RAM
        # excludes swap. Count active reservations conservatively even after part
        # of their allocation becomes resident; no unproved reuse credit.
        from pllm.runtime.benchmark_memory import host_reserve_bytes

        with self.lock:
            try:
                memory = psutil.virtual_memory()
                rss = int(psutil.Process().memory_info().rss)
                if not 0 <= memory.available <= memory.total or memory.total <= 0 or rss <= 0:
                    raise ValueError("invalid client memory telemetry")
                return ClientResources(time.time_ns() // 1_000_000, int(memory.available),
                    rss, host_reserve_bytes(int(memory.total)), sum(self._owners.values()),
                    _linux_memory_headroom() if platform.system() == "Linux" else None)
            except (OSError, ValueError, IndexError, psutil.Error) as exc:
                raise NetworkError("CLIENT_MEMORY_OBSERVATION_UNAVAILABLE") from exc

    @contextmanager
    def hold_bytes(self, incremental_bytes):
        """Reserve already-priced incremental owners without a planning record.

        Share the SDK reservation ledger and atomic live host/cgroup observation.
        Resident allocations never receive speculative reuse credit.
        """
        integer(incremental_bytes, "incremental_memory_bytes", minimum=1)
        token = object()
        with self.lock:
            resources = self.observe()
            if incremental_bytes > resources.headroom_bytes:
                raise NetworkError("CLIENT_MEMORY_CAPACITY_CHANGED")
            self._owners[token] = incremental_bytes
        try:
            yield resources
        finally:
            with self.lock:
                self._owners.pop(token, None)

    @contextmanager
    def hold(self, select):
        token = object()
        with self.lock:
            result = select()
            costs = result.costs
            amount = costs.get("client_incremental_memory_bytes")
            if amount is None:
                # Pinned and explicit-request SDK callers may bring historical
                # payload-only decisions. Reprice the selected execution locally;
                # do not rewrite its saved explanation or switch its placement.
                from pllm.runtime.benchmark_memory import estimate_plan_memory

                experiment = result.experiment
                profile = experiment.resolve()
                estimate = estimate_plan_memory(result.request.model_plan, experiment.pipeline,
                    max_input_tokens=experiment.budget.max_input_tokens,
                    max_output_tokens=experiment.budget.max_new_tokens,
                    inventory_rows=max(64, profile.prepared_inventory_rows or 0))
                if estimate["limitations"]:
                    raise NetworkError("CLIENT_MEMORY_OWNERS_UNAVAILABLE")
                by_role = {"client": estimate["client_peak_bytes"], **estimate["provider_peak_bytes"]}
                placement = result.native_placement
                local_roles = [row["role_id"] for row in placement["roles"]
                               if row["party_id"] == result.request.policy.client_party_id]
                amount = sum(by_role[role] + result.request.policy.buffer_reserve_bytes for role in local_roles)
            resources = self.observe()
            if amount > resources.headroom_bytes:
                raise NetworkError("CLIENT_MEMORY_CAPACITY_CHANGED")
            limit = result.request.policy.client_peak_memory_limit_bytes
            if limit is not None and (
                    amount + resources.process_memory_bytes + resources.reserved_memory_bytes > limit):
                raise NetworkError("CLIENT_MEMORY_CAPACITY_CHANGED")
            self._owners[token] = amount
        try:
            yield result
        finally:
            with self.lock:
                self._owners.pop(token, None)


_capacity = _ClientCapacity()


def reserve_incremental_memory(incremental_bytes):
    """Hold a conservative byte price through the lifetime of its local owners."""
    return _capacity.hold_bytes(incremental_bytes)
