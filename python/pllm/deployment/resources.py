"""Client-local resource observations and process-wide concurrent ownership."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import stat
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
        self._cleanup = {}

    def _cleanup_control_bytes(self):
        # Charge full current dictionary backing plus conservative 4x growth
        # and fixed transition scratch, NOT credit for already covered maps.
        # The ordinary memory ledger may contain non-cleanup reservations too.
        return (64 << 10) + 5 * (self._owners.__sizeof__() + self._cleanup.__sizeof__()) + 1024

    def cleanup_slot(self):
        # Slot/lock/callback allowance plus actual old/new bookkeeping backing.
        # One slot per exclusive client seat, bounded process ceiling.
        with self.lock:
            if len(self._cleanup) >= 1024:
                raise NetworkError("CLIENT_CLEANUP_CAPACITY")
            amount = self._cleanup_control_bytes()
            integer(amount, "cleanup_control_bytes", minimum=1)
            if amount > self.observe().headroom_bytes:
                raise NetworkError("CLIENT_MEMORY_CAPACITY_CHANGED")
            token = object()
            self._owners[token] = amount
            try:
                slot = _CleanupSlot(self, token)
                self._cleanup[token] = slot
            except BaseException:
                self._owners.pop(token, None)
                raise
        return slot

    def release_cleanup(self, token, slot):
        with self.lock:
            if self._cleanup.get(token) is not slot:
                raise NetworkError("CLIENT_CLEANUP_OWNER_CHANGED")
            retained = self._cleanup.pop(token)
            self._owners.pop(token)
        # Keep the popped strong reference alive until AFTER releasing the
        # admission lock. No payload destructor/callback runs under that lock.
        del retained

    def observe(self):
        import psutil

        # Share benchmark reclaimable-memory and pressure observations. Active
        # reservations remain charged even after some allocation is resident.
        from pllm.runtime.benchmark_memory import BenchmarkMemoryError, host_memory

        with self.lock:
            try:
                memory = host_memory()
                rss = int(psutil.Process().memory_info().rss)
                if not 0 <= memory.available <= memory.total or memory.total <= 0 or rss <= 0:
                    raise ValueError("invalid client memory telemetry")
                resources = ClientResources(time.time_ns() // 1_000_000, int(memory.available),
                    rss, memory.reserve, sum(self._owners.values()),
                    _linux_memory_headroom() if platform.system() == "Linux" else None)
            except (OSError, ValueError, IndexError, BenchmarkMemoryError, psutil.Error) as exc:
                raise NetworkError("CLIENT_MEMORY_OBSERVATION_UNAVAILABLE") from exc
            if memory.pressure in {"warning", "critical"}:
                raise NetworkError("CLIENT_MEMORY_PRESSURE")
            return resources

    @contextmanager
    def hold_bytes(self, incremental_bytes, *, _token=None):
        """Reserve already-priced incremental owners without a planning record.

        Share the SDK reservation ledger and atomic live host/cgroup observation.
        Resident allocations never receive speculative reuse credit.
        """
        integer(incremental_bytes, "incremental_memory_bytes", minimum=1)
        token = object() if _token is None else _token
        with self.lock:
            resources = self.observe()
            if incremental_bytes > resources.headroom_bytes:
                raise NetworkError("CLIENT_MEMORY_CAPACITY_CHANGED")
            self._owners[token] = incremental_bytes
        try:
            yield resources
        finally:
            with self.lock:
                if getattr(token, "retained", None) is None:
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
    return _MemoryReservation(_capacity, incremental_bytes)


class _MemoryToken:
    __slots__ = ("retained",)

    def __init__(self):
        self.retained = None


class _MemoryReservation:
    """Ordinary byte reservation, with bounded explicit failed-retirement custody.

    Failed ownership is rooted by the SAME admitted ledger token, not a second
    uncharged registry. At most two owners; fixed controls use the existing role
    allowance (minimum64KiB). No background retry/finalizer; explicit drain only.
    Successful ordinary context-manager behavior/amount remains unchanged.
    """
    def __init__(self, capacity, amount):
        self._capacity, self._amount = capacity, amount
        self._token = _MemoryToken()
        self._manager = capacity.hold_bytes(amount, _token=self._token)
        self._failed = ()
        self._exited = self._draining = False
        self._finish_lock = threading.Lock()
        self._drain_finish_state = None

    def __enter__(self):
        return self._manager.__enter__()

    def __exit__(self, *exc):
        with self._finish_lock:
            if self._token.retained is not None:
                return False  # failure is NOT retirement, including on caller GC
            if not self._exited:
                result = self._manager.__exit__(*exc)
                self._exited = True
                return result
            return False

    def retain_failed(self, owners):
        if type(owners) is not tuple or not 1 <= len(owners) <= 2 or self._amount < 64 << 10:
            raise NetworkError("CLIENT_FAILED_RETIREMENT_CUSTODY_INVALID")
        with self._capacity.lock:
            if (self._capacity is not _capacity or self._exited
                    or self._token not in self._capacity._owners or self._token.retained is not None):
                raise NetworkError("CLIENT_FAILED_RETIREMENT_CUSTODY_INVALID")
            self._failed = owners
            self._token.retained = self  # actual module admission-token root

    @property
    def pending_retirement(self):
        with self._capacity.lock:
            return (self._capacity is _capacity and self._token in self._capacity._owners
                    and self._token.retained is self and bool(self._failed))

    def drain_failed(self):
        # One external async BaseException per operation may dominate ordinary
        # failure, with available original context under Python's rules.
        # No universal bytecode/repeated-interruption/broken-lock guarantee.
        # One constant attempt object uses existing controls, not a new price.
        attempt = object()
        capacity = self._capacity
        body_error = None
        owner_error = None  # First known owner primary; never replaced by a secondary.
        first_pass_done = False
        try:
            try:
                with capacity.lock:  # drain admission
                    # Snapshot BEFORE any terminal clear; locals survive admission,
                    # finish, commit/recovery AND finalizer lock releases. Last shell
                    # decrefs therefore occur outside all reservation locks.
                    token, manager = self._token, self._manager
                    amount, owners = self._amount, self._failed
                    if self._draining:
                        raise NetworkError("CLIENT_FAILED_RETIREMENT_NOT_DRAINABLE")
                    recovering = (self._drain_finish_state in ("committing", "committed")
                                  and token not in capacity._owners)
                    if not recovering:
                        if self._drain_finish_state is not None:
                            raise NetworkError("CLIENT_FAILED_RETIREMENT_NOT_DRAINABLE")
                        if self._exited:
                            return
                        if (capacity is not _capacity or token.retained is not self
                                or not owners or type(amount) is not int
                                or capacity._owners.get(token) != amount):
                            raise NetworkError("CLIENT_FAILED_RETIREMENT_NOT_DRAINABLE")
                    self._draining = attempt
                if recovering:  # admission published; no owner/backend work yet
                    with self._finish_lock:
                        with capacity.lock:
                            if (self._capacity is not capacity or self._token is not token
                                    or self._manager is not manager or self._draining is not attempt
                                    or token in capacity._owners
                                    or self._drain_finish_state not in ("committing", "committed")):
                                raise NetworkError("CLIENT_FAILED_RETIREMENT_NOT_DRAINABLE")
                            # Original refund already linearized. Recovery ONLY
                            # finishes terminal bookkeeping; never owner/backend,
                            # second refund, new credit, reinsertion or re-rooting.
                            self._exited = True  # terminal exited
                            self._drain_finish_state = "committed"  # terminal state
                            token.retained = None  # terminal token
                            self._failed = ()  # terminal custody
                    return

                remaining, errors = [], []
                for owner in owners:
                    try:
                        # Count owners close an ExitStack containing THIS
                        # reservation: neither capacity nor finish lock may be held.
                        owner.close()
                    except BaseException as error:
                        # Capture BEFORE append/format/tuple/note allocations.
                        if owner_error is None:
                            owner_error = error
                        remaining.append(owner)
                        errors.append(error)
                if errors:
                    # All aggregation/optional diagnostics must finish before
                    # publishing a reduced tuple. Ordinary failure anywhere here
                    # retains ORIGINAL custody, including unvisited/retired owners.
                    first = owner_error
                    for error in errors[1:]:
                        first.add_note(f"retirement also failed: {type(error).__name__}")
                    failed_owners = tuple(remaining)
                    with capacity.lock:
                        self._failed = failed_owners
                    raise first

                with self._finish_lock:
                    with capacity.lock:
                        # A failed precommit validation is sealed too. Owner failure
                        # above, or latch cleanup alone, does NOT seal the backend.
                        self._drain_finish_state = "precommit"
                        if (self._capacity is not capacity or capacity is not _capacity
                                or self._token is not token or self._manager is not manager
                                or type(self._amount) is not int or self._amount != amount
                                or capacity._owners.get(token) != amount
                                or token.retained is not self or self._draining is not attempt
                                or self._failed is not owners or remaining or self._exited):
                            raise NetworkError("CLIENT_FAILED_RETIREMENT_NOT_DRAINABLE")
                        self._drain_finish_state = "backend_attempted"
                    # ORIGINAL backend ONCE, retaining ORIGINAL ledger/custody.
                    # Return truthiness has no suppression meaning. The existing
                    # hold_bytes finally defers refund while token.retained is self.
                    # Ordinary backend __exit__ re-entry under finish lock remains
                    # inherited/unsupported, NOT newly made deadlock-safe here.
                    manager.__exit__(None, None, None)
                    with capacity.lock:
                        if (self._capacity is not capacity or capacity is not _capacity
                                or self._token is not token or self._manager is not manager
                                or type(self._amount) is not int or self._amount != amount
                                or capacity._owners.get(token) != amount
                                or token.retained is not self or self._draining is not attempt
                                or self._failed is not owners or remaining or self._exited
                                or self._drain_finish_state != "backend_attempted"):
                            raise NetworkError("CLIENT_FAILED_RETIREMENT_NOT_DRAINABLE")
                        self._drain_finish_state = "committing"
                        # Sole irreversible refund/COMMITTED linearization point.
                        # No fallible payload callback/allocation after removal.
                        # Interrupted assignments leave uncleared fields rooted;
                        # explicit recovery completes them, never refunds again.
                        capacity._owners.pop(token)
                        self._exited = True  # terminal exited
                        self._drain_finish_state = "committed"  # terminal state
                        token.retained = None  # terminal token
                        self._failed = ()  # terminal custody
            except BaseException as error:
                body_error = error
                # Existing handler guards ordinary aggregation/lookup/format/
                # note failures. Keep first owner primary immutable; no frame
                # clearing or arbitrary callback/CPU guarantee. Process-control
                # BaseExceptions retain selected Python interruption precedence.
                if (owner_error is not None and isinstance(error, Exception)
                        and error is not owner_error):
                    body_error = owner_error
                    raise owner_error from error
                raise
            finally:
                # Fallback protection is already armed BEFORE admission. An
                # interruption at entry to this first pass cannot skip pass 2.
                with capacity.lock:  # latch cleanup pass 1 acquire
                    if self._draining is attempt:
                        self._draining = False  # latch cleanup pass 1 reset
                first_pass_done = True  # latch cleanup pass 1 released
        except BaseException as cleanup_error:
            # Bounded TWO passes, never a generic retry loop. Reset/release may
            # have let another drain publish its OWN flag: never clear that flag.
            if not first_pass_done:
                with capacity.lock:  # latch cleanup pass 2 acquire
                    if self._draining is attempt:
                        self._draining = False  # latch cleanup pass 2 reset
            if body_error is not None and body_error is not cleanup_error:
                # Only this established guarded association preserves exact body
                # primary with cleanup as explicit cause. External async errors
                # elsewhere may dominate, with available original context.
                # No universal bytecode protection or repeated-interruption claim.
                raise body_error from cleanup_error
            raise


class _CleanupSlot:
    """Admission-rooted failed cleanup custody; explicit drain only, no finalizer.

    Callers must drain executing producers before drain(). Failure retains the
    payload, callback and reservation in the module-owned capacity ledger even
    after all caller/client/traceback references disappear. No automatic retry
    or timeout-as-retirement; the caller must report shutdown failure.
    """
    __slots__ = ("_capacity", "_token", "_lock", "_payload", "_retire", "_closed", "_draining")

    def __init__(self, capacity, token):
        self._capacity, self._token = capacity, token
        self._lock = threading.Lock()
        self._payload = self._retire = None
        self._closed = self._draining = False

    @property
    def closed(self):
        with self._lock:
            return self._closed

    def owns(self, payload):
        with self._capacity.lock:
            if (self._capacity is not _capacity or self._capacity._cleanup.get(self._token) is not self
                    or self._token not in self._capacity._owners):
                return False
            with self._lock:
                return not self._closed and self._payload is payload

    def retain(self, payload, retire):
        with self._capacity.lock:
            if self._capacity is not _capacity or self._capacity._cleanup.get(self._token) is not self:
                raise NetworkError("CLIENT_CLEANUP_SLOT_UNAVAILABLE")
            with self._lock:
                if self._closed or self._draining or self._payload is not None or not callable(retire):
                    raise NetworkError("CLIENT_CLEANUP_SLOT_UNAVAILABLE")
                self._payload, self._retire = payload, retire

    def drain(self):
        with self._lock:
            if self._closed:
                return
            if self._draining:
                raise NetworkError("CLIENT_CLEANUP_ALREADY_DRAINING")
            self._draining = True
            retire = self._retire
        try:
            if retire is not None:
                retire()  # OUTSIDE slot/admission locks; may fail without refund.
        except BaseException:
            with self._lock:
                self._draining = False
            raise
        with self._lock:
            payload = self._payload
            self._payload = self._retire = None
            self._closed, self._draining = True, False
        self._capacity.release_cleanup(self._token, self)
        del payload, retire  # callbacks/destructors outside all reservation locks


def reserve_cleanup_slot():
    """Admit one bounded durable cleanup slot before native owner construction."""
    return _capacity.cleanup_slot()


@dataclass(frozen=True, slots=True)
class DiskResources:
    """Live free-space observation plus this process's logical device charges.

    Not a filesystem quota, allocation guarantee or cross-process reservation.
    Reservations retain their ownership until retirement. Only locally observed,
    deduplicated backing reduces pending bytes; logical file lengths and changes
    in global free space never earn backing credit.
    """

    device: int
    available_disk_bytes: int
    reserved_disk_bytes: int
    reserve_bytes: int
    backed_disk_bytes: int = 0

    @property
    def pending_disk_bytes(self):
        return self.reserved_disk_bytes - self.backed_disk_bytes

    @property
    def headroom_bytes(self):
        return max(0, self.available_disk_bytes - self.pending_disk_bytes - self.reserve_bytes)


class DiskLease:
    """Live local ownership, with the immutable observation made at admission.

    bind_backing accepts a synchronous LOCAL owner method, never a resource
    claim from a remote client. It must return a bounded list of plain records:
    {device: st_dev, file: st_ino, allocated_bytes: physical allocated bytes}.
    Files must stay owned until cleanup; closed owners must raise. Sparse length
    is not allocated backing. Callbacks run under the ledger lock and must only
    sample their owned backing, without modifying the reservation ledger.
    """

    __slots__ = ("_capacity", "_token", "_directory", "_bound", "_minimum_free",
                 "_observation", "_backing", "_closed", "_raw_holder")

    def __init__(self, capacity, token, directory, bound, minimum_free, observation):
        self._capacity, self._token, self._directory = capacity, token, directory
        self._bound, self._minimum_free = bound, minimum_free
        self._observation = observation
        self._backing, self._closed = None, False
        self._raw_holder = None

    @property
    def observation(self):
        return self._observation

    @property
    def closed(self):
        return self._closed

    def clear_backing(self):
        """Drain any observation before closing files; retain the full pending bound."""
        with self._capacity.lock:
            self._backing = None

    def bind_backing(self, local_callable):
        with self._capacity.lock:
            if (self._closed or self._capacity._owners.get(self._token) is not self
                    or self._raw_holder is not None
                    or self._backing is not None or not callable(local_callable)):
                raise NetworkError("DISK_BACKING_BINDING_INVALID")
            self._backing = local_callable
            try:
                # Validate immediately; later admissions always resample rather
                # than caching credit that could outlive its native owner.
                observed = self._capacity.observe(self._directory, self._minimum_free)
                if observed.device != self._observation.device:
                    raise NetworkError("DISK_BACKING_DEVICE_CHANGED")
            except BaseException:
                self._backing = None  # Failed binding leaves the full bound pending.
                raise


class _DiskCapacity:
    MAX_OWNERS = 1024
    MAX_BACKING_FILES = 4096  # total records in one device observation, not per owner

    def __init__(self):
        self.lock = threading.RLock()
        self._owners = {}

    def _pending(self, owners, device):
        """Bounded temporary file identities; never cache backing between observations."""
        snapshots, allocated, count = [], {}, 0
        try:
            for owner in owners:
                keys = set()
                if owner._backing is not None:
                    records = owner._backing()
                    if type(records) is not list or len(records) > self.MAX_BACKING_FILES - count:
                        raise ValueError("disk backing records exceed bounds")
                    count += len(records)
                    for record in records:
                        if type(record) is not dict or set(record) != {"device", "file", "allocated_bytes"}:
                            raise ValueError("disk backing fields differ")
                        dev, inode, size = record["device"], record["file"], record["allocated_bytes"]
                        for name, value in (("device", dev), ("file", inode), ("allocated_bytes", size)):
                            integer(value, name)
                        if dev != device:
                            raise ValueError("disk backing device differs")
                        identity = dev, inode
                        # Duplicate samples may straddle filesystem allocation.
                        # The smaller observation is the conservative credit.
                        allocated[identity] = min(allocated.get(identity, size), size)
                        keys.add(identity)
                snapshots.append((owner._bound, keys))
            pending = 0
            for bound, keys in snapshots:
                remaining = bound
                for identity in keys:
                    credit = min(remaining, allocated[identity])
                    remaining -= credit
                    allocated[identity] -= credit  # shared inodes can be credited only once
                pending += remaining
            return pending
        except Exception as error:
            raise NetworkError("DISK_BACKING_UNAVAILABLE") from error

    def observe(self, directory, minimum_free_bytes):
        with self.lock:
            try:
                before = directory.stat()
                owners = [owner for owner in self._owners.values()
                          if owner._observation.device == before.st_dev]
                pending = self._pending(owners, before.st_dev)
                # Backing MUST be sampled before this fresh free-space reading.
                usage = shutil.disk_usage(directory)
                after = directory.stat()
                if (not stat.S_ISDIR(before.st_mode) or before.st_dev != after.st_dev
                        or before.st_ino != after.st_ino
                        or type(usage.free) is not int or type(usage.total) is not int
                        or not 0 <= usage.free <= usage.total or usage.total <= 0):
                    raise ValueError("disk identity or free-space observation differs")
                reserved = sum(owner._bound for owner in owners)
                return DiskResources(before.st_dev, usage.free, reserved,
                                     max([minimum_free_bytes, *(owner._minimum_free for owner in owners)]),
                                     reserved - pending)
            except NetworkError:
                raise
            except (OSError, ValueError, TypeError, AttributeError) as error:
                raise NetworkError("DISK_OBSERVATION_UNAVAILABLE") from error

    @contextmanager
    def hold_bytes(self, incremental_bytes, *, directory, minimum_free_bytes=1 << 20):
        integer(incremental_bytes, "incremental_disk_bytes", minimum=1)
        integer(minimum_free_bytes, "minimum_free_disk_bytes", minimum=1)
        try:
            directory = Path(directory).resolve(strict=True)
        except (OSError, ValueError, TypeError, RuntimeError) as error:
            raise NetworkError("DISK_OBSERVATION_UNAVAILABLE") from error
        token = object()
        with self.lock:
            if len(self._owners) >= self.MAX_OWNERS:
                raise NetworkError("DISK_RESERVATION_CAPACITY_EXHAUSTED")
            resources = self.observe(directory, minimum_free_bytes)
            if incremental_bytes > resources.headroom_bytes:
                raise NetworkError("DISK_CAPACITY_CHANGED")
            lease = DiskLease(self, token, directory, incremental_bytes, minimum_free_bytes, resources)
            self._owners[token] = lease
        try:
            yield lease
        finally:
            with self.lock:
                self._owners.pop(token, None)
                lease._closed, lease._backing = True, None
                lease._raw_holder = None


_disk_capacity = _DiskCapacity()


def _check_raw_disk_lease(lease, holder):
    # Called only with the process ledger lock held. Neither observation/device
    # equality nor a duck-typed token establishes live reservation membership.
    if (type(lease) is not DiskLease or lease._capacity is not _disk_capacity
            or lease._closed or _disk_capacity._owners.get(lease._token) is not lease
            or lease._raw_holder is not holder):
        raise NetworkError("RAW_DISK_LEASE_OWNER_CHANGED")


@contextmanager
def _admit_raw_disk_lease(lease, holder, amount, directory):
    """Associate the actual live lease with one retained host; keep ingestion unbound.

    Caller retains the REAL reservation context manager before entering it. The
    returned device is only a native filesystem predicate. Hold the ledger lock
    through start so concurrent lease exit cannot invalidate the membership check.
    """
    with _disk_capacity.lock:
        if (type(lease) is not DiskLease or lease._capacity is not _disk_capacity
                or lease._closed or _disk_capacity._owners.get(lease._token) is not lease
                or lease._raw_holder is not None or lease._backing is not None
                or type(amount) is not int or amount <= 0 or lease._bound != amount
                or lease._directory != Path(directory)):
            raise NetworkError("RAW_DISK_LEASE_INVALID")
        lease._raw_holder = holder
        yield lease._observation.device


def _clear_raw_disk_backing(lease, holder):
    """Clear under the same ledger lock BEFORE any native file retirement."""
    with _disk_capacity.lock:
        _check_raw_disk_lease(lease, holder)
        lease._backing = None


def reserve_incremental_disk(incremental_bytes, *, directory, minimum_free_bytes=1 << 20):
    """Hold a bounded process-local charge on the observed filesystem device.

    Yields DiskLease; its observation records admission-time headroom. Unbound
    leases remain fully pending. Aliased directories share the device ledger;
    admissions sample local backing before live free space under the same lock.
    Other processes/writers remain outside this logical reservation's scope.
    """
    return _disk_capacity.hold_bytes(incremental_bytes, directory=directory,
                                     minimum_free_bytes=minimum_free_bytes)
