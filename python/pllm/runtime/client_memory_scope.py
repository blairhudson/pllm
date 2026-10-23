"""Bounded owner-declared client RSS scope, not a discovered process tree.

Only the existing target/launcher can declare helpers, using live process-owner
objects and its birth records. Parentage does not classify providers as clients
and this local accounting declaration is not operator/security authentication.
Without a sealed owner declaration helper membership stays unknown.

Sampling is sequential: a same-pass RSS SUM is not an atomic simultaneous SUM.
Its observed peak is an estimate, NOT a proved lower bound on simultaneous peak.
The largest individual RSS observation is a safe sampled lower bound, even when
other declared members are missing. Neither value measures physical used RAM.
"""
from __future__ import annotations

from copy import deepcopy
import math
from multiprocessing.process import BaseProcess
import os
import subprocess
import sys
import threading


SCHEMA = "pllm.client_owned_memory.v1"
HELPERS = tuple(f"client_helper_{index}" for index in range(1, 9))
WINDOW = "benchmark_sampler_start_through_responses_before_teardown"
_MAX = (1 << 63) - 1


def _bytes(value):
    return type(value) is int and 0 <= value <= _MAX


def _birth(value):
    return (type(value) in (int, float) and 0 <= value <= _MAX and math.isfinite(value))


class ClientMemoryScope:
    """One local owner's capability; not constructible from report/PID metadata.

    The benchmark owns the current-process root. Helper launchers must explicitly
    bind their own process objects and captured create_time; arbitrary PIDs and
    serialized owner claims are not accepted. Registration adds no process poll,
    reaper, discovery, watcher or sampling thread. Seal only a genuinely exhaustive
    client-helper manifest; it applies to the sampling window, not process birth.

    At most nine members, one latest snapshot and scalar maxima/counters. These
    fixed records/optional metrics share the existing coarse 256 MiB Python/client
    process allowances; no freed credits, changed caps or exact heap/RSS proof.
    """

    def __init__(self):
        self.root_pid = os.getpid()
        self._lock = threading.Lock()
        self._members = {"client": (self.root_pid, None, None, None)}
        self._sealed = False
        self._sealed_before_sampling = False
        self._sampling_claimed = self._sampling_started = False
        self._latest = {}
        self._samples = self._complete_samples = self._missing_samples = 0
        self._peak_sum = self._lower_bound = None
        self._member_peaks = {}
        self._maximum_span = 0.0
        self._conflicts = set()
        self._lifetime_declared = False
        self._lifetime_closed_family = False

    def seal_lifetime_membership(self, *, closed_family=False):
        """Owning launcher declares COMPLETE birth-through-exit client membership.

        Stronger than seal(): includes pre-sampler imports/setup, every client
        helper and final retirement. Never call for remote/unknown client helpers
        or merely because a PID is a descendant. Existing sampling seal is NOT it.
        closed_family additionally says ALL reaped descendants belong to this
        client group (not providers or benchmark bookkeeping). Default is unknown.
        Bind all genuine child owners/birth records first; no later additions.
        """
        if type(closed_family) is not bool:
            raise ValueError("closed family must be an explicit boolean")
        with self._lock:
            self._lifetime_declared = True
            self._lifetime_closed_family = closed_family
            if not self._sealed:
                self._sealed = True
                self._sealed_before_sampling = not self._sampling_started

    def _add(self, slot, declaration):
        if type(slot) is not str or slot not in HELPERS:
            raise ValueError("unsupported bounded client helper slot")
        with self._lock:
            if self._sealed or slot in self._members:
                raise ValueError("client helper manifest is sealed or slot already declared")
            pid = declaration[0]
            if pid is not None and any(member[0] == pid for member in self._members.values()):
                raise ValueError("duplicate client member identity")
            self._members[slot] = declaration

    def bind_helper(self, slot, process, *, created_at):
        """Called by owning launcher with its existing birth record, not ancestry."""
        if (not isinstance(process, (subprocess.Popen, BaseProcess))
                or type(process.pid) is not int or process.pid <= 0 or not _birth(created_at)):
            raise ValueError("client helper needs a live local process owner and birth record")
        identity_owner = getattr(process, "_popen", None) if isinstance(process, BaseProcess) else process
        captured = getattr(identity_owner, "_usage_identity", None)
        unavailable = ("member_owner_identity_conflict"
            if captured is not None and captured != (process.pid, created_at) else None)
        self._add(slot, (process.pid, created_at, process, unavailable))

    def expect_unavailable_helper(self, slot, *, remote=False):
        self._add(slot, (None, None, None,
                        "remote_helper_unobservable" if remote else "helper_identity_unavailable"))

    def seal(self):
        with self._lock:
            if not self._sealed:
                self._sealed_before_sampling = not self._sampling_started
                self._sealed = True

    def claim_sampling(self):
        """One scope per benchmark window; never replay another run's maxima."""
        with self._lock:
            if self._sampling_claimed or self._sampling_started:
                return False
            self._sampling_claimed = True
            return True

    def begin_sample(self):
        with self._lock:
            self._sampling_started = True

    def targets(self, provider_pids):
        """Only already-declared helpers; exclude conflicting provider identities."""
        with self._lock:
            targets = {}
            for name, (pid, _, _, unavailable) in self._members.items():
                if name == "client" or pid is None:
                    continue
                if pid in provider_pids:
                    if name not in self._conflicts:
                        # A later owner conflict invalidates earlier group sums;
                        # never retain a provider observation as a client bound.
                        self._peak_sum = None
                        self._complete_samples = 0
                        self._latest = {}
                    self._conflicts.add(name)
                    self._lower_bound = max((peak for member, peak in self._member_peaks.items()
                        if member not in self._conflicts), default=None)
                elif name not in self._conflicts and unavailable is None:
                    targets[name] = pid
            return targets

    def observe(self, readings, *, span_seconds):
        """Consume existing sampler's (PID, birth, RSS) tuples; no OS access."""
        if not _birth(span_seconds):
            raise ValueError("sample span must be finite nonnegative seconds")
        with self._lock:
            self._sampling_started = True
            observed, missing, bound = {}, [], None
            for name, (pid, born, owner, unavailable) in tuple(self._members.items()):
                value = readings.get(name)
                reason = ("provider_owner_conflict" if name in self._conflicts else unavailable)
                if reason is None:
                    if (type(value) is not tuple or len(value) != 3 or value[0] != pid
                            or not _birth(value[1]) or not _bytes(value[2])):
                        reason = "member_observation_unavailable"
                    elif born is not None and value[1] != born:
                        reason = "member_identity_changed"
                    else:
                        if born is None:  # Root is this live process, never an arbitrary PID.
                            self._members[name] = (pid, value[1], owner, None)
                        observed[name] = {"rss_bytes": value[2], "reason": None}
                        self._member_peaks[name] = max(self._member_peaks.get(name, 0), value[2])
                        bound = max(bound or 0, value[2])
                if reason is not None:
                    observed[name] = {"rss_bytes": None, "reason": reason}
                    missing.append(name)
            if not self._sealed:
                missing.append("undeclared_client_helpers")
            self._samples += 1
            self._missing_samples += bool(missing)
            if bound is not None:
                self._lower_bound = max(self._lower_bound or 0, bound)
            if not missing:
                total = sum(row["rss_bytes"] for row in observed.values())
                self._peak_sum = max(self._peak_sum or 0, total)
                self._complete_samples += 1
            self._maximum_span = max(self._maximum_span, span_seconds)
            self._latest = {"members": observed, "missing_members": missing,
                "same_pass_rss_sum_bytes": None if missing else total}

    def report(self, process_hwm=None):
        """Read bounded cache; existing RUSAGE_SELF endpoint stays individual only.

        No same-isolated-lifetime final individual HWM contract exists for the
        still-running client root. Never derive [max(HWM), sum(HWM)] by borrowing
        provider wait4 maxima, a last live snapshot or mismatched lifetime windows.
        """
        with self._lock:
            latest = deepcopy(self._latest)
            full = (self._sealed and self._sealed_before_sampling and self._samples > 0
                    and self._missing_samples == 0 and not self._conflicts)
            hwm = process_hwm if sys.platform in {"darwin", "linux"} and _bytes(process_hwm) else None
            return {"schema": SCHEMA, "window": WINDOW,
                "membership_declared": self._sealed,
                "membership_declared_before_sampling": self._sealed_before_sampling,
                "sampling_coverage": "all_declared_members_observed" if full else "unknown_or_partial",
                "sampled_peak_rss_sum_bytes": self._peak_sum,
                "sampled_peak_rss_sum_kind": "non_atomic_same_pass_estimate_not_simultaneous_peak_bound",
                "sampled_peak_rss_lower_bound_bytes": self._lower_bound,
                "lower_bound_source": "maximum_individual_RSS_observation",
                "samples": self._samples, "complete_member_samples": self._complete_samples,
                "missing_member_samples": self._missing_samples,
                "maximum_sample_span_seconds": self._maximum_span,
                "latest": latest or {"members": {}, "missing_members": ["client", "undeclared_client_helpers"],
                                     "same_pass_rss_sum_bytes": None},
                "individual_client_hwm": {"bytes": hwm,
                    "source": "resource.getrusage(RUSAGE_SELF)" if hwm is not None else None,
                    "window": "client_process_birth_through_report_endpoint",
                    "coverage": "individual_process_only_not_client_group",
                    "isolated_lifetime": None},
                "lifetime_peak_rss_sum_interval": {"lower_bound_bytes": None, "upper_bound_bytes": None,
                    "reason": "complete_isolated_lifetime_final_individual_HWM_unavailable"},
                "physical_used_ram_peak_bytes": None,
                "limitations": [
                    "Local owner declarations are accounting classifications, not operator trust proofs.",
                    "Sequential RSS reads have skew; same-pass sums can over- or underestimate true simultaneous peak.",
                    "Sampling misses short peaks; maximum individual observation is only a lower bound.",
                    "Shared pages count in each member RSS; sums are not physical used RAM.",
                    "Window excludes pre-sampler source resolution/import, teardown and unobserved helpers.",
                    "Current-process HWM includes earlier work; no fresh-isolation or workload-matched lifetime proof.",
                    "Remote/container providers are not client helpers by ancestry or role name.",
                    "Fixed metadata uses existing coarse process allowances; no cap reduction or allocator proof.",
                ]}


def client_memory_report(sampler, process_memory):
    scope = getattr(sampler, "client_memory_scope", None)
    if type(scope) is not ClientMemoryScope or scope.root_pid != os.getpid():
        scope = ClientMemoryScope()  # Explicitly unknown, not a zero-sized client group.
    return scope.report(process_memory.get("lifetime_peak_rss_bytes"))


def _builtin_local_client_topology(topology, *, retired=False):
    from .servers import LocalTopology
    if type(topology) is LocalTopology:
        return True
    from .native_wan import NativePartyNetwork, NativeWanTopology
    # Native WAN is a local in-process TCP relay, not a remote/container owner.
    # Qualify only its audited public client path, never arbitrary subclasses.
    if type(topology) is not NativeWanTopology or getattr(topology, "_privacy_mode", None) != "public":
        return False
    network = getattr(topology, "_native_network", None)
    if type(network) is not NativePartyNetwork:
        return False
    if not retired:
        return True
    if getattr(network, "closed", None) is not True or not hasattr(network, "thread"):
        return False
    thread = network.thread
    return thread is None or type(thread) is threading.Thread and not thread.is_alive()


def declare_builtin_client_lifetime(runtime):
    """Called by the dashboard's actual client creator, before prewarm/refill.

    This audited built-in public path creates client work in this process:
    dashboard/prewarm workers, provider-I/O and paging/refill executors, native
    client threads, and optional WAN relay threads. Their RSS is root RSS, not an
    omitted helper. LocalTopology._spawn alone creates the declared preparation
    and inference providers; they NEVER become client members through parentage.

    Existing foreign/helper launchers keep their own bind_helper/birth/final-usage
    contract. Never upgrade a sampling-only seal or an incomplete helper roster.
    No PID observation/discovery, new owner record, reserve or admission credit.
    """
    from pllm.configuration import Model
    from pllm.sources import BundleModel, TinyModel

    scope = getattr(runtime, "_benchmark_client_memory_owner", None)
    topology = getattr(runtime, "_topology", None)
    if (type(scope) is not ClientMemoryScope or scope.root_pid != os.getpid()
            or not _builtin_local_client_topology(topology)
            or getattr(topology, "_privacy_mode", None) != "public"):
        return False
    source = getattr(topology, "_model", None)
    offline_pin = (type(source) is Model and source.kind == "huggingface"
        and source.local_files_only is True and type(source.revision) is str
        and len(source.revision) == 40 and all(c in "0123456789abcdef" for c in source.revision))
    if type(source) not in (BundleModel, TinyModel) and not offline_pin:
        return False
    # Only this exact built-in creator has an exhaustive root-only declaration.
    # No helper launcher exists on this path; the concurrent sampler can only
    # update root readings. Other creators must bind helpers before sealing.
    with scope._lock:
        if scope._sealed or set(scope._members) != {"client"} or scope._conflicts:
            return False
    scope.seal_lifetime_membership(closed_family=False)
    return True


def client_lifetime_manifest(sampler, topology, roles, *, local_cpu_loopback):
    """Source-owner classification, not PID ancestry, discovery or a receipt.

    Audited built-in local-source CPU topology keeps its client in-process; its
    production subprocess owners create exactly the declared provider roles.
    Native WAN adds an in-process relay, not a client helper process. Its creator
    must explicitly declare the client lifetime and retire the relay first.
    Test-only compiler oracle subprocesses are NOT production helper evidence.
    Other/extra helper launchers must provide
    seal_lifetime_membership(), real birth-bound owners and their cached final
    usage; sampling-window seal alone supplies none of that lifetime coverage.
    """
    from .client_process_usage import WINDOW as LIFETIME_WINDOW, MAX_MEMBERS
    manifest = {"schema": "pllm.client_lifetime_owner.v1", "membership_complete": False,
        "closed_client_owned_family": False, "root_birth_bound": False,
        "window": LIFETIME_WINDOW, "members": {}, "reason": "lifetime_owner_classification_unavailable",
        "outside_isolated_role": "unknown_including_providers_controller_shared_tracker_parent_source_and_benchmark_metadata_helpers"}
    scope = getattr(sampler, "client_memory_scope", None)
    if (not local_cpu_loopback or type(scope) is not ClientMemoryScope or scope.root_pid != os.getpid()
            or type(roles) not in (tuple, list) or not 1 <= len(roles) <= 9):
        return manifest
    from .servers import LocalTopology
    from .process_usage import _PROVIDERS
    # Exact built-in closed owner only. Foreign/remote topologies, GPU, containers
    # and provider descendants cannot silently become the client's closed family.
    if not _builtin_local_client_topology(topology, retired=True):
        return manifest
    with topology._process_lock:
        closed = getattr(topology, "_closed", False) is True and getattr(topology, "_started", False) is True
        if (not closed or type(getattr(topology, "_role_ids", None)) not in (tuple, list)
                or type(getattr(topology, "_processes", None)) is not dict):
            return manifest
        providers = tuple(role for role in roles if role != "client")
        known_graph = (len(set(roles)) == len(roles) and tuple(roles).count("client") == 1
            and all(role in _PROVIDERS for role in providers)
            and set(topology._role_ids) == set(topology._processes) == set(providers))
        provider_pids = {process.pid for process in topology._processes.values()}
    with scope._lock:
        members = dict(scope._members)
        declared = scope._lifetime_declared
        closed_family = scope._lifetime_closed_family
        conflicts = bool(scope._conflicts)
    if type(topology) is not LocalTopology and not declared:
        return manifest
    if (not closed or not known_graph or conflicts or not 1 <= len(members) <= MAX_MEMBERS
            or members["client"][0] != os.getpid() or not _birth(members["client"][1])):
        return manifest
    if len(members) != 1 and not declared:
        return manifest
    if not declared:
        from pllm.sources import BundleModel, TinyModel
        if type(getattr(topology, "_model", None)) not in (BundleModel, TinyModel):
            manifest["reason"] = "source_helper_lifetime_owner_contract_unavailable"
            return manifest
    records = {"client": {"source": "parent_same_owner_reaper_pending", "coverage": "pending"}}
    for name, (pid, born, owner, missing) in members.items():
        if name == "client":
            continue
        from .client_process_usage import final_member_usage
        try:
            identity_owner = getattr(owner, "_popen", None) if isinstance(owner, BaseProcess) else owner
            matches = (owner is not None and owner.pid == pid
                and getattr(identity_owner, "_usage_identity", None) == (pid, born))
        except ValueError:  # A closed multiprocessing owner has released its cache.
            matches = False
        if missing is not None or pid is None or pid in provider_pids or not _birth(born) or not matches:
            return manifest
        records[name] = final_member_usage(owner)
    manifest.update(membership_complete=True, closed_client_owned_family=closed_family and not providers,
        root_birth_bound=False, _root_identity=(members["client"][0], members["client"][1]),
        members=records, reason=None,
        classification="builtin_in_process_client_owner" if len(members) == 1 else "explicit_closed_launcher_family")
    return manifest


def client_memory_metrics(report, measurement):
    """Optional distinct identities; historical reports/individual metric unchanged."""
    scope = report.get("client_owned_memory")
    if type(scope) is not dict or scope.get("schema") != SCHEMA:
        return
    for name, field, origin, semantics in (
        ("client.owned_sampled_peak_rss_sum", "sampled_peak_rss_sum_bytes", "estimate",
         "non_atomic_same_pass_owner_declared_RSS_sum"),
        ("client.owned_peak_rss_lower_bound", "sampled_peak_rss_lower_bound_bytes", "lower_bound",
         "maximum_individual_RSS_observation_not_group_peak"),
    ):
        measurement.metric(name, scope[field], kind="memory", window=WINDOW,
            scope=semantics, origin=origin, statistic="measurement")
