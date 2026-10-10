"""Bounded process-local cache of immutable native facts, never live admission."""
from collections import OrderedDict
import threading

from pllm.deployment.network import canonical, strict_load


class _PlanningCache:
    def __init__(self, *, max_entries=64, max_bytes=2 << 20):
        self.max_entries, self.max_bytes = max_entries, max_bytes
        self._entries = OrderedDict()
        self._bytes = 0
        self._lock = threading.RLock()

    def clear(self):
        with self._lock:
            self._entries.clear()
            self._bytes = 0

    def get(self, request, experiment):
        from pllm import _native
        from .placement import _geometry

        policy = request.policy
        key = (request.model_plan.digest, experiment.pipeline.digest(),
               canonical(experiment.budget.to_spec()), policy.allocation_model,
               policy.buffer_reserve_bytes, policy.remote_mac_denominator,
               policy.max_client_native_threads,
               tuple(row.digest for row in request.state_evidence
                     if row.configuration_digest == experiment.configuration_digest()))
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                admitted = _native.NetworkPlacementAdmission(
                    request.model_plan.canonical_bytes(), experiment.pipeline.canonical_bytes())
                requirements = admitted.requirements()
                geometry = canonical(_geometry(request, experiment, strict_load(requirements)))
                # Native owner retains only bounded role maps and digests. Charge
                # an additional 16 KiB for it, the key and Python/LRU metadata.
                size = len(requirements) + len(geometry) + (16 << 10)
                entry = (admitted, requirements, geometry, size)
                if size <= self.max_bytes:
                    self._entries[key] = entry
                    self._bytes += size
                    while len(self._entries) > self.max_entries or self._bytes > self.max_bytes:
                        self._bytes -= self._entries.popitem(last=False)[1][3]
            else:
                self._entries.move_to_end(key)
            admitted, requirements, geometry, _ = entry
            # Every caller gets fresh mutable estimates. Snapshot, policy, RSS,
            # concurrent reservations and prior evidence are always applied later.
            return admitted, strict_load(requirements), strict_load(geometry)


_cache = _PlanningCache()
