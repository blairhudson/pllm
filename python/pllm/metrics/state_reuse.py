"""Numeric/compatibility gates for new client-state reuse, without cache promotion."""
from __future__ import annotations

import hashlib
import json
import time
import secrets
from dataclasses import dataclass


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _same_state(left, right) -> bool:
    import numpy as np

    return (left.position == right.position and len(left.caches) == len(right.caches)
        and left.shared_kv.keys() == right.shared_kv.keys()
        and all(a.length == b.length and
                np.array_equal(a.key[:a.length].view(np.uint32), b.key[:b.length].view(np.uint32)) and
                np.array_equal(a.value[:a.length].view(np.uint32), b.value[:b.length].view(np.uint32))
                for a, b in zip(left.caches, right.caches, strict=True))
        and all(np.array_equal(left.shared_kv[k][i], right.shared_kv[k][i])
                for k in left.shared_kv for i in (0, 1)))


@dataclass(frozen=True, slots=True)
class GeneratedStateReuseProbe:
    """Compare generated execution with fresh W8A8 execution of executed tokens.

    The final selected-but-unexecuted token is excluded. This gate neither
    relabels an incremental snapshot nor authorizes an ordinary cache insertion.
    """

    steps: int = 4

    def __post_init__(self):
        if type(self.steps) is not int or not 1 <= self.steps <= 32:
            raise ValueError("generated-state gate needs 1..32 executed decode steps")

    def run(self, compiled, remote, token_ids) -> dict:
        import numpy as np

        from pllm.runtime.model_binding import CompiledRuntimeModel

        if type(compiled) is not CompiledRuntimeModel:
            raise ValueError("generated-state probe requires a checked compiled binding")
        compiled.validate()
        ids = list(token_ids)
        bound = compiled._plan.to_dict()["prefill"]["query_sequence"]
        if not ids or len(ids) + self.steps > min(4096, bound):
            raise ValueError("generated sequence exceeds the full-prefill comparison bound")
        if any(type(v) is not int or v < 0 or v >= int(compiled._bundle.cfg["vocab_size"]) for v in ids):
            raise ValueError("generated-state token cohort is invalid")
        start = time.process_time_ns()
        runtime = compiled.runtime(remote)
        logits = runtime.prepare_ids(ids)[1]
        rows = []
        peak_snapshot = 0
        for step in range(self.steps):
            ids.append(int(np.argmax(logits)))
            logits = runtime.decode_step(ids[-1], runtime.caches)[0]
            snapshot = runtime.snapshot()
            fresh = compiled.runtime(remote)
            golden = fresh.prepare_ids(ids)[1]
            reference = fresh.snapshot()
            if snapshot.state_basis is None or any(c.key is None or c.value is None for c in snapshot.caches):
                raise ValueError("generated-state gate requires qualified full KV state")
            peak_snapshot = max(peak_snapshot, sum(c.key.nbytes + c.value.nbytes for c in snapshot.caches
                if c.key is not None and c.value is not None))
            rows.append({"executed_decode_steps": step + 1,
                "bit_equal_logits": bool(np.array_equal(logits.view(np.uint32), golden.view(np.uint32))),
                "bit_equal_all_kv": _same_state(snapshot, reference),
                "max_absolute_logit_error": float(np.max(np.abs(logits - golden))),
                "original_basis_is_incremental": snapshot.state_basis.phase == "incremental"})
        return {"schema": "pllm.generated_state_reuse_probe.v1",
            "binding_digest": compiled.digest, "model_plan_digest": compiled.model_plan_digest,
            "cohort_digest": _digest([secrets.token_hex(32), ids]), "cohort_digest_salted": True,
            "input_tokens": len(ids) - self.steps,
            "executed_generated_tokens": self.steps, "pending_generated_tokens_excluded": 1,
            "cpu_seconds": (time.process_time_ns() - start) / 1e9,
            "snapshot_payload_bytes": peak_snapshot, "process_peak_memory_measured": False,
            "numeric_gate_passed": all(r["bit_equal_logits"] and r["bit_equal_all_kv"] for r in rows),
            "samples": rows, "live_cache_promotion": False,
            "scope": "same-token local numeric gate; source and snapshots stay private to this process"}


@dataclass(frozen=True, slots=True)
class StateCompatibilityProbe:
    """Compare complete public contracts before a separately measured state bridge.

    It deliberately does not erase source/verification commitments or make a
    successful comparison a transferable live-state authorization.
    """

    def compare(self, left, right) -> dict:
        from pllm.runtime.model_binding import CompiledRuntimeModel, _state_numeric_contract

        if type(left) is not CompiledRuntimeModel or type(right) is not CompiledRuntimeModel:
            raise ValueError("state compatibility needs two checked bindings")
        for value in (left, right):
            value.validate()
        rows = [_state_numeric_contract(value) for value in (left, right)]
        differences = [key for key in rows[0] if rows[0][key] != rows[1][key]]
        return {"schema": "pllm.state_compatibility_probe.v1", "left_binding": left.digest,
            "right_binding": right.digest, "contract_equal": not differences,
            "different_contract_fields": differences, "numeric_state_digest": _digest(rows[0]),
            "live_state_transfer_authorized": False,
            "scope": "checked public contract comparison, not state or verifier provenance transfer"}
