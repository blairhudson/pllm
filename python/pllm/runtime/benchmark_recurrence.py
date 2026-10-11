"""Source-shaped exact recurrence through the installed public state owners."""
from __future__ import annotations

import hashlib
import math
import time

import numpy as np

from pllm._benchmark_contract import TargetUnavailable, digest
from pllm._benchmark_worker import peak_rss


def execute_recurrence(p, m):
    from pllm.state import BoundedGatedDeltaRecurrence, OwnedGatedDeltaRecurrence
    from .benchmark_targets import model_context

    context = model_context(p, m)
    plan = context[2]
    operations = [op for op in plan.decode["operations"]
                  if op["operator"] == "gated_delta_state_update" and op["layer"] == p["layer"]]
    if len(operations) != 1:
        raise TargetUnavailable("exclusive_recurrence_required")
    operation = operations[0]
    _, vh, kd, vd = operation["output_shape"]
    repeats = operation["attributes"]["key_head_repeats"]
    geometry = (vh // repeats, vh, kd, vd)
    tokens = p["tokens"]
    if tokens > plan.decode["maximum_key_sequence"]:
        raise TargetUnavailable("recurrence_exceeds_compiled_bound", status="blocked")
    if tokens * vh * kd * vd * (m.sampling.repetitions + m.sampling.warmups) > 1 << 29:
        raise TargetUnavailable("recurrence_work_limit", status="blocked")
    state_bytes = 4 * vh * kd * vd
    workspace_bytes = 4 * vh * (kd * vd + 2 * kd + vd)
    m.admit((160 << 20) + 6 * state_bytes + workspace_bytes + 32 * vh * (kd + vd))
    rng = np.random.default_rng(p["seed"])
    q = rng.normal(size=(1, 1, geometry[0], kd)).astype(np.float32)
    k = rng.normal(size=q.shape).astype(np.float32)
    v = rng.normal(size=(1, 1, vh, vd)).astype(np.float32)
    g = -rng.random((1, 1, vh), dtype=np.float32)
    beta = rng.random(g.shape, dtype=np.float32)
    reference = BoundedGatedDeltaRecurrence(*geometry)

    def execute(_):
        h = hashlib.sha256()
        if p["owner"] == "owned":
            with OwnedGatedDeltaRecurrence(*geometry, maximum_tokens=tokens) as owner:
                for _ in range(tokens):
                    h.update(owner.evaluate(q, k, v, g, beta).tobytes())
                h.update(owner.snapshot().tobytes())
        else:
            state = reference.initial_state()
            try:
                for _ in range(tokens):
                    output, state = reference.evaluate(q, k, v, g, beta, state, mode="decode")
                    h.update(output.tobytes())
                h.update(state.tobytes())
            finally:
                state.fill(0)
        return h.hexdigest()

    observed = m.run(execute)
    m.details["target_peak_rss_bytes"] = peak_rss()
    m.details["target_cpu_from_birth_seconds"] = time.process_time()
    oracle_start = time.process_time()
    expected, state = hashlib.sha256(), reference.initial_state()
    for _ in range(tokens):
        output, state = reference.evaluate(q, k, v, g, beta, state, mode="decode")
        expected.update(output.tobytes())
    expected.update(state.tobytes())
    state.fill(0)
    passed = all(value == expected.hexdigest() for value in observed)
    m.oracle = {"kind": "float32_copy_reference_exact_bits", "outcome": "passed" if passed else "failed",
                "checked_values": len(observed) * (tokens * vh * vd + math.prod(operation["output_shape"]))}
    if not passed:
        raise TargetUnavailable("recurrence_parity_failed", status="failed")
    fixture = {"geometry": list(geometry), "layer": p["layer"], "seed": p["seed"]}
    m.fixture = {"kind": "synthetic", "digest": digest(fixture),
                 "metadata": {**fixture, "contract": "projected_gated_delta_recurrence"}}
    m.comparison.update(numeric_digest=digest({"operator": "float32_gated_delta_v1", "geometry": geometry}),
        state_digest=digest({"state": operation["output_shape"]}),
        workload_digest=digest({"fixture": fixture, "tokens": tokens}),
        output_digest=m.commitment(observed),
        measurement_digest=digest({"scope": "state_construction_transitions_hashing_retirement_v1"}))
    m.details.update(geometry=list(geometry), tokens=tokens, owner=p["owner"],
                     oracle_cpu_seconds=time.process_time() - oracle_start,
                     retained_state_bytes=state_bytes,
                     persistent_workspace_bytes=workspace_bytes if p["owner"] == "owned" else 0)
    m.metric("state.retained_bytes", state_bytes, kind="allocated", scope="owned_payload")
    m.metric("state.workspace_bytes", m.details["persistent_workspace_bytes"], kind="allocated", scope="owned_payload")
    m.limitations += ["Local clear projected inputs; weights, full decoder, role transport and serving metrics are excluded.",
                     "Timed work includes output hashing and retirement; the independent reference pass follows peak capture."]
