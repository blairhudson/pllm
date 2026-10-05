"""Independent, non-selectable exact-algebra and graph-demand research oracles.

Public weights/graphs determine every layout. Private values cannot select a
codec, mask width, parent, or traffic capacity. No protocol is activated here.
"""
from __future__ import annotations

from collections import deque
import hashlib
import math

import numpy as np


def terminal_stage_ids(compiled) -> frozenset[str]:
    """Conservative backward row demand, with persistent state as an ALL root.

    Stop LAST propagation at layout, attention, state and unknown operators.
    Only whole remote groups whose every output is LAST can lose rows. The
    research executor still runs local operations at their original shapes.
    """
    import json
    from pllm.configuration import Pipeline
    graph = compiled._plan.to_dict()["prefill"]
    composition = Pipeline.from_spec(json.loads(compiled._canonical_composition))
    schedule = compiled._plan.runtime_schedule(composition).to_dict()["prefill"]
    bindings = {operation: stage.stage_id for stage in compiled.stage_bindings
                for operation in stage.semantic_operations}
    return terminal_groups(graph, schedule, bindings)


def terminal_groups(graph: dict, schedule: dict, bindings: dict) -> frozenset[str]:
    operations = {op["id"]: op for op in graph["operations"]}
    pointwise = {"linear", "output_head", "rms_norm", "residual_add", "silu",
                 "elementwise_multiply", "gated_multiply", "multiply", "gelu_tanh", "softcap"}
    demand: dict[str, int] = {}
    queue = deque([(graph["output"], 2), *((s["id"], 2) for s in graph["state_outputs"])])
    while queue:
        identity, wanted = queue.popleft()
        if identity not in operations or demand.get(identity, 0) >= wanted:
            continue
        demand[identity] = wanted
        op = operations[identity]
        upstream = 1 if op["operator"] == "last_token" else wanted if op["operator"] in pointwise else 2
        queue.extend((source, upstream) for source in op["inputs"])
    result = set()
    for step in schedule["steps"]:
        ids = step["operation_ids"]
        if (step["executor"] == "remote_stage"
                and all(operations[key]["operator"] == "linear" and demand.get(key) == 1 for key in ids)):
            stages = {bindings[f"prefill:{key}"] for key in ids}
            if len(stages) != 1:
                raise ValueError("group lacks a unique bound stage")
            result.update(stages)
    return frozenset(result)


def residue_widths(weight: np.ndarray, qmax: int = 127) -> np.ndarray:
    value = np.asarray(weight)
    if value.ndim != 2 or value.dtype.kind not in "iu" or value.size == 0:
        raise ValueError("expected a nonempty integer matrix")
    bounds = np.sum(np.abs(value.astype(np.int64)), axis=1) * qmax
    return np.asarray([max(1, int(bound).bit_length() + 1) for bound in bounds], np.uint8)


def packed_size(widths: np.ndarray, rows: int) -> int:
    if type(rows) is not int or rows < 0:
        raise ValueError("rows must be nonnegative")
    return (int(np.sum(widths, dtype=np.int64)) * rows + 7) // 8


def integer_anchors(weight: np.ndarray, block: int) -> tuple[np.ndarray, np.ndarray]:
    """Exact W = R + A S; S sums disjoint public input-coordinate blocks.

    Choose lower medians, constrained to keep R signed-i8. No calibration,
    private inputs, floating-point folding or changed quantization boundaries.
    """
    weight = np.asarray(weight)
    if weight.dtype != np.int8 or weight.ndim != 2 or block < 1 or weight.shape[1] % block:
        raise ValueError("anchor blocks must exactly divide signed-i8 input width")
    groups = weight.astype(np.int16).reshape(weight.shape[0], -1, block)
    anchor = np.partition(groups, (block - 1) // 2, axis=2)[..., (block - 1) // 2]
    lower = groups.max(axis=2) - 127
    upper = groups.min(axis=2) + 128
    anchor = np.minimum(np.maximum(anchor, lower), upper).astype(np.int16)
    residual = groups - anchor[..., None]
    if residual.min() < -128 or residual.max() > 127:
        raise RuntimeError("anchor residual escaped the original native kernel domain")
    return residual.reshape(weight.shape).astype(np.int8), anchor


def anchor_output(anchor: np.ndarray, inputs: np.ndarray, block: int) -> np.ndarray:
    value = np.asarray(inputs, dtype=np.int64)
    sums = value.reshape(value.shape[0], -1, block).sum(axis=-1)
    return sums @ anchor.astype(np.int64).T


def lifting_forest(weight: np.ndarray, *, neighbours: int = 8) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    """Greedy unit-triangular row lifting; admit only an exact wire-bit saving.

    Bounded preceding neighbours avoid a quadratic full-checkpoint search. The
    original parent row is used, so inverse additions follow increasing rows.
    Identity is always available. Coefficients may be i16 in this cost oracle.
    """
    if weight.dtype != np.int8 or weight.ndim != 2 or not 1 <= neighbours <= 32:
        raise ValueError("invalid bounded row-lifting input")
    source = weight.astype(np.int16)
    residual = source.copy()
    widths = residue_widths(source)
    edges = []
    for offset in range(1, neighbours + 1):
        for sign in (-1, 1):
            candidate = source[offset:] - sign * source[:-offset]
            bits = residue_widths(candidate)
            improve = bits < widths[offset:]
            indices = np.flatnonzero(improve) + offset
            residual[indices] = candidate[improve]
            widths[indices] = bits[improve]
            for row in indices.tolist():
                edges.append((row, row - offset, sign))
    # A later, cheaper edge supersedes an earlier candidate for the same row.
    selected = {row: (row, parent, sign) for row, parent, sign in edges}
    return residual, [selected[row] for row in sorted(selected)]


def restore_lifted(outputs: np.ndarray, edges: list[tuple[int, int, int]]) -> np.ndarray:
    result = np.asarray(outputs, dtype=np.int64).copy()
    previous = -1
    for row, parent, sign in edges:
        if not previous < row < result.shape[-1] or not 0 <= parent < row or sign not in (-1, 1):
            raise ValueError("lifting forest is not public acyclic row order")
        result[..., row] += sign * result[..., parent]
        previous = row
    return result


def snapshot_digest(runtime) -> str:
    """Hash only executed KV, never uninitialized geometric spare capacity."""
    digest = hashlib.sha256()
    digest.update(runtime.position.to_bytes(8, "big"))
    for cache in runtime.caches:
        for value in (cache.key, cache.value):
            digest.update(value[:cache.length].astype("<f4", copy=False).tobytes())
    return digest.hexdigest()


def symmetry_privacy_witness() -> dict:
    """Conjugating a known public linear operator does not hide its basis key."""
    rng = np.random.default_rng(61007)
    public = rng.normal(size=(24, 8))
    basis, _ = np.linalg.qr(rng.normal(size=(8, 8)))
    transformed = public @ basis.T
    private = rng.normal(size=8)
    transcript = basis @ private
    recovered_basis_t = np.linalg.lstsq(public, transformed, rcond=None)[0]
    recovered = recovered_basis_t @ transcript
    error = float(np.max(np.abs(recovered - private)))
    if error > 1e-10:
        raise RuntimeError("public-weight basis recovery witness failed")
    return {"candidate": "secret-basis conjugate decoder", "public_operator_rank": 8,
            "private_input_recovery_max_error": error,
            "decision": "rejected: public operator pair reveals the supposedly secret orthogonal key"}


def rounding_debt_witness() -> dict:
    """Source-preserving layer-span fusion must carry intermediate rounding debt."""
    # ties-to-even: round(1/2)*3 = 0, but rounding the fused 3/2 gives 2.
    early = int(np.rint(1 / 2)) * 3
    late = int(np.rint(3 / 2))
    return {"candidate": "layer-span rounding-debt certificates",
            "early_rounding_output": early, "delayed_rounding_output": late,
            "exact_fusion_without_debt_is_false": early != late,
            "debt_bound_for_k_rescalings": "sum_i (0.5 * product_{j>i} ||W_j||_infinity)",
            "decision": "requires private certified debt settlement, not merely postponed rounding"}


def century_budgets(*, body_bytes: int, rows: int, layers: int, hidden: int, outputs: int) -> dict:
    """Necessary, optimistic costs; none substitutes for a private decoder."""
    target = body_bytes / 100
    one_opening = 2 * rows * hidden * 4
    return {"covered_control_bytes": body_bytes, "target_100x_bytes": target,
            "target_bytes_per_generated_token": target / outputs,
            "two_party_u32_full_width_opening_bytes": one_opening,
            "full_width_openings_affordable_if_all_else_free": math.floor(target / one_opening),
            "one_opening_per_layer_bytes": layers * one_opening,
            "one_opening_per_layer_already_exceeds_target": layers * one_opening > target,
            "full_wire_bytes": None}
