"""Pure metadata scratch estimates shared with local F32 execution.

No allocation authorization or protected-source ownership is conveyed here.
Output storage is counted separately by the live-root estimator.
"""
from __future__ import annotations

import math
import sys

from .bundle_storage import CHUNK_BYTES


def semantic_local_workspace_bytes(operation, actual_shape):
    """Additional scratch, excluding one F32 output of actual_shape.

    Actual dimensions may shrink compiled request bounds, never exceed them.
    LayerNorm uses one F32 row scratch and bounded finite/reduction temporaries;
    GeluTanh uses row scratch. Lookup scratch scales with requested IDs/results,
    never checkpoint table cardinality.
    """
    declared = operation.get("output_shape")
    if (type(actual_shape) not in (tuple, list) or not actual_shape
            or type(declared) not in (tuple, list) or len(actual_shape) != len(declared)
            or any(type(n) is not int or type(bound) is not int or not 0 < n <= bound
                   for n, bound in zip(actual_shape, declared))):
        raise ValueError("local workspace geometry exceeds compiled output bounds")
    elements = math.prod(actual_shape)
    if actual_shape[-1] != declared[-1]:
        raise ValueError("local workspace width differs from compiled output width")
    if elements > sys.maxsize // 4:
        raise ValueError("local workspace geometry exceeds addressable storage")
    kind = operation.get("operator")
    width = actual_shape[-1]
    # Finite-result masks are bounded by the shared import chunk, not tables.
    finite = min(elements, CHUNK_BYTES // 4)
    if kind in {"layer_norm", "gelu_tanh"}:
        return 4 * width + finite + 8 * 8
    if kind == "token_lookup":
        rows = math.prod(actual_shape[:-1])
        # intp ID conversion plus lower/upper/combined validity masks. The
        # result's finite mask may overlap those owners during error checks.
        # Lookup currently compares the full request's IDs, without chunking.
        return rows * (8 + 3) + finite + 8 * 8
    raise ValueError("local workspace operator is not qualified")
