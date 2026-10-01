"""Bounded decision-directed refinement of a single quantized linear head.

The body and KV must already be exact. This does not certify omitted decoder
layers. Public coefficient differences bound a shared unresolved input, retaining
correlation that independent logit intervals discard. No provider gets inputs.
"""

from __future__ import annotations

import numpy as np


def refine_linear_choice(weights, values, scales, activation_scale, *, block=128) -> dict:
    w, x = np.asarray(weights), np.asarray(values)
    s = np.asarray(scales)
    if (
        w.dtype != np.int8
        or w.ndim != 2
        or min(w.shape) < 1
        or w.shape[1] > 4096
        or x.dtype != np.int8
        or x.shape != (w.shape[1],)
        or s.dtype != np.float32
        or s.shape != (w.shape[0],)
        or not np.all(np.isfinite(s))
        or np.any(s <= 0)
        or np.any(s > 1)
        or not np.isfinite(activation_scale)
        or not 0 < activation_scale <= 65536
        or type(block) is not int
        or block < 1
        or np.any(x == -128)
    ):
        raise ValueError("bounded Q8 head requires weight scales <=1 and activation scale <=65536")
    a = float(np.float32(activation_scale))
    if a == 0:
        raise ValueError("activation scale underflows float32")
    partial = np.zeros(w.shape[0], dtype=np.int64)
    maximum = np.sum(np.abs(w.astype(np.int16)), axis=1, dtype=np.int64) * 127
    # Three float32 roundings (int cast, activation scale, weight scale).
    # Extra 1e-12 covers float64 bound arithmetic; additive term covers subnormals.
    u = 2.0**-24
    rounding = (3 * u / (1 - 3 * u) + 1e-12) * maximum * a * s.astype(np.float64)
    rounding += 4 * float(np.nextafter(np.float32(0), np.float32(1)))
    refinements = 0
    for start in range(0, w.shape[1], block):
        stop = min(start + block, w.shape[1])
        partial += w[:, start:stop].astype(np.int64) @ x[start:stop].astype(np.int64)
        refinements += 1
        approximate = partial.astype(np.float32) * np.float32(a) * s
        candidate = int(np.argmax(approximate))
        if stop == w.shape[1]:
            return {
                "selected": candidate,
                "resolved_features": stop,
                "refinements": refinements,
                "certified_before_full_head": False,
                "strict_margin_lower_bound": None,
            }
        ideal = partial.astype(np.float64) * a * s.astype(np.float64)
        worst = float("inf")
        for first in range(0, w.shape[0], 1024):
            last = min(first + 1024, w.shape[0])
            left = w[first:last, stop:].astype(np.float64) * s[first:last, None]
            right = w[candidate, stop:].astype(np.float64) * float(s[candidate])
            # Round subtraction outward before summing absolute coefficient differences.
            delta = np.abs(left - right) + (2.0**-52) * (np.abs(left) + np.abs(right))
            tail = np.nextafter(np.sum(delta, axis=1) * (127 * a) * (1 + 1e-12), np.inf)
            margins = np.nextafter(
                ideal[candidate]
                - ideal[first:last]
                - tail
                - rounding[candidate]
                - rounding[first:last],
                -np.inf,
            )
            if first <= candidate < last:
                margins[candidate - first] = np.inf
            worst = min(worst, float(np.min(margins)))
        if worst > 0:
            return {
                "selected": candidate,
                "resolved_features": stop,
                "refinements": refinements,
                "certified_before_full_head": True,
                "strict_margin_lower_bound": worst,
            }
    raise AssertionError("refinement did not reach full fallback")
