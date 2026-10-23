"""Tiny source-paper k=1..3 causal-row acceptance fixtures; NOT a protected region.

Equations: docs/research/sigma-normalization-softmax-audit.md, causal-softmax
section. Reuses committed D29 lane()/load_tables(); no exp evaluation or repeated
floor certification. Paper q(k)=6+ceil(log2(k+1)) is 7,8,8 for k=1,2,3;
the common q8 inverse embeds k=1's q7 indices without changing their values.
This is not the CPU artifact's fixed16 lookup or an adopted S-S numeric profile.

For all true signed38 scores, max-active minus an active score is in [0,2**38).
Signed subtraction/comparison needs signed39, NOT signed38. E is in [0,4096]
and some active maximum gives E=4096. Thus 4096<=D<=4096*k, 64<=Z<=64*k,
floor(2**18/(64*k))<=R<=4096 and 0<=E*R<=2**24. The final product therefore
fits the ring50 GapARS quarter-ring premise, independently of uniform masks.
D uses unsigned14 and Z unsigned8; they are not signed14/signed8 quantities.
All bounds are arithmetic implications of the clear-input premise, NOT an
original-owner input/source range certificate, composition/privacy proof, trained
quality target, model run, or implementation/adoption approval.

Run only small named fixtures: python -B -m benchmarks.research.sigma_causal_row_integer_oracle
"""
from __future__ import annotations

from hashlib import sha256
import json
import struct

from benchmarks.research.sigma_nexp_integer_classes import (
    INPUT_LIMIT,
    PINS as NEXP_PINS,
    lane as nexp_lane,
    load_tables,
)

SCALE = 4096
SCORE_MIN = -(1 << 37)
SCORE_LIMIT = 1 << 37
SOURCE_COMMIT = "aa0f2f860266fb79901405ef5e9965b8c9fe2e88"
PAPER_SHA256 = "f911b715b0e0523fa4d77f21c28c6a0440c5be781e67b5b30419f20e505a6ebf"
_TABLES = load_tables()  # Immutable pinned literals; identity check, NOT a floor proof.


def inverse_q8(index: int) -> int:
    """Complete proposed public table; index0 is an unreachable zero sentinel."""
    if type(index) is not int or not 0 <= index < 256:
        raise ValueError("q8 inverse index must be an exact unsigned8 integer")
    return 0 if index == 0 else (1 << 18) // index


def inverse_table_sha256() -> str:
    return sha256(struct.pack("<256Q", *(inverse_q8(i) for i in range(256)))).hexdigest()


def arithmetic_bounds(active_count: int) -> dict[str, list[int]]:
    """Inclusive endpoints, derived above; no admission or source authority."""
    if type(active_count) is not int or not 1 <= active_count <= 3:
        raise ValueError("public active count must be 1..3")
    return {
        "score": [SCORE_MIN, SCORE_LIMIT - 1],
        "active_delta": [0, INPUT_LIMIT - 1],
        "exp": [0, SCALE],
        "sum": [SCALE, SCALE * active_count],
        "denominator": [64, 64 * active_count],
        "reciprocal": [(1 << 18) // (64 * active_count), SCALE],
        "final_product": [0, 1 << 24],
        "output": [0, SCALE],
    }


def clear_row(scores: list[int] | tuple[int, ...], active_count: int | None = None) -> dict:
    """Exact floor map on a public active prefix; inactive scores never enter max/E/D.

    All stored score coordinates must satisfy the declared signed38 domain.
    None delta/coordinate denotes an excluded lane, not finite negative infinity.
    The returned clear witnesses are research data, NOT protected openings.
    """
    if type(scores) not in (tuple, list) or not 1 <= len(scores) <= 3:
        raise ValueError("fixture requires 1..3 score coordinates")
    if any(type(x) is not int or not SCORE_MIN <= x < SCORE_LIMIT for x in scores):
        raise ValueError("score outside exact signed38 domain")
    k = len(scores) if active_count is None else active_count
    if type(k) is not int or not 1 <= k <= len(scores):
        raise ValueError("a nonempty public causal prefix is required")
    maximum = max(scores[:k])
    deltas, coordinates, exps = [], [], []
    for i, score in enumerate(scores):
        if i >= k:
            deltas.append(None)
            coordinates.append(None)
            exps.append(0)
            continue
        delta = maximum - score
        trace = nexp_lane(delta, _TABLES)
        deltas.append(delta)
        coordinates.append(trace["c16"])
        exps.append(trace["e"])
    total = sum(exps)
    denominator = total // 64  # Sum first; never sum individually truncated E.
    reciprocal = inverse_q8(denominator)
    products = [e * reciprocal for e in exps]
    return {
        "scores": list(scores), "active_count": k, "maximum": maximum,
        "delta": deltas, "coordinate": coordinates, "exp": exps,
        "sum": total, "denominator": denominator, "reciprocal": reciprocal,
        "product": products, "output": [p // SCALE for p in products],
    }


def named_fixtures() -> dict[str, dict]:
    """Thirteen fixed examples, not a quality campaign or full-domain sweep."""
    cases = {
        "first_public_row": ((-1, SCORE_LIMIT - 1, SCORE_MIN), 1),
        "equal_two": ((0, 0), 2),
        "equal_three": ((0, 0, 0), 3),
        "not_renormalized": ((0, -4096), 2),
        "sum_before_tr6": ((0, -4096, -4096), 3),
        "carry_before": ((0, -1279), 2),
        "carry_after": ((0, -1280), 2),
        "first_exp_zero": ((0, -31233), 2),
        "zero_before_reset": ((0, -31487), 2),
        "zero_reset": ((0, -31488), 2),
        "clip_last": ((0, -65535), 2),
        "clip_next": ((0, -65536), 2),
        "signed38_endpoints": ((SCORE_MIN, SCORE_LIMIT - 1), 2),
    }
    return {name: clear_row(scores, k) for name, (scores, k) in cases.items()}


if __name__ == "__main__":
    print(json.dumps({
        "scope": "Source-paper integer fixtures only; no protected S-S implementation/adoption.",
        "source_commit": SOURCE_COMMIT, "paper_sha256": PAPER_SHA256,
        "exp_table_u64le_sha256": NEXP_PINS,
        "inverse_q8_u64le_sha256": inverse_table_sha256(),
        "bounds": {str(k): arithmetic_bounds(k) for k in (1, 2, 3)},
        "fixtures": named_fixtures(),
    }, indent=2))
