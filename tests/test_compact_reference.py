"""The Compact slice is a numeric reference, not an executable component."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from pllm.components import NotYetImplementedError, get, planned_component
from pllm.nonlinear import CompactPiecewiseActivation, fit_compact_silu_q7_reference


def _counts(values: dict[int, int] | None = None) -> bytes:
    counts = [0] * 257
    for encoded, count in (values or {}).items():
        counts[encoded + 128] = count
    return struct.pack("<257I", *counts)


def test_bounded_native_profile_is_immutable_and_deterministic() -> None:
    profile = fit_compact_silu_q7_reference(_counts(), max_pieces=4)
    assert len(profile.digest) == len(profile.calibration_digest) == 32
    assert profile.digest == fit_compact_silu_q7_reference(_counts()).digest
    assert profile.pieces[0][0] == -128
    assert profile.pieces[-1][1] == 128
    assert 1 <= len(profile.pieces) <= 4
    assert profile.evaluate(0) == 0
    assert profile.weighted_error[1] == 257
    assert profile.maximum_encoded_error <= 2
    registry = json.loads(
        (Path(__file__).resolve().parents[1] / "docs/data/research/papers.json").read_text()
    )
    source = next(paper for paper in registry["papers"] if paper["id"] == "R18")
    assert profile.source_sha256 == source["source_lock"]["paper_sha256"]
    with pytest.raises(AttributeError):
        profile.digest = b"forged"
    visible = profile.pieces
    visible.clear()
    assert profile.pieces


def test_public_calibration_changes_the_locked_profile() -> None:
    histogram = _counts({encoded: 1_000_000 for encoded in range(-100, -79)})
    adapted = fit_compact_silu_q7_reference(histogram, max_pieces=2)
    uniform = fit_compact_silu_q7_reference(_counts(), max_pieces=2)
    assert adapted.calibration_digest != uniform.calibration_digest
    assert adapted.digest != uniform.digest
    assert adapted.weighted_error[1] == 21_000_257


def test_boundary_rejects_invalid_material_and_out_of_domain_inputs() -> None:
    with pytest.raises(TypeError, match="immutable bytes"):
        fit_compact_silu_q7_reference(bytearray(_counts()))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="1028 bytes"):
        fit_compact_silu_q7_reference(b"\x00" * 1027)
    with pytest.raises(ValueError, match="PieceBudget"):
        fit_compact_silu_q7_reference(_counts(), max_pieces=0)
    with pytest.raises(ValueError, match="CalibrationCount"):
        fit_compact_silu_q7_reference(_counts({-128: 1_000_001}))
    profile = fit_compact_silu_q7_reference(_counts())
    for encoded in (-129, 129):
        with pytest.raises(ValueError, match="InputOutOfRange"):
            profile.evaluate(encoded)


def test_numeric_reference_does_not_promote_compact_to_an_executable_component() -> None:
    record = planned_component("compact")
    assert record.status == "pending"
    with pytest.raises(NotYetImplementedError, match="compact"):
        CompactPiecewiseActivation()
    with pytest.raises(NotYetImplementedError, match="compact"):
        get(record.identity)
