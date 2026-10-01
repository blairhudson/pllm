import numpy as np


def test_wire_threshold_uses_weight_l1_not_nominal_bit_count(monkeypatch):
    from pathlib import Path

    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    from probe_wire_precision import wire_bits

    high = np.full((2, 896), 100, np.int8)
    assert wire_bits(high, 8) == 32
    assert wire_bits(high, 7) == 24
    assert wire_bits(high, 6) == 24
    small = np.ones((2, 896), np.int8)
    assert wire_bits(small, 8) == 24
    assert wire_bits(small, 6) == 16


def test_ring_bound_covers_extreme_dot_before_center_decode(monkeypatch):
    from pathlib import Path

    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    from probe_wire_precision import wire_bits

    rng = np.random.default_rng(911)
    for bits in (4, 6, 7, 8):
        qmax = (1 << (bits - 1)) - 1
        weight = rng.integers(-128, 128, (5, 896), dtype=np.int8)
        inputs = qmax * np.sign(weight[0]).astype(np.int64)
        dot = weight.astype(np.int64) @ inputs
        ring = 1 << wire_bits(weight, bits)
        residues = dot % ring
        decoded = np.where(residues >= ring // 2, residues - ring, residues)
        np.testing.assert_array_equal(decoded, dot)


def test_locked_mixed_precision_cohort_and_rejection_are_consistent(monkeypatch):
    import json
    from pathlib import Path

    root = Path(__file__).parents[1]
    monkeypatch.syspath_prepend(str(root / "scripts"))
    from decoder_probe_support import digest

    report = json.loads(
        (root / "docs/evidence/conditional-execution-qwen25-2026-10-01.json").read_text()
    )["mixed_precision"]
    cohorts = json.loads((root / report["public_cohort"]).read_text())
    assert digest(cohorts) == report["public_cohort_digest"]
    assert digest({"mlp_down": report["selected_override"]}) == report["override_digest"]
    cohort = report["matched_requests"]["39+32"]
    saving = (
        cohort["baseline_online_arithmetic_bytes"]
        + cohort["baseline_correction_arithmetic_bytes"]
        - cohort["mixed_online_arithmetic_bytes"]
        - cohort["mixed_correction_arithmetic_bytes"]
    )
    assert saving == report["selected_override"]["saved_bytes"]
    assert cohort["free_generation_matches"] < cohort["outputs"]
    assert report["heldout"]["prefill_matches"] < report["heldout"]["samples"]


def test_locked_speculation_charges_every_rejected_row():
    import json
    from pathlib import Path

    report = json.loads(
        (
            Path(__file__).parents[1] / "docs/evidence/conditional-execution-qwen25-2026-10-01.json"
        ).read_text()
    )
    for row in report["speculation"]["requests"].values():
        assert row["greedy_parity"]
        assert row["target_decode_rows"] == row["baseline_decode_rows"] + row["rejected_rows"]
        assert row["stage_rows"] == 96 * (39 + row["target_decode_rows"])
        assert row["stage_calls"] == 96 * (1 + row["target_decode_rows"])
    assert report["certification"]["certified_before_full_head"] == 0
    assert report["certification"]["provider_bytes_avoided"] == 0
