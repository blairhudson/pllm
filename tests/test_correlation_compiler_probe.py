"""Small exact algebra and rejection checks; no measurements or model loads."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

PATH = Path(__file__).resolve().parents[1] / "scripts/probe_correlation_compiler.py"
spec = importlib.util.spec_from_file_location("correlation_probe", PATH)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


@pytest.mark.parametrize("block", [1, 4, 8, 12])
def test_subset_exact_full_ring_and_partial_block(block):
    weights = np.random.default_rng(19).integers(-128, 128, (7, 13), dtype=np.int16).astype(np.int8)
    compiler = probe.SubsetCompiler(weights, block)
    for r in [np.full(13, 0xffffffff, dtype=np.uint32), np.zeros(13, dtype=np.uint32),
              np.arange(13, dtype=np.uint32) * np.uint32(0x12345678)]:
        expected = [sum(int(w) * int(v) for w, v in zip(row, r)) & probe.MASK for row in weights]
        assert compiler.project(r).tolist() == expected
    assert probe.oracle_check(weights, compiler.project) == 21
    assert sum(table.nbytes for _, _, table in compiler.tables) == compiler.cost["table_bytes"]


def test_preflight_mlp_and_validation():
    # Regression fixtures only; production geometry comes from compiler, not these constants.
    for m, n in [(9728, 896), (896, 4864)]:
        assert probe.preflight(m, n, 8)["status"] == "rejected_before_weights"
        assert probe.preflight(m, n, 4)["status"] == "admitted_geometry"
    assert probe.preflight(896, 896, 12)["status"] == "rejected_before_weights"
    for args in [(1, 1, 13), (0, 1, 4), (True, 1, 4)]:
        with pytest.raises(ValueError):
            probe.preflight(*args)


def test_failed_response_and_cancel_burn():
    for failure in ("shape", "binding", "dtype", "cancel"):
        material = probe.ClientMaterial(np.ones(3, dtype=np.uint32), np.ones(2, dtype=np.uint32), b"bound")
        if failure == "cancel":
            material.cancel()
        reply = np.ones(3 if failure == "shape" else 2, dtype=np.int64 if failure == "dtype" else np.uint32)
        with pytest.raises(ValueError):
            material.finish(reply, b"wrong" if failure == "binding" else b"bound")
        assert not material.live
        assert not material.r.any() and not material.s.any()
        with pytest.raises(ValueError):
            material.finish(np.ones(2, dtype=np.uint32), b"bound")


def test_xof_separation_and_fresh_issue():
    seed = bytes(32)
    r = probe.expand(seed, b"input-mask", b"context", 16)
    assert not np.array_equal(r, probe.expand(seed, b"output-mask", b"context", 16))
    assert not np.array_equal(r, probe.expand(seed, b"input-mask", b"other", 16))
    weights = np.ones((3, 4), dtype=np.int8)
    a, _ = probe.issue(lambda r: probe.dense_project(weights, r), 3, 4, b"a")
    b, _ = probe.issue(lambda r: probe.dense_project(weights, r), 3, 4, b"b")
    assert not np.array_equal(a.r, b.r) and not np.array_equal(a.s, b.s)
    a.cancel()
    b.cancel()


def test_horizon_keeps_regression_and_mismatch():
    base = dict(rows=3, columns=4, block=4, weights_sha256="public", setup_elapsed_s=1,
                mode="native_wrap32", observations=[dict(issuance_elapsed_s=2)])
    faster = base | dict(mode="subset", setup_elapsed_s=6, observations=[dict(issuance_elapsed_s=1)])
    assert probe.paired_horizons([base, faster] * 3)[0]["empirical_elapsed_amortization_samples"] == 5
    same = base | dict(mode="subset")
    assert probe.paired_horizons([base, same] * 3)[0]["empirical_elapsed_amortization_samples"] is None
    with pytest.raises(ValueError):
        probe.paired_horizons([base, faster | dict(weights_sha256="different")] * 3)
    with pytest.raises(ValueError):
        probe.paired_horizons([base, faster])


def test_balanced_fresh_jobs_and_declared_pruning():
    cases = [(64, 64, 4), (32, 32, 4)]
    for modes in [("native_wrap32", "subset"), probe.MODES]:
        jobs = list(probe.cohort_jobs(cases, 4, modes=modes))
        assert len(jobs) == len(cases) * len(modes) * 3
        for m, n, b in cases:
            cell = [row for row in jobs if (row["rows"], row["columns"], row["block"]) == (m, n, b)]
            for mode in modes:
                matching = [row for row in cell if row["mode"] == mode]
                assert {row["process_repetition"] for row in matching} == {0, 1, 2}
                assert len({row["execution_order"] for row in matching}) == len(modes)
    gates = probe.timing_selection([probe.preflight(896, 896, b) for b in (4, 8, 12)])
    assert [g["timing_retained"] for g in gates] == [True, False, False]
    assert gates[2]["timing_gate"] == "resource_preflight_rejection"


def test_source_change_rejected(tmp_path):
    import hashlib
    path = tmp_path / "helper.py"
    path.write_text("before")
    frozen = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()}
    probe.assert_sources_unchanged(frozen)
    path.write_text("after")
    with pytest.raises(RuntimeError, match="cohort source changed"):
        probe.assert_sources_unchanged(frozen)


def test_installed_native_exact_full_ring():
    # Functional only: no timing cohort, build or checkpoint loads.
    from pllm.runtime.native import capabilities
    weights = np.array([[-128, 127, 0], [23, -9, 3]], dtype=np.int8)
    r = np.array([0xffffffff, 0x80000000, 7], dtype=np.uint32)
    modes = ["native_wrap32"] + (["native_digit8"] if capabilities().get("digit8_cpu") else [])
    for mode in modes:
        projector, library = probe.native_projector(weights, mode)
        assert library.is_file()
        assert projector(r).tolist() == probe.dense_project(weights, r).tolist()
        assert probe.oracle_check(weights, projector, 1) == 2
    frozen = probe.freeze_sources([library])
    from pllm.runtime import coded_resources
    assert str(Path(coded_resources.__file__).resolve()) in frozen
