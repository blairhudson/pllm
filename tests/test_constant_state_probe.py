"""Low-memory checks: NumPy only, no checkpoint, Torch or native build."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("constant_state_probe",Path(__file__).resolve().parents[1]/"scripts/probe_constant_state.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)
SOURCE_IDENTITY = probe.source_identity()


@pytest.fixture(autouse=True)
def freeze_sources_before_and_after_checks():
    with probe.frozen_sources(SOURCE_IDENTITY):
        yield


def test_matches_existing_semantic_reference():
    assert probe.functional()["checks"] == 17


def test_real_checkpoint_state_geometry_without_weights():
    result = probe.functional(geometry=(16,32,128,128),steps=4)
    assert result["owned_bytes"] == 4259840


def test_state_and_workspace_do_not_grow():
    owner = probe.OwnedGatedDelta(2,4,8,6)
    pointers = [x.ctypes.data for x in (owner.state,owner.outer,owner.output)]
    size = owner.owned_bytes
    for _ in range(300):
        owner.step(*probe.inputs())
    assert owner.owned_bytes == size
    assert pointers == [x.ctypes.data for x in (owner.state,owner.outer,owner.output)]


def test_invalid_input_does_not_mutate():
    owner = probe.OwnedGatedDelta(2,4,8,6)
    x = list(probe.inputs())
    owner.step(*x)
    prior = owner.state.copy()
    x[-1].fill(2)
    with pytest.raises(ValueError,match="invalid"):
        owner.step(*x)
    np.testing.assert_array_equal(prior,owner.state)
    assert owner.position == 1


def test_horizon_fail_closed():
    owner = probe.OwnedGatedDelta(2,4,8,6)
    owner.position = 4096
    with pytest.raises(ValueError,match="horizon"):
        owner.step(*probe.inputs())


def test_bad_geometry_rejected():
    with pytest.raises(ValueError):
        probe.OwnedGatedDelta(3,4,8,6)


def test_numeric_failure_burns_owner():
    owner = probe.OwnedGatedDelta(2,4,8,6)
    owner.state.fill(np.finfo(np.float32).max)
    x = list(probe.inputs())
    x[1].fill(1)
    x[3].fill(0)
    with pytest.raises(ValueError,match="burned"):
        owner.step(*x)
    assert not owner.active
    assert not np.any(owner.state)


def test_source_audit_preserves_hybrid_distinction():
    if not probe.SOURCE_DIR.exists():
        pytest.skip("run --audit to retrieve pinned small source/config artifacts")
    models = probe.audit()["models"]
    assert [m["float32_kv_bytes_per_token"] for m in models] == [0,0,65536]
    assert [m["float32_retained_state_bytes"] for m in models] == [41943040,184320,53477376]


def test_weighted_exp_matches_direct_high_precision_sum():
    log_decay = np.array([-.7,.2],np.float32)
    first = np.array([.3,-.1],np.float32)
    owner = probe.StableWeightedExpState(log_decay,first)
    numerator = np.zeros(2,np.float64)
    denominator = np.zeros(2,np.float64)
    for i in range(19):
        key = np.array([np.sin(i),np.cos(i)],np.float32)
        value = np.array([np.cos(i),np.sin(i)],np.float32)
        current = np.exp(key.astype(np.float64)+first)
        expected = (numerator+current*value)/(denominator+current)
        actual = owner.step(key,value)
        np.testing.assert_allclose(actual,expected,rtol=2e-5,atol=2e-6)
        keep = np.exp(-np.exp(log_decay.astype(np.float64)))
        numerator = numerator*keep+np.exp(key.astype(np.float64))*value
        denominator = denominator*keep+np.exp(key.astype(np.float64))


def test_weighted_exp_extreme_keys_remain_finite():
    owner = probe.StableWeightedExpState(np.array([0],np.float32),np.array([0],np.float32))
    for key in (10000,-10000,10001):
        assert np.all(np.isfinite(owner.step(np.array([key],np.float32),np.array([.5],np.float32))))
    assert owner.state.nbytes == 12


def test_timing_commitments_reject_numeric_difference():
    shared = {"tokens":64,"repeat":0,"output_commitment":"output",
              "state_commitment":"state","actual_imported_semantic_hybrid_sha256":"source",
              "source_identity":SOURCE_IDENTITY}
    control = shared | {"mode":"copy_reference"}
    owned = shared | {"mode":"owned"}
    assert probe.verify_pairs([control,owned])["exact_output_and_state_bits"]
    with pytest.raises(AssertionError,match="numeric/source failure"):
        probe.verify_pairs([control,owned | {"state_commitment":"different"}])


def test_source_change_rejected_before_work(monkeypatch):
    changed = dict(SOURCE_IDENTITY)
    changed[next(iter(changed))] = "changed"
    monkeypatch.setattr(probe,"source_identity",lambda:changed)
    with pytest.raises(ValueError,match="differs"):
        with probe.frozen_sources(SOURCE_IDENTITY):
            pytest.fail("must not enter source-mismatched work")
    monkeypatch.undo()


def test_commitment_distinguishes_signed_zero():
    assert probe.commitment(np.array([0],np.float32)) != probe.commitment(np.array([-0.0],np.float32))
