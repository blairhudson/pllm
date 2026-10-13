"""Public geometry conservation and software transport controls, not TEE tests."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "attested_budget", Path(__file__).resolve().parents[1] / "scripts/probe_attested_linear_budget.py")
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


@pytest.mark.parametrize("bits", [1, 8, 17, 23, 24, 26, 32])
def test_fixed_signed_width_extremes_and_canonical_tail(bits):
    values = [-(1 << (bits - 1)), (1 << (bits - 1)) - 1, 0]
    raw = probe.pack_signed(values, bits)
    assert len(raw) == (len(values) * bits + 7) // 8
    assert probe.unpack_signed(raw, len(values), bits) == values
    with pytest.raises(ValueError, match="out of range"):
        probe.pack_signed([1 << (bits - 1)], bits)
    if len(values) * bits % 8:
        with pytest.raises(ValueError, match="padding"):
            probe.unpack_signed(raw[:-1] + bytes([raw[-1] | 128]), len(values), bits)


def test_all_i8_and_bounded_parser():
    values = list(range(-128, 128))
    assert probe.unpack_signed(probe.pack_signed(values, 8), 256, 8) == values
    for raw, count, bits in [(b"", 0, 8), (b"", 1, 8), (b"x", 1, 33), (b"x", 65537, 8)]:
        with pytest.raises(ValueError):
            probe.unpack_signed(raw, count, bits)


def channel(**overrides):
    pytest.importorskip("cryptography")
    params = dict(key=b"k" * 32, session=b"s" * 16, epoch=b"e" * 32, nonce_prefix=b"UP00")
    params.update(overrides)
    return probe.SoftwareChannel(**params)


def test_aead_size_tamper_replay_and_failure_does_not_advance():
    from cryptography.exceptions import InvalidTag
    sender, receiver = channel(), channel()
    context = dict(phase=1, stage=3, rows=1, bits=8)
    frame = sender.seal(b"hello", **context)
    assert len(frame) == 5 + probe.FRAME_OVERHEAD
    with pytest.raises(InvalidTag):
        receiver.open(frame[:-1] + bytes([frame[-1] ^ 1]), **context)
    assert receiver.received == 0
    assert receiver.open(frame, **context) == b"hello"
    with pytest.raises(ValueError, match="replay"):
        receiver.open(frame, **context)


@pytest.mark.parametrize("field,value", [("phase", 2), ("stage", 4), ("rows", 2), ("bits", 9)])
def test_authenticated_operation_context(field, value):
    sender, receiver = channel(), channel()
    context = dict(phase=1, stage=3, rows=1, bits=8)
    frame = sender.seal(b"x", **context)
    with pytest.raises(ValueError, match="context"):
        receiver.open(frame, **{**context, field: value})


@pytest.mark.parametrize("overrides", [{"session": b"t" * 16}, {"epoch": b"f" * 32},
                                      {"key": b"j" * 32}, {"nonce_prefix": b"DN00"}])
def test_wrong_session_epoch_key_or_direction(overrides):
    from cryptography.exceptions import InvalidTag
    sender, receiver = channel(), channel(**overrides)
    context = dict(phase=1, stage=3, rows=1, bits=8)
    frame = sender.seal(b"x", **context)
    with pytest.raises((ValueError, InvalidTag)):
        receiver.open(frame, **context)


def test_close_counter_exhaustion_and_truncation_fail_closed():
    sender, receiver = channel(), channel()
    context = dict(phase=1, stage=3, rows=1, bits=8)
    frame = sender.seal(b"x", **context)
    with pytest.raises(ValueError):
        receiver.open(frame[:-1], **context)
    receiver.close()
    with pytest.raises(ValueError, match="closed"):
        receiver.open(frame, **context)
    sender.sent = (1 << 64) - 1
    with pytest.raises(ValueError, match="exhausted"):
        sender.seal(b"x", **context)


def documents():
    return [json.loads(p.read_text()) for p in (probe.CONTROL, probe.GEOMETRY, probe.CONFIRM)]


def test_historical_rows_weights_and_aead_accounting():
    result = probe.budget(*documents())
    totals = result["totals"]
    assert totals["pair_rows"] == 15297
    assert totals["input_payload_bytes"] == 28732928
    assert totals["output_packed_bytes"] == 144546192
    assert totals["frames_each_direction"] == 93 * 24 + 3 * 15
    assert totals["aead_overhead_bytes"] == 2 * totals["frames_each_direction"] * probe.FRAME_OVERHEAD
    assert result["raw_body_i8_weight_bytes"] == 357826560
    for mode in result["alternatives"].values():
        for row in mode["horizons"]:
            h = row["pairs"]
            assert row["baseline_body_bytes"] == 124905091 + h * (493575247 - 124905091)
            assert row["candidate_body_bytes_before_extra_setup"] == 124905091 + h * mode["recurring_pair_bytes"]


@pytest.mark.parametrize("mutation", ["body", "ledger", "rows", "bound", "stages"])
def test_changed_source_geometry_or_ledger_rejected(mutation):
    control, geometry, confirmation = copy.deepcopy(documents())
    if mutation == "body":
        geometry["source"]["body_fingerprint"] = "bad"
    elif mutation == "ledger":
        control["control"]["pair_bytes"] += 1
    elif mutation == "rows":
        report = next(r for r in confirmation["results"] if r["execution"]["case"]["name"] == "prepared/digit8")
        report["execution"]["details"]["report"]["prepared_material_accounting"]["counts"]["claimed"] += 1
    elif mutation == "bound":
        geometry["stages"][0]["odd_radix"]["modulus"] = 2
    else:
        geometry["stages"].pop()
    with pytest.raises(ValueError):
        probe.budget(control, geometry, confirmation)


@pytest.mark.parametrize("target", [.2, .35, .5])
@pytest.mark.parametrize("extra", [0, 65536, 357826560])
def test_horizon_matches_integer_exhaustive_oracle(target, extra):
    public, baseline, candidate = 124905091, 368670156, 174000000
    expected = next(h for h in range(1, 1000)
                    if public + h * candidate + extra <= (1 - target) * (public + h * baseline))
    assert probe.minimum_horizon(public, baseline, candidate, extra, target) == expected
    assert probe.minimum_horizon(public, baseline, baseline, extra, target) is None


def test_tiny_exact_integer_aead_control_is_never_tee_evidence():
    pytest.importorskip("cryptography")
    result = probe.synthetic_control()
    assert result["executed"] and not result["tee_proof"]
    assert result["exact_output_integers"] == 32 * 3 * 7
    assert result["tamper_rejected"] and result["replay_rejected"]
