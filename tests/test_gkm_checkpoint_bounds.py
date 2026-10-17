"""Independent source-bit, exact-integer, file-integrity and resource-gate tests."""

from collections import Counter
from contextlib import ExitStack
from fractions import Fraction
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import shutil
import struct
import subprocess
import sys

import numpy as np
import pytest

from scripts.gkm_checkpoint_oracle import scalar_bits
from scripts import probe_gkm_checkpoint_bounds as probe


class NoPressure:
    def check(self, **kwargs):
        pass


def safetensor(path, entries):
    header, payload = {}, bytearray()
    for key, dtype, shape, raw in entries:
        header[key] = {"dtype": dtype, "shape": shape,
                       "data_offsets": [len(payload), len(payload) + len(raw)]}
        payload.extend(raw)
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 8)
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + payload)
    return path


@pytest.mark.parametrize("dtype", ["BF16", "F16"])
def test_every_16_bit_pattern_against_independent_integer_oracle(dtype):
    raw = b"".join(n.to_bytes(2, "little") for n in range(65536))
    with np.errstate(invalid="ignore"):
        values, rounded, finite = probe.decode_round(raw, dtype)
    for bits in range(65536):
        oracle = scalar_bits(bits.to_bytes(2, "little"), dtype)
        assert bool(finite[bits]) == oracle["finite"]
        if oracle["finite"]:
            assert Fraction(float(values[bits])) == Fraction(oracle["numerator"], oracle["denominator"])
            assert int(rounded[bits]) == oracle["rounded"]
            assert struct.pack("<f", float(values[bits])) == oracle["float32_bits"].to_bytes(4, "little")


def test_float32_edges_and_bit_samples_against_two_scalar_references():
    # Halfway ties on both signs, rejection thresholds, signed zeros, subnormals,
    # finite extremes, infinities and several NaN payloads. No host-endian loads.
    values = [0.0, -0.0, 0.5 / 2048, 1.5 / 2048, 2.5 / 2048, -0.5 / 2048,
              -1.5 / 2048, -2.5 / 2048, 32766.5 / 2048, 32767.5 / 2048,
              -32768.5 / 2048, -32769.5 / 2048]
    bits = [int.from_bytes(struct.pack("<f", value), "little") for value in values]
    bits += [1, 0x80000001, 0x007FFFFF, 0x00800000, 0x7F7FFFFF, 0xFF7FFFFF,
             0x7F800000, 0xFF800000, 0x7F800001, 0xFFC00001]
    random_bits = random.Random(1849)
    bits += [random_bits.getrandbits(32) for _ in range(4096)]
    raw = b"".join(value.to_bytes(4, "little") for value in bits)
    with np.errstate(invalid="ignore"):
        decoded, rounded, finite = probe.decode_round(raw, "F32")
    checked = probe.compare_oracle(raw, "F32", decoded, rounded, range(len(bits)))
    assert len(checked) == len(bits)
    for i, value_bits in enumerate(bits):
        scalar = struct.unpack("<f", value_bits.to_bytes(4, "little"))[0]
        assert bool(finite[i]) == math.isfinite(scalar)
        if math.isfinite(scalar):
            assert int(rounded[i]) == round(Fraction(scalar) * 2048)


def test_streamed_l1_hashes_tied_owners_and_conditional_input_maxima(tmp_path, monkeypatch):
    # Rows include cancellation, >i8 weights, and the admitted -32768 endpoint.
    encoded = [[0, -1, 2, 0], [-32768, 32767, -256, 512], [0, 0, 0, 0]]
    values = [value / 2048 for row in encoded for value in row]
    values[0] = -0.0
    raw = struct.pack("<12f", *values)
    path = safetensor(tmp_path / "weights", [("opaque.table", "F32", [3, 4], raw)])
    original = os.pread
    requests = []

    def bounded(fd, size, offset):
        requests.append((size, offset))
        assert size <= probe.BUFFER_BYTES
        return original(fd, size, offset)

    monkeypatch.setattr(os, "pread", bounded)
    io = Counter()
    with probe.InputFile(path, io, NoPressure()) as source:
        metadata, _ = probe.parse_header(source)
        result = probe.scan_tensor(source, "opaque.table", metadata["opaque.table"],
            [{"operator": "token_lookup"}, {"operator": "output_head"}], buffer_bytes=16)
    sums = [sum(abs(value) for value in row) for row in encoded]
    assert result["max_row_L1"] == max(sums) == 66303
    assert result["max_row_L1_index"] == 1
    assert result["safe_uniform_encoded_input_max"] == ((1 << 63) - 1) // 66303
    assert max(sums) * result["safe_uniform_encoded_input_max"] < 1 << 63
    assert max(sums) * (result["safe_uniform_encoded_input_max"] + 1) >= 1 << 63
    assert result["i16_le_sha256"] == hashlib.sha256(struct.pack("<12h", *(v for row in encoded for v in row))).hexdigest()
    assert result["float32_le_sha256"] == hashlib.sha256(raw).hexdigest()
    row_hashes = b"".join(hashlib.sha256(raw[start:start + 16]).digest() for start in range(0, len(raw), 16))
    assert result["table_row_sha256_vector_sha256"] == hashlib.sha256(row_hashes).hexdigest()
    assert result["negative_zero_count"] == 1
    assert io["source_value_scan"] == len(raw)
    assert sum(offset >= metadata["opaque.table"]["absolute_offset"] for _, offset in requests) == 3


def test_zero_row_bound_and_int16_tie_rejection_are_not_clipped(tmp_path):
    cases = [("zeros", [0.0] * 4),
             ("edges", [32766.5 / 2048, 32767.5 / 2048, -32768.5 / 2048, -32769.5 / 2048]),
             ("nan", [math.nan, math.inf, -math.inf, 1.0])]
    entries = [(name, "F32", [1, 4], struct.pack("<4f", *values)) for name, values in cases]
    with probe.InputFile(safetensor(tmp_path / "weights", entries), Counter(), NoPressure()) as source:
        metadata, _ = probe.parse_header(source)
        results = {name: probe.scan_tensor(source, name, metadata[name], [{"operator": "linear"}]) for name, _ in cases}
    assert results["zeros"]["safe_uniform_encoded_input_max"] == (1 << 63) - 1
    assert results["edges"]["finite_out_of_i16_count"] == 2
    assert results["edges"]["rounded_min"] == -32770
    assert results["edges"]["rounded_max"] == 32768
    assert results["edges"]["i16_le_sha256"] is None
    assert results["edges"]["max_row_L1"] is None
    assert "safe_uniform_encoded_input_max" not in results["edges"]
    assert results["nan"]["nonfinite_count"] == 3
    assert results["nan"]["float32_le_sha256"] is None


def test_bias_values_retain_separate_float_owner(tmp_path):
    raw = struct.pack("<2f", 32.0, -64.0)
    with probe.InputFile(safetensor(tmp_path / "w", [("opaque.bias", "F32", [2], raw)]), Counter(), NoPressure()) as source:
        metadata, _ = probe.parse_header(source)
        result = probe.scan_tensor(source, "opaque.bias", metadata["opaque.bias"], [{"operator": "client_local_bias"}])
    assert result["finite_out_of_i16_count"] == 2  # Diagnostic, not an i16 weight admission.
    assert result["linear_i16_owner"] is False
    assert result["float32_le_sha256"] == hashlib.sha256(raw).hexdigest()
    assert result["max_row_L1"] is None


@pytest.mark.parametrize("header,payload", [
    ('{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"a":{}}', bytes(4)),
    ('{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,4]}}', bytes(4)),
    ('{"a":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}}', bytes(8)),
    ('{"a":{"dtype":"I16","shape":[2],"data_offsets":[0,4]}}', bytes(4)),
    ('{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}', bytes(3)),
    ('{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"b":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}', bytes(4)),
])
def test_bad_headers_reject_without_payload_reads(tmp_path, header, payload):
    raw = header.encode()
    raw += b" " * (-len(raw) % 8)
    path = tmp_path / "bad"
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + payload)
    io = Counter()
    with probe.InputFile(path, io, NoPressure()) as source, pytest.raises(probe.Rejected):
        probe.parse_header(source)
    assert io["source_value_scan"] == 0


def test_oversized_header_and_short_pread_reject(tmp_path, monkeypatch):
    path = tmp_path / "bad"
    path.write_bytes(struct.pack("<Q", probe.MAX_HEADER + 8))
    with probe.InputFile(path, Counter(), NoPressure()) as source, pytest.raises(probe.Rejected, match="header"):
        probe.parse_header(source)
    path.write_bytes(bytes(8))
    monkeypatch.setattr(os, "pread", lambda *args: b"short")
    with probe.InputFile(path, Counter(), NoPressure()) as source, pytest.raises(probe.Rejected, match="short pread"):
        source.read(0, 8, "test")


def test_input_mutation_and_symlink_retarget_reject(tmp_path):
    path = tmp_path / "source"
    path.write_bytes(b"abcd")
    with probe.InputFile(path, Counter(), NoPressure()) as source:
        path.write_bytes(b"abce")
        with pytest.raises(probe.Rejected, match="changed"):
            source.check_identity()
    other = tmp_path / "other"
    other.write_bytes(b"abcd")
    link = tmp_path / "link"
    link.symlink_to(path)
    with probe.InputFile(link, Counter(), NoPressure()) as source:
        link.unlink()
        link.symlink_to(other)
        with pytest.raises(probe.Rejected, match="no longer identifies"):
            source.check_identity()


@pytest.mark.parametrize("reader", ["source", "code_receipt", "archive"])
def test_fifo_inputs_reject_without_waiting_for_writer(tmp_path, reader):
    path = tmp_path / "fifo"
    os.mkfifo(path)
    # An external timeout makes a regression an ordinary failure, not a hung
    # suite. No writer is ever opened, and no checkpoint files are inspected.
    code = """
from collections import Counter
import importlib.util
from pathlib import Path
import sys
sys.path.insert(0, sys.argv[1])
from scripts import probe_gkm_checkpoint_bounds as probe
path, reader = Path(sys.argv[2]), sys.argv[3]
try:
    if reader == 'source':
        with probe.InputFile(path, Counter(), None):
            raise AssertionError('FIFO admitted')
    elif reader == 'code_receipt':
        probe.__file__ = str(path)
        probe.receipt_code_identity()
        raise AssertionError('FIFO admitted')
    else:
        filename = Path(sys.argv[1]) / 'paper/artifacts/gkm-checkpoint-bounds/verify.py'
        spec = importlib.util.spec_from_file_location('archive_verifier', filename)
        verifier = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(verifier)
        verifier.read_bounded(path)
        raise AssertionError('FIFO admitted')
except ValueError as exc:
    assert 'regular file' in str(exc), str(exc)
"""
    subprocess.run([sys.executable, "-I", "-B", "-c", code, str(probe.ROOT), str(path), reader],
                   check=True, capture_output=True, text=True, timeout=3)


def test_atomic_output_preserves_failed_attempt(tmp_path):
    path = tmp_path / "report.json"
    probe.atomic_json(path, {"status": "failed", "reason": "reserve"})
    with pytest.raises(FileExistsError):
        probe.atomic_json(path, {"status": "complete"})
    assert json.loads(path.read_text())["status"] == "failed"
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("mutation,reason", [
    ({"available_bytes": probe.HOST_RESERVE - 1}, "host_reserve_breach"),
    ({"swap_used_bytes": 101}, "new_swap_usage"),
    ({"swapout_bytes": 201}, "new_swapout"),
    ({"rss_bytes": probe.PROCESS_CEILING + 1}, "process_estimate_exceeded"),
])
def test_resource_guard_has_no_swap_allowance(mutation, reason):
    baseline = {"total_bytes": 32 * probe.GIB, "available_bytes": 10 * probe.GIB,
                "swap_used_bytes": 100, "swapout_bytes": 200, "rss_bytes": 50 * probe.MIB}
    assert probe.resource_reason(baseline | mutation, baseline, probe.PROCESS_CEILING, elapsed=0, cpu=0) == reason
    assert probe.resource_reason(baseline, baseline, probe.PROCESS_CEILING, elapsed=301, cpu=0) == "publication_scan_time_budget_exceeded"


def test_lowering_inventory_uses_opaque_weight_identities_and_separates_biases():
    from pllm.modeling import ModelPlan

    operations = [
        {"id": "lookup", "operator": "token_lookup", "inputs": ["input.tokens"], "layer": None,
         "output_shape": [1, 1, 2], "attributes": {"weight": "opaque.shared"}},
        {"id": "project", "operator": "linear", "inputs": ["lookup"], "layer": 0,
         "output_shape": [1, 1, 2], "attributes": {"weight": "opaque.matrix", "bias": "opaque.offset"}},
        {"id": "head", "operator": "output_head", "inputs": ["project"], "layer": None,
         "output_shape": [1, 1, 3], "attributes": {"weight": "opaque.shared"}},
    ]
    plan = ModelPlan(probe.canonical({"schema_version": "pllm.decoder_plan.v1",
        "prefill": {"operations": operations}, "decode": {"operations": operations}}))
    tensors = {"opaque.shared": {"shape": [3, 2], "elements": 6, "dtype": "BF16"},
               "opaque.matrix": {"shape": [2, 2], "elements": 4, "dtype": "F16"},
               "opaque.offset": {"shape": [2], "elements": 2, "dtype": "F32"}}
    inventory = probe.semantic_inventory(plan, tensors)
    geometry = probe.geometry(inventory, tensors, 1000)
    assert inventory["fused_groups"]["status"] == "blocked"
    assert geometry["body_i16_bytes_per_worker"] == 8
    assert geometry["head_i16_bytes_per_worker"] == 12
    assert geometry["raw_float32_table_bytes_per_worker"] == 24
    assert geometry["table_plus_linear_payload_bytes_per_worker"] == 44
    assert geometry["client_bias_float32_bytes"] == 8
    assert geometry["two_workers_table_plus_linear_payload_bytes"] == 88
    with pytest.raises(probe.Rejected, match="not owned"):
        probe.semantic_inventory(plan, tensors | {"unowned": {"shape": [1]}})


def test_official_source_hash_mismatch_rejects(tmp_path):
    path = tmp_path / "source"
    path.write_bytes(b"public fixture")
    with probe.InputFile(path, Counter(), NoPressure()) as source:
        actual = source.hashes("before")
        probe.verify_expected("source", source, {"size": 14, "blob_id": actual["git_blob_sha1"]}, actual)
        with pytest.raises(probe.Rejected, match="LFS SHA-256"):
            probe.verify_expected("source", source, {"size": 14, "lfs_size": 14, "lfs_sha256": "0" * 64}, actual)


def test_failed_preflight_keeps_receipt_and_never_enters_source_scan(tmp_path, monkeypatch):
    baseline = {"total_bytes": 32 * probe.GIB, "available_bytes": probe.HOST_RESERVE,
                "swap_used_bytes": 100, "swapout_bytes": 200, "rss_bytes": 50 * probe.MIB}
    original_guard = probe.Guard
    monkeypatch.setattr(probe, "ROOT", tmp_path)
    monkeypatch.setattr(probe, "receipt_code_identity", lambda: {"oracle": {"sha256": "a" * 64}})
    monkeypatch.setattr(probe, "Guard", lambda start: original_guard(start, sample=lambda: baseline))
    monkeypatch.setattr(probe.sys, "platform", "darwin")
    monkeypatch.setattr(probe.os, "uname", lambda: type("Host", (), {"machine": "arm64"})())
    monkeypatch.setattr(probe, "scan", lambda *args: pytest.fail("value scan entered after rejection"))
    monkeypatch.setattr(probe.sys, "path", list(probe.sys.path))
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                "NUMEXPR_NUM_THREADS", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        monkeypatch.setenv(key, "1")
    output = tmp_path / "target" / "rejected"
    assert probe.main(["--output", str(output)]) == 2
    receipt = json.loads((output / "report.json").read_bytes())
    assert receipt["failure"]["reason"] == "insufficient_host_headroom_before_metadata_import"
    assert receipt["resource_guard_before_report_serialization"]["baseline"] == baseline
    assert receipt["receipt_code_hashes"]["oracle"]["sha256"] == "a" * 64
    assert not (output / "admission.json").exists()
    assert json.loads((output / "finalization.json").read_bytes())["status"] == "failed"


def test_imported_module_receipts_bind_installed_files_not_checkout(tmp_path, monkeypatch):
    checkout = tmp_path / "checkout"
    installed = tmp_path / "installed-wheel" / "site-packages" / "provenance_fixture"
    installed.mkdir(parents=True)
    monkeypatch.setattr(probe, "ROOT", checkout)
    modules = {}
    for name, filename in (("provenance_fixture", "__init__.py"),
                           ("provenance_fixture.modeling", "modeling.py"),
                           ("provenance_fixture.model_loader", "model_loader.py")):
        actual = installed / filename
        actual.write_text('def origin():\n    return "installed-wheel"\n')
        reference = checkout / "python" / "pllm" / filename
        reference.parent.mkdir(parents=True, exist_ok=True)
        reference.write_text('def origin():\n    return "different-checkout"\n')
        spec = importlib.util.spec_from_file_location(name, actual)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        modules[name] = module
    receipt = {}
    with ExitStack() as stack:
        bindings = probe.bind_imported_modules(modules,
            lambda path: stack.enter_context(probe.InputFile(path, Counter(), NoPressure())), receipt)
        assert receipt["checks"]["before_semantic_lowering"] == "complete"
        for name, module in modules.items():
            assert module.origin() == "installed-wheel"
            row = receipt["modules"][name]
            assert Path(row["module_file"]).is_relative_to(installed)
            assert row["artifacts"][0]["sha256_before"] == hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
            assert all(not Path(artifact["path"]).is_relative_to(checkout) for artifact in row["artifacts"])
        (checkout / "python/pllm/modeling.py").write_text("changed reference, not executed\n")
        probe.recheck_imported_modules(bindings, receipt, "after_semantic_lowering")
        assert receipt["checks"]["after_semantic_lowering"] == "complete"
        (installed / "modeling.py").write_text("changed imported artifact\n")
        with pytest.raises(probe.Rejected, match="changed"):
            probe.recheck_imported_modules(bindings, receipt, "after_source_scan")
        assert receipt["checks"]["after_source_scan"] == "incomplete"


def test_metadata_import_binding_includes_packages_native_and_actual_oracle(monkeypatch):
    requested = []
    monkeypatch.setattr(probe.importlib, "import_module", lambda name: requested.append(name) or object())
    modules = probe.imported_metadata_modules()
    assert set(modules) == {"pllm", "pllm.runtime", "pllm.modeling", "pllm.model_loader",
                            "pllm.runtime.semantic_tensors", "pllm._native", "scripts.gkm_checkpoint_oracle"}
    assert requested == list(modules)


class Clock:
    def __init__(self, now=1.0):
        self.now = now

    def __call__(self):
        return self.now


def healthy_host():
    return {"total_bytes": 32 * probe.GIB, "available_bytes": 10 * probe.GIB,
            "swap_used_bytes": 100, "swapout_bytes": 200, "rss_bytes": 50 * probe.MIB}


def test_final_guard_observation_crossing_deadline_rejects_and_counts_duration():
    clock = Clock()
    delay = [0]

    def sample():
        clock.now += delay[0]
        return healthy_host()

    guard = probe.Guard(0, sample=sample, clock=clock)
    guard.check(force=True)
    assert guard.last_time == 1
    clock.now = probe.WALL_SECONDS - 1
    delay[0] = 2
    with pytest.raises(probe.Rejected, match="time_budget"):
        guard.check(force=True)
    assert guard.last_time == probe.WALL_SECONDS + 1
    assert guard.report()["maximum_sample_gap_seconds"] == probe.WALL_SECONDS
    assert guard.report()["maximum_observation_duration_seconds"] == 2
    assert guard.last_work["observed_through_elapsed_seconds"] == probe.WALL_SECONDS + 1


def test_final_pread_crossing_deadline_rejects_after_read(tmp_path, monkeypatch):
    clock = Clock()
    guard = probe.Guard(0, sample=healthy_host, clock=clock)
    path = tmp_path / "fixture"
    path.write_bytes(bytes(8))
    original = os.pread

    def slow_read(*args):
        raw = original(*args)
        clock.now += 2
        return raw

    monkeypatch.setattr(os, "pread", slow_read)
    io = Counter()
    with probe.InputFile(path, io, guard) as source:
        clock.now = probe.WALL_SECONDS - 1
        with pytest.raises(probe.Rejected, match="time_budget"):
            source.read(0, 8, "last_read")
    assert io["last_read"] == 8


@pytest.mark.parametrize("serialization_seconds,expected", [(0.5, "complete"), (301.0, "failed")])
def test_finalization_measures_report_window_and_rejects_late_serialization(tmp_path, monkeypatch,
                                                                         serialization_seconds, expected):
    clock = Clock(10)
    guard = probe.Guard(0, sample=healthy_host, clock=clock)
    original = probe.atomic_json

    def publication(path, value):
        result = original(path, value)
        # The main report is measured; the companion's own publication is an
        # explicitly excluded tail, so no self-referential lifetime claim arises.
        clock.now += serialization_seconds if path.name == "report.json" else 2
        return result

    monkeypatch.setattr(probe, "atomic_json", publication)
    report = {"status": "complete", "source_numeric_admission": "pass", "complete_gkm_admission": "blocked"}
    final = probe.finalize_report(report, tmp_path, guard, 0, clock=clock)
    saved = json.loads((tmp_path / "report.json").read_bytes())
    assert saved["status"] == saved["source_numeric_admission"] == "pending_receipt_finalization"
    assert final["status"] == expected
    assert final["source_numeric_admission"] == ("pass" if expected == "complete" else "blocked")
    assert final["report_serialization_and_atomic_publication"]["duration_seconds"] == serialization_seconds
    assert final["publication_work_after_report_serialization"]["observed_through_elapsed_seconds"] == 10 + serialization_seconds
    assert final["report_artifact"]["sha256"] == hashlib.sha256((tmp_path / "report.json").read_bytes()).hexdigest()
    assert final["complete_gkm_admission"] == "blocked"
    assert final["unobserved_tail"]
    assert clock.now == 12 + serialization_seconds
    if expected == "failed":
        assert final["final_observation_error"]["reason"] == "publication_scan_time_budget_exceeded"


def test_failed_final_observation_does_not_relabel_an_earlier_sample(tmp_path):
    clock = Clock()
    calls = [0]

    def sample():
        calls[0] += 1
        if calls[0] > 2:
            raise RuntimeError("final telemetry read failed")
        return healthy_host()

    guard = probe.Guard(0, sample=sample, clock=clock)
    guard.check(force=True)
    final = probe.finalize_report({"status": "complete", "source_numeric_admission": "pass"},
                                  tmp_path, guard, 0, clock=clock)
    assert final["status"] == "failed"
    assert final["publication_work_after_report_serialization"] is None
    assert final["final_observation_error"]["reason"] == "final telemetry read failed"


def test_negative_evidence_archive_verifies_and_rejects_byte_tampering(tmp_path):
    verifier_path = Path(__file__).resolve().parents[1] / "paper/artifacts/gkm-checkpoint-bounds/verify.py"
    spec = importlib.util.spec_from_file_location("gkm_negative_archive_verifier", verifier_path)
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    result = verifier.verify()
    assert result["verified_files"] == 7
    assert result["attempt_statuses"] == ["failed", "failed"]
    assert result["source_numeric_eligibility"] == "not_established"
    copied = tmp_path / "archive"
    shutil.copytree(verifier.ARCHIVE, copied)
    altered = copied / "receipts/v2/report.json"
    altered.chmod(0o600)
    altered.write_bytes(altered.read_bytes() + b" ")
    with pytest.raises(ValueError, match="byte hash mismatch"):
        verifier.verify(copied)
