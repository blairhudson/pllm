"""Accounting tests; optional native baseline never asserts that fusion exists."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/probe_boundary_scan_fusion.py"
spec = importlib.util.spec_from_file_location("boundary_scan_fusion_probe", SCRIPT)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


@pytest.mark.parametrize("rows,columns", [(1, 1), (127, 17), (128, 16), (129, 19), (257, 33)])
def test_analytical_budget(rows, columns):
    m = probe.material_model(rows, columns)
    assert m["baseline_weight_scan_bytes"] == 2 * rows * columns
    assert m["candidate_weight_scan_bytes_lower_bound"] == rows * columns
    assert m["scale_scan_bytes_both"] == 4 * rows
    assert m["candidate_tile_mask_bytes_upper_bound"] == min(rows, 128)
    assert m["query_packet_bytes"] == 97 + 17 * (((rows + 127) // 128 - 1).bit_length())
    assert m["head_result_bytes"] == 4 * rows
    assert not m["fused_scan_executed"]
    assert not m["physical_read_instrumentation"]


@pytest.mark.parametrize("args", [(0, 1), (1, 0), (True, 1), (262145, 1), (1, 65533), (262144, 65532), (129, 19, 127), (129, 19, 128, 4097)])
def test_reject_unbounded_geometry(args):
    with pytest.raises(ValueError):
        probe.material_model(*args)


def test_qwen_model_only():
    m = probe.material_model(151936, 896, pending=8)
    assert m["matrix_owner_bytes"] == 136134656
    assert m["candidate_tile_mask_bytes_upper_bound"] == 128
    assert m["pending_client_r_and_Wr_bytes"] == 8 * 4 * (151936 + 896)
    assert 0.49 < m["weight_scan_saving_fraction"] < 0.5
    assert 0.24 < m["two_role_token_scan_saving_fraction"] < 0.25


def test_cli_explicit_blocker():
    p = subprocess.run([sys.executable, str(SCRIPT)], check=True, capture_output=True, text=True)
    row = json.loads(p.stdout)
    assert row["status"] == "test-hook-available-not-executed"
    assert "native_baseline" not in row


@pytest.mark.skipif(os.environ.get("PLLM_BOUNDARY_SCAN_FUSION_NATIVE") != "1", reason="explicit opt-in Rust compile/smoke")
@pytest.mark.parametrize("rows,columns,index", [(1, 1, 0), (127, 17, 126), (128, 16, 127), (129, 19, 127), (129, 19, 128), (257, 33, 0), (257, 33, 256)])
def test_native_baseline(rows, columns, index):
    p = subprocess.run([sys.executable, str(SCRIPT), "--native", "--rows", str(rows), "--columns", str(columns), "--index", str(index)], check=True, capture_output=True, text=True)
    row = json.loads(p.stdout)["native_baseline"]
    assert row["baseline_correct"] and row["parties_checked"] == 2
    assert not row["fused_scan_executed"]


@pytest.mark.skipif(os.environ.get("PLLM_BOUNDARY_SCAN_FUSION_NATIVE") != "1", reason="explicit opt-in cfg(test) fused reference")
def test_real_fused_reference():
    p = subprocess.run([sys.executable, str(SCRIPT), "--fused"], check=True, capture_output=True, text=True)
    row = json.loads(p.stdout)["native_fused_reference"]
    assert row["correct"] and row["fused_scan_executed"]
    assert row["source_weight_values_visited"] == 2 * 129 * 19
    assert row["source_scale_bytes_visited"] == 2 * 129 * 4
    assert row["terminal_mask_bytes"] == 16
    assert not row["full_vocabulary_mask_allocated"]
    assert not row["sdk_selectable"]
    assert not row["assembly_single_load_verified"]
