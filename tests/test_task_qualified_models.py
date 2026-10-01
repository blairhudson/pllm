"""Objective grading and failed-attempt/EOS accounting for the public model screen."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "task_probe", ROOT / "scripts/probe_task_qualified_models.py"
)
assert spec is not None and spec.loader is not None
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def test_importing_task_graders_does_not_change_process_environment() -> None:
    script = ROOT / "scripts/probe_task_qualified_models.py"
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import os, runpy, sys; before = dict(os.environ); "
            "runpy.run_path(sys.argv[1]); assert dict(os.environ) == before",
            str(script),
        ],
        check=True,
    )


def test_public_canonical_example_explicitly_opts_into_digest_capture(monkeypatch):
    captured = {}

    class StopBeforeExecution(Exception):
        pass

    def benchmark(**kwargs):
        captured.update(kwargs)
        raise StopBeforeExecution

    monkeypatch.setattr("pllm.runtime.benchmark_cli.run_loopback_benchmark", benchmark)
    document = probe.load_tasks()
    with pytest.raises(StopBeforeExecution):
        probe.canonical_example("smol", document["tasks"][10], document["policy"], {})
    assert captured["temperature"] == 0.0
    assert captured["capture_output_digest"] is True
    assert captured["inventory_policy"] == "request-sized"


def test_fixed_disjoint_contract_and_objective_answers():
    document = probe.load_tasks()
    tasks = document["tasks"]
    assert len(tasks) == 16
    assert document["policy"]["canonical_task_ids"] == ["screen-extract-1", "review-arithmetic-1"]
    assert [probe.expected_answer(task["grader"]) for task in tasks] == [
        "K47",
        "BM3",
        "43",
        "56",
        "EVEN",
        "NEGATIVE",
        "MAPLE",
        "2,5,9",
        "Lima",
        "EU7",
        "35",
        "54",
        "ODD",
        "ZERO",
        "cedar",
        "1,4,8",
    ]
    for cohort in ("screen", "review"):
        rows = [task for task in tasks if task["cohort"] == cohort]
        assert len(rows) == 8
        assert {
            category: sum(row["category"] == category for row in rows)
            for category in ("extraction", "arithmetic", "classification", "formatting")
        } == {"extraction": 2, "arithmetic": 2, "classification": 2, "formatting": 2}


@pytest.mark.parametrize(
    "text,eos,correct,success,failure",
    [
        ("  K47\n", True, True, True, None),
        ("K47", False, True, False, "truncation"),
        ("The code is K47", True, False, False, "wrong_answer"),
        ("k47", True, False, False, "wrong_answer"),
        ("", True, False, False, "wrong_answer"),
    ],
)
def test_strict_objective_grader_and_termination(text, eos, correct, success, failure):
    result = probe.grade(probe.load_tasks()["tasks"][0], text, eos=eos)
    assert result["answer_correct"] is correct
    assert result["success"] is success
    assert result["failure"] == failure
    assert result["empty_eos_failure"] == (eos and not text.strip())


def test_actual_ring_width_projection_and_eos_rows():
    stages = [
        {"in_features": 3, "out_features": 5, "wire_bits": 16},
        {"in_features": 7, "out_features": 11, "wire_bits": 24},
    ]
    # 13 input, 4 answer tokens, then EOS: 13 + 5 selections - 1 = 17 rows.
    result = probe.arithmetic_cost(stages, 17)
    assert result == dict(zip(probe.COST_KEYS, (1190, 731, 1921), strict=True))
    # Cap reached after 4 answer tokens: no extra EOS prediction row.
    assert probe.arithmetic_cost(stages, 16)[probe.COST_KEYS[2]] == 1808
    # Fixed cap of 32 reserves 13 + 31 = 44 rows; charge all correction rows.
    assert probe.arithmetic_cost(stages, 17, preparation_rows=44) == dict(
        zip(probe.COST_KEYS, (1190, 1892, 3082), strict=True)
    )
    with pytest.raises(ValueError, match="cover all consumed"):
        probe.arithmetic_cost(stages, 17, preparation_rows=16)


def record(task_id, *, success, amount, tokens=3):
    return {
        "task_id": task_id,
        "success": success,
        "failure": None if success else "wrong_answer",
        "eos": True,
        "empty_eos_failure": False,
        "input_tokens": 10,
        "output_tokens": tokens,
        "selected_tokens": tokens + 1,
        "consumed_rows_per_stage": 10 + tokens,
        "prepared_rows_per_stage": 41,
        "cpu_seconds": 1,
        **dict(zip(probe.COST_KEYS, (amount, amount, amount * 2), strict=True)),
    }


def test_fallback_charges_failed_small_and_failed_large_attempts():
    small = [
        record("a", success=True, amount=10),
        record("b", success=False, amount=20),
        record("c", success=False, amount=30),
    ]
    large = [
        record("a", success=True, amount=100),
        record("b", success=True, amount=200),
        record("c", success=False, amount=300),
    ]
    result = probe.compare(large, small, probe.load_tasks()["policy"])
    summary = result["policies"]["smol_then_qwen"]
    assert result["fallback_attempts"] == 2
    assert result["total_attempts_with_fallback"] == 5
    assert summary["successes"] == 2
    assert summary["projected_all_link_arithmetic_bytes"] == 1120
    assert summary["projected_bytes_per_successful_task"][probe.COST_KEYS[2]] == 560
    assert summary["input_tokens"] == 50
    assert summary["output_tokens_excluding_eos"] == 15
    assert summary["eos_count"] == 5
    assert result["decisions"]["smol_then_qwen"]["quality_qualified"] is False


def test_fixed_quality_gate_and_zero_success_denominator():
    policy = probe.load_tasks()["policy"]
    large = [record(str(i), success=True, amount=100) for i in range(4)]
    small = [record(str(i), success=False, amount=10) for i in range(4)]
    result = probe.compare(large, small, policy)
    assert (
        result["policies"]["smol"]["projected_bytes_per_successful_task"][probe.COST_KEYS[2]]
        is None
    )
    assert result["decisions"]["smol"]["promising_under_fixed_gate"] is False
    assert result["decisions"]["smol_then_qwen"]["quality_qualified"] is True
    assert result["decisions"]["smol_then_qwen"]["promising_under_fixed_gate"] is False
    assert (
        result["decisions"]["smol_then_qwen"]["projected_all_link_bytes_per_success_ratio_to_qwen"]
        == 1.1
    )


def test_fallback_retains_failed_attempt_termination_and_inventory_charge():
    small = record("a", success=False, amount=10, tokens=32)
    small.update(eos=False, failure="truncation", selected_tokens=32, consumed_rows_per_stage=41)
    large = record("a", success=True, amount=100)
    summary = probe.compare([large], [small], probe.load_tasks()["policy"])["policies"][
        "smol_then_qwen"
    ]
    assert summary["successes"] == 1
    assert summary["failures"] == {}
    assert summary["attempt_truncations"] == 1
    assert summary["eos_count"] == 1
    assert summary["prepared_rows_per_stage"] == 82
    assert summary["projected_all_link_arithmetic_bytes"] == 220


def test_persisted_real_evidence_token_inventory_and_content_boundaries():
    document = probe.load_tasks()
    seen = set()
    for cohort in ("screen", "review"):
        path = ROOT / f"docs/evidence/task-qualified-models-{cohort}-2026-10-01.json"
        report = json.loads(path.read_text())
        assert report["task_contract_digest"] == probe.digest(document)
        assert report["policy"] == document["policy"]
        assert report["full_wire_bytes"] is None
        for name, model in report["models"].items():
            assert (model["model"], model["revision"]) == probe.MODELS[name]
            assert model["native_capabilities"]["compiled"] is True
            assert model["native_threads"] == 1
            assert (model["weight_bits"], model["activation_bits"]) == (8, 8)
            expected_ids = {t["id"] for t in document["tasks"] if t["cohort"] == cohort}
            assert {r["task_id"] for r in model["records"]} == expected_ids
            for row in model["records"]:
                assert (name, row["task_id"]) not in seen
                seen.add((name, row["task_id"]))
                assert 1 <= row["input_tokens"] <= 512
                assert 0 <= row["output_tokens"] <= 32
                assert row["selected_tokens"] == row["output_tokens"] + int(row["eos"])
                assert (
                    row["consumed_rows_per_stage"]
                    == row["input_tokens"] + row["selected_tokens"] - 1
                )
                assert row["prepared_rows_per_stage"] == row["input_tokens"] + 31
                assert row["stage_rows"] == len(model["stages"]) * row["consumed_rows_per_stage"]
                assert row["success"] == (row["answer_correct"] and row["eos"])
                assert {
                    "prompt",
                    "output_text",
                    "output_token_ids",
                    "text",
                    "generated",
                }.isdisjoint(row)
                assert len(row["output_text_digest"]) == len(row["output_token_digest"]) == 64
        assert path.with_suffix(".md").read_text() == probe.markdown(report)
        capture_policy = report["output_digest_capture_policy"]
        assert capture_policy["capture_output_digest"] is True
        assert capture_policy["ordinary_benchmark_default"] is False
        assert isinstance(capture_policy["canonical_examples_rerun_for_opt_in_policy"], bool)
        assert report["canonical_requested"] is True
        correction = report["sampling_correction"]
        assert correction["task_contract_unchanged"] is True
        assert correction["controlled_canonical_temperature"] == 0.0
        assert correction["historical_effective_temperature"] == 0.8
        assert correction["historical_examples_are_greedy_parity_evidence"] is False
        assert len(report["canonical_examples"]) == 2
        for example in report["canonical_examples"]:
            assert example["status"] == "validated"
            assert all(example["checks"].values())
            assert example["clear_output_content_parity_measured"] is True
            assert example["sampling"] == {
                "requested_temperature": 0.0,
                "effective_temperature": 0.0,
                "mode": "greedy",
                "top_p": None,
            }
            row = next(
                row
                for row in report["models"][example["model"]]["records"]
                if row["task_id"] == example["task_id"]
            )
            assert example["output_text_digest"] == row["output_text_digest"]
            assert example["eos"] == row["eos"]
            assert example["selected_tokens_including_eos"] == row["selected_tokens"]
            assert example["observed_consumed_rows_per_stage"] == row["consumed_rows_per_stage"]
            assert example["observed_prepared_rows_per_stage"] == row["prepared_rows_per_stage"]
            assert example["observed_burned_rows_per_stage"] == (
                row["prepared_rows_per_stage"] - row["consumed_rows_per_stage"]
            )
        if cohort == "review":
            previous = next(
                entry
                for entry in correction["historical_canonical_examples"]
                if entry["model"] == "smol"
            )
            assert previous["status"] == "cohort_mismatch"
            assert previous["observed_output_tokens"] == 32
            assert previous["clear_output_tokens"] == 10
    assert len(seen) == 32
