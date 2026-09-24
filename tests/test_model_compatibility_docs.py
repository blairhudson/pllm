"""Keep published architecture claims below exact native adapter/runtime evidence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import pllm
from pllm.profiles import MaskedLinearCpu

ROOT = Path(__file__).resolve().parents[1]
INVENTORY = json.loads((ROOT / "docs/data/model-compatibility.json").read_text(encoding="utf-8"))
EXPECTED_ADAPTERS = {
    "pllm.qwen2.v1", "pllm.qwen3.v1", "pllm.qwen3_5_text.v1",
    "pllm.phi4_mini.v1", "pllm.gemma4_e2b_text.v1", "pllm.gemma4_e4b_text.v1",
}


def test_model_inventory_has_all_checked_source_readers_without_invented_runtime_paths() -> None:
    adapters = INVENTORY["adapters"]
    assert {row["adapter"] for row in adapters} == EXPECTED_ADAPTERS
    assert {row["adapter"] for row in adapters if row["baseline_schedule"]} == {
        "pllm.qwen2.v1", "pllm.qwen3.v1",
    }
    assert len({row["model_type"] for row in INVENTORY["candidates"]}) == len(INVENTORY["candidates"])


@pytest.mark.parametrize("row", INVENTORY["adapters"], ids=lambda row: row["adapter"])
def test_documented_adapters_and_compiled_baseline_scope(row: dict) -> None:
    source = (ROOT / row["fixture"]).read_bytes() if "fixture" in row else row["config"]
    plan = pllm.lower_model(source, batch=1, max_input_tokens=4, max_new_tokens=2)
    assert plan.to_dict()["adapter"] == row["adapter"]
    assert plan.to_dict()["model_family"] == row["model_family"]
    operators = {operation["operator"] for operation in plan.prefill["operations"]}
    assert set(row["baseline_blockers"]) <= set(row["requires"])
    assert row["baseline_schedule"] is (not row["baseline_blockers"])
    for identity in row["requires"]:
        assert identity in INVENTORY["capabilities"]
        operator = INVENTORY["capabilities"][identity]["operator"]
        if operator is not None:
            assert operator in operators, f"{row['name']} lacks declared {identity} operator {operator}"
    baseline = MaskedLinearCpu(pllm.Model("org/model"))
    if row["baseline_schedule"]:
        schedule = plan.runtime_schedule(baseline)
        assert schedule.complete is True
        assert schedule.protected_execution is False
    else:
        assert plan.coverage(baseline).complete is False
        with pytest.raises(ValueError):
            plan.runtime_schedule(baseline)


@pytest.mark.parametrize("row", INVENTORY["candidates"], ids=lambda row: row["model_type"])
def test_candidate_architectures_cannot_impersonate_qwen2(row: dict) -> None:
    assert set(row["requires"]) <= INVENTORY["capabilities"].keys()
    qwen2 = INVENTORY["adapters"][0]["config"]
    with pytest.raises(ValueError, match="no decoder adapter"):
        pllm.lower_model(
            {**qwen2, "model_type": row["model_type"]},
            batch=1, max_input_tokens=4, max_new_tokens=2,
        )
