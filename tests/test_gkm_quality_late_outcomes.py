"""Late-position quality adversaries; synthetic execution, never trained evidence.

Reuse the ordinary CLI/factories/scorer and existing bounded source/transport
doubles. No checkpoint, Torch model or numerical native job is loaded.
"""
import json
import math

import numpy as np
import pytest

# Re-export existing fixtures; do not create another quality harness.
from test_paper_quality_cli import documented_argv, lane as lane, quality as quality


def argv_for(lane, tmp_path, mode):
    argv = documented_argv(lane.output)
    if mode == "record":
        key = tmp_path / "synthetic-only.key"
        key.write_bytes(b"k" * 32)
        teacher = tmp_path / "synthetic-only-teacher.json"
        teacher.write_text(json.dumps([[2, 2, 2]] * 12))
        argv += ["--outcomes", "record", "--cohort-key-file", str(key),
                 "--cohort-key-id", "gkm-late-test", "--continuations-file", str(teacher)]
    return argv


def assert_unranked_record(lane, capsys):
    from pllm.runtime.quality_outcomes import read_report, require_complete_for_ranking

    captured = capsys.readouterr()
    assert json.loads(captured.err)["error"]["code"] == "QUALITY_EXECUTION_FAILED"
    report = read_report(lane.output.read_bytes())
    assert json.loads(captured.out)["data"]["report"] == report
    assert report["schema_version"] == "pllm.reference_quality_benchmark.v2"
    assert report["terminal_state"] == "execution_failed"
    assert report["reference_status"] == "unavailable"
    assert report["cohort"]["prompt_count"] == 12 and report["cohort"]["position_count"] == 48
    for row in report["candidates"]:
        assert row["all_positions"]["planned_sample_count"] == 48
        assert len(row["all_positions"]["outcomes"]) == 48
        for section in (row, row["all_positions"], row["same_token_decode"]):
            assert section["sample_count"] == section["accepted_agreement"]["sample_count"] == 0
            for metric in ("top1_agreement", "top_k_recall", "max_abs_logit_error"):
                assert section[metric] is section["accepted_agreement"][metric] is None
    with pytest.raises(ValueError, match="unavailable"):
        require_complete_for_ranking(report)
    return report


@pytest.mark.parametrize("mode", ["abort", "record"])
@pytest.mark.parametrize("sharing", ["full", "seeded"])
@pytest.mark.parametrize("fault", ["nonfinite", "short-head"])
def test_last_candidate_request_last_decode_never_leaves_a_success_subset(
        lane, tmp_path, capsys, mode, sharing, fault):
    from pllm.cli import main

    calls = {"full": 0, "seeded": 0}

    def accepted(experiment, *args, **kwargs):
        profile = experiment.pipeline.components["linear"].params["sharing"]
        calls[profile] += 1
        result = lane.accepted(experiment, *args, **kwargs)
        if profile == sharing and calls[profile] == 12:
            # First eleven requests and first three vectors look correct. Only
            # the final logical position is corrupt, after synthetic acceptance.
            assert len(result.logits) == 4
            result.logits[-1] = (np.full(5, np.inf, dtype="<f4") if fault == "nonfinite"
                                else np.zeros(4, dtype="<f4")).tobytes()
        return result

    lane.execute.side_effect = accepted
    with pytest.raises(SystemExit) as stopped:
        main(argv_for(lane, tmp_path, mode))
    assert stopped.value.code == 6
    assert calls == ({"full": 12, "seeded": 0} if sharing == "full"
                     else {"full": 12, "seeded": 12})
    lane.streamed.assert_not_called()
    lane.forbidden.assert_not_called()
    if mode == "abort":
        captured = capsys.readouterr()
        assert json.loads(captured.err)["error"]["code"] == "QUALITY_FAILED"
        assert captured.out == "" and not lane.output.exists()
        return
    report = assert_unranked_record(lane, capsys)
    failed_index = ("full", "seeded").index(sharing)
    section = report["candidates"][failed_index]["all_positions"]
    assert section["coverage"] == {"total": 48, "accepted": 44, "rejected": 0, "failed": 4}
    assert all(row["status"] == "accepted" for row in section["outcomes"][:44])
    assert section["outcomes"][44] == {
        "position": 44, "status": "failed", "code": "execution_failed", "attempted": True}
    assert all(row["status"] == "failed" and row["code"] == "candidate_stopped"
               and not row["attempted"] for row in section["outcomes"][45:])
    other = report["candidates"][1 - failed_index]["all_positions"]
    assert other["coverage"] == ({"total": 48, "accepted": 0, "rejected": 0, "failed": 48}
                                 if sharing == "full" else
                                 {"total": 48, "accepted": 48, "rejected": 0, "failed": 0})


@pytest.mark.parametrize("mode", ["abort", "record"])
@pytest.mark.parametrize("fault", ["nonfinite", "missing-row", "extra-row"])
def test_last_reference_decode_cannot_score_complete_candidate_heads(
        lane, tmp_path, capsys, mode, fault):
    from pllm.cli import main

    def trajectory(ids, continuation):
        rows = [lane.vector.copy() for _ in range(4)]
        if lane.reference.trajectory.call_count == 12:
            if fault == "nonfinite":
                rows[-1][-1] = np.nan
            elif fault == "missing-row":
                rows.pop()
            else:
                rows.append(lane.vector.copy())
        return rows

    lane.reference.trajectory.side_effect = trajectory
    with pytest.raises(SystemExit) as stopped:
        main(argv_for(lane, tmp_path, mode))
    assert stopped.value.code == 6
    assert lane.execute.call_count == 24 and lane.reference.trajectory.call_count == 12
    assert lane.source_reader.call_count == 2  # Actual shared before/after pin guards still run.
    lane.forbidden.assert_not_called()
    if mode == "abort":
        captured = capsys.readouterr()
        assert json.loads(captured.err)["error"]["code"] == "QUALITY_FAILED"
        assert captured.out == "" and not lane.output.exists()
        return
    report = assert_unranked_record(lane, capsys)
    for candidate in report["candidates"]:
        assert candidate["all_positions"]["coverage"] == {
            "total": 48, "accepted": 48, "rejected": 0, "failed": 0}


def test_complete_gkm_record_scores_exactly_48_positions_each(lane, tmp_path, capsys):
    from pllm.cli import main
    from pllm.runtime.quality_outcomes import read_report, require_complete_for_ranking

    main(argv_for(lane, tmp_path, "record"))
    captured = capsys.readouterr()
    assert captured.err == ""
    report = read_report(lane.output.read_bytes())
    assert json.loads(captured.out)["data"]["report"] == report
    require_complete_for_ranking(report)
    assert report["terminal_state"] == report["reference_status"] == "complete"
    assert report["cohort"]["position_count"] == 48
    assert [row["configuration_digest"] for row in report["candidates"]] == [
        lane.quality.experiment().configuration_digest(), lane.quality.seeded().configuration_digest()]
    assert report["model"]["checkpoint_digest"] == lane.source.source.checkpoint_digest
    assert report["model"]["source_lock_digest"] == lane.source.source.source_lock_digest
    for candidate in report["candidates"]:
        for section, total in ((candidate, 12), (candidate["same_token_decode"], 36),
                               (candidate["all_positions"], 48)):
            assert section["planned_sample_count"] == section["sample_count"] == total
            assert section["coverage"] == {"total": total, "accepted": total, "rejected": 0, "failed": 0}
            for metric in ("top1_agreement", "top_k_recall", "max_abs_logit_error"):
                assert math.isfinite(section[metric])
        assert candidate["all_positions"]["outcomes"] == [
            {"position": i, "status": "accepted", "code": None, "attempted": True} for i in range(48)]
    assert lane.execute.call_count == 24 and lane.reference.trajectory.call_count == 12
    assert lane.source_reader.call_count == 2
    lane.forbidden.assert_not_called()
