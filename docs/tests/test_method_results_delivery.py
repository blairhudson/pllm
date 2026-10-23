"""Optional result-asset routing; real report parsing, no PDF/model/native jobs."""

from __future__ import annotations

import csv
from dataclasses import replace
import importlib.util
from importlib.abc import MetaPathFinder
import io
import json
from pathlib import Path
import sys
from unittest.mock import Mock
from zipfile import ZipFile

import pytest


ROOT = Path(__file__).resolve().parents[2]
GKM = ROOT / "paper/artifacts/gkm-sdk/qwen-response-v1/benchmark.json"


@pytest.fixture
def builder(monkeypatch, tmp_path):
    class NoExecutionImports(MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if any(fullname == name or fullname.startswith(name + ".") for name in (
                "pllm._native", "pllm.model_loader", "torch", "transformers", "numpy",
            )):
                raise AssertionError(f"asset delivery attempted execution import: {fullname}")
            return None

    monkeypatch.setattr(sys, "meta_path", [NoExecutionImports(), *sys.meta_path])
    spec = importlib.util.spec_from_file_location("method_results_builder", ROOT / "docs/scripts/build-papers.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "BUILD_DIR", tmp_path / "build")
    monkeypatch.setattr(module, "DOWNLOADS", tmp_path / "downloads")
    monkeypatch.setattr(module, "WEB_DIR", tmp_path / "web")
    return module


@pytest.fixture
def cells(tmp_path):
    document = json.loads(GKM.read_text())
    selection = {"cells": [{"name": "gkm-frozen", "method": "gupta-katz-miers",
        "era": "historical", "scope": "trained_checkpoint_response", "publication_qualified": False,
        "report": str(GKM), "case": document["suite"]["cases"][0]["name"], "cohort": document["cohort"]},
        {"name": "fss-unaccepted", "method": "fusefss", "era": "current_candidate",
         "scope": "trained_checkpoint_response", "publication_qualified": False,
         "status": "failed", "reason": "no accepted response; cause unassigned"}]}
    path = tmp_path / "explicit-cells.json"
    path.write_text(json.dumps(selection))
    return path


def mock_build(builder, monkeypatch, tmp_path):
    """External document compilation is mocked, never ordinary report parsing."""
    pdf = tmp_path / "mock.pdf"
    pdf.write_bytes(b"document compiler mocked; not a PDF fixture result")
    # The external figure job is mocked too. Fresh worktrees have no ignored
    # plot products; never depend on a previous real paper build for this test.
    figures = tmp_path / "mock-figures"
    figures.mkdir(exist_ok=True)
    monkeypatch.setattr(builder, "FIGURES", figures)
    for spec in builder.PAPERS.values():
        for figure in spec.figures:
            (figures / f"{figure}.png").write_bytes(b"mocked figure-copy input; not rendered plot evidence")
    jobs = {}
    for name in ("run", "build_web", "source_archive"):
        jobs[name] = Mock()
        monkeypatch.setattr(builder, name, jobs[name])
    for name in ("pdf_engine", "browser_engine"):
        jobs[name] = Mock(return_value="unused")
        monkeypatch.setattr(builder, name, jobs[name])
    for name in ("build_pdf", "build_whitepaper_pdf"):
        jobs[name] = Mock(return_value=pdf)
        monkeypatch.setattr(builder, name, jobs[name])

    def arxiv(name, spec):
        output = builder.DOWNLOADS / f"{name}-arxiv-source.zip"
        with ZipFile(output, "w") as archive:
            archive.writestr("main.tex", "No result asset is referenced in this mock document.")
        return output

    jobs["arxiv_archive"] = Mock(side_effect=arxiv)
    monkeypatch.setattr(builder, "arxiv_archive", jobs["arxiv_archive"])
    return jobs


def test_default_build_preserves_jobs_limits_and_does_not_import_renderer(builder, monkeypatch, tmp_path):
    jobs = mock_build(builder, monkeypatch, tmp_path)
    loader = Mock(side_effect=AssertionError("default build must not load result inputs"))
    monkeypatch.setattr(builder, "method_results_renderer", loader)
    builder.main([])
    loader.assert_not_called()
    assert not (builder.BUILD_DIR / "method-results").exists()
    assert {key: value.max_pages for key, value in builder.PAPERS.items()} == {
        "paper": 20, "low-memory": 16, "whitepaper": 2}
    assert all(s.method_results_cells is None for s in builder.PAPERS.values())
    assert jobs["build_pdf"].call_count == 2 and jobs["build_whitepaper_pdf"].call_count == 1
    assert jobs["build_web"].call_count == jobs["source_archive"].call_count == 3
    assert jobs["arxiv_archive"].call_count == 2
    jobs["run"].assert_called_once_with([sys.executable, str(ROOT / "docs/scripts/render-paper-figures.py"),
        "--only", "low-memory-history", "mechanics", "research-loop"])
    assert not list(builder.DOWNLOADS.glob("*method-results*"))


def test_cli_help_documents_opt_in_without_loading_reports(builder, monkeypatch, capsys):
    loader = Mock(side_effect=AssertionError("help must not load data"))
    monkeypatch.setattr(builder, "method_results_renderer", loader)
    with pytest.raises(SystemExit) as stopped:
        builder.main(["--help"])
    assert stopped.value.code == 0
    help_text = capsys.readouterr().out
    assert "--method-results-cells PAPER=PATH" in help_text
    assert "--method-results-downloads" in help_text
    loader.assert_not_called()


@pytest.mark.parametrize("target", ["paper", "low-memory", "all"])
def test_explicit_assets_obey_selected_papers_and_pdf_only_filter(builder, cells, monkeypatch, tmp_path, target):
    jobs = mock_build(builder, monkeypatch, tmp_path)
    builder.main([target, "--pdf-only", "--method-results-cells", f"paper={cells}"])
    assets = builder.BUILD_DIR / "method-results"
    assert (assets / "paper.csv").exists() == (target != "low-memory")
    assert not (assets / "low-memory.csv").exists()
    assert not builder.DOWNLOADS.exists()
    jobs["source_archive"].assert_not_called()
    jobs["build_web"].assert_not_called()
    if target == "low-memory":
        jobs["run"].assert_called_once_with([sys.executable, str(ROOT / "docs/scripts/render-paper-figures.py"),
                                           "--only", "low-memory-history"])


def test_unselected_missing_source_is_not_read_and_selected_missing_source_fails_before_build(builder, monkeypatch, tmp_path):
    jobs = mock_build(builder, monkeypatch, tmp_path)
    builder.main(["low-memory", "--pdf-only", "--method-results-cells", "paper=/Users/private/missing.json"])
    jobs["build_pdf"].assert_called_once()
    jobs["build_pdf"].reset_mock()
    with pytest.raises(SystemExit) as stopped:
        builder.main(["paper", "--pdf-only", "--method-results-cells", "paper=/Users/private/missing.json"])
    assert "method-result assets for paper" in str(stopped.value)
    assert "/Users" not in str(stopped.value)
    jobs["build_pdf"].assert_not_called()


def test_paper_spec_and_download_flag_route_actual_sanitized_assets(builder, cells, monkeypatch, tmp_path):
    jobs = mock_build(builder, monkeypatch, tmp_path)
    papers = {**builder.PAPERS, "paper": replace(builder.PAPERS["paper"], method_results_cells=str(cells))}
    monkeypatch.setattr(builder, "PAPERS", papers)
    builder.main(["paper", "--method-results-downloads"])
    for suffix in ("md", "csv"):
        source = builder.BUILD_DIR / "method-results" / f"paper.{suffix}"
        download = builder.DOWNLOADS / f"paper-method-results.{suffix}"
        assert source.read_bytes() == download.read_bytes()
    table = list(csv.DictReader(io.StringIO((builder.DOWNLOADS / "paper-method-results.csv").read_text())))
    accepted, failed = table[:2]
    assert accepted["status"] == "accepted_bounded_response"
    assert accepted["qualification"] == "not established by this renderer"
    assert accepted["input_tokens"] == "[4]" and accepted["output_tokens"] == "[2]"
    assert accepted["decode_tokens_per_second"] == "unknown"
    source = json.loads(accepted["source"])
    assert source["paper_version"] == "ePrint 2026/1849, hash-pinned revision"
    assert source["model"]["id"] == "7ae557604adf67be50417f59c2c2f167def9a775"
    assert source["model"]["source_lock_digest"] == "880f80ec61274d3c80e2a0c1336394b9e955d53ff98436a426a680a460a592e7"
    assert json.loads(accepted["numeric"])["runtime"]["local_numeric_profile"] == "semantic.float32.dense-full-kv.client-bias-after-div22.v1"
    assert json.loads(accepted["privacy"])["roles"] == ["client", "worker_a", "worker_b"]
    assert json.loads(accepted["privacy"])["sharing"] == "full"
    assert json.loads(accepted["workload"])["runtime_contract"]["vocabulary"] == 151936
    assert json.loads(accepted["workload"])["configuration"]["output_delivery"] == "buffered_until_verified"
    assert failed["status"] == "failed" and failed["full_seconds"] == "unknown"
    assert json.loads(failed["provenance"])["report_supplied"] is False
    assert json.loads(failed["provenance"])["local_report_path"] == "not supplied"
    assert all(json.loads(r["comparisons"]) == [] for r in table)
    metrics = json.loads(accepted["metric_contracts"])
    assert metrics["process.peak_rss"]["unit"] == "bytes"
    assert metrics["process.peak_rss"]["measurement_scope"] == "process_lifetime_high_water"
    assert metrics["network.covered_body_bytes"]["window"] == "setup_through_scenario"
    assert "client_group_sampled_rss_sum_bytes" in accepted and "client_group_os_hwm_lower_bound_bytes" in accepted
    assert jobs["source_archive"].call_args.kwargs["result_tex"].startswith("No result asset")


def test_public_projection_never_publishes_private_paths_context_or_active_markdown(builder, cells, monkeypatch):
    renderer = builder.method_results_renderer()
    original = renderer.method_rows

    def contaminated(*args, **kwargs):
        rows = original(*args, **kwargs)
        row = rows[0]
        secret = "/Users/private-person/secret-context [click](javascript:alert(1)) <img src=x onerror=alert(1)>"
        row["source"]["source_specification"] = {"source": secret}
        row["privacy"]["components"]["linear"]["params"]["private_token"] = secret
        row["workload"]["configuration"]["prompt"] = "do-not-publish-this-prompt"
        row["workload"]["requests"][0]["generation"]["output_text"] = "do-not-publish-this-response"
        row["provenance"]["report"] = secret
        row["limitations"].append(secret)
        row["limitations"].append("private plain narrative not intended for public delivery")
        row["metric_contracts"]["request.0.decode_tps"]["unavailable_reason"] = secret
        row["reason"] = secret
        return rows

    monkeypatch.setattr(renderer, "method_rows", contaminated)
    monkeypatch.setattr(builder, "method_results_renderer", lambda: renderer)
    paths = builder.build_method_results("paper", replace(builder.PAPERS["paper"], method_results_cells=str(cells)))
    for path in paths:
        text = path.read_text()
        for forbidden in ("/Users", "private-person", "secret-context", "javascript:", "<img", "[click]",
                          "do-not-publish-this-prompt", "do-not-publish-this-response"):
            assert forbidden not in text
        assert "private plain narrative" not in text
        assert "withheld" in text


@pytest.mark.parametrize("extra", [{"metrics": {"copied_result": 1}}, {"report_payload": {}}, {"publication_qualified": True}])
def test_selection_rejects_inline_report_dumps_and_completion_assertions(builder, cells, extra):
    value = json.loads(cells.read_text())
    value["cells"][0].update(extra)
    cells.write_text(json.dumps(value))
    with pytest.raises(SystemExit, match="inspect selected cells/reports locally"):
        builder.build_method_results("paper", replace(builder.PAPERS["paper"], method_results_cells=str(cells)))
    assert not (builder.BUILD_DIR / "method-results").exists()


def test_frozen_cohort_guard_is_reused_not_bypassed(builder, cells):
    value = json.loads(cells.read_text())
    value["cells"][0]["cohort"] = "0" * 64
    cells.write_text(json.dumps(value))
    with pytest.raises(SystemExit, match="inspect selected cells/reports locally"):
        builder.build_method_results("paper", replace(builder.PAPERS["paper"], method_results_cells=str(cells)))
    assert not (builder.BUILD_DIR / "method-results").exists()


def test_selection_never_copies_comparison_records_into_config_or_assets(builder, cells):
    value = json.loads(cells.read_text())
    value["comparisons"] = [{"record": {"status": "comparable", "ratio": 123}}]
    cells.write_text(json.dumps(value))
    with pytest.raises(SystemExit, match="inspect selected cells/reports locally"):
        builder.build_method_results("paper", replace(builder.PAPERS["paper"], method_results_cells=str(cells)))


def test_report_errors_are_sanitized_before_any_external_jobs(builder, cells, monkeypatch, tmp_path, capsys):
    jobs = mock_build(builder, monkeypatch, tmp_path)
    renderer = builder.method_results_renderer()
    monkeypatch.setattr(renderer, "method_rows", Mock(side_effect=ValueError(
        "/Users/private/report.json secret-prompt-markers <script>")))
    monkeypatch.setattr(builder, "method_results_renderer", lambda: renderer)
    with pytest.raises(SystemExit) as stopped:
        builder.main(["paper", "--method-results-cells", f"paper={cells}"])
    assert "secret-prompt" not in str(stopped.value) and "/Users" not in str(stopped.value)
    assert stopped.value.__suppress_context__
    assert capsys.readouterr().out == ""
    jobs["run"].assert_not_called()
    jobs["pdf_engine"].assert_not_called()


@pytest.mark.parametrize("args", [["--pdf-only", "--method-results-downloads"], ["--method-results-downloads"]])
def test_download_switch_requires_explicit_selected_non_pdf_build(builder, monkeypatch, args):
    monkeypatch.setattr(builder, "method_results_renderer", Mock(side_effect=AssertionError("unexpected renderer")))
    with pytest.raises(SystemExit) as stopped:
        builder.main(["paper", *args])
    assert stopped.value.code == 2


@pytest.mark.parametrize("reference", ["none", "comment", "csv"])
def test_both_archives_include_only_assets_referenced_by_exported_tex(builder, monkeypatch, tmp_path, reference):
    root = tmp_path / "repository"
    monkeypatch.setattr(builder, "ROOT", root)
    monkeypatch.setattr(builder, "PAPER_DIR", root / "paper")
    monkeypatch.setattr(builder, "BUILD_DIR", root / "docs/build/papers")
    monkeypatch.setattr(builder, "DOWNLOADS", root / "downloads")
    builder.DOWNLOADS.mkdir(parents=True)
    spec = replace(builder.PAPERS["paper"], method_results_cells="explicit-cells-not-archived.json")
    for path in ("LICENSE", "docs/scripts/build-papers.py", "paper/manuscript.md", "paper/web.lua",
                 "paper/web.template.md", "paper/header.tex", "paper/pdf.lua", "paper/references.bib"):
        file = root / path
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("ordinary archive plumbing test fixture; no copied source/report")
    directory = builder.BUILD_DIR / "method-results"
    directory.mkdir(parents=True)
    for suffix in ("md", "csv"):
        (directory / f"paper.{suffix}").write_text("harmless result-asset fixture")
    relative = "docs/build/papers/method-results/paper.csv"
    tex = {"none": "No optional asset.", "comment": "% \\csvautotabular{" + relative + "}",
           "csv": "\\csvautotabular{" + relative + "}"}[reference]

    def export(command):
        destination = next(arg.split("=", 1)[1] for arg in command if arg.startswith("--output="))
        Path(destination).write_text(tex)

    monkeypatch.setattr(builder, "pandoc_args", lambda spec: ["pandoc-mocked"])
    monkeypatch.setattr(builder, "run", export)
    arxiv = builder.arxiv_archive("paper", spec)
    technical = builder.source_archive("paper", spec, result_tex=tex)
    for path in (arxiv, technical):
        with ZipFile(path) as archive:
            assert (relative in archive.namelist()) == (reference == "csv")
            assert "docs/build/papers/method-results/paper.md" not in archive.namelist()
            assert not any(name.endswith(".json") for name in archive.namelist())
            assert not any("explicit-cells" in name for name in archive.namelist())
