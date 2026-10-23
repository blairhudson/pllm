"""Render selected ordinary reports as Markdown/CSV; never execute experiments.

Usage (report paths in cells are relative to the working directory)::

    python docs/scripts/render-method-results.py --cells target/cells.json \
        --format markdown --output target/method-results.md

The presentation selection is an object with a ``cells`` list. Each cell has
``name``, a method id from the existing paper source lock, ``era`` (historical or
current_candidate), and ``scope`` (trained_checkpoint_response, historical_control,
generated_fixture or component). A measured selection also supplies ``report``,
``case`` and the explicitly frozen ``cohort``. An unmeasured selection supplies
``status`` (blocked, unsupported, failed, timeout or unmeasured) and ``reason``.
No report is discovered automatically. Missing methods get blocked rows.

For example, this selection retains the existing bounded GKM functionality cell::

    {"cells": [{"name": "gkm-frozen", "method": "gupta-katz-miers",
      "era": "historical", "scope": "trained_checkpoint_response",
      "report": "paper/artifacts/gkm-sdk/qwen-response-v1/benchmark.json",
      "case": "gkm-pinned-qwen-capacity-smoke",
      "cohort": "ce21d5fc51cfde677946baeeae3f3d9d75d156c61ec598c242d13900a8035eb7"}]}

Optional ``comparisons`` select cell names as ``control``/``candidate``, a metric
id and an existing ordinary BenchmarkRun.compare return value in ``record``.
Only that record is displayed, after reuse of the ordinary comparison helper;
no ranks, cross-report pools, ratios or throughput are manufactured here.
Historical controls outside the ten methods require ``source_reference`` naming
their retained source/contract and must use historical_control/historical scope.

This is a trusted-publisher presentation adapter, not source authentication,
receipt verification, cryptographic review or a publication completion gate.
Accepted response checks do not establish whole-method completeness, source-model
quality, independent operators or publication qualification. Source-lock entries
identify governing papers, not executed checkpoint bytes. No model/native imports.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path

from pllm.metrics.benchmark_runner import BenchmarkRun


ROOT = Path(__file__).resolve().parents[2]
RUNTIME_METHODS = {"gkm": "gupta-katz-miers", "fss": "fusefss", "replicated": "puma"}
METRICS = {
    "full_seconds": ("request.0.full", "seconds"),
    "covered_body_bytes": ("network.covered_body_bytes", "bytes"),
    "process_peak_rss_bytes": ("process.peak_rss", "bytes"),
    "client_group_sampled_rss_sum_bytes": ("client.owned_sampled_peak_rss_sum", "bytes"),
    "client_group_os_hwm_lower_bound_bytes": ("client.owned_peak_rss_lower_bound", "bytes"),
    "decode_tokens_per_second": ("request.0.decode_tps", "per_second"),
}
NOTE = (
    "Scoped observations, not whole-method acceptance or SOTA. One row per declared attempt; "
    "no pooling across cohorts or filtering failures into throughput. Unknown is not zero. "
    "Rows without ordinary report provenance are declarations, not measurements. "
    "RSS bytes distinguish process lifetime OS high-water, sampled helper-group sums and "
    "group high-water lower bounds through their original metric contracts; none is a "
    "physical whole-client peak. Windows/exclusions remain report-specific. "
    "Report hashes identify bytes, not authenticity or cryptographic security; immutable "
    "source-bound inputs and publisher declarations remain trusted."
)


def _json(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))


def _hash(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _validate_cell(cell, sources):
    required = {"name", "method", "era", "scope"}
    optional = {"report", "case", "cohort", "status", "reason", "source_reference"}
    if (not isinstance(cell, dict) or not required <= cell.keys()
            or cell.keys() - required - optional):
        raise ValueError("cell fields differ; whole-method/publication assertions are not inferred")
    if any(not isinstance(cell[k], str) or not cell[k].strip() for k in required):
        raise ValueError("cell identities must be nonempty strings")
    if cell["method"] not in sources and not (
            cell["scope"] == "historical_control" and cell["era"] == "historical"
            and isinstance(cell.get("source_reference"), str) and cell["source_reference"].strip()):
        raise ValueError("method must identify an existing source lock; historical controls need a source reference")
    if cell["era"] not in {"historical", "current_candidate"} or cell["scope"] not in {
        "trained_checkpoint_response", "historical_control", "generated_fixture", "component",
    }:
        raise ValueError("unsupported era or bounded evidence scope")
    if "report" in cell:
        if (not {"case", "cohort"} <= cell.keys() or not _hash(cell["cohort"])
                or {"status", "reason"} & cell.keys()):
            raise ValueError("report cells require exact case/cohort, never an overridden status")
    elif (cell.get("status") not in {"blocked", "unsupported", "failed", "timeout", "unmeasured"}
          or not isinstance(cell.get("reason"), str) or not cell["reason"].strip()):
        raise ValueError("unmeasured cells require an explicit non-success status and reason")


def _empty(cell, paper, *, status, reason):
    return {
        "method": cell["method"], "cell": cell["name"], "era": cell["era"], "scope": cell["scope"],
        "status": status, "reason": reason, "replicate": None,
        "qualification": "not established by this renderer",
        "source": {"paper_source_lock_entry": paper, "historical_source_reference": cell.get("source_reference")},
        "numeric": None, "privacy": None,
        "workload": None, "input_tokens": None, "output_tokens": None, "provenance": None,
        "checks": None, "metric_contracts": {}, "limitations": [], "comparisons": [],
        **dict.fromkeys(METRICS),
    }


def _observed(cell, paper, document, result, path, report_sha256):
    e = result["execution"]
    target = e["case"]["target"]
    parameters = target["parameters"]
    experiment = parameters.get("model", {}).get("experiment", {})
    pipeline = experiment.get("pipeline", {})
    report = e["details"].get("report", {})
    config = report.get("configuration", {})
    runtime = report.get("runtime_contract", e["details"].get("runtime_contract", {}))
    runs = report.get("runs", [])
    row = _empty(cell, paper, status=result["status"], reason=e["failure_code"])
    row.update(replicate=e["replicate"], checks=report.get("checks"),
        source={"paper_source_lock_entry": paper, "historical_source_reference": cell.get("source_reference"),
                "model": result["model"],
                "source_specification": pipeline.get("model"),
                "fixture": e["fixture"], "runtime_checkpoint": runtime.get("checkpoint_digest")},
        numeric={"commitment": e["comparison"]["numeric_digest"], "runtime": runtime.get("numeric"),
                 "quantization": pipeline.get("components", {}).get("quantization")},
        privacy={"commitment": e["comparison"]["privacy_digest"],
                 "components": pipeline.get("components"), "roles": config.get("roles"),
                 "admission": report.get("privacy_admission"), "sharing": runtime.get("sharing")},
        workload={"scenario": parameters.get("scenario"), "configuration": config,
                  "sampling": e["sampling"], "backend": parameters.get("backend"),
                  "runtime_contract": runtime,
                  "requests": [{k: run.get(k) for k in (
                      "status", "tokens", "model_fingerprint", "generation", "sampling", "warm",
                      "max_output_tokens")} for run in runs]},
        provenance={"report": str(path), "report_sha256": report_sha256, "result_id": result["id"],
                    "cohort": document["cohort"], "cohort_complete": document["complete"],
                    "build": e["build"]["digest"], "environment": result["environment"]["digest"],
                    "comparison_commitments": e["comparison"], "oracle": e["oracle"],
                    "budget_assessments": e["budgets"]},
        limitations=result["limitations"] + report.get("limitations", []))
    tokens = [run.get("tokens", {}) for run in runs]
    row["input_tokens"] = [t.get("input_tokens") for t in tokens] or None
    row["output_tokens"] = [t.get("output_tokens") for t in tokens] or None
    trained = cell["scope"] == "trained_checkpoint_response"
    if trained and (target["kind"] != "response" or e["fixture"]["kind"] not in {"checkpoint", "none"}
                    or pipeline.get("model", {}).get("kind") == "tiny" or config.get("tiny") is True):
        raise ValueError("component/generated probes cannot be presented as trained checkpoint responses")
    owner = RUNTIME_METHODS.get(runtime.get("runtime_kind"))
    if trained and owner is not None and owner != cell["method"]:
        raise ValueError("reported runtime belongs to another method, not the selected governing source")
    source = result["model"]["source_lock_digest"]
    bound = (_hash(source) and source == config.get("source_lock_digest") == e["fixture"]["digest"]
             == e["comparison"]["source_digest"] == runtime.get("source_lock_digest")
             and _hash(result["model"]["checkpoint_digest"])
             and result["model"]["checkpoint_digest"] == runtime.get("checkpoint_digest"))
    observed = (bool(tokens) and len(runs) == parameters.get("scenario", {}).get("count")
        and all(run.get("status") == "completed" and t.get("authoritative") is True
                and all(type(t.get(k)) is int and t[k] > 0 for k in ("input_tokens", "output_tokens"))
                and _hash(run.get("model_fingerprint"))
                and run["model_fingerprint"] == runtime.get("model_fingerprint")
                and _hash(run.get("generation", {}).get("output_text_digest"))
                for run, t in zip(runs, tokens, strict=True)))
    accepted = (result["status"] == "completed" and e["fixture"]["kind"] == "checkpoint"
        and owner == cell["method"] and report.get("checks", {}).get("passed") is True
        and config.get("tiny") is False
        and parameters.get("backend") == "native" and config.get("provider_backend") == "native"
        and bound and observed and type(runtime.get("vocabulary")) is int and runtime["vocabulary"] > 0)
    if result["status"] == "completed":
        row["status"] = "accepted_bounded_response" if trained and accepted else "completed_scoped_only"
        row["reason"] = None if trained and accepted else "complete trained identity/acceptance not established"
    for metric in result["metrics"]:
        preserved = dict(metric)
        # A failed attempt cannot donate a successful-subset throughput estimate.
        if metric["component"] == "pllm/throughput" and not accepted:
            preserved.update(value=None, unavailable_reason="attempt not accepted; no filtered-failure rate")
        row["metric_contracts"][metric["id"]] = preserved
    for column, (name, unit) in METRICS.items():
        metric = row["metric_contracts"].get(name)
        if metric is not None:
            if metric["unit"] != unit:
                raise ValueError(f"metric unit differs: {name}")
            value = metric["value"]
            if value is not None and (type(value) not in (int, float) or not math.isfinite(value)):
                raise ValueError("nonfinite metric")
            row[column] = value
    return row


def method_rows(selection, *, source_lock, base=None):
    """Read only explicitly selected canonical reports; preserve attempts and scope."""
    base = Path.cwd() if base is None else Path(base)
    if set(selection) - {"cells", "comparisons"} or not isinstance(selection.get("cells"), list):
        raise ValueError("expected presentation cells and optional ordinary comparison records")
    if source_lock.get("schema") != "pllm.ten_method_source_lock.v1":
        raise ValueError("expected existing ten-method paper source lock")
    sources = {entry["id"]: entry for entry in source_lock["sources"]}
    if len(sources) != len(source_lock["sources"]) or any(not _hash(p["sha256"]) for p in sources.values()):
        raise ValueError("ambiguous/unpinned governing paper source")
    rows, cells, loaded, grouped = [], {}, {}, {}
    for cell in selection["cells"]:
        _validate_cell(cell, sources)
        name = cell["name"]
        if name in cells:
            raise ValueError("duplicate cell name")
        cells[name] = cell
        selected = []
        if "report" not in cell:
            selected.append(_empty(cell, sources.get(cell["method"]), status=cell["status"], reason=cell["reason"]))
        else:
            path = (base / cell["report"]).resolve()
            if path not in loaded:
                raw = path.read_bytes()
                loaded[path] = (BenchmarkRun(json.loads(raw)), hashlib.sha256(raw).hexdigest())
            run, checksum = loaded[path]
            document = run.to_dict()
            if cell["cohort"] != document["cohort"]:
                raise ValueError("selection differs from explicitly frozen cohort")
            if cell["case"] not in {c["name"] for c in document["suite"]["cases"]}:
                raise ValueError("case not declared in ordinary report")
            trials = {r["execution"]["replicate"]: r for r in document["results"]
                      if r["execution"]["case"]["name"] == cell["case"]}
            for replicate in range(document["suite"]["sampling"]["processes"]):
                if replicate in trials:
                    selected.append(_observed(cell, sources.get(cell["method"]), document,
                                              trials[replicate], path, checksum))
                else:
                    missing = _empty(cell, sources.get(cell["method"]), status="unmeasured", reason="trial not recorded")
                    missing.update(replicate=replicate, provenance={"report": str(path), "cohort": document["cohort"]})
                    selected.append(missing)
        grouped[name] = selected
        rows.extend(selected)
    for method, paper in sources.items():
        if method not in {cell["method"] for cell in cells.values() if cell["scope"] == "trained_checkpoint_response"}:
            cell = {"name": method, "method": method, "era": "current_candidate", "scope": "trained_checkpoint_response"}
            rows.append(_empty(cell, paper, status="blocked", reason="no report/cell supplied; not zero performance"))
    for comparison in selection.get("comparisons", []):
        if set(comparison) != {"control", "candidate", "metric", "record"}:
            raise ValueError("comparison requires an existing ordinary comparison record")
        left, right = (cells[comparison[key]] for key in ("control", "candidate"))
        outcome = {"status": "unranked", "reason": "separate_or_unaccepted_cells"}
        if (all("report" in c for c in (left, right))
                and (base / left["report"]).resolve() == (base / right["report"]).resolve()
                and (left["era"], left["scope"]) == (right["era"], right["scope"])
                and all(r["status"] == "accepted_bounded_response" or (
                    r["scope"] == "historical_control" and r["status"] == "completed_scoped_only"
                    and (r["checks"] or {}).get("passed") is True
                    and r["workload"]["backend"] == "native"
                    and r["workload"]["requests"]
                    and all(request["status"] == "completed" and _hash(request["model_fingerprint"])
                        and request["tokens"].get("authoritative") is True
                        and all(type(request["tokens"].get(k)) is int and request["tokens"][k] > 0
                            for k in ("input_tokens", "output_tokens"))
                        for request in r["workload"]["requests"]))
                    for c in (left, right) for r in grouped[c["name"]])):
            run = loaded[(base / left["report"]).resolve()][0]
            outcome = run.compare(left["case"], right["case"], comparison["metric"])
            if outcome != comparison["record"]:
                raise ValueError("comparison record differs from ordinary comparison helper")
        shown = {**comparison, "record": outcome}
        for c in (left, right):
            for row in grouped[c["name"]]:
                row["comparisons"].append(shown)
    return rows


def render(rows, format="markdown"):
    """CSV retains full JSON-valued identity/metric contracts; Markdown adds them below."""
    columns = ["method", "cell", "era", "scope", "status", "replicate", "input_tokens", "output_tokens", *METRICS]
    metadata = ["qualification", "reason", "source", "numeric", "privacy", "workload", "provenance",
                "checks", "metric_contracts", "limitations", "comparisons"]

    def text(value):
        return "unknown" if value is None else _json(value) if isinstance(value, (dict, list)) else str(value)

    if format == "csv":
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=columns + metadata + ["presentation_limits"], lineterminator="\n")
        writer.writeheader()
        writer.writerows({**{key: text(row[key]) for key in columns + metadata}, "presentation_limits": NOTE}
                         for row in rows)
        return output.getvalue()
    if format != "markdown":
        raise ValueError("format must be markdown or csv")

    def safe(value):
        return text(value).replace("&", "&amp;").replace("<", "&lt;").replace("|", "&#124;").replace("\n", "<br>")

    lines = ["# Scoped method observations", "", NOTE, "", "| " + " | ".join(columns) + " |",
             "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(safe(row[key]) for key in columns) + " |")
    lines += ["", "## Identity, measurement windows and exclusions", ""]
    for row in rows:
        lines += [f"### {safe(row['cell'])} / replicate {safe(row['replicate'])}", ""]
        lines.extend(f"- **{key}**: {safe(row[key])}" for key in metadata)
        lines.append("")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cells", required=True, type=Path)
    parser.add_argument("--source-lock", type=Path, default=ROOT / "paper/methods/source-lock.v1.json")
    parser.add_argument("--format", choices=("markdown", "csv"), default="markdown")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        rows = method_rows(json.loads(args.cells.read_text()), source_lock=json.loads(args.source_lock.read_text()))
        content = render(rows, args.format)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    if args.output is None:
        print(content, end="")
    else:
        args.output.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    main()
