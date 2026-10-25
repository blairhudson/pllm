"""Build the PLLM papers from Markdown with Pandoc.

Optional ordinary-report assets (never inserted into manuscript/PDF)::

    python docs/scripts/build-papers.py paper \
        --method-results-cells paper=target/cells.json --method-results-downloads

Cells use the existing render-method-results.py selection fields; paths are
repository-relative or absolute. Only a cells list of report references/scoped
declarations is accepted, not copied report/comparison data. Optional per-cell
publication_qualified=false is supported; this build cannot establish positive
qualification. MD/CSV assets go to docs/build/papers/method-results; sanitized
downloads require the explicit switch. Unselected papers' inputs are not read.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import signal
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from dataclasses import dataclass, replace
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[2]
PAPER_DIR = ROOT / "paper"
BUILD_DIR = ROOT / "docs" / "build" / "papers"
FIGURES = BUILD_DIR / "figures"
DOWNLOADS = ROOT / "docs" / "public" / "downloads"
WEB_DIR = ROOT / "docs" / "content" / "research"


@dataclass(frozen=True)
class Paper:
    source: str
    output: str
    web: str
    max_pages: int
    bibliography: bool = False
    figures: tuple[str, ...] = ()
    figure_inputs: tuple[str, ...] = ()
    # Explicit presentation input only; never discover reports or alter the paper.
    method_results_cells: str | None = None


PAPERS = {
    # Total-document guards include references/appendices, not venue page limits.
    "paper": Paper("manuscript.md", "paper.pdf", "paper.mdx", 20, True),
    "low-memory": Paper("low-memory.md", "low-memory.pdf", "low-memory.mdx", 16, True,
                        figures=("low-memory-history",),
                        figure_inputs=("paper/artifacts/historical-results-v1.json",)),
    "whitepaper": Paper("whitepaper.md", "whitepaper.pdf", "whitepaper.mdx", 2,
                        figures=("mechanics", "research-loop"),
                        figure_inputs=("docs/evidence/current-runtime-2026-09-11.json",)),
}


def run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def executable(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise SystemExit(f"Required executable not found: {name}")
    return path


def pdf_engine(requested: str | None) -> str:
    if requested:
        return executable(requested)
    for name in ("tectonic", "pdflatex"):
        if path := shutil.which(name):
            return path
    raise SystemExit("Install Tectonic or pdfLaTeX to build the papers")


def browser_engine(requested: str | None) -> str:
    if requested:
        if path := shutil.which(requested):
            return path
        raise SystemExit(f"Browser executable not found: {requested}")
    for name in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser"):
        if path := shutil.which(name):
            return path
    mac_chrome = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
    if mac_chrome.is_file():
        return str(mac_chrome)
    raise SystemExit("Install Chrome or Chromium to build the whitepaper PDF")


def pandoc_args(spec: Paper) -> list[str]:
    args = [executable("pandoc"), str(PAPER_DIR / spec.source),
            "--from=markdown+raw_tex", "--fail-if-warnings"]
    if spec.bibliography:
        args.append("--citeproc")
    return args


def validate_pdf(output: Path, spec: Paper) -> None:
    info = subprocess.check_output([executable("pdfinfo"), str(output)], text=True)
    pages_match = re.search(r"^Pages:\s+(\d+)$", info, re.MULTILINE)
    if pages_match is None:
        raise SystemExit(f"Could not read page count from {output}")
    pages = int(pages_match.group(1))
    if pages > spec.max_pages:
        raise SystemExit(f"{output} has {pages} pages; limit is {spec.max_pages}")
    if not re.search(r"^Page size:\s+612 x 792 pts", info, re.MULTILINE):
        raise SystemExit(f"{output} is not US Letter")


def build_pdf(spec: Paper, engine: str) -> Path:
    output = BUILD_DIR / spec.output
    env = os.environ.copy()
    env.setdefault("SOURCE_DATE_EPOCH", "0")
    run(
        [
            *pandoc_args(spec),
            "--standalone",
            f"--pdf-engine={engine}",
            f"--include-in-header={PAPER_DIR / 'header.tex'}",
            f"--lua-filter={PAPER_DIR / 'pdf.lua'}",
            f"--output={output}",
        ],
        env=env,
    )
    validate_pdf(output, spec)
    return output


def whitepaper_profile_processes(profile: Path) -> list[int]:
    """Read-only coverage for detached helpers retaining this unique profile."""
    result = subprocess.run(["ps", "-axo", "pid=,command="], capture_output=True,
                            text=True, check=True, timeout=1)
    pattern = re.compile(re.escape(str(profile)) + r"(?=$|[/\s\"'])")
    return [int(fields[0]) for line in result.stdout.splitlines()
            if len(fields := line.split(None, 1)) == 2
            and fields[0].isdigit() and pattern.search(fields[1])]


def cleanup_whitepaper_browser(process: subprocess.Popen, profile: Path) -> None:
    """Reap our child, drain its private group, and check visible profile helpers."""
    ownership = f"owned browser PID/group {process.pid}, profile {profile}"
    try:
        # The PDF trailer can arrive just before Chrome's ordinary exit. Let that
        # exit finish before signalling a sandboxed process group unnecessarily.
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        for sig in (signal.SIGTERM, signal.SIGKILL):
            if process.poll() is not None:
                break
            try:
                os.killpg(process.pid, sig)
            except ProcessLookupError:
                # The leader/group may exit between poll and killpg.
                pass
            except PermissionError:
                # Group denial does not authorize signalling discovered helpers.
                # Popen still owns this exact unreaped leader; ask it to retire
                # its children, then verify group drain rather than assuming it.
                try:
                    process.send_signal(sig)
                except ProcessLookupError:
                    pass
                except PermissionError as error:
                    raise RuntimeError(f"Cannot signal {ownership}; cleanup unverified") from error
            try:
                process.wait(timeout=5)
                break
            except subprocess.TimeoutExpired as error:
                if sig == signal.SIGKILL:
                    raise RuntimeError(f"Timed out reaping {ownership}; cleanup unverified") from error

    deadline = time.monotonic() + 5
    while True:
        denied = False
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            try:
                helpers = whitepaper_profile_processes(profile)
            except (OSError, subprocess.SubprocessError) as error:
                raise RuntimeError(f"Cannot inspect helpers for {ownership}; cleanup unverified") from error
            if not helpers:
                return
            detail = f"profile-associated helpers remain: {helpers}"
        except PermissionError:
            denied = True
        else:
            detail = "group members remain"
        if time.monotonic() >= deadline:
            if denied:
                detail = "group visibility denied"
            raise RuntimeError(f"Cannot confirm drain of {ownership}: {detail}; helper cleanup unverified")
        time.sleep(0.1)


def build_whitepaper_pdf(spec: Paper, browser: str) -> Path:
    output = BUILD_DIR / spec.output
    with tempfile.TemporaryDirectory(prefix="pllm-whitepaper-") as temporary:
        temp = Path(temporary)
        html = temp / "whitepaper.html"
        pdf = temp / spec.output
        run([
            *pandoc_args(spec),
            "--standalone",
            "--to=html5",
            "--embed-resources",
            f"--resource-path={BUILD_DIR}",
            f"--template={PAPER_DIR / 'whitepaper_print.html'}",
            f"--css={PAPER_DIR / 'whitepaper_print.css'}",
            f"--lua-filter={PAPER_DIR / 'whitepaper_print.lua'}",
            f"--output={html}",
        ])
        command = [
            browser,
            "--headless=new",
            "--disable-gpu",
            "--disable-dev-shm-usage",
            "--disable-background-networking",
            "--disable-component-update",
            "--disable-extensions",
            "--disable-sync",
            "--no-default-browser-check",
            "--no-first-run",
            "--no-pdf-header-footer",
            f"--user-data-dir={temp / 'profile'}",
            f"--print-to-pdf={pdf}",
            html.as_uri(),
        ]
        # Chrome on macOS sometimes writes a complete PDF but leaves background
        # services running. Wait for the PDF trailer, then own process shutdown.
        with (temp / "browser.log").open("wb") as log:
            process = subprocess.Popen(
                command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                deadline = time.monotonic() + 40
                while time.monotonic() < deadline:
                    if pdf.is_file() and b"%%EOF" in pdf.read_bytes()[-1024:]:
                        break
                    if process.poll() is not None:
                        break
                    time.sleep(0.2)
                else:
                    raise SystemExit(
                        "Browser timed out printing whitepaper:\n"
                        + (temp / "browser.log").read_text(encoding="utf-8", errors="replace")[-1200:]
                    )
                if not pdf.is_file() or b"%%EOF" not in pdf.read_bytes()[-1024:]:
                    raise SystemExit(
                        "Browser did not print whitepaper:\n"
                        + (temp / "browser.log").read_text(encoding="utf-8", errors="replace")[-1200:]
                    )
            finally:
                render_error = sys.exception()
                try:
                    cleanup_whitepaper_browser(process, temp / "profile")
                except Exception as cleanup_error:
                    if render_error is None:
                        raise
                    message = f"Browser cleanup also failed: {cleanup_error}"
                    render_error.add_note(message)
                    print(message, file=sys.stderr)
        if preview := os.environ.get("PLLM_WHITEPAPER_PREVIEW"):
            shutil.copyfile(pdf, preview)
        validate_pdf(pdf, spec)
        shutil.copyfile(pdf, output)
    return output


def build_web(spec: Paper) -> Path:
    output = WEB_DIR / spec.web
    run(
        [
            *pandoc_args(spec),
            "--standalone",
            "--to=gfm",
            "--wrap=none",
            f"--template={PAPER_DIR / 'web.template.md'}",
            f"--lua-filter={PAPER_DIR / 'web.lua'}",
            f"--output={output}",
        ]
    )
    return output


def method_results_renderer():
    """Import the existing data-only renderer only for explicitly requested assets."""
    module = importlib.util.spec_from_file_location(
        "paper_method_results", ROOT / "docs/scripts/render-method-results.py",
    )
    if module is None or module.loader is None:
        raise ValueError("method-result renderer unavailable")
    renderer = importlib.util.module_from_spec(module)
    module.loader.exec_module(renderer)
    return renderer


def public_method_result_rows(rows: list[dict]) -> list[dict]:
    """Project public table metadata, not report payloads, paths, or arbitrary text.

    Keep renderer outcomes and values unchanged. Reports remain authoritative for
    full detail; downloads retain byte identities, scope, counts and cost windows.
    This projection is not a source-authentication or qualification mechanism.
    """
    def label(value):
        if value is None or type(value) in (int, float, bool):
            return value
        if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 ._+:;(),=!?%'\-]{0,1023}", value):
            return value
        return "withheld_nonpublic_metadata"

    def selected(value, keys):
        return {key: label(value[key]) for key in keys if key in (value or {})}

    def public_reference(value):
        if not isinstance(value, str):
            return None
        try:
            relative = (ROOT / value).resolve(strict=True).relative_to(ROOT.resolve())
        except (OSError, ValueError):
            return None
        text = relative.as_posix()
        return text if (text.startswith(("paper/", "docs/evidence/", "docs/research/"))
                        and re.fullmatch(r"[A-Za-z0-9._/\-]+", text)) else None

    def component(value):
        if not isinstance(value, dict):
            return None
        identifier = value.get("component")
        return {"component": identifier if isinstance(identifier, str) and re.fullmatch(
            r"pllm/[A-Za-z0-9_-]+(?:/[A-Za-z0-9_-]+)*", identifier) else "withheld_nonpublic_metadata",
            "params": selected(value.get("params"), (
                "sharing", "generation", "capacity", "acceptance", "weight_bits", "activation_bits"))}

    def paper_text(value):
        if isinstance(value, str) and len(value) <= 1024 and not re.search(
                r"[<>\[\]`\\\n\r\t]|(?:^|\s)(?:/|~/|[A-Za-z]:)", value):
            return value
        return "withheld_nonpublic_metadata"

    published = []
    for row in rows:
        # Only fixed renderer columns and selected scalar report metadata survive.
        out = {key: label(row.get(key)) for key in (
            "method", "cell", "era", "scope", "status", "reason", "qualification", "replicate",
            "full_seconds", "covered_body_bytes", "process_peak_rss_bytes",
            "client_group_sampled_rss_sum_bytes", "client_group_os_hwm_lower_bound_bytes", "decode_tokens_per_second")}
        for key in ("input_tokens", "output_tokens"):
            out[key] = [v if type(v) is int else None for v in row[key]] if row.get(key) is not None else None
        source = row.get("source") or {}
        paper = source.get("paper_source_lock_entry") or {}
        # Governing paper metadata comes from the existing repository source lock.
        # URLs are retained only as credential-free public HTTPS references.
        url = urlsplit(paper.get("url", ""))
        out["source"] = {"paper": {k: paper_text(paper[k]) for k in ("id", "name", "sha256") if k in paper},
            "paper_version": paper_text(paper["version"]) if "version" in paper else None,
            "paper_url": paper.get("url") if url.scheme == "https" and url.netloc
                and not url.username and not url.password and not re.search(r"[<>\[\]`\s]", paper["url"]) else None,
            "historical_reference": public_reference(source.get("historical_source_reference")),
            "model": selected(source.get("model"), ("id", "checkpoint_digest", "source_lock_digest")),
            "declared_revision": label((source.get("source_specification") or {}).get("revision")),
            "fixture_kind": label((source.get("fixture") or {}).get("kind")),
            "local_source_path": "withheld; use checkpoint/source-lock identities"
                if source.get("source_specification") else "not supplied"}
        numeric, privacy, workload = (row.get(key) or {} for key in ("numeric", "privacy", "workload"))
        out["numeric"] = {"commitment": label(numeric.get("commitment")),
            "runtime": selected(numeric.get("runtime"), (
                "numeric_profile", "local_numeric_profile", "lookup", "kv", "output_head")),
            "quantization": component(numeric.get("quantization"))}
        out["privacy"] = {"commitment": label(privacy.get("commitment")),
            "sharing": label(privacy.get("sharing")), "roles": [label(v) for v in privacy.get("roles") or []],
            "components": {label(k): component(v) for k, v in (privacy.get("components") or {}).items()}}
        configuration = workload.get("configuration") or {}
        out["workload"] = {"backend": label(workload.get("backend")),
            "scenario": selected(workload.get("scenario"), ("kind", "name", "count", "max_output_tokens")),
            "configuration": selected(configuration, (
                "prompt_digest", "prompt_sequence_digest", "source_lock_digest", "wan_emulation_digest",
                "link_conditions_digest", "provider_backend", "warmups", "repetitions", "max_output_tokens",
                "tiny", "output_delivery", "inventory_policy", "prefill_cache_mode", "runtime_kind")),
            "sampling": selected(workload.get("sampling"), (
                "memory_bytes", "processes", "repetitions", "warmups", "threads", "timeout_seconds", "order")),
            "runtime_contract": selected(workload.get("runtime_contract"), (
                "schema", "client_runtime", "runtime_kind", "vocabulary", "width", "matrix_count",
                "numeric_profile", "local_numeric_profile", "fractional_bits", "ring_bits",
                "generation_policy", "eos_policy", "sharing", "verification_window", "output_delivery",
                "decode_tps_supported", "decode_tps_unavailable_reason", "ttft_scope", "kv_reuse")),
            "requests": [{**selected(request, ("status", "model_fingerprint", "max_output_tokens", "warm")),
                "tokens": selected(request.get("tokens"), ("authoritative", "input_tokens", "output_tokens", "total_tokens")),
                "generation": selected(request.get("generation"), ("output_text_digest",)),
                "sampling": selected(request.get("sampling"), ("mode", "effective_temperature", "requested_temperature", "top_p"))}
                for request in workload.get("requests", [])]}
        provenance = row.get("provenance") or {}
        if provenance and row.get("reason") is not None:
            reason = row["reason"]
            out["reason"] = reason if isinstance(reason, str) and re.fullmatch(
                r"[a-z][a-z0-9_.-]{0,95}", reason) else "source report context withheld; inspect locally"
        out["provenance"] = {**selected(provenance, (
            "report_sha256", "cohort", "cohort_complete", "build", "environment")),
            "report": public_reference(provenance.get("report")),
            "report_supplied": bool(provenance.get("report")),
            "local_report_path": "withheld; use report_sha256" if provenance.get("report") else "not supplied",
            "comparison_commitments": selected(provenance.get("comparison_commitments"), (
                "source_digest", "numeric_digest", "privacy_digest", "state_digest", "workload_digest",
                "output_digest", "measurement_digest")),
            "oracle": selected(provenance.get("oracle"), ("kind", "outcome", "checked_values")),
            "budget_assessments": [selected(v, ("metric", "minimum", "maximum", "outcome", "observed"))
                                   for v in provenance.get("budget_assessments", [])]}
        out["checks"] = {label(k): v for k, v in (row.get("checks") or {}).items() if type(v) is bool}
        out["metric_contracts"] = {}
        for key, metric in row.get("metric_contracts", {}).items():
            out["metric_contracts"][label(key)] = {**selected(metric, (
                "id", "value", "unit", "unit_detail", "statistic", "phase", "role", "origin",
                "unavailable_reason", "sample_count", "window", "measurement_scope")),
                "component": (component({"component": metric.get("component")}) or {}).get("component"),
                "parameters": selected(metric.get("parameters"), ("kind", "statistic", "phase", "direction", "basis")),
                "denominator": selected(metric.get("denominator"), ("unit", "value")) if metric.get("denominator") else None,
                "owners": [label(v) for v in metric.get("owners", [])]}
            if metric.get("unavailable_reason"):
                reason = metric["unavailable_reason"]
                out["metric_contracts"][label(key)]["unavailable_reason"] = reason if isinstance(reason, str) and re.fullmatch(
                    r"[a-z][a-z0-9_.-]{0,95}", reason) else "unavailable in source report; inspect locally"
        out["limitations"] = [
            "Public projection withholds local paths and report narrative or exception payloads.",
            "Original report limitations are not copied; consult the identified canonical report before use."]
        out["comparisons"] = []  # Selection files contain references, not copied comparison data/rankings.
        published.append(out)
    return published


def build_method_results(name: str, spec: Paper) -> tuple[Path, ...]:
    """Optional normal MD/CSV build assets; no experiment execution or PDF insertion."""
    if spec.method_results_cells is None:
        return ()
    context = name if name in PAPERS else "selected paper"
    try:
        if name not in PAPERS:
            raise ValueError("unknown paper")
        path = ROOT / spec.method_results_cells
        with path.open("rb") as source:
            raw = source.read((1 << 20) + 1)
        if len(raw) > 1 << 20:
            raise ValueError("selection exceeds bound")
        selection = json.loads(raw)
        if set(selection) != {"cells"} or not isinstance(selection["cells"], list):
            raise ValueError("only explicit cell references/declarations are supported")
        cells = []
        for cell in selection["cells"]:
            normalized = dict(cell)
            if normalized.pop("publication_qualified", False) is not False:
                raise ValueError("publication qualification cannot be established by this asset")
            if not all(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}", normalized.get(k, "")) for k in ("name", "method")):
                raise ValueError("public cell labels must be bounded identifiers")
            cells.append(normalized)
        renderer = method_results_renderer()
        rows = renderer.method_rows({"cells": cells},
            source_lock=json.loads((ROOT / "paper/methods/source-lock.v1.json").read_text()), base=ROOT)
        rows = public_method_result_rows(rows)
        contents = {"md": renderer.render(rows, "markdown"), "csv": renderer.render(rows, "csv")}
        directory = BUILD_DIR / "method-results"
        directory.mkdir(parents=True, exist_ok=True)
        paths = tuple(directory / f"{name}.{extension}" for extension in contents)
        for output, content in zip(paths, contents.values(), strict=True):
            output.write_text(content, encoding="utf-8")
        return paths
    except Exception:
        # Neither OS paths, arbitrary exception text nor report context belong on
        # the public build log. Original reports stay local for diagnosis.
        raise SystemExit(f"Could not render method-result assets for {context}; inspect selected cells/reports locally.") from None


def referenced_method_results(name: str, spec: Paper, tex: str | None) -> tuple[Path, ...]:
    """Bundle only known generated assets actually read by the exported TeX."""
    if spec.method_results_cells is None or tex is None:
        return ()
    tex = re.sub(r"(?<!\\)%[^\n]*", "", tex)
    paths = (BUILD_DIR / "method-results" / f"{name}.{extension}" for extension in ("md", "csv"))
    return tuple(path for path in paths if re.search(
        r"\\(?:input|include|csvreader|csvautotabular|VerbatimInput|lstinputlisting)"
        r"(?:\[[^\]]*\])?\{" + re.escape(path.relative_to(ROOT).as_posix()) + r"\}", tex))


def source_archive(name: str, spec: Paper, *, result_tex: str | None = None) -> Path:
    output = DOWNLOADS / f"{name}-source.zip"
    files = [
        Path("LICENSE"),
        Path("docs/scripts/build-papers.py"),
        Path("paper") / spec.source,
        Path("paper/web.lua"),
        Path("paper/web.template.md"),
    ]
    if name == "paper":
        files.append(Path("paper/CLAIMS-AND-EVIDENCE.md"))
    if spec.bibliography:
        files.extend((Path("paper/header.tex"), Path("paper/pdf.lua"), Path("paper/references.bib")))
    if name == "whitepaper":
        files.extend([
            Path("paper/whitepaper_print.lua"),
            Path("paper/whitepaper_print.html"),
            Path("paper/whitepaper_print.css"),
        ])
    if spec.figures:
        files.extend([Path("docs/scripts/render-paper-figures.py"),
                      *(Path(relative) for relative in spec.figure_inputs)])
        for figure in spec.figures:
            files.extend([FIGURES.relative_to(ROOT) / f"{figure}.{extension}"
                          for extension in ("svg", "png")])
    files.extend(path.relative_to(ROOT) for path in referenced_method_results(name, spec, result_tex))
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for relative in sorted(files):
            info = zipfile.ZipInfo(relative.as_posix(), (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, (ROOT / relative).read_bytes())
    return output


def arxiv_archive(name: str, spec: Paper) -> Path:
    """Export standalone, citeproc-resolved TeX; arXiv need not run Pandoc."""
    output = DOWNLOADS / f"{name}-arxiv-source.zip"
    with tempfile.TemporaryDirectory(prefix="pllm-arxiv-") as temporary:
        source = Path(temporary) / "main.tex"
        run([
            *pandoc_args(spec), "--standalone", "--to=latex",
            f"--include-in-header={PAPER_DIR / 'header.tex'}",
            f"--lua-filter={PAPER_DIR / 'pdf.lua'}", f"--output={source}",
        ])
        files = {
            "main.tex": source.read_bytes(),
            "LICENSE": (ROOT / "LICENSE").read_bytes(),
            "README.txt": (
                f"PLLM {name}: standalone arXiv source\n\n"
                "Compile main.tex with pdfLaTeX twice. References are already\n"
                "resolved and embedded; no Pandoc, Python, BibTeX, external\n"
                "bibliography, or figure generation are required. Referenced\n"
                "images are bundled at their TeX-relative paths.\n\n"
                f"Canonical authoring source: paper/{spec.source} in the PLLM\n"
                "repository, https://github.com/blairhudson/pllm\n\n"
                "A buildable archive does not establish submission readiness.\n"
                "Working-draft evidence placeholders remain in main.tex.\n"
            ).encode(),
        }
        if spec.method_results_cells is not None:
            for asset in referenced_method_results(name, spec, source.read_text()):
                files[asset.relative_to(ROOT).as_posix()] = asset.read_bytes()
        for figure in spec.figures:
            relative = FIGURES.relative_to(ROOT) / f"{figure}.png"
            files[relative.as_posix()] = (ROOT / relative).read_bytes()
        with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for relative, content in sorted(files.items()):
                info = zipfile.ZipInfo(relative, (1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                archive.writestr(info, content)
    return output


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", choices=["all", *PAPERS], nargs="?", default="all")
    parser.add_argument("--pdf-engine")
    parser.add_argument("--browser", help="Chrome or Chromium binary for whitepaper print layout")
    parser.add_argument("--pdf-only", action="store_true")
    parser.add_argument("--method-results-cells", action="append", default=[], metavar="PAPER=PATH",
                        help="Explicit renderer cells JSON for a selected paper; no manuscript insertion")
    parser.add_argument("--method-results-downloads", action="store_true",
                        help="Also publish sanitized optional MD/CSV downloads (not with --pdf-only)")
    args = parser.parse_args(argv)

    names = PAPERS if args.target == "all" else (args.target,)
    overrides = {}
    for value in args.method_results_cells:
        name, separator, path = value.partition("=")
        if not separator or name not in PAPERS or not path or name in overrides:
            parser.error("method-results-cells requires one PAPER=PATH per known paper")
        overrides[name] = path
    specs = {name: replace(PAPERS[name], method_results_cells=overrides[name]) if name in overrides
             else PAPERS[name] for name in names}
    if args.method_results_downloads and (args.pdf_only or not any(s.method_results_cells for s in specs.values())):
        parser.error("method-result downloads require selected cells and a non-PDF-only build")
    results = {name: build_method_results(name, spec) for name, spec in specs.items() if spec.method_results_cells}
    engine = pdf_engine(args.pdf_engine) if any(PAPERS[name].bibliography for name in names) else None
    browser = browser_engine(args.browser) if "whitepaper" in names else None
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    figures = tuple(dict.fromkeys(figure for name in names for figure in PAPERS[name].figures))
    if figures:
        run([sys.executable, str(ROOT / "docs/scripts/render-paper-figures.py"), "--only", *figures])
    if not args.pdf_only:
        DOWNLOADS.mkdir(parents=True, exist_ok=True)
        WEB_DIR.mkdir(parents=True, exist_ok=True)
    for name in names:
        spec = specs[name]
        if name == "whitepaper":
            assert browser is not None
            pdf = build_whitepaper_pdf(spec, browser)
        else:
            assert engine is not None
            pdf = build_pdf(spec, engine)
        if not args.pdf_only:
            shutil.copyfile(pdf, DOWNLOADS / spec.output)
            if args.method_results_downloads:
                for asset in results.get(name, ()):
                    shutil.copyfile(asset, DOWNLOADS / f"{name}-method-results{asset.suffix}")
            if spec.figures:
                figure_downloads = DOWNLOADS / "figures"
                figure_downloads.mkdir(parents=True, exist_ok=True)
                for figure in spec.figures:
                    shutil.copyfile(FIGURES / f"{figure}.png",
                                    figure_downloads / f"{figure}.png")
            build_web(spec)
            if spec.method_results_cells and spec.bibliography:
                archive = arxiv_archive(name, spec)
                with zipfile.ZipFile(archive) as exported:
                    source_archive(name, spec, result_tex=exported.read("main.tex").decode("utf-8"))
            else:
                source_archive(name, spec)
                if spec.bibliography:
                    arxiv_archive(name, spec)
        print(f"Built {name} from paper/{spec.source}")


if __name__ == "__main__":
    main()
