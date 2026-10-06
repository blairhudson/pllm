"""Inventory the paper landing map and fetch verified primary PDFs when available.

This is an explicit, networked maintenance command, never part of docs builds.
Research records and implementation claims remain independent of this library.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher
from hashlib import sha256
import json
from pathlib import Path
import re
import subprocess
import tempfile
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
MAP = ROOT / "docs/research/paper-module-landing-map-2026-09-23.md"
RECORDS = ROOT / "docs/data/research/papers.json"
LIBRARY = ROOT / "docs/data/research/paper-library.json"
PAPERS = ROOT / "papers"
HEADING = re.compile(r"^#### \[(.+)]\((https://[^)]+)\) \((\d{4})\)$")
ID = re.compile(r"\*\*Paper ID:\*\* `([^`]+)` · \*\*Track:\*\* `([^`]+)` · \*\*Source evidence:\*\* `([^`]+)`")
UA = "PLLM research bibliography (https://github.com/blairhudson/pllm)"
OPEN_PDFS = {
    "sort-sweep-mirror": ["https://staff.ie.cuhk.edu.hk/~smchow/sp2026c1-final1484.pdf"],
    "game-of-arrows": ["https://www.usenix.org/system/files/usenixsecurity25-wang-pengli.pdf"],
    "matrix-coding-verification": [
        "https://drops.dagstuhl.de/storage/00lipics/lipics-vol317-approx-random2024/"
        "LIPIcs.APPROX-RANDOM.2024.42/LIPIcs.APPROX-RANDOM.2024.42.pdf",
    ],
    "cheetah": ["https://www.usenix.org/system/files/sec22-huang-zhicong.pdf"],
    "iron": [
        "https://proceedings.neurips.cc/paper_files/paper/2022/file/"
        "64e2449d74f84e5b1a5c96ba7b3d308e-Paper-Conference.pdf",
    ],
    "aby2": ["https://www.usenix.org/system/files/sec21-patra.pdf"],
    "cryptonets": [
        "https://www.microsoft.com/en-us/research/wp-content/uploads/2016/04/"
        "CryptonetsTechReport.pdf",
    ],
    # Open full version of the mapped 2016 CCS paper. Publication DOI stays primary.
    "fss-extensions": ["https://eprint.iacr.org/2018/707.pdf"],
}
# Transcribed from the publisher pages when bibliographic APIs did not return
# a sufficiently close title match. MOAI stays without unverified authors.
AUTHOR_OVERRIDES = {
    "game-of-arrows": [
        "Pengli Wang", "Bingyou Dong", "Yifeng Cai", "Zheng Zhang", "Junlin Liu",
        "Huanran Xue", "Ye Wu", "Yao Zhang", "Ziqi Zhang",
    ],
    "cheetah": ["Zhicong Huang", "Wen-jie Lu", "Cheng Hong", "Jiansheng Ding"],
    "matrix-coding-verification": [
        "Huck Bennett", "Karthik Gajulapalli", "Alexander Golovnev", "Evelyn Warton",
    ],
}


def norm(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold()))


def parse_map() -> list[dict]:
    entries: list[dict] = []
    section = ""
    for line in MAP.read_text().splitlines():
        if line.startswith("### `"):
            section = line.removeprefix("### `").removesuffix("`")
        elif match := HEADING.match(line):
            title, url, year = match.groups()
            entries.append({"title": title, "source_url": url, "year": int(year), "owner": section})
        elif entries and (match := ID.search(line)):
            entries[-1].update(zip(("id", "track", "source_evidence"), match.groups(), strict=True))
        elif entries:
            for label, field in (
                ("Suggested contribution", "contribution"),
                ("First experiment", "first_experiment"),
                ("Promotion gate", "promotion_gate"),
                ("Original evidence boundary", "evidence_boundary"),
            ):
                if line.startswith(f"**{label}:** "):
                    entries[-1][field] = line.split("** ", 1)[1]
    if len(entries) < 84 or len({entry.get("id") for entry in entries}) != len(entries):
        raise ValueError("Expected distinct paper placements retaining the original library")
    required = {"id", "track", "source_evidence", "contribution", "first_experiment",
                "promotion_gate", "evidence_boundary"}
    for entry in entries:
        if missing := required - entry.keys():
            raise ValueError(f"Missing fields for {entry['title']}: {sorted(missing)}")
        if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", entry["id"]):
            raise ValueError(f"Invalid paper slug: {entry['id']}")
        if not entry["source_url"].startswith("https://"):
            raise ValueError(f"Paper source must use HTTPS: {entry['id']}")
    return entries


def candidates(url: str) -> list[str]:
    parsed = urlparse(url)
    host = parsed.hostname or ""
    path = parsed.path
    if host == "arxiv.org":
        return [f"https://arxiv.org/pdf/{path.split('/')[-1]}"]
    if host == "eprint.iacr.org":
        return [f"https://eprint.iacr.org{path.rstrip('/')}.pdf"]
    if host == "openreview.net":
        return [url.replace("/forum", "/pdf")]
    if host == "proceedings.mlr.press" and path.endswith(".html"):
        return [url.removesuffix(".html").replace("/v267/", "/v267/") + "/" + path.split("/")[-1].removesuffix(".html") + ".pdf"]
    if host == "link.springer.com" and "/chapter/" in path:
        return ["https://link.springer.com/content/pdf/" + path.split("/chapter/", 1)[1] + ".pdf"]
    if host in {"doi.org", "www.doi.org"} and "10.1007/" in path:
        return ["https://link.springer.com/content/pdf/" + path.lstrip("/") + ".pdf"]
    if path.lower().endswith(".pdf"):
        return [url]
    return []


def metadata(title: str, year: int) -> dict:
    url = "https://api.openalex.org/works?search=" + quote(title) + "&per-page=5"
    try:
        request = Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
        with urlopen(request, timeout=18) as response:
            results = json.load(response)["results"]
    except (OSError, ValueError, KeyError):
        return {}
    scored = sorted(
        ((SequenceMatcher(None, norm(title), norm(hit.get("title") or "")).ratio(), hit)
         for hit in results if abs((hit.get("publication_year") or year) - year) <= 2),
        key=lambda pair: pair[0], reverse=True,
    )
    if not scored or scored[0][0] < 0.82:
        return {}
    hit = scored[0][1]
    links = [loc.get("pdf_url") for loc in (hit.get("locations") or []) if loc.get("pdf_url")]
    authors = [item["author"]["display_name"] for item in hit.get("authorships", [])
               if item.get("author", {}).get("display_name")]
    return {"authors": authors, "metadata_url": hit["id"], "oa_urls": list(dict.fromkeys(links))}


def source_authors(url: str) -> list[str]:
    parsed = urlparse(url)
    if parsed.hostname == "arxiv.org":
        url = "https://arxiv.org/abs/" + parsed.path.split("/")[-1]
    elif parsed.hostname == "eprint.iacr.org":
        url = url.removesuffix(".pdf")
    elif parsed.hostname in {"doi.org", "www.doi.org"}:
        url = "https://api.crossref.org/works/" + quote(parsed.path.lstrip("/"), safe="")
        try:
            request = Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
            with urlopen(request, timeout=18) as response:
                people = json.load(response)["message"].get("author", [])
            return [" ".join(filter(None, (p.get("given"), p.get("family")))) for p in people]
        except (OSError, ValueError, KeyError):
            return []
    else:
        return []
    try:
        request = Request(url, headers={"User-Agent": UA})
        with urlopen(request, timeout=18) as response:
            html = response.read().decode("utf-8")
        return re.findall(r'<meta name="citation_author" content="([^"]+)"', html)
    except (OSError, UnicodeError, ValueError):
        return []


def is_pdf(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size < 2048:
        return False
    with path.open("rb") as source:
        if source.read(5) != b"%PDF-":
            return False
    return subprocess.run(["pdfinfo", str(path)], capture_output=True, check=False).returncode == 0


def title_matches(path: Path, title: str) -> bool:
    result = subprocess.run(["pdftotext", "-f", "1", "-l", "1", str(path), "-"],
                            capture_output=True, text=True, check=False)
    if result.returncode or not result.stdout.strip():
        return True  # Scanned or graphics-only PDFs need separate manual review.
    words = set(norm(title).split()) - {"the", "a", "an", "of", "for", "and", "in", "to", "with"}
    return len(words & set(norm(result.stdout).split())) / max(len(words), 1) >= 0.55


def acquire(entry: dict, old: dict, *, fetch: bool) -> dict:
    saved = {key: old[key] for key in ("file", "pdf_url", "sha256", "bytes", "status",
                                       "authors", "metadata_url", "note", "attempted_pdf_urls") if key in old}
    name = saved.get("file") or (f"papers/{entry['id']}.pdf" if not entry.get("registry_id") else None)
    if name and (ROOT / name).exists() and not is_pdf(ROOT / name):
        raise ValueError(f"Invalid previously downloaded PDF: {entry['id']}")
    if name and is_pdf(ROOT / name):
        binary = ROOT / name
        digest = sha256(binary.read_bytes()).hexdigest()
        if saved.get("sha256") and saved["sha256"] != digest:
            raise ValueError(f"Previously inventoried PDF changed: {entry['id']}")
        if fetch and not saved.get("authors"):
            saved["authors"] = AUTHOR_OVERRIDES.get(entry["id"]) or source_authors(entry["source_url"])
        return {**saved, "file": name, "sha256": digest,
                 "bytes": binary.stat().st_size, "status": "available"}
    if entry["id"] == "ripple":
        return {**saved, "file": None, "status": "unverified_primary",
                "note": "The mapped URL is Curl's PDF, not an identified standalone Ripple paper."}
    if not fetch:
        if saved.get("status") == "available" and saved.get("file") and saved.get("sha256"):
            return saved  # The local cache is optional; the verified fingerprint is not.
        return {**saved, "file": None, "status": saved.get("status", "not_downloaded")}
    info = metadata(entry["title"], entry["year"])
    authors = info.get("authors") or AUTHOR_OVERRIDES.get(entry["id"]) or source_authors(entry["source_url"])
    urls = list(dict.fromkeys(([saved["pdf_url"]] if saved.get("pdf_url") else [])
                              + OPEN_PDFS.get(entry["id"], [])
                              + candidates(entry["source_url"]) + info.get("oa_urls", [])))
    if not urls:
        return {**saved, "authors": authors or [],
                 "metadata_url": info.get("metadata_url"), "file": None, "status": "no_pdf_url"}
    target = ROOT / name if name else PAPERS / f"{entry['id']}.pdf"
    changed_pinned_pdf = False
    with tempfile.TemporaryDirectory() as directory:
        candidate = Path(directory) / "paper.pdf"
        for url in urls[:6]:
            if not url.startswith("https://"):
                continue
            result = subprocess.run(["curl", "--fail", "--location", "--silent", "--show-error",
                                     "--max-time", "35", "--retry", "1", "--user-agent", UA,
                                     "--output", str(candidate), url], capture_output=True, text=True,
                                    check=False)
            if result.returncode or not is_pdf(candidate) or not title_matches(candidate, entry["title"]):
                candidate.unlink(missing_ok=True)
                continue
            digest = sha256(candidate.read_bytes()).hexdigest()
            if saved.get("sha256") and saved["sha256"] != digest:
                changed_pinned_pdf = True
                candidate.unlink(missing_ok=True)
                continue
            candidate.replace(target)
            saved.pop("attempted_pdf_urls", None)
            return {**saved, "authors": authors or [],
                      "metadata_url": info.get("metadata_url"), "file": f"papers/{target.name}",
                     "pdf_url": url, "sha256": digest,
                     "bytes": target.stat().st_size, "status": "available"}
    if saved.get("status") == "available":
        detail = "source bytes changed" if changed_pinned_pdf else "source unavailable"
        raise ValueError(f"Pinned PDF cannot be restored for {entry['id']}: {detail}")
    return {**saved, "authors": authors or [], "metadata_url": info.get("metadata_url"),
            "file": None, "status": "pdf_unavailable", "attempted_pdf_urls": urls[:6]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch", action="store_true", help="Fetch missing PDFs and OA metadata")
    args = parser.parse_args()
    registry = json.loads(RECORDS.read_text())
    known = {paper["slug"].replace("_", "-"): paper for paper in registry["papers"]}
    known["bumblebee"] = known["openbumblebee"]
    previous = {row["id"]: row for row in json.loads(LIBRARY.read_text())["papers"]} if LIBRARY.exists() else {}
    entries = parse_map()
    rows = []
    for entry in entries:
        entry["map_year"] = entry["year"]
        paper = known.get(entry["id"])
        if paper:
            entry["registry_id"] = paper["id"]
            entry["title"] = paper["title"]
            entry["year"] = paper["year"]
            entry["authors"] = paper["authors"]
            entry["source_url"] = paper["primary_url"]
            filename = "bumblebee" if entry["id"] == "bumblebee" else entry["id"]
            entry["file"] = f"papers/{paper['id'].lower()}-{filename}.pdf"
        rows.append(entry)
    if len([row for row in rows if row.get("registry_id")]) != 24:
        raise ValueError("The landing map must cover all 24 curated research records")
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(acquire, row, {**previous.get(row["id"], {}),
                                             **({"file": row["file"]} if "file" in row else {})},
                               fetch=args.fetch): row for row in rows}
        for future in as_completed(futures):
            row = futures[future]
            row.update(future.result())
            print(f"{row['id']}: {row['status']}")
    rows.sort(key=lambda row: (-row["year"], row["title"].casefold()))
    LIBRARY.write_text(json.dumps({"schema_version": "pllm.paper_library.v1", "as_of": "2026-09-23",
                                   "papers": rows}, indent=2, ensure_ascii=False) + "\n")
    print(f"Library: {sum(row['status'] == 'available' for row in rows)}/{len(rows)} PDFs")


if __name__ == "__main__":
    main()
