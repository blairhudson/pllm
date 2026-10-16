"""Read-only source/hash/citation check; no upstream imports or runtime execution."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
from urllib.parse import urlparse, unquote
import urllib.request


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
COMMIT = "271637be7196d26a5433ca1e62b1a5357e769a5b"


def require(condition, message):
    if not condition:
        raise SystemExit(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def unique(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def parse(data):
    def invalid(value):
        raise ValueError(f"non-finite JSON value: {value}")
    return json.loads(data, object_pairs_hook=unique, parse_constant=invalid)


def load(path):
    return parse(path.read_bytes())


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def confined(base, name):
    result = (base / name).resolve()
    require(result.is_relative_to(base.resolve()), f"path escapes archive/root: {name}")
    return result


def get(url):
    request = urllib.request.Request(url, headers={
        "User-Agent": "PLLM-documentary-source-check/1", "Accept-Encoding": "identity"})
    with urllib.request.urlopen(request, timeout=60) as response:
        require(response.status == 200, f"HTTP failure: {url}")
        return response.read()


def check_links(path):
    text = path.read_text()
    definitions = dict(re.findall(r"^\[([^\]]+)\]:\s+(\S+)", text, re.M))
    used = re.findall(r"\[[^\]\n]+\]\[([^\]\n]+)\]", text)
    used += re.findall(r"(?<!\])\[(P\d+)\](?![:(])", text)
    for key in used:
        require(key in definitions, f"undefined citation {key} in {path.name}")
    targets = list(definitions.values()) + re.findall(r"\]\(([^\s)]+)\)", text)
    for target in targets:
        url = urlparse(target)
        if url.scheme in {"http", "https"}:
            require(bool(url.netloc), f"invalid external URL: {target}")
        elif url.path:
            dest = (path.parent / unquote(url.path)).resolve()
            require(dest.is_relative_to(ROOT), f"link escapes repository: {target}")
            require(dest.exists(), f"missing local citation target: {target}")
    return text


def main(online, pdf_text):
    records = {path.name: load(path) for path in sorted(HERE.glob("*.json"))}
    receipt = records["receipt.json"]
    expected = set(receipt["artifacts"]) | {"receipt.json"}
    actual = {p.name for p in HERE.iterdir() if p.is_file()}
    require(actual == expected, "archive file inventory differs from receipt")
    for name, digest in receipt["artifacts"].items():
        require(sha(confined(HERE, name).read_bytes()) == digest, f"archive hash mismatch: {name}")
    for name, digest in receipt["documents"].items():
        require(sha(confined(ROOT, name).read_bytes()) == digest, f"document changed: {name}")

    manifest = load(ROOT / "paper/prior-art-sources.json")
    additions = manifest.pop("source_code_reviews")
    primary = records["primary-locators.json"]
    require(sha(canonical(manifest)) == primary["frozen_primary_records_canonical_sha256"],
            "frozen primary records changed")
    addition = next(item for item in additions if item["id"] == "2026-10-16-gkm-qualification-v1")
    require(addition["commit"] == COMMIT, "appended commit mismatch")
    for name, digest in addition["records"].items():
        require(sha(confined(ROOT, name).read_bytes()) == digest, f"appended record mismatch: {name}")
    pdf = primary["primary_record"]
    require(pdf in manifest["local_pdf_sources"], "frozen PDF identity mismatch")
    require(sha(confined(ROOT, pdf["cache"]).read_bytes()) == pdf["sha256"], "primary PDF changed")
    require(primary["page_numbering"].startswith("PDF page selected"), "unknown page numbering")
    for page in primary["pages"]:
        require(sha(confined(HERE, page["path"]).read_bytes()) == page["sha256"],
                f"page extract changed: {page['page']}")
    if pdf_text:
        tool = shutil.which("pdftotext")
        require(tool is not None, "pdftotext missing; no installation attempted")
        require(sha(Path(tool).read_bytes()) == primary["pdftotext"]["sha256"],
                "pdftotext binary differs; preserve old extract and review new tool separately")
        for page in primary["pages"]:
            argv = [tool, "-layout", "-f", str(page["page"]), "-l", str(page["page"]), pdf["cache"], "-"]
            result = subprocess.run(argv, cwd=ROOT, capture_output=True, check=True)
            require(sha(result.stdout) == page["sha256"], f"PDF text replay mismatch: {page['page']}")

    code = records["code-lock.json"]
    require(code["commit"] == COMMIT, "code commit mismatch")
    require(code["status"] == "quarantined_specification_oracle_only", "quarantine label changed")
    upstream = {item["path"]: item for item in code["files"]}
    for path, item in upstream.items():
        expected_url = f"https://raw.githubusercontent.com/kanav99/dvllm/{COMMIT}/{path}"
        require(item["retrieval"]["requested_url"] == expected_url, f"unpinned URL: {path}")
        require(item["retrieval"]["sha256"] == item["sha256"], f"retrieval digest differs: {path}")
        for extract in item["extracts"]:
            start, end = extract["start_line"], extract["end_line"]
            require(1 <= start <= end <= item["line_count"], f"extract out of range: {path}")
            require(len(extract["text"].splitlines()) == end - start + 1, f"extract line count: {path}")
            require(sha(extract["text"].encode()) == extract["sha256"], f"extract hash: {path}")
        if online:
            data = get(expected_url)
            require(sha(data) == item["sha256"] and len(data) == item["bytes"], f"upstream bytes changed: {path}")
            blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
            require(blob == item["git_blob_sha1"], f"git blob mismatch: {path}")
            lines = data.decode().splitlines(keepends=True)
            require(len(lines) == item["line_count"], f"upstream line count: {path}")
            for extract in item["extracts"]:
                text = "".join(lines[extract["start_line"] - 1:extract["end_line"]])
                require(text == extract["text"], f"upstream extract mismatch: {path}")
    if online:
        commit = parse(get(f"https://api.github.com/repos/kanav99/dvllm/commits/{COMMIT}"))
        require(commit["sha"] == COMMIT and commit["commit"]["tree"]["sha"] == code["tree"],
                "upstream commit/tree mismatch")
        # Mutable API response bytes/headers need not equal the historical receipt.

    local = records["local-locators.json"]
    local_files = {item["path"]: item for item in local["files"]}
    for path, item in local_files.items():
        data = confined(ROOT, path).read_bytes()
        require(sha(data) == item["sha256"], f"local source changed; requalify citation: {path}")
        require(len(data.decode().splitlines()) == item["line_count"], f"local line count: {path}")
    allowed = set()
    for kind, document, files in (("U", code, upstream), ("L", local, local_files)):
        for cite in document["citations"]:
            path, start, end = cite["path"], cite["start_line"], cite["end_line"]
            require(path in files and 1 <= start <= end <= files[path]["line_count"],
                    f"citation range: {cite['id']}")
            expected_url = (f"{code['repository']}/blob/{COMMIT}/{path}" if kind == "U" else f"../{path}")
            require(cite["url"] == f"{expected_url}#L{start}-L{end}", f"citation URL: {cite['id']}")
            allowed.add((kind, path, start, end))
    text = check_links(ROOT / "paper/COMPETITOR-QUALIFICATION.md")
    citations = re.findall(r"\b([UL]) ([\w./-]+):(\d+)-(\d+)", text)
    require(bool(citations), "no code citations found")
    for kind, path, start, end in citations:
        require((kind, path, int(start), int(end)) in allowed, f"unregistered citation: {kind} {path}:{start}-{end}")
    check_links(ROOT / "paper/PRIOR-ART-REVIEW.md")
    check_links(HERE / "README.md")
    print(json.dumps({
        "status": "passed", "json_documents": len(records) + 1,
        "upstream_files": len(upstream), "local_source_files": len(local_files),
        "code_citations_checked": len(citations), "pdf_pages": len(primary["pages"]),
        "frozen_primary_records_unchanged": True,
        "full_upstream_bytes_retrieved_and_checked": online,
        "pdf_text_replayed": pdf_text, "upstream_code_executed": False,
        "scope": "Documentary integrity, locator bounds and links; not semantic proof, runtime conformance or security/performance qualification",
    }, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--online", action="store_true", help="GET pinned source bytes in memory and compare full-file hashes/extracts")
    parser.add_argument("--pdf-text", action="store_true", help="re-extract existing primary pages to stdout with the pinned pdftotext")
    args = parser.parse_args()
    main(args.online, args.pdf_text)
