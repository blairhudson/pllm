"""Capture documentary inputs only; never import, compile or execute upstream code.

Generated records are create-only. Run in a new archive directory for a new review.
Network GETs, local byte reads and the existing pdftotext executable are the only
inputs. This is not the read-only verifier; see check.py for that command.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
COMMIT = "271637be7196d26a5433ca1e62b1a5357e769a5b"
REPO = "https://github.com/kanav99/dvllm"
RAW = f"https://raw.githubusercontent.com/kanav99/dvllm/{COMMIT}/"
API = f"https://api.github.com/repos/kanav99/dvllm/commits/{COMMIT}"
HEADERS = {"User-Agent": "PLLM-documentary-source-review/1", "Accept-Encoding": "identity"}

# Necessary bounded extracts; inclusive upstream line numbers, not a vendored tree.
EXTRACTS = {
    "README.md": [(1, 32)],
    "LICENSE": [(1, 19)],
    "include/constants.hpp": [(1, 8)],
    "include/common.hpp": [(1, 35)],
    "include/client.hpp": [(56, 76), (142, 175), (211, 239), (263, 279),
                           (288, 340), (377, 399), (403, 411), (480, 491),
                           (529, 538), (682, 689)],
    "src/quantize.cpp": [(123, 145)],
    "src/llama3.2q.cpp": [(313, 324), (367, 387), (403, 436), (627, 640),
                         (662, 695)],
    "include/verification.hpp": [(59, 64), (113, 120), (147, 165)],
    "ext/simdcrypt/src/PRNG.cpp": [(7, 13), (43, 55), (68, 75)],
    "include/networking.hpp": [(117, 150), (190, 218), (298, 328)],
    "include/async_verify.hpp": [(20, 47), (60, 81)],
    "CMakeLists.txt": [(1, 30)],
}

# IDs resolve documentary citations; larger locator ranges do not imply exhaustive audit.
UPSTREAM_CITATIONS = {
    "U-readme": ("README.md", 1, 32),
    "U-types": ("include/common.hpp", 1, 35),
    "U-scale": ("include/constants.hpp", 1, 8),
    "U-quant": ("src/quantize.cpp", 123, 145),
    "U-linear": ("include/client.hpp", 211, 279),
    "U-lookup": ("include/client.hpp", 288, 340),
    "U-lookup-abort": ("include/client.hpp", 377, 399),
    "U-state": ("include/client.hpp", 56, 76),
    "U-nonlinear": ("include/client.hpp", 142, 175),
    "U-decoder": ("include/client.hpp", 403, 538),
    "U-head": ("include/client.hpp", 682, 689),
    "U-sampler": ("src/llama3.2q.cpp", 313, 436),
    "U-seeds": ("src/llama3.2q.cpp", 627, 640),
    "U-session": ("src/llama3.2q.cpp", 662, 695),
    "U-prng": ("ext/simdcrypt/src/PRNG.cpp", 43, 75),
    "U-verifier-seed": ("include/verification.hpp", 59, 64),
    "U-verifier-abort": ("include/verification.hpp", 147, 165),
    "U-queue": ("include/async_verify.hpp", 20, 81),
    "U-network": ("include/networking.hpp", 117, 218),
}
LOCAL_CITATIONS = {
    "L-codec": ("crates/pllm-core/src/codec.rs", 142, 174),
    "L-quant": ("python/pllm/runtime/quantization.py", 179, 202),
    "L-boundary-wire": ("python/pllm/runtime/private_boundary_transport.py", 307, 383),
    "L-sampler": ("python/pllm/runtime/transformer_client.py", 1828, 1846),
    "L-boundary-compiler": ("crates/pllm-compiler/src/decoder_boundary.rs", 160, 195),
    "L-offset-compiler": ("crates/pllm-compiler/src/lib.rs", 2123, 2206),
    "L-profiles": ("python/pllm/profiles/__init__.py", 690, 712),
    "L-binding": ("python/pllm/runtime/model_binding.py", 368, 409),
    "L-client": ("python/pllm/runtime/client.py", 1109, 1159),
    "L-semantic": ("python/pllm/runtime/semantic_executor.py", 1131, 1155),
    "L-worker-bundle": ("python/pllm/runtime/offset_worker.py", 157, 163),
    "L-bundle-binding": ("python/pllm/runtime/model_binding.py", 1928, 1960),
    "L-native-pages": ("crates/pllm-python/src/private_pages.rs", 26, 48),
    "L-native-owner": ("crates/pllm-garble/src/private_pages.rs", 413, 485),
    "L-native-matrix": ("python/pllm/runtime/native.py", 72, 178),
    "L-worker-bounds": ("python/pllm/runtime/offset_worker.py", 31, 37),
    "L-network-placement": ("crates/pllm-compiler/src/network_placement.rs", 233, 241),
}
PAGES = {
    6: "Section 3; Section 4 encoding and Section 4.1",
    7: "Figure 1; Section 4.2 linear verification",
    8: "Figure 2; verification completeness and security discussion",
    9: "Section 4.3; Figure 3 dense XOR-PIR",
    10: "Figure 4 embedding verification; Section 4.4",
    11: "Figure 5 complete private/verifiable inference; batching",
    12: "Section 5 implementation, PRG optimization and experimental setup",
    13: "Tables 1-2; per-token costs amortized over 100 tokens and separate 100-token verification",
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def save(name, value):
    with (HERE / name).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=True, allow_nan=False)
        stream.write("\n")


def get(url):
    started = datetime.now(timezone.utc).isoformat()
    with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=60) as response:
        data = response.read()
        provenance = {
            "requested_url": url, "resolved_url": response.url,
            "method": "GET", "status": response.status, "retrieved_utc": started,
            "request_headers": HEADERS,
            "response_headers": {k: response.headers[k] for k in
                                 ("Date", "ETag", "Last-Modified", "Content-Type")
                                 if k in response.headers},
            "bytes": len(data), "sha256": sha(data),
        }
    return data, provenance


def capture_discovery():
    records = []
    for url, fields in (
        ("https://api.github.com/repos/kanav99/dvllm",
         ("full_name", "html_url", "homepage", "description", "created_at", "license")),
        ("https://api.github.com/users/kanav99", ("login", "name", "html_url", "blog")),
    ):
        data, provenance = get(url)
        document = json.loads(data)
        records.append({"retrieval": provenance,
                        "fields": {key: document.get(key) for key in fields}})
    save("repository-record.json", {
        "schema": "pllm.paper_competitor_discovery.v1", "records": records,
        "scope": "Bounded fields from mutable GitHub metadata: author/repository self-identification, not proof of the evaluated build or a complete license/dependency audit",
    })


def main():
    for name in ("code-lock.json", "primary-locators.json", "local-locators.json"):
        if (HERE / name).exists():
            raise SystemExit(f"refusing to overwrite frozen record: {name}")
    old = (ROOT / "paper/prior-art-sources.json").read_bytes()
    primary = json.loads(old)
    if "source_code_reviews" in primary:
        raise SystemExit("capture must precede appending this dated review")
    capture_discovery()
    commit_bytes, commit_receipt = get(API)
    commit = json.loads(commit_bytes)
    assert commit["sha"] == COMMIT
    records = []
    for path, ranges in EXTRACTS.items():
        data, provenance = get(RAW + path)
        lines = data.decode("utf-8").splitlines(keepends=True)
        extracts = []
        for start, end in ranges:
            assert 1 <= start <= end <= len(lines), (path, start, end, len(lines))
            text = "".join(lines[start - 1:end])
            extracts.append({"start_line": start, "end_line": end, "text": text,
                             "sha256": sha(text.encode())})
        blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        records.append({"path": path, "sha256": sha(data), "git_blob_sha1": blob,
                        "bytes": len(data), "line_count": len(lines),
                        "retrieval": provenance, "extracts": extracts})
    save("code-lock.json", {
        "schema": "pllm.paper_competitor_source_lock.v1", "review_date": "2026-10-16",
        "status": "quarantined_specification_oracle_only", "repository": REPO,
        "commit": COMMIT, "tree": commit["commit"]["tree"]["sha"],
        "commit_date": commit["commit"]["committer"]["date"],
        "commit_record_retrieval": commit_receipt,
        "scope": "Selected source files read as bytes/text; no clone, import, linking, build, execution or dependency installation. Identity with authors' evaluated build is unresolved.",
        "files": records,
        "citations": [{"id": key, "path": path, "start_line": start, "end_line": end,
                       "url": f"{REPO}/blob/{COMMIT}/{path}#L{start}-L{end}"}
                      for key, (path, start, end) in UPSTREAM_CITATIONS.items()],
    })
    pdf = next(item for item in primary["local_pdf_sources"] if item["id"] == "open-weight-verifiable")
    pdf_bytes = (ROOT / pdf["cache"]).read_bytes()
    assert sha(pdf_bytes) == pdf["sha256"]
    tool = shutil.which("pdftotext")
    if tool is None:
        raise SystemExit("existing pdftotext required; installation is out of scope")
    version = subprocess.run([tool, "-v"], capture_output=True, check=True)
    pages = []
    for page, locator in PAGES.items():
        args = [tool, "-layout", "-f", str(page), "-l", str(page), pdf["cache"], "-"]
        result = subprocess.run(args, cwd=ROOT, capture_output=True, check=True)
        name = f"gkm-pdf-page-{page}.txt"
        with (HERE / name).open("xb") as stream:
            stream.write(result.stdout)
        pages.append({"page": page, "locator": locator, "path": name,
                      "argv": args, "cwd": "repository root", "exit_code": result.returncode,
                      "stderr": result.stderr.decode(), "sha256": sha(result.stdout)})
    save("primary-locators.json", {
        "schema": "pllm.paper_competitor_primary_locators.v1",
        "captured_utc": datetime.now(timezone.utc).isoformat(), "primary_record": pdf,
        "primary_pdf_bytes": len(pdf_bytes),
        "prior_manifest_raw_sha256_before_append": sha(old),
        "frozen_primary_records_canonical_sha256": sha(canonical(primary)),
        "canonicalization": "json.dumps(sort_keys=True, separators=(',', ':'), ensure_ascii=True); UTF-8; exclude newly appended source_code_reviews",
        "pdf_version": "Existing unversioned cache pinned by SHA-256; no fresh PDF identity asserted",
        "pdftotext": {"path": tool, "sha256": sha(Path(tool).read_bytes()),
                      "version": (version.stdout + version.stderr).decode()},
        "page_numbering": "PDF page selected with -f/-l, not count of form-feed characters",
        "pages": pages,
    })
    files = []
    for path in sorted({v[0] for v in LOCAL_CITATIONS.values()}):
        data = (ROOT / path).read_bytes()
        files.append({"path": path, "sha256": sha(data), "bytes": len(data),
                      "line_count": len(data.decode().splitlines())})
    save("local-locators.json", {
        "schema": "pllm.paper_competitor_local_locators.v1",
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Current working-tree source bytes, read only; not an installed-build or runtime claim",
        "files": files,
        "citations": [{"id": key, "path": path, "start_line": start, "end_line": end,
                       "url": f"../{path}#L{start}-L{end}"}
                      for key, (path, start, end) in LOCAL_CITATIONS.items()],
    })
    print(json.dumps({"files": len(records), "pdf_pages": len(pages),
                      "upstream_execution": False, "output": str(HERE)}))


if __name__ == "__main__":
    if sys.argv[1:] == ["--discovery-only"]:
        capture_discovery()
    elif sys.argv[1:]:
        raise SystemExit("usage: capture.py [--discovery-only]")
    else:
        main()
