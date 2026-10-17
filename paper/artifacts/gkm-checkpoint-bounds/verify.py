"""Read-only byte-integrity check of the interrupted GKM source-scan archive.

This verifier never imports archived Python, opens model-cache files, downloads,
builds, or runs the scanner. Hash consistency is not source-numeric eligibility.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat

ARCHIVE = Path(__file__).resolve().parent / "2026-10-17-interrupted-v1-v2"
MAX_FILE_BYTES = 2 << 20


def read_bounded(path):
    fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NONBLOCK | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or not 0 <= info.st_size <= MAX_FILE_BYTES:
            raise ValueError("archive member is not a bounded regular file")
        raw = stream.read(MAX_FILE_BYTES + 1)
    if len(raw) > MAX_FILE_BYTES:
        raise ValueError("archive member exceeds verification bound")
    return raw


def verify(root=ARCHIVE):
    root = Path(root)
    index_raw = read_bounded(root / "index.json")
    index = json.loads(index_raw)
    if index.get("schema") != "pllm.gkm_checkpoint_negative_archive.v1":
        raise ValueError("unknown negative-evidence archive schema")
    rows = index["files"]
    if not 1 <= len(rows) <= 16:
        raise ValueError("invalid archive member count")
    declared = {row["path"] for row in rows}
    if len(declared) != len(rows):
        raise ValueError("duplicate archive member")
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    if actual != declared | {"index.json"}:
        raise ValueError("unindexed or missing archive member")
    payloads, total = {}, 0
    for row in rows:
        relative = Path(row["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("archive path escape")
        path = root / relative
        if any(part.is_symlink() for part in (path, *path.parents) if part != root.parent):
            raise ValueError("archive symlinks are not permitted")
        before = path.stat()
        raw = read_bounded(path)
        after = path.stat()
        if (before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
                after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            raise ValueError("archive member changed while verifying")
        if len(raw) != row["bytes"] or hashlib.sha256(raw).hexdigest() != row["sha256"]:
            raise ValueError(f"archive byte hash mismatch: {row['path']}")
        payloads[row["path"]] = raw
        total += len(raw)
    reports = [json.loads(payloads[f"receipts/{attempt}/report.json"]) for attempt in ("v1", "v2")]
    for report in reports:
        if (report["status"] != "failed" or report["source_numeric_admission"] != "blocked"
                or report["complete_gkm_admission"] != "blocked"
                or report["failure"]["reason"] != "new_swapout" or report["tensors"]
                or report.get("immutable_inputs_verified") is True):
            raise ValueError("historical failure was promoted or altered")
        for original, identity in report["receipt_code_hashes"].items():
            relative = next((key for key in payloads if key.startswith("historical/")
                             and original.endswith("/" + key.removeprefix("historical/"))), None)
            if relative is None or hashlib.sha256(payloads[relative]).hexdigest() != identity["sha256"]:
                raise ValueError("historical scanner/oracle/test identity mismatch")
    admission = json.loads(payloads["receipts/v2/admission.json"])
    if admission["status"] != "admitted_before_value_reads":
        raise ValueError("historical admission status changed")
    return {"schema": index["schema"], "verified_files": len(rows), "verified_bytes": total,
            "index_sha256": hashlib.sha256(index_raw).hexdigest(), "attempt_statuses": ["failed", "failed"],
            "source_numeric_eligibility": "not_established",
            "scope": "archived byte consistency only; historical code was not imported or executed"}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
