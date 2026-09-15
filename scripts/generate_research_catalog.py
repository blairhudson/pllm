#!/usr/bin/env python3
"""Generate or verify wheel-safe canonical research metadata."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from pllm.research import _checkout_payload  # noqa: E402
from pllm.research._validation import CATALOG_VERSION, payload_digest  # noqa: E402

OUTPUT = ROOT / "python/pllm/research/_catalog.json"
OBSOLETE_OUTPUT = ROOT / "python/pllm/_cli/_research_catalog.py"


def generated_bytes() -> bytes:
    payload = _checkout_payload(ROOT)
    document = {
        "_generated": {
            "catalog_version": CATALOG_VERSION,
            "generator": "scripts/generate_research_catalog.py",
            "notice": "Generated file; do not edit.",
            "payload_sha256": payload_digest(payload),
        },
        "payload": payload,
    }
    return (json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if generated catalog is stale")
    args = parser.parse_args()
    expected = generated_bytes()
    if args.check:
        if OBSOLETE_OUTPUT.exists() or not OUTPUT.is_file() or OUTPUT.read_bytes() != expected:
            print(
                "research catalog is stale; run scripts/generate_research_catalog.py",
                file=sys.stderr,
            )
            return 1
        print("research catalog is current")
        return 0
    OUTPUT.write_bytes(expected)
    OBSOLETE_OUTPUT.unlink(missing_ok=True)
    print(f"wrote {OUTPUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
