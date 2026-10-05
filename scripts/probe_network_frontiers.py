#!/usr/bin/env python3
"""Run five synthetic whole-decoder network gates; writes no private data."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import network_frontier_hypotheses as hypotheses


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = hypotheses.report()
    result["created_at"] = datetime.now(timezone.utc).isoformat()
    result["code_sha256"] = hashlib.sha256(Path(hypotheses.__file__).read_bytes()).hexdigest()
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2))
