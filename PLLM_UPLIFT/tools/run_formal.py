#!/usr/bin/env python3
import json
import sys
from pathlib import Path
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/"python"))
from pllm_uplift.formal import run_formal
result=run_formal(root)
print(json.dumps(result,indent=2))
raise SystemExit(0 if all(c["expectation_met"] for c in result["checks"]) else 1)
