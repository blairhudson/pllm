"""Replay the retained ordinary-SDK validation driver for this installed wheel."""
import hashlib
import json
from pathlib import Path

base = Path("paper/artifacts/puma-sdk/local-v10/validation/run_validation.py")
source = base.read_bytes()
out = Path(__file__).resolve().parent
(out / "base-harness.json").write_text(json.dumps({
    "path": str(base), "sha256": hashlib.sha256(source).hexdigest(),
    "replacement": {"puma-attention-range-build-v1": "puma-guarded-activation-build-v2"},
}, indent=2) + "\n")
text = source.decode().replace("puma-attention-range-build-v1", "puma-guarded-activation-build-v2")
exec(compile(text, str(base), "exec"), {"__file__": __file__})
