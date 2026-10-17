"""Record the GKM capacity checks against one frozen native input set."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent / (sys.argv[1] if len(sys.argv) > 1 else "native")
OUT.mkdir(exist_ok=False)


def write(name, value):
    with (OUT / name).open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def snapshot():
    paths = sorted(path for path in (ROOT / "crates").rglob("*")
                   if path.is_file() and path.suffix in {".rs", ".toml"}
                   and "target" not in path.relative_to(ROOT).parts)
    paths += [ROOT / name for name in ("Cargo.toml", "Cargo.lock", "rust-toolchain.toml")]
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths}


commands = json.loads((ROOT / "paper/artifacts/gkm-sdk/source-stream-v1/native/commands.json").read_text())
commands["compiler-tests"][commands["compiler-tests"].index("--"):commands["compiler-tests"].index("--")] = ["--test", "delegated_capacity"]
commands["wire-tests"][commands["wire-tests"].index("--"):commands["wire-tests"].index("--")] = ["--test", "gkm_capacity"]
write("runner.json", {"argv": sys.argv, "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
(OUT / "runner.py").write_bytes(Path(__file__).read_bytes())
write("commands.json", commands)
before = snapshot()
write("sources-before.json", before)
env = dict(os.environ, CARGO_BUILD_JOBS="2", CARGO_NET_OFFLINE="true", CARGO_TERM_COLOR="never")
codes = {}
for name, argv in commands.items():
    with (OUT / (name + ".log")).open("x") as handle:
        codes[name] = subprocess.run(argv, cwd=ROOT, env=env, stdout=handle,
                                     stderr=subprocess.STDOUT, check=False).returncode
    print(name, codes[name], flush=True)
    if codes[name]:
        break
after = snapshot()
write("sources-after.json", after)
receipt = {"exit_codes": codes, "source_changed": before != after}
write("receipt.json", receipt)
print(json.dumps(receipt, indent=2))
raise SystemExit(int(before != after or len(codes) != len(commands) or any(codes.values())))
