"""Two sequential Cargo compilations under the outer aggregate resource guard."""
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
PREFIX = ["rustup", "run", "nightly-2026-05-14"]

if __name__ == "__main__":
    subprocess.run([*PREFIX, "rustc", "--version", "--verbose"], cwd=HERE, check=True)
    common = ["--release", "--locked", "--target", "aarch64-apple-darwin",
              "--jobs", "2", "--manifest-path", str(HERE / "Cargo.toml")]
    for args in (["build", "--bins"], ["test", "--no-run"]):
        command = [*PREFIX, "cargo", *args, *common]
        print(json.dumps({"build_only_command": command}), flush=True)
        subprocess.run(command, cwd=HERE, check=True)
