"""Record this interpreter's complete installed PLLM package against a wheel."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import pllm
from pllm import _native
from pllm.runtime import delegated_decoder


def check(wheel):
    package = Path(pllm.__file__).resolve().parent
    if not package.is_relative_to(Path(sys.prefix).resolve()):
        raise ValueError("PLLM is not installed inside the selected environment")
    files = {}
    with zipfile.ZipFile(wheel) as packed:
        members = {n for n in packed.namelist() if n.startswith("pllm/") and not n.endswith("/")}
        actual = {
            "pllm/" + p.relative_to(package).as_posix()
            for p in package.rglob("*")
            if p.is_file() and "__pycache__" not in p.parts and p.suffix not in {".pyc", ".pyo"}
        }
        if members != actual:
            raise ValueError("installed package file set differs from wheel")
        for name in sorted(members):
            path = package.parent / name
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            with packed.open(name) as stream:
                original = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest != original or path.stat().st_size != packed.getinfo(name).file_size:
                raise ValueError(f"installed package differs from wheel: {name}")
            files[name] = {"bytes": path.stat().st_size, "sha256": digest}
    with wheel.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {
        "executable": sys.executable,
        "python": sys.version,
        "package": str(package),
        "native": str(Path(_native.__file__).resolve()),
        "decoder": str(Path(delegated_decoder.__file__).resolve()),
        "wheel_sha256": digest,
        "files": files,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = check(args.wheel)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(f"Verified {len(result['files'])} installed package files.")
