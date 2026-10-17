"""Retain evidence while reclaiming stale, reconstructible target caches."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from email.parser import Parser
import fcntl
import hashlib
import json
import os
from pathlib import Path
import plistlib
import re
import shutil
import stat
import subprocess
import sys
import time
import uuid
import zipfile

import psutil

ROOT = Path(__file__).resolve().parents[1]
GIB = 1 << 30
DAY = 86400
PIN = ".pllm-keep"
VENV_ENTRIES = {"bin", "lib", "lib64", "include", "share", "pyvenv.cfg",
                ".gitignore", ".lock", "CACHEDIR.TAG"}


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def regular(path):
    return stat.S_ISREG(path.lstat().st_mode)


def safe_directory(path):
    """Do not traverse symlinks, including intermediate components."""
    path = Path(os.path.abspath(path))
    for component in [*reversed(path.parents), path]:
        if not stat.S_ISDIR(component.lstat().st_mode):
            raise ValueError(f"not a real directory: {component}")
    return path


def tree_info(path):
    """Allocated blocks and last modification; never follow links."""
    allocated, newest, payload_newest, pinned = 0, 0.0, 0.0, False
    seen = set()
    for directory, dirs, files in os.walk(path, followlinks=False):
        base = Path(directory)
        for entry in [base, *(base / name for name in dirs + files)]:
            item = entry.lstat()
            identity = (item.st_dev, item.st_ino)
            if identity in seen:
                continue
            seen.add(identity)
            allocated += item.st_blocks * 512
            newest = max(newest, item.st_mtime, item.st_ctime)
            # Creating our environment lock changes the root timestamp. Keep a
            # separate payload clock so we can recheck after acquiring that lock.
            if entry != path and entry != path / ".lock":
                payload_newest = max(payload_newest, item.st_mtime, item.st_ctime)
            pinned |= entry.name == PIN
    return {"allocated_bytes": allocated, "newest": newest,
            "payload_newest": payload_newest, "pinned": pinned}


def pinned(path, target):
    return any(os.path.lexists(parent / PIN)
               for parent in [path, *path.parents] if parent == target or target in parent.parents)


def discover(target):
    """Only Cargo incremental units and recognisable virtual environments."""
    units, wheels = [], []
    for directory, dirs, files in os.walk(target, followlinks=False):
        base = Path(directory)
        dirs[:] = [name for name in dirs if not (base / name).is_symlink()
                   and name not in {".maintenance", ".traces"}]
        if "pyvenv.cfg" in files:
            if set(dirs + files) <= VENV_ENTRIES | {PIN}:
                units.append((base, "venv", "venv"))
            dirs[:] = []
            continue
        wheels.extend(base / name for name in files
                      if name.startswith("pllm_run-") and name.endswith(".whl")
                      and regular(base / name))
        if base.name == "incremental" and (base.parent / ".cargo-lock").is_file():
            for name in dirs:
                path = base / name
                if re.fullmatch(r"[A-Za-z0-9_]+-[a-z0-9]+", name):
                    children = list(path.iterdir())
                    if all(child.name == PIN or child.name.startswith("s-") for child in children):
                        units.append((path, "incremental", str(base / name.rsplit("-", 1)[0])))
            dirs[:] = []
        elif base.name in {"deps", "build", ".fingerprint"} and (base.parent / ".cargo-lock").is_file():
            dirs[:] = []
    return units, sorted(wheels)


def active_paths():
    """Retain same-user process arguments, cwd and open files; uncertainty blocks."""
    paths, compilers = [], []
    for process in psutil.process_iter():
        try:
            if process.pid == os.getpid() or process.uids().effective != os.getuid():
                continue
            if process.status() == psutil.STATUS_ZOMBIE:
                continue
            command = process.cmdline()
            cwd = Path(process.cwd())
            paths.append(cwd)
            for arg in command:
                if "/" in arg and not arg.startswith("-"):
                    value = Path(arg)
                    paths.append(Path(os.path.abspath(value if value.is_absolute() else cwd / value)))
            try:
                paths.extend(Path(item.path) for item in process.open_files())
            except psutil.AccessDenied:
                # SIP denies fd inspection for OS services such as lsd. Their
                # arguments/cwd are checked above; developer processes still
                # fail closed if their ownership cannot be inspected.
                if not process.exe().startswith(("/usr/libexec/", "/System/Library/", "/usr/sbin/")):
                    raise
            if process.name() in {"cargo", "rustc", "maturin"}:
                compilers.append(process.pid)
        except psutil.NoSuchProcess:
            continue
        except psutil.AccessDenied as exc:
            raise RuntimeError("cannot inspect same-user process ownership") from exc
    return paths, compilers


def owner_ancestors(paths):
    return {parent for item in paths for parent in (item, *item.parents)}


def package_manifest(package):
    result = {}
    for directory, dirs, files in os.walk(package, followlinks=False):
        base = Path(directory)
        if any((base / name).is_symlink() for name in dirs + files):
            raise ValueError("linked package")
        dirs[:] = [name for name in dirs if name != "__pycache__"]
        for name in files:
            path = base / name
            if not regular(path):
                raise ValueError("nonregular package file")
            result["pllm/" + path.relative_to(package).as_posix()] = sha256(path)
    return result


def wheel_manifest(path):
    with zipfile.ZipFile(path) as wheel:
        members = [item for item in wheel.infolist() if item.filename.startswith("pllm/")
                   and not item.is_dir()]
        if len(members) > 4096 or sum(item.file_size for item in members) > 128 << 20:
            raise ValueError("wheel package bound")
        return {item.filename: hashlib.sha256(wheel.read(item)).hexdigest() for item in members}


def environment_receipt(path, wheels, wheel_cache):
    sites = list(path.glob("lib/python*/site-packages"))
    if len(sites) != 1:
        raise ValueError("unknown environment layout")
    site = safe_directory(sites[0])
    package = safe_directory(site / "pllm")
    manifest = package_manifest(package)
    native = [name for name in manifest if "/_native" in name and name.endswith((".so", ".pyd"))]
    if not native:
        raise ValueError("no installed native package")
    for wheel in wheels:
        if wheel not in wheel_cache:
            wheel_cache[wheel] = wheel_manifest(wheel)
        if wheel_cache[wheel] == manifest:
            distributions = []
            for metadata in sorted(site.glob("*.dist-info/METADATA")):
                if not regular(metadata) or metadata.stat().st_size > 4 << 20:
                    raise ValueError("unknown dependency metadata")
                parsed = Parser().parsestr(metadata.read_text())
                if not parsed["Name"] or not parsed["Version"]:
                    raise ValueError("incomplete dependency metadata")
                distributions.append(f'{parsed["Name"]}=={parsed["Version"]}')
            return {"wheel": str(wheel), "wheel_sha256": sha256(wheel),
                    "package_sha256": manifest, "requirements": distributions,
                    "pyvenv_cfg": (path / "pyvenv.cfg").read_text()}
    raise ValueError("no retained byte-matching wheel")


def plan(target, *, budget_bytes=20 * GIB, free_floor_bytes=30 * GIB,
         now=None, ownership=None):
    target = safe_directory(target)
    now = time.time() if now is None else now
    paths, compilers = active_paths() if ownership is None else ownership
    owners = owner_ancestors(paths)
    total = tree_info(target)["allocated_bytes"]
    free = shutil.disk_usage(target).free
    pressure = total > budget_bytes or free < free_floor_bytes
    units, wheels = discover(target)
    rows, cache = [], {}
    for path, kind, group in units:
        info = tree_info(path)
        rows.append({"path": str(path.relative_to(target)), "kind": kind, "group": group,
                     "age_days": max(0, (now - info["newest"]) / DAY), **info,
                     "identity": [path.stat().st_dev, path.stat().st_ino]})
    newest = set()
    groups = {row["group"] for row in rows}
    for group in groups:
        members = sorted((row for row in rows if row["group"] == group
                          and not row["pinned"] and not pinned(target / row["path"], target)),
                         key=lambda row: -row["newest"])
        newest.update(row["path"] for row in members[:3 if group == "venv" else 2])
    selected, retained = [], []
    for row in sorted(rows, key=lambda row: row["newest"]):
        path = target / row["path"]
        minimum = (3 if pressure else 7) if row["kind"] == "incremental" else (7 if pressure else 14)
        reason = None
        if row["pinned"] or pinned(path, target):
            reason = "pinned"
        elif path in owners or (row["kind"] == "incremental" and compilers):
            reason = "active"
        elif row["path"] in newest:
            reason = "newest_versions"
        elif row["age_days"] < minimum:
            reason = "recent"
        elif row["kind"] == "venv":
            try:
                row["recreate"] = environment_receipt(path, wheels, cache)
            except (OSError, ValueError, zipfile.BadZipFile) as exc:
                reason = str(exc)
        if reason:
            retained.append({"path": row["path"], "reason": reason})
        else:
            selected.append(row)
    return {"schema": "pllm.target_retention.v1", "target": str(target), "pressure": pressure,
            "target_allocated_bytes": total, "free_bytes_before": free,
            "soft_budget_bytes": budget_bytes, "free_floor_bytes": free_floor_bytes,
            "selected": selected, "retained": retained,
            "selected_allocated_bytes": sum(row["allocated_bytes"] for row in selected)}


@contextmanager
def locked(path):
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError("nonregular lock")
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        os.close(descriptor)


def write_json(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def remove_tree(target, path, identity):
    """Anchor deletion below target, even if an ancestor is replaced concurrently."""
    if not shutil.rmtree.avoids_symlink_attacks:
        raise RuntimeError("descriptor-relative tree removal unavailable")
    parts = path.relative_to(target).parts
    descriptor = os.open(target, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in parts[:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        current = os.stat(parts[-1], dir_fd=descriptor, follow_symlinks=False)
        if not stat.S_ISDIR(current.st_mode) or [current.st_dev, current.st_ino] != identity:
            raise ValueError("candidate replaced")
        shutil.rmtree(parts[-1], dir_fd=descriptor)
    finally:
        os.close(descriptor)


def apply(report, *, ownership=None):
    target = safe_directory(Path(report["target"]))
    state = target / ".maintenance"
    state.mkdir(exist_ok=True)
    safe_directory(state)
    removed, skipped = [], []
    with locked(state / "cleanup.lock"):
        receipt = state / f"{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}-{uuid.uuid4().hex}.json"
        # Commit recovery details before removing the first byte.
        write_json(receipt, report)
        paths, compilers = active_paths() if ownership is None else ownership
        owners = owner_ancestors(paths)
        for row in report["selected"]:
            path = target / row["path"]
            if not path.is_relative_to(target) or ".." in path.parts or path == target:
                raise ValueError("candidate escapes target")
            try:
                safe_directory(path)
                identity = [path.stat().st_dev, path.stat().st_ino]
                info = tree_info(path)
                if row["kind"] == "venv" and ownership is None:
                    paths, compilers = active_paths()
                    owners = owner_ancestors(paths)
                if identity != row["identity"] or info["newest"] != row["newest"]:
                    raise ValueError("candidate changed")
                if info["pinned"] or pinned(path, target) or path in owners:
                    raise ValueError("candidate pinned or active")
                if row["kind"] == "incremental":
                    if path.parent.name != "incremental" or compilers:
                        raise ValueError("compiler active or invalid cache")
                    with locked(path.parent.parent / ".cargo-lock"):
                        check_locked_tree(path, target, row)
                        remove_tree(target, path, row["identity"])
                elif row["kind"] == "venv":
                    recreate = row["recreate"]
                    if sha256(Path(recreate["wheel"])) != recreate["wheel_sha256"]:
                        raise ValueError("retained wheel changed")
                    # Serialize against uv installations, which use this environment lock.
                    with locked(path / ".lock"):
                        check_locked_tree(path, target, row)
                        remove_tree(target, path, row["identity"])
                else:
                    raise ValueError("unknown cleanup class")
                removed.append({"path": row["path"], "allocated_bytes": row["allocated_bytes"]})
            except (OSError, ValueError, RuntimeError) as exc:
                skipped.append({"path": row["path"], "reason": str(exc)})
        result = {"removed": removed, "skipped": skipped, "plan_receipt": str(receipt),
                  "free_bytes_after": shutil.disk_usage(target).free,
                  "removed_allocated_bytes": sum(row["allocated_bytes"] for row in removed),
                  "scope": "allocated file blocks; APFS clones/hardlinks are not physical savings"}
        write_json(receipt.with_suffix(".result.json"), result)
        return result


def check_locked_tree(path, target, row):
    current = tree_info(path)
    if (current["payload_newest"] != row["payload_newest"] or current["pinned"]
            or pinned(path, target)):
        raise ValueError("candidate changed while acquiring lock")


def install_daily(root=ROOT):
    if sys.platform != "darwin":
        raise ValueError("daily installation currently requires macOS launchd")
    label = "run.pllm.target-maintenance." + hashlib.sha256(str(root).encode()).hexdigest()[:12]
    state = root / "target/.maintenance"
    state.mkdir(parents=True, exist_ok=True)
    safe_directory(state)
    agents = Path.home() / "Library/LaunchAgents"
    agents.mkdir(parents=True, exist_ok=True)
    destination = agents / f"{label}.plist"
    arguments = [str(root / ".venv/bin/python"), "-B", str(root / "scripts/maintain_target.py"), "--apply"]
    definition = {"Label": label, "ProgramArguments": arguments, "WorkingDirectory": str(root),
                  "StartInterval": DAY, "RunAtLoad": True, "ProcessType": "Background",
                  "LowPriorityIO": True, "Nice": 10,
                  "StandardOutPath": str(state / "launchd.stdout.log"),
                  "StandardErrorPath": str(state / "launchd.stderr.log")}
    if os.path.lexists(destination):
        if not regular(destination) or plistlib.loads(destination.read_bytes()).get("ProgramArguments") != arguments:
            raise ValueError("refusing to replace an unrelated launch agent")
        subprocess.run(["/bin/launchctl", "bootout", f"gui/{os.getuid()}/{label}"],
                       capture_output=True, check=False)
    destination.write_bytes(plistlib.dumps(definition))
    subprocess.run(["/bin/launchctl", "bootstrap", f"gui/{os.getuid()}", str(destination)], check=True)
    return {"label": label, "plist": str(destination), "interval_seconds": DAY}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="remove eligible caches; default only reports")
    parser.add_argument("--install-daily", action="store_true", help="install this checkout's macOS daily job")
    parser.add_argument("--budget-gib", type=int, default=20, help="soft target budget; never overrides retention")
    parser.add_argument("--free-gib", type=int, default=30, help="disk-free floor for shorter cache retention")
    args = parser.parse_args()
    if args.budget_gib < 1 or args.free_gib < 1:
        parser.error("budgets must be positive")
    if args.install_daily:
        result = install_daily()
    else:
        report = plan(ROOT / "target", budget_bytes=args.budget_gib * GIB, free_floor_bytes=args.free_gib * GIB)
        if args.apply:
            outcome = apply(report)
            result = {"removed_count": len(outcome["removed"]), "skipped_count": len(outcome["skipped"]),
                      "removed_allocated_bytes": outcome["removed_allocated_bytes"],
                      "free_bytes_before": report["free_bytes_before"],
                      "free_bytes_after": outcome["free_bytes_after"], "receipt": outcome["plan_receipt"]}
        else:
            result = {**report, "selected": [{key: value for key, value in row.items() if key != "recreate"}
                                              for row in report["selected"]]}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
