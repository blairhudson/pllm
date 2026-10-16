"""Research-only Python launch isolation and per-process installed-byte receipts.

No runtime/package files are changed. The controller and every captured Python
child install a process-local Popen wrapper. An explicit separate reference
interpreter remains outside PLLM inspection. This is local provenance, not a
hermetic build or remote attestation.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
from pathlib import Path
import runpy
import shutil
import site
import subprocess
import sys
import uuid


_POPEN = subprocess.Popen
_ACTIVE = None
PREFIXES = ("PYTHON", "DYLD_", "LD_")


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def clean_environment(environment):
    return {key: value for key, value in environment.items() if not key.startswith(PREFIXES)}


def interpreter(path):
    value = os.fspath(path)
    if not os.path.dirname(value):
        value = shutil.which(value) or value
    # Preserve venv entry-point identity: resolving its symlink selects base Python.
    return os.path.abspath(value)


def parse_command(argv):
    flags, rest = [], list(map(os.fspath, argv[1:]))
    while rest and rest[0].startswith("-") and rest[0] not in ("-m", "-c"):
        option = rest.pop(0)
        if option in ("-I", "-E", "-s", "-P"):
            continue
        if option in ("-u", "-B", "-O", "-OO", "-q"):
            flags.append(option)
        elif option in ("-W", "-X") and rest:
            flags.extend((option, rest.pop(0)))
        elif option.startswith(("-W", "-X")):
            flags.append(option)
        else:
            raise RuntimeError(f"uncaptured Python invocation option: {option}")
    if not rest:
        raise RuntimeError("interactive Python is outside the bounded audit launcher")
    if rest[0] in ("-m", "-c"):
        if len(rest) < 2:
            raise RuntimeError("Python entry point missing")
        mode, entry, arguments = rest[0][1:], rest[1], rest[2:]
    else:
        mode, entry, arguments = "file", rest[0], rest[1:]
        if not Path(entry).is_file():
            raise RuntimeError("audit launcher supports Python source files, modules or -c only")
        entry = os.path.abspath(entry)
    public = {"mode": mode, "entry": entry if mode != "c" else "<inline-code>"}
    role = None
    if mode == "m" and entry == "pllm" and len(arguments) > 1 and arguments[0] == "serve":
        role = arguments[1]
    elif mode == "m" and entry in ("pllm.runtime.party", "pllm.runtime.private_boundary_role"):
        if "--role" in arguments:
            role = arguments[arguments.index("--role") + 1]
    if role in {"inference", "preparation", "worker_a", "worker_b", "boundary_a", "boundary_b"}:
        public["role"] = role
    public["arguments_sha256"] = hashlib.sha256(json.dumps(arguments).encode()).hexdigest()
    if mode == "c":
        public["code_sha256"] = hashlib.sha256(entry.encode()).hexdigest()
    return flags, mode, entry, arguments, public


def identity(expected):
    errors = []
    import pllm
    native = importlib.import_module("pllm._native")
    package = Path(pllm.__file__).resolve().parent
    wanted = Path(expected["package_path"]).resolve()
    package_files = {}
    for relative, digest in expected["wheel_package_files"].items():
        path = package.parent / relative
        observed = sha(path) if path.is_file() else None
        package_files[relative] = observed
        if observed != digest:
            errors.append(f"package file differs: {relative}")
    extra = {"pllm/" + path.relative_to(package).as_posix(): sha(path)
             for path in package.rglob("*") if path.is_file() and "__pycache__" not in path.parts
             and path.suffix not in {".pyc", ".pyo"}
             and "pllm/" + path.relative_to(package).as_posix() not in expected["wheel_package_files"]}
    if extra:
        errors.append("uninspected extra package files")
    native_path = Path(native.__file__).resolve()
    native_sha = sha(native_path)
    if package != wanted:
        errors.append("package origin differs from inspected installation")
    if native_sha != expected["native_sha256"]:
        errors.append("loaded native binary differs")
    if interpreter(sys.executable) != interpreter(expected["python"]):
        errors.append("interpreter entry point differs")
    flags = {name: getattr(sys.flags, name) for name in ("isolated", "ignore_environment", "no_user_site", "safe_path")}
    if not all(flags.values()) or site.ENABLE_USER_SITE:
        errors.append("isolated interpreter flags/user-site policy missing")
    remaining = sorted(key for key in os.environ if key.startswith(PREFIXES))
    if remaining:
        errors.append("import/loader environment remains")
    origins = {}
    for name, module in sorted(sys.modules.items()):
        if name != "pllm" and not name.startswith("pllm."):
            continue
        origin = getattr(module, "__file__", None)
        if not origin:
            continue
        path = Path(origin).resolve()
        origins[name] = str(path)
        if not path.is_relative_to(wanted):
            errors.append(f"loaded module outside installed package: {name}")
    return {"schema": "pllm.paper_python_identity.v1", "pid": os.getpid(), "parent_pid": os.getppid(),
            "python": sys.executable, "python_binary_sha256": sha(sys.executable),
            "prefix": sys.prefix, "base_prefix": sys.base_prefix, "flags": flags,
            "user_site_enabled": site.ENABLE_USER_SITE, "sys_path": list(sys.path), "cwd": os.getcwd(),
            "remaining_import_environment_keys": remaining, "package_path": str(package),
            "native_path": str(native_path), "native_sha256": native_sha,
            "wheel_sha256": expected["wheel_sha256"], "wheel_package_files": package_files,
            "extra_package_files": extra,
            "loaded_pllm_module_origins": origins, "errors": errors,
            "status": "passed" if not errors else "failed"}


def install(directory, expected_path, helper):
    global _ACTIVE
    if _ACTIVE is not None:
        raise RuntimeError("Python isolation wrapper already active")
    _ACTIVE = (Path(directory), Path(expected_path), Path(helper))
    for key in tuple(os.environ):
        if key.startswith(PREFIXES):
            del os.environ[key]

    class CapturedPopen(_POPEN):
        def __init__(self, args, *positional, **kwargs):
            if not isinstance(args, (list, tuple)) or not args:
                return super().__init__(args, *positional, **kwargs)
            if interpreter(args[0]) != interpreter(sys.executable):
                return super().__init__(args, *positional, **kwargs)
            if positional or kwargs.get("shell") or kwargs.get("executable"):
                raise RuntimeError("ambiguous Python subprocess launch rejected by audit")
            directory, expected_path, helper = _ACTIVE
            flags, mode, entry, arguments, public = parse_command(args)
            env = dict(kwargs.get("env") if kwargs.get("env") is not None else os.environ)
            removed = sorted(key for key in env if key.startswith(PREFIXES))
            kwargs["env"] = clean_environment(env)
            # A deliberately separate, no-PLLM upstream reference must not import
            # PLLM merely to measure its own provenance. Its driver checks that.
            if mode == "file" and Path(entry).name == "audit_paper_conformance.py" and "--reference-model" in arguments:
                return super().__init__([args[0], "-I", *flags, entry, *arguments], **kwargs)
            key = uuid.uuid4().hex
            envelope = {"directory": str(directory), "expected_path": str(expected_path), "helper": str(helper),
                        "key": key, "mode": mode, "entry": entry, "arguments": arguments}
            code = f"import runpy; runpy.run_path({str(helper)!r})['run_child']({envelope!r})"
            save(directory / f"{key}.launch.json", {
                "schema": "pllm.paper_python_launch.v1", "launcher_pid": os.getpid(), "entry": public,
                "expected_sha256": sha(expected_path), "helper_sha256": sha(helper),
                "removed_environment_keys": removed, "isolated_flag_enforced": True,
                "environment_values_recorded": False,
            })
            super().__init__([args[0], "-I", *flags, "-c", code], **kwargs)
            save(directory / f"{key}.spawn.json", {"pid": self.pid, "launcher_pid": os.getpid()})

    subprocess.Popen = CapturedPopen


def activate(directory, provenance_path):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    provenance = json.loads(Path(provenance_path).read_text())
    expected = {key: provenance[key] for key in ("python", "package_path", "wheel_sha256", "native_sha256", "wheel_package_files")}
    expected_path = directory / "expected.json"
    save(expected_path, expected)
    helper = directory / "python-child-isolation.py"
    helper.write_bytes(Path(__file__).read_bytes())
    removed = sorted(key for key in os.environ if key.startswith(PREFIXES))
    save(directory / "policy.json", {
        "schema": "pllm.paper_python_isolation.v1", "helper_sha256": sha(helper),
        "expected_sha256": sha(expected_path), "removed_parent_environment_keys": removed,
        "removed_prefixes": list(PREFIXES), "child_flags": ["-I"],
        "capture": "same-interpreter Python subprocesses and nested launches; pre-entry and end receipts",
        "separate_reference": "explicit separate interpreter remains independent; reference driver records no PLLM imports",
        "scope": "research-only process-local launch wrapper; no runtime edits; not hermetic/CI attestation",
    })
    install(directory, expected_path, helper)
    record = identity(expected)
    save(directory / "controller-start.json", record)
    if record["status"] != "passed":
        raise RuntimeError("controller installed identity/isolation mismatch")
    return {"directory": str(directory), "expected_sha256": sha(expected_path)}


def run_child(envelope):
    directory = Path(envelope["directory"])
    expected = json.loads(Path(envelope["expected_path"]).read_text())
    install(directory, envelope["expected_path"], envelope["helper"])
    start = identity(expected)
    save(directory / f"{envelope['key']}.start.json", start)
    if start["status"] != "passed":
        raise RuntimeError("child installed identity mismatch before entry point")
    error_type, exit_code = None, 0
    try:
        entry, arguments = envelope["entry"], envelope["arguments"]
        if envelope["mode"] == "m":
            sys.argv = [entry, *arguments]
            runpy.run_module(entry, run_name="__main__", alter_sys=True)
        elif envelope["mode"] == "file":
            sys.argv = [entry, *arguments]
            runpy.run_path(entry, run_name="__main__")
        else:
            sys.argv = ["-c", *arguments]
            exec(compile(entry, "<isolated-audit-child>", "exec"), {"__name__": "__main__", "__package__": None})
    except BaseException as exc:
        error_type = type(exc).__name__
        exit_code = exc.code if isinstance(exc, SystemExit) and isinstance(exc.code, int) else (0 if isinstance(exc, SystemExit) and exc.code is None else 1)
        raise
    finally:
        record = identity(expected)
        record.update(entry_exit_code=exit_code, entry_exception_type=error_type)
        save(directory / f"{envelope['key']}.end.json", record)


def summarize(directory, *, require_roles=()):
    directory = Path(directory)
    expected = json.loads((directory / "expected.json").read_text())
    failures, entries, roles = [], [], {}
    for path in sorted(directory.glob("*.launch.json")):
        key = path.name.removesuffix(".launch.json")
        launch = json.loads(path.read_text())
        observed = {}
        for phase in ("spawn", "start", "end"):
            target = directory / f"{key}.{phase}.json"
            if not target.is_file():
                failures.append(f"missing {key}.{phase} receipt")
                continue
            observed[phase] = json.loads(target.read_text())
        if len(observed) == 3:
            if len({observed[phase]["pid"] for phase in observed}) != 1:
                failures.append(f"process ID mismatch: {key}")
            for phase in ("start", "end"):
                record = observed[phase]
                if record["status"] != "passed" or record["native_sha256"] != expected["native_sha256"] or record["wheel_package_files"] != expected["wheel_package_files"]:
                    failures.append(f"identity mismatch: {key}.{phase}")
        role = launch["entry"].get("role")
        if role:
            roles[role] = roles.get(role, 0) + 1
        entries.append({"key": key, "entry": launch["entry"], "pid": observed.get("spawn", {}).get("pid")})
    failures.extend(f"required provider role absent: {role}" for role in require_roles if not roles.get(role))
    return {"schema": "pllm.paper_python_identity_summary.v1", "status": "failed" if failures else "passed",
            "captured_python_processes": len(entries), "roles": roles, "entries": entries,
            "failures": failures, "expected_sha256": sha(directory / "expected.json")}


def finish(directory, *, require_roles=()):
    directory = Path(directory)
    result = summarize(directory, require_roles=require_roles)
    expected = json.loads((directory / "expected.json").read_text())
    record = identity(expected)
    save(directory / "controller-end.json", record)
    if record["status"] != "passed":
        result["status"] = "failed"
        result["failures"].append("controller final identity differs")
    save(directory / "summary.json", result)
    return result
