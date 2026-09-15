"""Safe, explicit CLI target resolution."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import re
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from .errors import LocalIOError, ResolutionError

_DOTTED_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_]*(?:\.[A-Za-z][A-Za-z0-9_]*)*")
_DATA_SUFFIXES = {".json", ".yaml", ".yml"}
_SECRET_KEYS = {
    "api_key",
    "credential",
    "credentials",
    "password",
    "private_key",
    "secret",
    "seed",
    "token",
}
_SECRET_SUFFIXES = tuple(f"_{key}" for key in _SECRET_KEYS)


@dataclass(frozen=True, slots=True)
class ResolvedTarget:
    configuration: Any
    kind: str
    source: str
    python_executed: bool


def _reject_secret_fields(value: object, path: str = "configuration") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).lower().replace("-", "_")
            if normalized in _SECRET_KEYS or normalized.endswith(_SECRET_SUFFIXES):
                raise ResolutionError(
                    "CONFIGURATION_SECRET_FIELD",
                    f"public configuration contains forbidden secret field: {path}.{key}",
                )
            _reject_secret_fields(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_secret_fields(item, f"{path}[{index}]")


def _public_dotted_name(value: str, label: str) -> None:
    if _DOTTED_NAME.fullmatch(value) is None:
        raise ResolutionError("TARGET_SYNTAX", f"invalid {label}: {value!r}")
    if any(part.startswith("_") for part in value.split(".")):
        raise ResolutionError("TARGET_PRIVATE_ATTRIBUTE", f"private {label} is not allowed")


def _confirm_python(
    source: str,
    *,
    no_input: bool,
    trust_python: bool,
    output_format: str,
) -> None:
    if trust_python:
        return
    if no_input or not sys.stdin.isatty() or output_format != "human":
        raise ResolutionError(
            "PYTHON_TRUST_REQUIRED",
            "Python targets require --trust-python when input is disabled or noninteractive",
        )
    message = f"Python target {source!r} will execute local code."
    print(f"warning[PYTHON_CODE_EXECUTION]: {message}", file=sys.stderr)
    print("Continue? [y/N] ", end="", file=sys.stderr, flush=True)
    answer = sys.stdin.readline().strip().lower()
    if answer not in {"y", "yes"}:
        raise ResolutionError("PYTHON_TRUST_DECLINED", "Python target execution was declined")


def _load_file_module(path: Path) -> ModuleType:
    name = f"_pllm_cli_target_{hashlib.sha256(str(path).encode()).hexdigest()[:16]}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ResolutionError("TARGET_IMPORT", "Python target file cannot be imported")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except SystemExit as exc:
        sys.modules.pop(name, None)
        raise ResolutionError("TARGET_EXECUTION", "Python target attempted to exit") from exc
    except Exception as exc:
        sys.modules.pop(name, None)
        raise ResolutionError(
            "TARGET_EXECUTION", f"Python target execution failed ({type(exc).__name__})"
        ) from exc
    return module


def _load_python_target(left: str, *, is_file: bool) -> ModuleType:
    if is_file:
        path = Path(left).expanduser()
        try:
            if not path.is_file():
                raise LocalIOError("TARGET_NOT_FOUND", f"target file does not exist: {path}")
            resolved = path.resolve(strict=True)
        except OSError as exc:
            raise LocalIOError("TARGET_READ", f"cannot access target file: {path}") from exc
        return _load_file_module(resolved)
    _public_dotted_name(left, "module name")
    try:
        return importlib.import_module(left)
    except SystemExit as exc:
        raise ResolutionError("TARGET_EXECUTION", "Python target attempted to exit") from exc
    except ModuleNotFoundError as exc:
        raise ResolutionError("TARGET_IMPORT", f"cannot import Python module: {left}") from exc
    except Exception as exc:
        raise ResolutionError(
            "TARGET_EXECUTION", f"Python target execution failed ({type(exc).__name__})"
        ) from exc


def resolve_target(
    target: str,
    *,
    factory: bool,
    no_input: bool,
    trust_python: bool,
    output_format: str,
) -> ResolvedTarget:
    """Resolve strict data or one explicit public Python object."""
    from pllm.config import ConfigurationError, Experiment, load_configuration

    suffix = Path(target).suffix.lower()
    if ":" not in target and suffix in _DATA_SUFFIXES:
        if factory or trust_python:
            raise ResolutionError(
                "TARGET_OPTION_MISMATCH",
                "--factory and --trust-python apply only to Python targets",
            )
        path = Path(target).expanduser()
        try:
            configuration = load_configuration(path)
        except FileNotFoundError as exc:
            raise LocalIOError("TARGET_NOT_FOUND", f"target file does not exist: {path}") from exc
        except OSError as exc:
            raise LocalIOError("TARGET_READ", f"cannot read target file: {path}") from exc
        except ConfigurationError as exc:
            raise ResolutionError("CONFIGURATION_INVALID", str(exc)) from exc
        _reject_secret_fields(configuration.to_spec())
        return ResolvedTarget(configuration, "declarative", str(path), False)

    if target.count(":") != 1:
        raise ResolutionError(
            "TARGET_SYNTAX",
            "TARGET must be a .json/.yaml file or explicit path.py:object/module:object",
        )
    left, object_path = target.split(":", 1)
    is_file = Path(left).suffix.lower() == ".py"
    if not is_file:
        _public_dotted_name(left, "module name")
    _public_dotted_name(object_path, "object path")

    if is_file:
        path = Path(left).expanduser()
        try:
            exists = path.is_file()
        except OSError as exc:
            raise LocalIOError("TARGET_READ", f"cannot access target file: {path}") from exc
        if not exists:
            raise LocalIOError("TARGET_NOT_FOUND", f"target file does not exist: {path}")
    _confirm_python(
        target,
        no_input=no_input,
        trust_python=trust_python,
        output_format=output_format,
    )
    module = _load_python_target(left, is_file=is_file)
    value: object = module
    try:
        for attribute in object_path.split("."):
            value = getattr(value, attribute)
    except AttributeError as exc:
        raise ResolutionError("TARGET_OBJECT", f"Python target has no object {object_path!r}") from exc

    if factory:
        if not callable(value):
            raise ResolutionError("TARGET_FACTORY", "--factory target is not callable")
        try:
            value = value()
        except SystemExit as exc:
            raise ResolutionError("TARGET_FACTORY", "Python target factory attempted to exit") from exc
        except Exception as exc:
            raise ResolutionError(
                "TARGET_FACTORY", f"Python target factory failed ({type(exc).__name__})"
            ) from exc
    if type(value) is not Experiment:
        raise ResolutionError(
            "TARGET_TYPE", "Python target must resolve to a public pllm.config.Experiment"
        )
    _reject_secret_fields(value.to_spec())
    return ResolvedTarget(value, "python-factory" if factory else "python-object", target, True)
