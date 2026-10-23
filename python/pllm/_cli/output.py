"""Canonical CLI result and diagnostic rendering."""

from __future__ import annotations

import json
import sys
from collections.abc import Iterable, Mapping
from typing import Any

from .errors import CLIError

RESULT_SCHEMA = "pllm.cli.result.v1"
ERROR_SCHEMA = "pllm.cli.error.v1"


def _json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def result_envelope(command: str, data: Mapping[str, Any]) -> dict[str, object]:
    return {"schema_version": RESULT_SCHEMA, "command": command, "data": dict(data)}


def emit_machine(
    command: str,
    data: Mapping[str, Any],
    output_format: str,
    *,
    items: Iterable[Mapping[str, Any]] | None = None,
) -> None:
    if output_format == "jsonl" and items is not None:
        shared = {key: value for key, value in data.items() if key != "items"}
        for item in items:
            print(_json(result_envelope(command, {**shared, "item": dict(item)})))
        return
    payload = result_envelope(command, data)
    print(
        _json(payload)
        if output_format == "jsonl"
        else json.dumps(payload, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True)
    )


def _quality_note_uint(text: str, maximum: int = (1 << 63) - 1) -> bool:
    return (0 < len(text) <= 19 and text.isdecimal()
            and (len(text) == 1 or text[0] != "0") and int(text) <= maximum)


def _quality_memory_note(note: object) -> bool:
    """Validate complete bounded public grammar, not merely a reserved prefix."""
    if type(note) is not str or not 0 < len(note) <= 256 or not note.isascii():
        return False
    admission = "quality memory admission: "
    watchdog = "memory watchdog observed: "
    if note.startswith(admission):
        parts = note[len(admission):].split("; ")
        if not 3 <= len(parts) <= 5 or parts[0] != "backend=native":
            return False
        fields = ("estimated_peak_bytes", "safe_headroom_bytes", "available_bytes", "margin_percent")
        previous = -1
        for index, part in enumerate(parts[1:]):
            key, separator, value = part.partition("=")
            if not separator or key not in fields:
                return False
            position = fields.index(key)
            if position <= previous or (index < 2 and position != index):
                return False
            if not _quality_note_uint(value, 100 if key == "margin_percent" else (1 << 63) - 1):
                return False
            previous = position
        return True
    if note.startswith(watchdog):
        parts = note[len(watchdog):].split("; ")
        if not 1 <= len(parts) <= 4:
            return False
        reasons = (
            "benchmark stopped: macOS memory pressure is warning",
            "benchmark stopped: macOS memory pressure is critical",
            "benchmark stopped: host memory fell below runtime reserve",
            "benchmark stopped: host swap grew by more than 256 MiB",
            "unclassified failure (text omitted)",
        )
        previous = -1
        for part in parts:
            if part in reasons:
                position = 0
            else:
                key, separator, value = part.partition("=")
                fields = ("pressure", "available_bytes", "swap_used_bytes")
                if not separator or key not in fields:
                    return False
                position = fields.index(key) + 1
                if (key == "pressure" and value not in ("normal", "warning", "critical")
                        or key != "pressure" and not _quality_note_uint(value)):
                    return False
            if position <= previous:
                return False
            previous = position
        return True
    return False


def _quality_memory_notes(error: CLIError) -> tuple[str, ...]:
    """Inspect only two trailing slots on the immediate cause and wrapper."""
    result = []
    try:
        if error.code not in ("QUALITY_MEMORY", "QUALITY_FAILED"):
            return ()
        for source in (error.__cause__, error):
            notes = getattr(source, "__notes__", None)
            if type(notes) is not list:
                continue
            for note in notes[-2:]:
                if _quality_memory_note(note) and not any(
                        item.partition(": ")[0] == note.partition(": ")[0] for item in result):
                    result.append(note)
                    if len(result) == 2:
                        return tuple(result)
    except BaseException:
        pass  # Observation must never replace the original CLI failure.
    return tuple(result)


def emit_error(error: CLIError, output_format: str, *, quality_notes: bool = False) -> None:
    message = error.message
    if quality_notes:
        try:
            notes = _quality_memory_notes(error)
            if notes:
                message = "\n".join((message, *notes))
        except BaseException:
            pass
    if output_format in {"json", "jsonl"}:
        payload = {
            "schema_version": ERROR_SCHEMA,
            "error": {
                "code": error.code,
                "details": {},
                "exit_code": error.exit_code,
                "message": message,
                "stage": error.stage,
            },
        }
        print(_json(payload), file=sys.stderr)
        return
    print(f"error[{error.code}]: {message}", file=sys.stderr)
