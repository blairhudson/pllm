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


def emit_error(error: CLIError, output_format: str) -> None:
    if output_format in {"json", "jsonl"}:
        payload = {
            "schema_version": ERROR_SCHEMA,
            "error": {
                "code": error.code,
                "details": {},
                "exit_code": error.exit_code,
                "message": error.message,
                "stage": error.stage,
            },
        }
        print(_json(payload), file=sys.stderr)
        return
    print(f"error[{error.code}]: {error.message}", file=sys.stderr)
