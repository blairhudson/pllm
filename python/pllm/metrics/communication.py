"""Normalise recorded protocol bodies without mixing token or lifecycle scopes."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def _count(value: object) -> int | None:
    return value if type(value) is int and value >= 0 else None


def _tokens(record: Mapping[str, Any]) -> int | None:
    tokens = record.get("tokens", {})
    if record.get("status") != "completed" or tokens.get("authoritative") is not True:
        return None
    return _count(tokens.get("output_tokens"))


def _sum(values: list[int | None]) -> int | None:
    if not values or any(value is None for value in values):
        return None
    return sum(value for value in values if value is not None)


def _mb(amount: int | None, tokens: int | None) -> float | None:
    return amount / tokens / 1_000_000 if amount is not None and tokens else None


def _body(ledger: Mapping[str, Any] | None, field: str) -> int | None:
    if (
        not isinstance(ledger, Mapping)
        or ledger.get("tracked_body_counter_set_present") is not True
    ):
        return None
    return _count(ledger.get(field))


def communication_per_token(report: Mapping[str, Any]) -> dict[str, Any]:
    """Return decimal MB/generated-token for a canonical loopback report.

    Online includes prefill. Covered adds in-window offline/distribution bodies.
    Setup-inclusive charges startup and discarded warmups once to measured output.
    Decode uses an observed first-output boundary and N-1; never an estimate from
    total bytes or from the requested output cap. Unknowns remain ``None``.
    """
    if report.get("schema_version") != "pllm.loopback_benchmark.v1":
        raise ValueError("communication_per_token requires a loopback benchmark report")
    records = report.get("runs", [])
    warmups = report.get("warmup_runs", [])
    if type(records) is not list or type(warmups) is not list or len(records) + len(warmups) > 1024:
        raise ValueError("communication report exceeds bounded run count")
    if any(
        not isinstance(row, Mapping)
        or not isinstance(row.get("tokens", {}), Mapping)
        or not isinstance(row.get("network", {}), Mapping)
        for row in (*records, *warmups)
    ):
        raise ValueError("invalid communication run record")
    accounting = report.get("topology_accounting") or {}
    if not isinstance(accounting, Mapping):
        raise ValueError("invalid communication accounting")
    ledgers = accounting.get("runs", [])
    warm_ledgers = accounting.get("warmups", [])
    if type(ledgers) is not list or type(warm_ledgers) is not list:
        raise ValueError("invalid communication accounting windows")
    aligned = len(ledgers) == len(records)
    runs = []
    for index, record in enumerate(records):
        tokens = _tokens(record)
        ledger = ledgers[index] if aligned else None
        online = _body(ledger, "online_all_link_serialized_body_bytes")
        covered = _body(ledger, "all_link_serialized_body_bytes")
        decode = record.get("network", {})
        decode_tokens = _count(decode.get("decode_output_tokens"))
        decode_bytes = _count(decode.get("decode_online_body_bytes"))
        if (
            tokens is None
            or tokens < 1
            or decode_tokens != tokens - 1
            or decode.get("decode_scope") != "after-first-output-through-completion"
            or online is None
            or decode_bytes is None
            or decode_bytes > online
        ):
            decode_tokens = decode_bytes = None
        runs.append(
            {
                "generated_output_tokens": tokens,
                "online_body_bytes": online,
                "covered_body_bytes": covered,
                "online_mb_per_output_token": _mb(online, tokens),
                "covered_mb_per_output_token": _mb(covered, tokens),
                "decode_output_tokens": decode_tokens,
                "decode_online_body_bytes": decode_bytes,
                "decode_online_mb_per_output_token": _mb(decode_bytes, decode_tokens),
            }
        )
    tokens = _sum([item["generated_output_tokens"] for item in runs])
    online = _sum([item["online_body_bytes"] for item in runs])
    covered = _sum([item["covered_body_bytes"] for item in runs])
    setup = (
        _sum(
            [
                _body(item, "all_link_serialized_body_bytes")
                for item in (
                    accounting.get("startup"),
                    *warm_ledgers,
                    *ledgers,
                )
            ]
        )
        if aligned and len(warm_ledgers) == len(warmups) and records
        else None
    )
    decode_tokens = _sum([item["decode_output_tokens"] for item in runs])
    decode_bytes = _sum([item["decode_online_body_bytes"] for item in runs])
    first = (warmups or records or [{}])[0]
    first_ledger = (warm_ledgers or ledgers or [None])[0]
    cold = (
        _sum(
            [
                _body(accounting.get("startup"), "all_link_serialized_body_bytes"),
                _body(first_ledger, "all_link_serialized_body_bytes"),
            ]
        )
        if aligned and len(warm_ledgers) == len(warmups)
        else None
    )
    return {
        "schema": "pllm.communication_per_generated_token.v1",
        "byte_unit": "decimal MB (1000000 bytes)",
        "denominator": "authoritative generated output tokens, not input tokens or output cap",
        "scope": "covered application bodies; no full wire or checkpoint download",
        "setup_scope": "startup and discarded warmups charged once to measured-run output tokens",
        "decode_scope": "online bodies after first output through completion, divided by N-1",
        "runs": runs,
        "summary": {
            "generated_output_tokens": tokens,
            "online_mb_per_output_token": _mb(online, tokens),
            "covered_mb_per_output_token": _mb(covered, tokens),
            "setup_inclusive_mb_per_output_token": _mb(setup, tokens),
            "cold_first_response_mb_per_output_token": _mb(cold, _tokens(first)),
            "decode_online_mb_per_output_token": _mb(decode_bytes, decode_tokens),
        },
    }
