"""Strict, allocation-aware public inputs for stateless forward execution."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

MAX_INPUT_JSON_BYTES = 1 << 20


class ForwardInputError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ForwardInputs:
    """Borrow exact [S] or [1,S] lists/arrays, without copying or coercion.

    Flat [S] is SDK shorthand. Admitted owned inputs always have native [1,S]
    geometry. Validation and snapshots belong to admission.
    """

    tokens: object
    positions: object = field(default=None, kw_only=True)
    segments: object = field(default=None, kw_only=True)
    attention_mask: object = field(default=None, kw_only=True)


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ForwardInputError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _constant(value):
    raise ForwardInputError(f"nonfinite JSON constant {value!r}")


def parse_forward_inputs(data: bytes) -> ForwardInputs:
    """Production JSON parser; accepts bounded UTF-8 bytes, never coercion hooks."""
    if type(data) is not bytes or not 0 < len(data) <= MAX_INPUT_JSON_BYTES:
        raise ForwardInputError("forward input JSON must be 1..1048576 bytes")
    try:
        value = json.loads(data, object_pairs_hook=_object, parse_constant=_constant)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise ForwardInputError("invalid forward input JSON") from exc
    allowed = {"schema_version", "tokens", "positions", "segments", "attention_mask"}
    if (type(value) is not dict or set(value) - allowed
            or value.get("schema_version") != "pllm.forward_inputs.v1" or "tokens" not in value
            or any(value[name] is None for name in value if name not in {"schema_version", "tokens"})):
        raise ForwardInputError("invalid pllm.forward_inputs.v1 fields")
    result = ForwardInputs(value["tokens"], **{k: v for k, v in value.items()
                                             if k not in {"schema_version", "tokens"}})
    # Reject incompatible geometry across every field before reading scalars.
    length = _geometry(result)
    _validate_values(result, length, domains=None)
    return result


def read_forward_inputs(path: str | Path) -> ForwardInputs:
    """Read at most limit+1 bytes before parsing (also safe for special files)."""
    with Path(path).open("rb") as stream:
        data = stream.read(MAX_INPUT_JSON_BYTES + 1)
    return parse_forward_inputs(data)


def _list_row(value):
    """Borrow a literal scalar row; never flatten or call container hooks."""
    if type(value) is not list or not value:
        raise ForwardInputError("inputs require nonempty exact [S] or [1,S] lists")
    try:
        first = value[0]
    except IndexError as exc:
        raise ForwardInputError("input list changed during validation") from exc
    if type(first) is list:
        if len(value) != 1 or not first:
            raise ForwardInputError("nested inputs require exactly one nonempty batch row")
        return first
    return value


def _extent(value, *, mask):
    """Inspect structure only: never traverse a caller-owned scalar row."""
    if type(value) is list:
        row = _list_row(value)
        size = len(row)
        if size < 1:
            raise ForwardInputError("inputs require nonempty literal int/bool rows")
        return size
    if type(value) is not np.ndarray:
        raise ForwardInputError("inputs require exact list or numpy.ndarray containers")
    if (value.dtype != np.dtype(np.bool_ if mask else np.int64)
            or not value.dtype.isnative or value.ndim not in (1, 2)
            or (value.ndim == 2 and value.shape[0] != 1) or value.shape[-1] < 1):
        raise ForwardInputError("inputs require native int64/bool [S] or [1,S] arrays")
    return value.shape[-1]


def _geometry(inputs, *, maximum=None, sequence=None):
    """Admit all extents before any value check or destination allocation."""
    if type(inputs) is not ForwardInputs:
        raise ForwardInputError("forward requires exact ForwardInputs")
    size = _extent(inputs.tokens, mask=False)
    if maximum is not None and not 1 <= size <= maximum:
        raise ForwardInputError("sequence length exceeds forward budget")
    if sequence is not None and size != sequence:
        raise ForwardInputError("inputs changed during admission")
    for name in ("positions", "segments", "attention_mask"):
        value = getattr(inputs, name)
        if value is not None and _extent(value, mask=name == "attention_mask") != size:
            raise ForwardInputError("forward input lengths differ")
    return size


def _scalar(value, index, *, mask, bound):
    """Read one bounded index; exact scalar type precedes all scalar operations."""
    literal = type(value) is list
    try:
        if literal:
            item = _list_row(value)[index]
        else:
            item = value[index] if value.ndim == 1 else value[0, index]
    except (IndexError, TypeError) as exc:
        raise ForwardInputError("inputs changed during scalar access") from exc
    wanted = (bool if mask else int) if literal else (np.bool_ if mask else np.int64)
    if type(item) is not wanted:
        raise ForwardInputError("inputs require exact native int64/bool scalars")
    if mask:
        return bool(item)
    if literal and not -(1 << 63) <= item < (1 << 63):
        raise ForwardInputError("input integer exceeds int64")
    if bound is not None and not 0 <= item < bound:
        raise ForwardInputError("input IDs exceed their lookup table domain")
    return int(item)


def _validate_values(inputs, sequence, *, domains):
    for name in ("tokens", "positions", "segments", "attention_mask"):
        value = getattr(inputs, name)
        if value is None:
            continue
        mask = name == "attention_mask"
        if _extent(value, mask=mask) != sequence:
            raise ForwardInputError("inputs changed during validation")
        bound = None if mask or domains is None else domains[name]
        present = False
        # Fixed count, not an iterator that could follow concurrent list growth.
        for index in range(sequence):
            item = _scalar(value, index, mask=mask, bound=bound)
            if mask:
                present = item or present
        if _extent(value, mask=mask) != sequence:
            raise ForwardInputError("inputs changed during validation")
        if mask and not present:
            raise ForwardInputError("all-false attention mask is unsupported")
    _geometry(inputs, sequence=sequence)


def inspect_forward_inputs(inputs, *, maximum, domains):
    """No input-sized arrays, casts, flattening, or unbounded row iteration."""
    size = _geometry(inputs, maximum=maximum)
    if inputs.positions is None and size > domains["positions"]:
        raise ForwardInputError("default positions exceed their lookup table domain")
    _validate_values(inputs, size, domains=domains)
    return size


def snapshot_forward_inputs(inputs, *, maximum, domains, sequence):
    """Called only after quote and host admission. Revalidate each owned copy."""
    if inspect_forward_inputs(inputs, maximum=maximum, domains=domains) != sequence:
        raise ForwardInputError("inputs changed during admission")
    result = {}
    for name in ("tokens", "positions", "segments", "attention_mask"):
        _geometry(inputs, maximum=maximum, sequence=sequence)
        value = getattr(inputs, name)
        mask = name == "attention_mask"
        # Caller mutation cannot determine any allocation extent, even after
        # reinspection. Never copy/flatten/reshape a mutable caller container.
        array = np.empty((1, sequence), dtype=np.bool_ if mask else np.int64)
        if value is not None and _extent(value, mask=mask) != sequence:
            raise ForwardInputError("inputs changed during snapshot")
        for index in range(sequence):
            if value is None:
                item = index if name == "positions" else (True if mask else 0)
            else:
                item = _scalar(value, index, mask=mask,
                               bound=None if mask else domains[name])
            array[0, index] = item
        if value is not None and _extent(value, mask=mask) != sequence:
            raise ForwardInputError("inputs changed during snapshot")
        result["input." + name] = array
    _geometry(inputs, maximum=maximum, sequence=sequence)
    snapshot = ForwardInputs(result["input.tokens"], positions=result["input.positions"],
                             segments=result["input.segments"],
                             attention_mask=result["input.attention_mask"])
    inspect_forward_inputs(snapshot, maximum=maximum, domains=domains)
    for array in result.values():
        array.flags.writeable = False
    return result


__all__ = ["ForwardInputs", "ForwardInputError", "parse_forward_inputs", "read_forward_inputs"]
