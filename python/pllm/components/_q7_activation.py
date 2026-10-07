"""Shared public-calibration contract for bounded Q7 activation methods."""

from collections.abc import Sequence

from pllm.configuration import ConfigurationError


def parameter_schema(*, wavelet: bool = False) -> dict:
    properties = {
        "max_pieces": {"type": "integer", "minimum": 1, "maximum": 8},
        "public_calibration_counts": {
            "type": "array", "minItems": 257, "maxItems": 257,
            "items": {"type": "integer", "minimum": 0, "maximum": 1_000_000},
        },
    }
    if wavelet:
        properties["levels"] = {"type": "integer", "minimum": 0, "maximum": 8}
    return {"type": "object", "properties": properties,
            "required": list(properties), "additionalProperties": False}


def parameters(max_pieces: int, public_calibration_counts: Sequence[int] | None) -> dict:
    if type(max_pieces) is not int or not 1 <= max_pieces <= 8:
        raise ConfigurationError("max_pieces must be an integer in [1, 8]")
    if public_calibration_counts is None:
        counts = [0] * 257  # The fitter adds one public pseudocount per code.
    else:
        if (not isinstance(public_calibration_counts, Sequence)
                or isinstance(public_calibration_counts, (str, bytes, bytearray))
                or len(public_calibration_counts) != 257):
            raise ConfigurationError("public_calibration_counts requires 257 public offline counts")
        counts = [public_calibration_counts[i] for i in range(257)]
        if (any(type(value) is not int or not 0 <= value <= 1_000_000 for value in counts)
                or sum(counts) > 100_000_000):
            raise ConfigurationError("public calibration counts exceed the bounded fitting contract")
    return {"max_pieces": max_pieces, "public_calibration_counts": counts}
