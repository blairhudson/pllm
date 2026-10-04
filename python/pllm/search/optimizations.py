"""Bounded model-neutral proposals for the existing placement planner."""

from __future__ import annotations

import itertools
import json
import math
from typing import TYPE_CHECKING

from pllm.configuration import Experiment, Pipeline

if TYPE_CHECKING:
    from pllm.search import SearchSpace


def optimization_space(
    experiment: Experiment,
    *,
    allow_client_weights: bool = False,
    client_prefix_layers: tuple[int, ...] = (),
    prefix_cache_bytes: int = 0,
    metal_min_rows: int | None = None,
    max_candidates: int = 64,
) -> SearchSpace:
    """Propose choices without declaring their cost or enabling them.

    Model, precision, verification and topology remain fixed. Additional client
    weights and equality-revealing cache reuse require explicit permission.
    Native model/placement admission remains the compatibility authority.
    """
    from pllm.kernels import AppleMetal
    from pllm.preparation import PreparedInventory
    from pllm.protocols import ClientBundleTransport, TwoOnlineOffsetLinear
    from pllm.roles import ClientLinearRoles, ClientPrefixLayers
    from pllm.search import SearchSpace
    from pllm.state import ClientPrefixReuse

    if not isinstance(experiment, Experiment):
        raise TypeError("optimization_space requires an Experiment")
    if type(allow_client_weights) is not bool:
        raise TypeError("allow_client_weights must be boolean")
    if (type(client_prefix_layers) is not tuple or len(client_prefix_layers) > 8
            or len(set(client_prefix_layers)) != len(client_prefix_layers)
            or any(type(value) is not int or not 1 <= value <= 8 for value in client_prefix_layers)):
        raise ValueError("client_prefix_layers must be a unique tuple of up to eight bounds in [1,8]")
    if client_prefix_layers and not allow_client_weights:
        raise ValueError("prefix placement proposals require permission for client weights")
    if type(prefix_cache_bytes) is not int or prefix_cache_bytes < 0:
        raise ValueError("prefix_cache_bytes must be a non-negative integer")
    if type(max_candidates) is not int or not 1 <= max_candidates <= 1024:
        raise ValueError("max_candidates must be in [1,1024]")
    experiment.resolve()
    original = dict(experiment.pipeline.components)
    axes = {}

    def choices(slot, *values):
        unique = {}
        for row in (original.get(slot), *values):
            key = json.dumps(None if row is None else row.to_spec(), sort_keys=True)
            unique.setdefault(key, row)
        axes[slot] = tuple(unique.values())

    if metal_min_rows is not None:
        choices("kernels", AppleMetal(min_rows=metal_min_rows))
    linear = original.get("linear")
    prepared = linear is not None and linear.component == "pllm/masked-linear"
    offset = linear is not None and linear.component == TwoOnlineOffsetLinear.descriptor.component
    if prepared or offset:
        choices("delivery", None, ClientBundleTransport("zlib"), ClientBundleTransport("artifacts"),
                ClientBundleTransport("artifacts", compression="zlib"))
        if prefix_cache_bytes:
            choices("cache", ClientPrefixReuse(max_bytes=prefix_cache_bytes,
                    fixed_input_tokens=experiment.budget.max_input_tokens))
    if prepared:
        choices("inventory", PreparedInventory("request-sized", rows=1, refill="on-demand"))
        if allow_client_weights:
            roles = ("qkv_projection", "attention_output")
            choices("placement", ClientLinearRoles(roles), *(
                component for layers in client_prefix_layers
                for component in (ClientPrefixLayers(layers), ClientLinearRoles(roles, prefix_layers=layers))))
    elif offset:
        if allow_client_weights:
            raise ValueError("worker topology cannot select client body placement")
        choices(
            "linear",
            *(
                TwoOnlineOffsetLinear(
                    input_encoding=input_encoding, output_encoding=output_encoding
                )
                for input_encoding in ("raw", "seeded")
                for output_encoding in ("raw", "row_residues")
            ),
        )
    elif allow_client_weights or prefix_cache_bytes:
        raise ValueError("requested ownership/cache proposals require ordinary prepared execution")

    size = math.prod(len(values) for values in axes.values())
    if size > max_candidates:
        raise ValueError(f"optimization space needs {size} candidates; bound is {max_candidates}")
    pipelines = {}
    for values in itertools.product(*axes.values()):
        components = dict(original)
        for slot, value in zip(axes, values, strict=True):
            if value is None:
                components.pop(slot, None)
            else:
                components[slot] = value
        pipeline = Pipeline(model=experiment.pipeline.model, components=components)
        pipelines[pipeline.digest()] = pipeline
    return SearchSpace(experiment, {"pipeline": tuple(pipelines[key] for key in sorted(pipelines))})


__all__ = ["optimization_space"]
