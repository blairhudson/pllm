"""Matched prepared setup/lifecycle ablations; explicit public artifact inputs."""
from benchmarks.research import sota
from pllm.preparation import PreparedInventory
from pllm.state import ClientPrefixReuse


def serial():
    return sota.experiment().with_params(name="paged-demand-serial")


def windowed():
    e = serial()
    return e.with_params(name="paged-demand-window64", pipeline=e.pipeline.with_params(
        inventory=PreparedInventory(refill="on-demand", allocation="demand", stage_window=4,
                                    stage_window_bytes=64 << 20)))


def reused():
    e = windowed()
    return e.with_params(name="paged-demand-window64-reuse", pipeline=e.pipeline.with_params(
        cache=ClientPrefixReuse(max_bytes=32 << 20, fixed_input_tokens=256)))


def resident():
    e = windowed()
    return e.with_params(name="resident-demand-window64", pipeline=e.pipeline.with_params(
        delivery=e.pipeline.components["delivery"].with_params(storage="memory")))
